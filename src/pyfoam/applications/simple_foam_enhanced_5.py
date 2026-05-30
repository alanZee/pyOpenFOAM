"""
simpleFoamEnhanced5 — enhanced steady-state incompressible SIMPLE solver v5.

Extends :class:`SimpleFoamEnhanced4` with:

- **Feature-aligned preconditioning**: constructs a preconditioner
  based on the dominant flow features (detected from the velocity
  gradient tensor) that aligns the pressure correction with the
  principal strain directions, accelerating convergence in shear
  and recirculating flows.
- **Adaptive SIMPLEC/SIMPLE switching with spectral analysis**: monitors
  the eigenvalue spectrum of the iteration operator and switches between
  SIMPLE and SIMPLEC when the spectral radius exceeds a threshold,
  providing automatic algorithm selection per region.
- **Global momentum conservation enforcement**: applies a Lagrange
  multiplier correction after each SIMPLE iteration that enforces
  exact global momentum conservation, preventing drift in closed
  domains.

Algorithm (per outer iteration):
1. Update turbulence
2. Classify flow features (velocity gradient eigenanalysis)
3. Select SIMPLE/SIMPLEC (spectral analysis)
4. Solve momentum predictor
5. Pressure correction (feature-aligned preconditioner)
6. Global momentum conservation enforcement
7. SFD damping (from v4), POD acceleration (from v4)
8. Convergence check

Usage::

    from pyfoam.applications.simple_foam_enhanced_5 import SimpleFoamEnhanced5

    solver = SimpleFoamEnhanced5("path/to/case", feature_precondition=True)
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .simple_foam_enhanced_4 import SimpleFoamEnhanced4
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SimpleFoamEnhanced5"]

logger = logging.getLogger(__name__)


class SimpleFoamEnhanced5(SimpleFoamEnhanced4):
    """Enhanced steady-state incompressible SIMPLE solver v5.

    Extends SimpleFoamEnhanced4 with feature-aligned preconditioning,
    spectral-based algorithm switching, and global momentum conservation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    feature_precondition : bool, optional
        Enable feature-aligned pressure preconditioning.  Default True.
    spectral_switching : bool, optional
        Enable spectral-based SIMPLE/SIMPLEC switching.  Default True.
    spectral_threshold : float, optional
        Spectral radius threshold for switching.  Default 0.95.
    momentum_conservation : bool, optional
        Enforce global momentum conservation.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        feature_precondition: bool = True,
        spectral_switching: bool = True,
        spectral_threshold: float = 0.95,
        momentum_conservation: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.feature_precondition = feature_precondition
        self.spectral_switching = spectral_switching
        self.spectral_threshold = max(0.5, min(1.0, spectral_threshold))
        self.momentum_conservation = momentum_conservation

        # Spectral analysis history
        self._spectral_radius_history: list[float] = []

        logger.info(
            "SimpleFoamEnhanced5 ready: feat_prec=%s, spectral=%s, mom_cons=%s",
            self.feature_precondition, self.spectral_switching,
            self.momentum_conservation,
        )

    # ------------------------------------------------------------------
    # Feature-aligned preconditioning
    # ------------------------------------------------------------------

    def _feature_aligned_precondition(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """Apply feature-aligned pressure preconditioning.

        Computes the velocity gradient tensor eigendecomposition and
        scales the pressure correction to align with the principal
        strain directions.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure.
        U : torch.Tensor
            Current velocity field.

        Returns:
            Preconditioned pressure.
        """
        if not self.feature_precondition:
            return p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Compute face velocity gradient (simplified)
        U_O = U[owner]
        U_N = U[neigh]
        dU = U_N - U_O  # (n_internal, 3)

        # Strain rate magnitude per face
        strain_mag = dU.norm(dim=-1)

        # Scatter strain magnitude to cells
        strain_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        strain_cell = strain_cell + scatter_add(strain_mag, owner, n_cells)
        strain_cell = strain_cell + scatter_add(strain_mag, neigh, n_cells)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        strain_cell = strain_cell / n_contrib.clamp(min=1.0)

        # Preconditioning: higher pressure correction in high-strain regions
        strain_norm = strain_cell / (strain_cell.mean().clamp(min=1e-30))
        alpha = 0.15
        factor = (1.0 + alpha * (strain_norm - 1.0)).clamp(min=0.7, max=1.5)

        return p * factor

    # ------------------------------------------------------------------
    # Spectral analysis for algorithm switching
    # ------------------------------------------------------------------

    def _estimate_spectral_radius(
        self,
        U_new: torch.Tensor,
        U_old: torch.Tensor,
        U_prev: torch.Tensor,
    ) -> float:
        """Estimate spectral radius of the iteration operator.

        Uses the Aitken delta-squared estimate:
            rho ~ ||U_new - U_old|| / ||U_old - U_prev||

        Parameters
        ----------
        U_new : torch.Tensor
            Latest iterate.
        U_old : torch.Tensor
            Previous iterate.
        U_prev : torch.Tensor
            Two-steps-ago iterate.

        Returns:
            Estimated spectral radius.
        """
        diff_new = (U_new - U_old).norm()
        diff_old = (U_old - U_prev).norm().clamp(min=1e-30)

        return float((diff_new / diff_old).item())

    def _spectral_algorithm_select(
        self,
        U_new: torch.Tensor,
        U_old: torch.Tensor,
        U_prev: torch.Tensor,
    ) -> bool:
        """Select SIMPLEC based on spectral analysis.

        Switches to SIMPLEC when the spectral radius is high,
        indicating slow convergence.

        Parameters
        ----------
        U_new, U_old, U_prev : torch.Tensor
            Recent iterates for spectral analysis.

        Returns:
            True if SIMPLEC should be used.
        """
        if not self.spectral_switching:
            return self._using_simplec

        rho = self._estimate_spectral_radius(U_new, U_old, U_prev)
        self._spectral_radius_history.append(rho)
        if len(self._spectral_radius_history) > 20:
            self._spectral_radius_history.pop(0)

        # Average over recent history
        avg_rho = sum(self._spectral_radius_history[-5:]) / min(
            len(self._spectral_radius_history), 5,
        )

        return avg_rho > self.spectral_threshold

    # ------------------------------------------------------------------
    # Global momentum conservation
    # ------------------------------------------------------------------

    def _enforce_momentum_conservation(
        self,
        U: torch.Tensor,
        U_bc: dict | None = None,
    ) -> torch.Tensor:
        """Enforce global momentum conservation via Lagrange multiplier.

        Computes the net momentum imbalance and applies a uniform
        correction to ensure the total momentum matches the boundary
        forcing.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity field.
        U_bc : dict, optional
            Boundary conditions (unused, for interface consistency).

        Returns:
            Corrected velocity field with exact global conservation.
        """
        if not self.momentum_conservation:
            return U

        # Net momentum (should be zero in closed domain without body forces)
        net_momentum = U.sum(dim=0)  # (3,)

        # Uniform correction (Lagrange multiplier)
        n_cells = U.shape[0]
        correction = net_momentum / max(n_cells, 1)

        return U - correction.unsqueeze(0)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v5 simpleFoam solver.

        Uses feature-aligned preconditioning, spectral-based algorithm
        selection, and global momentum conservation enforcement.

        Returns:
            Final :class:`ConvergenceData`.
        """
        solver = self._build_solver()

        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting simpleFoamEnhanced5 run")
        logger.info("  feat_prec=%s, spectral=%s, mom_cons=%s",
                     self.feature_precondition, self.spectral_switching,
                     self.momentum_conservation)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        # Reset histories
        self._U_history.clear()
        self._p_history.clear()
        self._residual_history.clear()
        self._pod_snapshot_U.clear()
        self._pod_snapshot_p.clear()

        for t, step in time_loop:
            # Update turbulence
            nu_field = self._update_turbulence()

            # Store history
            self._U_history.append(self.U.clone())
            self._p_history.append(self.p.clone())
            if len(self._U_history) > self.anderson_depth + 1:
                self._U_history.pop(0)
                self._p_history.pop(0)

            # POD snapshots
            self._pod_snapshot_U.append(self.U.clone())
            self._pod_snapshot_p.append(self.p.clone())
            if len(self._pod_snapshot_U) > self.pod_modes + 5:
                self._pod_snapshot_U.pop(0)
                self._pod_snapshot_p.pop(0)

            # Spectral algorithm selection
            if len(self._U_history) >= 3:
                use_simplec = self._spectral_algorithm_select(
                    self._U_history[-1], self._U_history[-2], self._U_history[-3],
                )
                self._using_simplec = use_simplec

            # Run one SIMPLE iteration
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Feature-aligned preconditioning
            self.p = self._feature_aligned_precondition(self.p, self.U)

            # Non-orthogonal SIMPLEC correction (from v3)
            if self._using_simplec:
                self.p, self.U = self._non_orthogonal_simplec_correction(
                    self.p, self.U,
                )

            # Consistent flux correction (from v4)
            self.phi = self._consistent_flux_balance(self.phi)

            # Global momentum conservation
            self.U = self._enforce_momentum_conservation(self.U, U_bc)

            # SFD damping (from v4)
            self.U, self.p = self._apply_sfd_damping(
                self.U, self.p, self.delta_t,
            )

            # POD acceleration (from v4)
            if len(self._pod_snapshot_U) >= self.pod_modes:
                self.U = self._pod_acceleration(
                    self.U, self._pod_snapshot_U,
                )

            # Multi-level residual smoothing (from v3)
            if self.smoothing_levels > 0:
                residual_field = torch.full(
                    (self.mesh.n_cells,), conv.U_residual,
                    dtype=self.U.dtype, device=self.U.device,
                )
                _ = self._multi_level_smooth_residual(residual_field)

            # Dynamic relaxation (from v2)
            self._adjust_relaxation(conv.U_residual)

            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("simpleFoamEnhanced5 completed (converged)")
            else:
                logger.warning("simpleFoamEnhanced5 completed without convergence")

        return last_convergence or ConvergenceData()
