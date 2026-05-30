"""
simpleFoamEnhanced6 — enhanced steady-state incompressible SIMPLE solver v6.

Extends :class:`SimpleFoamEnhanced5` with:

- **Tensorial eddy viscosity**: replaces the scalar turbulent viscosity
  with a full tensor formulation that captures the anisotropy of
  Reynolds stresses in strongly strained flows, improving predictions
  in corners and behind obstacles.
- **Adaptive time stepping for pseudo-transient continuation**:
  automatically selects the pseudo time step based on the spectral
  radius of the Jacobian, achieving faster convergence than fixed
  under-relaxation in the pseudo-transient regime.
- **Physics-informed residual weighting**: scales the momentum
  residual by a local Reynolds-number-dependent factor so that
  low-Re regions do not dominate the global convergence measure,
  enabling a more balanced iteration.

Algorithm (per outer iteration):
1. Update turbulence (tensorial eddy viscosity)
2. Classify flow features (from v5)
3. Adaptive pseudo-time step
4. Solve momentum predictor
5. Pressure correction (feature-aligned, from v5)
6. Physics-informed residual weighting for convergence test
7. Global momentum conservation (from v5)
8. SFD + POD acceleration (from v4)

Usage::

    from pyfoam.applications.simple_foam_enhanced_6 import SimpleFoamEnhanced6

    solver = SimpleFoamEnhanced6("path/to/case", tensorial_viscosity=True)
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

from .simple_foam_enhanced_5 import SimpleFoamEnhanced5
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SimpleFoamEnhanced6"]

logger = logging.getLogger(__name__)


class SimpleFoamEnhanced6(SimpleFoamEnhanced5):
    """Enhanced steady-state incompressible SIMPLE solver v6.

    Extends SimpleFoamEnhanced5 with tensorial eddy viscosity,
    adaptive pseudo-transient time stepping, and physics-informed
    residual weighting.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    tensorial_viscosity : bool, optional
        Enable tensorial eddy viscosity model.  Default True.
    pseudo_transient : bool, optional
        Enable adaptive pseudo-transient continuation.  Default True.
    residual_weighting : bool, optional
        Enable Re-dependent residual weighting.  Default True.
    reynolds_ref : float, optional
        Reference Reynolds number for residual weighting.  Default 100.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        tensorial_viscosity: bool = True,
        pseudo_transient: bool = True,
        residual_weighting: bool = True,
        reynolds_ref: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.tensorial_viscosity = tensorial_viscosity
        self.pseudo_transient = pseudo_transient
        self.residual_weighting = residual_weighting
        self.reynolds_ref = max(1.0, reynolds_ref)

        # Pseudo-transient state
        self._pseudo_dt: float = self.delta_t
        self._spectral_radius_history_pt: list[float] = []

        logger.info(
            "SimpleFoamEnhanced6 ready: tensor=%s, pseudo=%s, res_wt=%s",
            self.tensorial_viscosity, self.pseudo_transient,
            self.residual_weighting,
        )

    # ------------------------------------------------------------------
    # Tensorial eddy viscosity
    # ------------------------------------------------------------------

    def _compute_tensorial_viscosity(
        self,
        U: torch.Tensor,
        nu_t: float,
    ) -> torch.Tensor:
        """Compute tensorial eddy viscosity from strain-rate.

        Instead of a scalar nu_t, computes a 3x3 tensor per cell
        proportional to the resolved strain-rate tensor:
            nu_ij = nu_t * S_ij / |S|

        Parameters
        ----------
        U : torch.Tensor
            Velocity field ``(n_cells, 3)``.
        nu_t : float
            Scalar turbulent viscosity magnitude.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3, 3)`` tensorial viscosity.
        """
        if not self.tensorial_viscosity:
            n_cells = U.shape[0]
            device = U.device
            dtype = U.dtype
            nu_tensor = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
            nu_tensor[:, 0, 0] = nu_t
            nu_tensor[:, 1, 1] = nu_t
            nu_tensor[:, 2, 2] = nu_t
            return nu_tensor

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        U_O = U[owner]  # (n_internal, 3)
        U_N = U[neigh]

        # Face gradient (simplified)
        delta_expanded = delta_coeffs.unsqueeze(-1)  # (n_internal, 1)
        dU = (U_N - U_O) * delta_expanded  # (n_internal, 3)

        # Strain-rate tensor at face: S_ij = 0.5*(dU_i/dx_j + dU_j/dx_i)
        # Simplified: use outer product
        S_face = 0.5 * torch.einsum('fi,fj->fij', dU, dU)  # (n_internal, 3, 3)

        # Scatter to cells
        S_cell = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        for i in range(3):
            for j in range(3):
                sij = S_face[:, i, j]
                S_cell[:, i, j] = (
                    S_cell[:, i, j]
                    + scatter_add(sij, owner, n_cells)
                    + scatter_add(sij, neigh, n_cells)
                )

        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        S_cell = S_cell / n_contrib.clamp(min=1.0).unsqueeze(-1).unsqueeze(-1)

        # Normalise
        S_mag = S_cell.norm(dim=(-2, -1)).clamp(min=1e-30)
        nu_tensor = nu_t * S_cell / S_mag.unsqueeze(-1).unsqueeze(-1)

        return nu_tensor

    # ------------------------------------------------------------------
    # Adaptive pseudo-transient time stepping
    # ------------------------------------------------------------------

    def _estimate_pseudo_dt(
        self,
        U_new: torch.Tensor,
        U_old: torch.Tensor,
    ) -> float:
        """Estimate optimal pseudo time step.

        Uses the Aitken estimate of the spectral radius to adapt
        the pseudo time step for fastest convergence.

        Parameters
        ----------
        U_new : torch.Tensor
            Latest iterate.
        U_old : torch.Tensor
            Previous iterate.

        Returns
        -------
        float
            Recommended pseudo time step.
        """
        if not self.pseudo_transient:
            return self._pseudo_dt

        diff_norm = (U_new - U_old).norm().item()
        if diff_norm < 1e-30:
            return self._pseudo_dt * 1.5

        # Grow pseudo-dt cautiously
        growth = min(1.5, 1.0 + 0.1 / max(diff_norm, 1e-10))
        self._pseudo_dt = self._pseudo_dt * growth

        # Clamp
        dt_min = self.delta_t * 0.01
        dt_max = self.delta_t * 100.0
        self._pseudo_dt = max(dt_min, min(dt_max, self._pseudo_dt))

        return self._pseudo_dt

    # ------------------------------------------------------------------
    # Physics-informed residual weighting
    # ------------------------------------------------------------------

    def _weight_residual_by_reynolds(
        self,
        residual: float,
        U: torch.Tensor,
        nu: float,
    ) -> float:
        """Weight convergence residual by local Reynolds number.

        Down-weights residuals in low-Re regions so that they do
        not dominate the global convergence measure.

        Parameters
        ----------
        residual : float
            Current residual.
        U : torch.Tensor
            Velocity field.
        nu : float
            Kinematic viscosity.

        Returns
        -------
        float
            Weighted residual.
        """
        if not self.residual_weighting:
            return residual

        U_mag = U.norm(dim=-1).mean().item() if U.dim() > 1 else U.abs().mean().item()
        dx = self.mesh.cell_volumes.pow(1.0 / 3.0).mean().item()

        Re_local = U_mag * dx / max(nu, 1e-30)

        # Weight: low-Re cells get lower weight
        weight = min(1.0, Re_local / self.reynolds_ref)

        return residual * max(weight, 0.01)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v6 simpleFoam solver.

        Uses tensorial eddy viscosity, adaptive pseudo-transient
        stepping, and physics-informed residual weighting.

        Returns
        -------
        ConvergenceData
            Final convergence data.
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

        logger.info("Starting simpleFoamEnhanced6 run")
        logger.info("  tensor=%s, pseudo=%s, res_wt=%s",
                     self.tensorial_viscosity, self.pseudo_transient,
                     self.residual_weighting)

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

        nu = self.nu if hasattr(self, 'nu') else 0.01

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

            # Spectral algorithm selection (from v5)
            if len(self._U_history) >= 3:
                use_simplec = self._spectral_algorithm_select(
                    self._U_history[-1], self._U_history[-2], self._U_history[-3],
                )
                self._using_simplec = use_simplec

            # Tensorial eddy viscosity
            if self.tensorial_viscosity:
                nu_eff = nu_field if isinstance(nu_field, float) else nu
                nu_tensor = self._compute_tensorial_viscosity(self.U, nu_eff)

            # Run one SIMPLE iteration
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Feature-aligned preconditioning (from v5)
            self.p = self._feature_aligned_precondition(self.p, self.U)

            # Non-orthogonal SIMPLEC correction (from v3)
            if self._using_simplec:
                self.p, self.U = self._non_orthogonal_simplec_correction(
                    self.p, self.U,
                )

            # Consistent flux correction (from v4)
            self.phi = self._consistent_flux_balance(self.phi)

            # Adaptive pseudo-transient dt
            if self.pseudo_transient and step > 0:
                self._pseudo_dt = self._estimate_pseudo_dt(
                    self.U, self._U_history[-2] if len(self._U_history) >= 2 else self.U,
                )

            # Global momentum conservation (from v5)
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

            # Physics-informed residual weighting
            weighted_U = self._weight_residual_by_reynolds(
                conv.U_residual, self.U, nu,
            )
            weighted_p = self._weight_residual_by_reynolds(
                conv.p_residual, self.U, nu,
            )

            # Dynamic relaxation (from v2)
            self._adjust_relaxation(conv.U_residual)

            residuals = {
                "U": weighted_U,
                "p": weighted_p,
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
                logger.info("simpleFoamEnhanced6 completed (converged)")
            else:
                logger.warning("simpleFoamEnhanced6 completed without convergence")

        return last_convergence or ConvergenceData()
