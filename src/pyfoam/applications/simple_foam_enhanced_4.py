"""
simpleFoamEnhanced4 — enhanced steady-state incompressible SIMPLE solver v4.

Extends :class:`SimpleFoamEnhanced3` with:

- **Proper Orthogonal Decomposition (POD) acceleration**: extracts dominant
  modes from the residual history and uses them to accelerate convergence
  by projecting the solution onto a low-dimensional subspace.
- **Selective frequency damping (SFD)**: applies a temporal low-pass filter
  to the velocity field to damp out high-frequency oscillations while
  preserving the steady-state solution.
- **SIMPLEC with consistent flux correction**: extends v3's non-orthogonal
  SIMPLEC with a global flux balance correction that ensures net mass
  conservation across all boundaries after each pressure solve.

Algorithm (per outer iteration):
1. Solve momentum predictor
2. SIMPLEC pressure equation (with non-orthogonal correction from v3)
3. Consistent flux correction
4. SFD damping
5. POD acceleration
6. Turbulence update and convergence check

Usage::

    from pyfoam.applications.simple_foam_enhanced_4 import SimpleFoamEnhanced4

    solver = SimpleFoamEnhanced4("path/to/case", pod_modes=5, sfd_enabled=True)
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

from .simple_foam_enhanced_3 import SimpleFoamEnhanced3
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SimpleFoamEnhanced4"]

logger = logging.getLogger(__name__)


class SimpleFoamEnhanced4(SimpleFoamEnhanced3):
    """Enhanced steady-state incompressible SIMPLE solver v4.

    Extends SimpleFoamEnhanced3 with POD acceleration, selective
    frequency damping, and consistent flux correction.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    pod_modes : int, optional
        Number of POD modes for acceleration.  Default 5.
    sfd_enabled : bool, optional
        Enable selective frequency damping.  Default True.
    sfd_coeff : float, optional
        SFD filter coefficient.  Default 0.1.
    flux_correction : bool, optional
        Enable global flux balance correction.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        pod_modes: int = 5,
        sfd_enabled: bool = True,
        sfd_coeff: float = 0.1,
        flux_correction: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.pod_modes = max(2, min(20, pod_modes))
        self.sfd_enabled = sfd_enabled
        self.sfd_coeff = max(0.01, min(1.0, sfd_coeff))
        self.flux_correction = flux_correction

        # POD history
        self._pod_snapshot_U: list[torch.Tensor] = []
        self._pod_snapshot_p: list[torch.Tensor] = []

        # SFD filter state
        self._sfd_U_filtered: torch.Tensor | None = None
        self._sfd_p_filtered: torch.Tensor | None = None

        logger.info(
            "SimpleFoamEnhanced4 ready: pod=%d, sfd=%s, flux_corr=%s",
            self.pod_modes, self.sfd_enabled, self.flux_correction,
        )

    # ------------------------------------------------------------------
    # Selective frequency damping
    # ------------------------------------------------------------------

    def _apply_sfd_damping(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply selective frequency damping.

        Uses a temporal low-pass filter:
            U_filtered = U_filtered + dt/delta * (U - U_filtered)
            U_damped = U - chi * (U - U_filtered)

        where chi controls the damping strength.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity.
        p : torch.Tensor
            Current pressure.
        dt : float
            Time step.

        Returns:
            Tuple of (damped_U, damped_p).
        """
        if not self.sfd_enabled:
            return U, p

        # Initialise filter on first call
        if self._sfd_U_filtered is None:
            self._sfd_U_filtered = U.clone()
            self._sfd_p_filtered = p.clone()
            return U, p

        # Update filter (exponential moving average)
        alpha = self.sfd_coeff * dt
        alpha = min(alpha, 0.5)  # Limit filter update rate

        self._sfd_U_filtered = (1.0 - alpha) * self._sfd_U_filtered + alpha * U
        self._sfd_p_filtered = (1.0 - alpha) * self._sfd_p_filtered + alpha * p

        # Damped fields
        chi = 0.5 * self.sfd_coeff
        U_damped = U - chi * (U - self._sfd_U_filtered)
        p_damped = p - chi * (p - self._sfd_p_filtered)

        return U_damped, p_damped

    # ------------------------------------------------------------------
    # POD acceleration
    # ------------------------------------------------------------------

    def _pod_acceleration(
        self,
        U: torch.Tensor,
        pod_snapshots: list[torch.Tensor],
    ) -> torch.Tensor:
        """Apply POD-based convergence acceleration.

        Extracts the dominant POD modes from the snapshot history
        and projects the current solution onto the subspace spanned
        by these modes.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity iterate.
        pod_snapshots : list[torch.Tensor]
            History of velocity iterates.

        Returns:
            POD-accelerated velocity.
        """
        n_snapshots = len(pod_snapshots)
        if n_snapshots < self.pod_modes:
            return U

        # Build snapshot matrix (flattened)
        X = torch.stack([s.flatten() for s in pod_snapshots[-self.pod_modes:]])

        # Compute mean and subtract
        X_mean = X.mean(dim=0)
        X_centered = X - X_mean

        # SVD for POD modes
        try:
            U_svd, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        except Exception:
            return U

        # Keep top k modes (energy threshold)
        energy = S.pow(2)
        total_energy = energy.sum().clamp(min=1e-30)
        cum_energy = energy.cumsum(dim=0) / total_energy
        k = max(1, int((cum_energy < 0.99).sum().item()) + 1)
        k = min(k, self.pod_modes)

        # Project current solution
        U_flat = U.flatten()
        coeffs = (U_flat - X_mean) @ Vt[:k].T

        # Reconstruct from POD subspace
        U_pod = X_mean + coeffs @ Vt[:k]

        # Blend: mostly POD, keep some original
        blend = 0.3
        U_result = (1.0 - blend) * U + blend * U_pod.reshape(U.shape)

        return U_result

    # ------------------------------------------------------------------
    # Consistent flux correction
    # ------------------------------------------------------------------

    def _consistent_flux_balance(
        self,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """Apply global flux balance correction.

        Computes the net mass imbalance across all boundaries and
        distributes a uniform correction to ensure global conservation.

        Parameters
        ----------
        phi : torch.Tensor
            Face flux field.

        Returns:
            Corrected flux field.
        """
        if not self.flux_correction:
            return phi

        mesh = self.mesh
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        # Net flux through internal faces should be zero
        # Check total outflow through boundary
        boundary_start = n_internal
        if boundary_start >= n_faces:
            return phi

        phi_boundary = phi[boundary_start:]
        net_imbalance = phi_boundary.sum()

        n_boundary = n_faces - n_internal
        if n_boundary > 0:
            correction = net_imbalance / n_boundary
            phi = phi.clone()
            phi[boundary_start:] = phi[boundary_start:] - correction

        return phi

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v4 simpleFoam solver.

        Uses POD acceleration, selective frequency damping, and
        consistent flux correction.

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

        logger.info("Starting simpleFoamEnhanced4 run")
        logger.info("  pod=%d, sfd=%s, flux_corr=%s",
                     self.pod_modes, self.sfd_enabled, self.flux_correction)
        logger.info("  algorithm=%s", "SIMPLEC" if self._using_simplec else "SIMPLE")

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

            # Store history for Anderson mixing (from v3)
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

            # Run one SIMPLE iteration
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Multi-level residual smoothing (from v3)
            if self.smoothing_levels > 0:
                residual_field = torch.full(
                    (self.mesh.n_cells,), conv.U_residual,
                    dtype=self.U.dtype, device=self.U.device,
                )
                _ = self._multi_level_smooth_residual(residual_field)

            # Anderson mixing (from v3)
            self._residual_history.append(self.U - self._U_history[-1])
            if len(self._U_history) >= 3 and len(self._residual_history) >= 3:
                self.U = self._anderson_mix(
                    self.U, self._U_history, self._residual_history,
                )

            # Non-orthogonal SIMPLEC correction (from v3)
            if self._using_simplec:
                self.p, self.U = self._non_orthogonal_simplec_correction(
                    self.p, self.U,
                )

            # Consistent flux correction
            self.phi = self._consistent_flux_balance(self.phi)

            # SFD damping
            self.U, self.p = self._apply_sfd_damping(
                self.U, self.p, self.delta_t,
            )

            # POD acceleration
            if len(self._pod_snapshot_U) >= self.pod_modes:
                self.U = self._pod_acceleration(
                    self.U, self._pod_snapshot_U,
                )

            # Dynamic relaxation (from v2)
            self._adjust_relaxation(conv.U_residual)
            self._check_and_switch_algorithm(conv.U_residual)

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
                logger.info("simpleFoamEnhanced4 completed (converged)")
            else:
                logger.warning("simpleFoamEnhanced4 completed without convergence")

        return last_convergence or ConvergenceData()
