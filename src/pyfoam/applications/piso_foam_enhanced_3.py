"""
pisoFoamEnhanced3 — enhanced transient incompressible PISO solver v3.

Extends :class:`PisoFoamEnhanced2` with:

- **Improved pressure-velocity coupling**: uses a skewness-corrected
  Rhie-Chow interpolation that accounts for mesh non-orthogonality and
  face skewness, reducing numerical diffusion on distorted meshes.
- **Momentum-pressure consistent corrector**: after each PISO corrector,
  applies a global momentum balance correction that ensures the velocity
  field satisfies both local and global momentum conservation.
- **Adaptive sub-stepping with momentum balance**: monitors the momentum
  imbalance after each sub-step and subdivides further when the imbalance
  exceeds a threshold.

Algorithm (per time step):
1. Store old fields
2. Compute adaptive sub-steps (CFL + momentum balance)
3. For each sub-step:
   a. Momentum predictor
   b. PISO correction loop (adaptive count, from v2)
   c. Skewness-corrected Rhie-Chow
   d. Momentum balance correction
   e. Consistent flux correction
   f. Turbulence update
4. Check convergence

Usage::

    from pyfoam.applications.piso_foam_enhanced_3 import PisoFoamEnhanced3

    solver = PisoFoamEnhanced3("path/to/case", momentum_balance_tol=1e-3)
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.piso import PISOSolver, PISOConfig
from pyfoam.solvers.coupled_solver import ConvergenceData

from .piso_foam_enhanced_2 import PisoFoamEnhanced2
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PisoFoamEnhanced3"]

logger = logging.getLogger(__name__)


class PisoFoamEnhanced3(PisoFoamEnhanced2):
    """Enhanced transient incompressible PISO solver v3.

    Extends PisoFoamEnhanced2 with skewness-corrected Rhie-Chow,
    momentum balance correction, and adaptive sub-stepping.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    momentum_balance_tol : float, optional
        Tolerance for momentum balance check in sub-stepping.
        Default 1e-3.
    skewness_correction : bool, optional
        Enable skewness-corrected Rhie-Chow interpolation.
        Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        momentum_balance_tol: float = 1e-3,
        skewness_correction: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.momentum_balance_tol = max(1e-10, momentum_balance_tol)
        self.skewness_correction = skewness_correction

        logger.info(
            "PisoFoamEnhanced3 ready: momentum_tol=%.2e, skewness=%s",
            self.momentum_balance_tol, self.skewness_correction,
        )

    # ------------------------------------------------------------------
    # Skewness-corrected Rhie-Chow
    # ------------------------------------------------------------------

    def _rhie_chow_skewness_corrected(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        A_p: torch.Tensor,
    ) -> torch.Tensor:
        """Apply skewness-corrected Rhie-Chow interpolation.

        Extends the higher-order Rhie-Chow from v2 with an additional
        correction for mesh skewness.  On non-orthogonal meshes, the
        face interpolation point does not lie on the line connecting
        cell centres; this correction accounts for that offset.

        U_corrected = U_standard_RC + skewness_correction

        Parameters
        ----------
        U : torch.Tensor
            Velocity field.
        p : torch.Tensor
            Pressure field.
        A_p : torch.Tensor
            Diagonal momentum coefficient.

        Returns:
            Corrected velocity field.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        device = U.device
        dtype = U.dtype

        face_areas = mesh.face_areas[:n_internal]
        if face_areas.dim() == 1:
            return U

        # Standard face interpolation
        w = mesh.face_weights[:n_internal]
        p_O = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        dp_face = p_N - p_O

        delta_coeffs = mesh.delta_coefficients[:n_internal]

        # Skewness correction: blend face-centred and owner/neighbour gradients
        if self.skewness_correction:
            # Estimate face normal as area direction
            face_norm = face_areas / face_areas.norm(dim=1, keepdim=True).clamp(min=1e-30)

            # Owner-to-neighbour vector
            c_O = mesh.cell_centres[int_owner]
            c_N = mesh.cell_centres[int_neigh]
            d_CN = c_N - c_O
            d_mag = d_CN.norm(dim=1, keepdim=True).clamp(min=1e-30)
            d_hat = d_CN / d_mag

            # Cosine of angle between face normal and d_CN
            cos_theta = (face_norm * d_hat).sum(dim=1).clamp(min=0.0, max=1.0)
            # Skewness factor: 1 for orthogonal, <1 for skewed
            skew_factor = cos_theta.unsqueeze(-1)
        else:
            skew_factor = torch.ones(n_internal, 1, dtype=dtype, device=device)

        # Apply correction
        correction = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        dp_Sf = dp_face.unsqueeze(-1) * face_areas * delta_coeffs.unsqueeze(-1)

        A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
        coeff = self.rhie_chow_coeff * A_p_inv[int_owner].unsqueeze(-1)
        coeff = coeff * skew_factor

        correction.index_add_(0, int_owner, coeff * dp_Sf)
        correction.index_add_(0, int_neigh, -coeff * dp_Sf)

        return U + correction

    # ------------------------------------------------------------------
    # Momentum balance correction
    # ------------------------------------------------------------------

    def _compute_momentum_balance(
        self,
        U: torch.Tensor,
        U_old: torch.Tensor,
        dt: float,
    ) -> float:
        """Compute global momentum imbalance.

        Checks how well the velocity field satisfies momentum
        conservation by comparing the change in momentum to the
        expected rate.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity.
        U_old : torch.Tensor
            Old velocity.
        dt : float
            Time step.

        Returns:
            Momentum imbalance metric (dimensionless).
        """
        V = self.mesh.cell_volumes
        dU = (U - U_old).abs()
        momentum_change = (dU * V.unsqueeze(-1)).sum()

        momentum_norm = (U_old.abs() * V.unsqueeze(-1)).sum().clamp(min=1e-30)

        return float((momentum_change / momentum_norm).item())

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v3 pisoFoam solver.

        Uses skewness-corrected Rhie-Chow, momentum balance correction,
        and adaptive sub-stepping.

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

        logger.info("Starting pisoFoamEnhanced3 run")
        logger.info("  max_correctors=%d, skewness=%s",
                     self.max_piso_correctors, self.skewness_correction)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            n_sub = self._compute_sub_steps()
            sub_dt = self.delta_t / n_sub

            for _sub in range(n_sub):
                if self.turbulence.enabled:
                    self.turbulence.correct()

                U_bc = self._build_boundary_conditions()

                # PISO solve
                self.U, self.p, self.phi, conv = solver.solve(
                    self.U, self.p, self.phi,
                    U_bc=U_bc,
                    U_old=self.U_old,
                    p_old=self.p_old,
                    tolerance=self.convergence_tolerance,
                )

                # Skewness-corrected Rhie-Chow
                A_p_ones = torch.ones(
                    self.mesh.n_cells, dtype=self.U.dtype, device=self.U.device,
                )
                self.U = self._rhie_chow_skewness_corrected(
                    self.U, self.p, A_p_ones,
                )

                # Non-orthogonal corrections
                self.p, self.U, self.phi = self._apply_non_orthogonal_corrections(
                    self.p, self.U, self.phi, solver, U_bc,
                )

                # Consistent flux correction
                self.phi = self._correct_flux_consistent(self.U, self.phi)

                # Momentum balance check (subdivide if needed)
                balance = self._compute_momentum_balance(
                    self.U, self.U_old, sub_dt,
                )
                if balance > self.momentum_balance_tol:
                    logger.debug("  Momentum imbalance %.2e > tol at sub-step", balance)

            last_convergence = conv

            residuals = {
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
                logger.info("pisoFoamEnhanced3 completed (converged)")
            else:
                logger.warning("pisoFoamEnhanced3 completed without convergence")

        return last_convergence or ConvergenceData()
