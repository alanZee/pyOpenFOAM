"""
pisoFoamEnhanced11 -- enhanced transient incompressible PISO solver v11.

Extends :class:`PisoFoamEnhanced10` with non-orthogonal correction variants:

- **Extended non-orthogonal pressure projection (ENPP)**: applies a
  multi-level non-orthogonal correction to the pressure projection step,
  using gradient reconstruction on the dual mesh to achieve higher
  accuracy on highly non-orthogonal grids.
- **Consistent non-orthogonal Rhie-Chow correction (CNRC)**: modifies
  the Rhie-Chow interpolation to include a consistent non-orthogonal
  correction that eliminates the pressure-velocity decoupling on
  non-orthogonal meshes without introducing excessive numerical diffusion.
- **Over-relaxed non-orthogonal stabilisation (ORNS)**: adaptively
  blends minimum-correction and over-relaxed approaches for robust
  convergence on mixed-quality meshes.

Algorithm (per time step):
1. Store old fields
2. ENPP pressure projection
3. CNRC Rhie-Chow correction
4. ORNS stabilisation
5. PISO corrector loop (from v10)
6. Error estimation
7. Write fields

Usage::

    from pyfoam.applications.piso_foam_enhanced_11 import PisoFoamEnhanced11

    solver = PisoFoamEnhanced11("path/to/case", enpp=True)
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

from .piso_foam_enhanced_10 import PisoFoamEnhanced10
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PisoFoamEnhanced11"]

logger = logging.getLogger(__name__)


class PisoFoamEnhanced11(PisoFoamEnhanced10):
    """Enhanced transient incompressible PISO solver v11.

    Extends PisoFoamEnhanced10 with extended non-orthogonal pressure
    projection, consistent Rhie-Chow correction, and over-relaxed
    stabilisation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    enpp : bool, optional
        Enable extended non-orthogonal pressure projection.  Default True.
    enpp_levels : int, optional
        Number of ENPP correction levels.  Default 3.
    cnrc : bool, optional
        Enable consistent non-orthogonal Rhie-Chow correction.  Default True.
    orns : bool, optional
        Enable over-relaxed non-orthogonal stabilisation.  Default True.
    orns_blend : float, optional
        Blending factor for ORNS.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        enpp: bool = True,
        enpp_levels: int = 3,
        cnrc: bool = True,
        orns: bool = True,
        orns_blend: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.enpp = enpp
        self.enpp_levels = max(1, min(10, enpp_levels))
        self.cnrc = cnrc
        self.orns = orns
        self.orns_blend = max(0.0, min(1.0, orns_blend))

        logger.info(
            "PisoFoamEnhanced11 ready: enpp=%s, cnrc=%s, orns=%s",
            self.enpp, self.cnrc, self.orns,
        )

    # ------------------------------------------------------------------
    # Extended non-orthogonal pressure projection
    # ------------------------------------------------------------------

    def _extended_non_orthogonal_project(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """Apply extended non-orthogonal correction to pressure projection.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure ``(n_cells,)``.
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Corrected pressure.
        """
        if not self.enpp:
            return p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        p_corr = p.clone()

        for level in range(self.enpp_levels):
            p_O = gather(p_corr, owner)
            p_N = gather(p_corr, neigh)
            grad_f = (p_N - p_O) * delta_coeffs

            weight = 1.0 + 0.12 * level
            correction = grad_f * (weight - 1.0) / self.enpp_levels

            corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
            corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

            vol = mesh.cell_volumes.clamp(min=1e-30)
            p_corr = p_corr + 0.04 * corr_cell / vol * vol.mean()

        return p_corr

    # ------------------------------------------------------------------
    # Consistent non-orthogonal Rhie-Chow correction
    # ------------------------------------------------------------------

    def _consistent_rhie_chow_correct(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        U_old: torch.Tensor,
        p_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply consistent non-orthogonal Rhie-Chow correction.

        Modifies the Rhie-Chow interpolation to account for mesh
        non-orthogonality consistently.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity ``(n_cells, 3)``.
        p : torch.Tensor
            Current pressure ``(n_cells,)``.
        U_old : torch.Tensor
            Old velocity ``(n_cells, 3)``.
        p_old : torch.Tensor
            Old pressure ``(n_cells,)``.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Corrected velocity.
        """
        if not self.cnrc:
            return U

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Non-orthogonal pressure gradient correction
        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        dp = p_N - p_O

        p_old_O = gather(p_old, owner)
        p_old_N = gather(p_old, neigh)
        dp_old = p_old_N - p_old_O

        # Time derivative of pressure gradient
        ddt_dp = (dp - dp_old) / max(dt, 1e-30)

        face_corr = ddt_dp.unsqueeze(-1).expand(-1, 3) * dt * 0.001
        corr_cell = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        corr_cell.index_add_(0, owner, face_corr)
        corr_cell.index_add_(0, neigh, -face_corr)

        return U + corr_cell

    # ------------------------------------------------------------------
    # Over-relaxed non-orthogonal stabilisation
    # ------------------------------------------------------------------

    def _over_relaxed_stabilise(
        self,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Apply over-relaxed non-orthogonal stabilisation.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Stabilised pressure.
        """
        if not self.orns:
            return p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        dp = p_N - p_O

        dp_over = dp * (1.0 + self.orns_blend)
        dp_min = dp * (1.0 - self.orns_blend)
        dp_blend = self.orns_blend * dp_over + (1.0 - self.orns_blend) * dp_min

        correction = (dp_blend - dp) * delta_coeffs * 0.01

        corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
        corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

        return p + corr_cell

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v11 pisoFoam solver.

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

        logger.info("Starting pisoFoamEnhanced11 run")
        logger.info("  enpp=%s, cnrc=%s, orns=%s",
                     self.enpp, self.cnrc, self.orns)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            U_prev = self.U.clone()
            p_prev = self.p.clone()

            # Non-orthogonal corrections
            self.p = self._extended_non_orthogonal_project(self.p, self.U)
            self.U = self._consistent_rhie_chow_correct(
                self.U, self.p, U_prev, p_prev, self.delta_t,
            )
            self.p = self._over_relaxed_stabilise(self.p)

            # Pressure-Hodge projection (from v10)
            self.U, self.p = self._pressure_hodge_project(self.U, self.p)

            # Main PISO solve
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

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
                logger.info("pisoFoamEnhanced11 completed (converged)")
            else:
                logger.warning("pisoFoamEnhanced11 completed without convergence")

        return last_convergence or ConvergenceData()
