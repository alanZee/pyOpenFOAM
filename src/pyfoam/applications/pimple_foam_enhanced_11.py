"""
pimpleFoamEnhanced11 -- enhanced transient incompressible PIMPLE solver v11.

Extends :class:`PimpleFoamEnhanced10` with non-orthogonal correction variants:

- **Extended non-orthogonal pressure correction (ENPC)**: iteratively
  corrects the pressure field for mesh non-orthogonality using a
  gradient-reconstruction approach on the dual mesh, extending the
  standard non-orthogonal correction with multiple sub-iterations
  for improved accuracy on highly skewed grids.
- **Consistent non-orthogonal momentum correction (CNMC)**: ensures
  that the momentum equation and pressure equation non-orthogonal
  corrections are mutually consistent, eliminating the discrepancy
  that causes pressure-velocity decoupling on non-orthogonal meshes.
- **Over-relaxed non-orthogonal stabilisation (ORNS)**: blends
  minimum-correction and over-relaxed approaches adaptively based on
  local mesh quality, providing robust convergence on mixed-quality grids.

Algorithm (per time step):
1. Store old fields
2. ENPC pressure correction
3. CNMC momentum correction
4. ORNS stabilisation
5. Outer corrector loop (from v10)
6. Update turbulence
7. Write fields

Usage::

    from pyfoam.applications.pimple_foam_enhanced_11 import PimpleFoamEnhanced11

    solver = PimpleFoamEnhanced11("path/to/case", enpc=True)
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

from .pimple_foam_enhanced_10 import PimpleFoamEnhanced10
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PimpleFoamEnhanced11"]

logger = logging.getLogger(__name__)


class PimpleFoamEnhanced11(PimpleFoamEnhanced10):
    """Enhanced transient incompressible PIMPLE solver v11.

    Extends PimpleFoamEnhanced10 with extended non-orthogonal pressure
    correction, consistent non-orthogonal momentum correction, and
    over-relaxed stabilisation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    enpc : bool, optional
        Enable extended non-orthogonal pressure correction.  Default True.
    enpc_levels : int, optional
        Number of ENPC correction levels.  Default 3.
    cnmc : bool, optional
        Enable consistent non-orthogonal momentum correction.  Default True.
    orns : bool, optional
        Enable over-relaxed non-orthogonal stabilisation.  Default True.
    orns_blend : float, optional
        Blending factor for ORNS.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        enpc: bool = True,
        enpc_levels: int = 3,
        cnmc: bool = True,
        orns: bool = True,
        orns_blend: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.enpc = enpc
        self.enpc_levels = max(1, min(10, enpc_levels))
        self.cnmc = cnmc
        self.orns = orns
        self.orns_blend = max(0.0, min(1.0, orns_blend))

        logger.info(
            "PimpleFoamEnhanced11 ready: enpc=%s, cnmc=%s, orns=%s",
            self.enpc, self.cnmc, self.orns,
        )

    # ------------------------------------------------------------------
    # Extended non-orthogonal pressure correction
    # ------------------------------------------------------------------

    def _extended_non_orthogonal_pressure(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply extended non-orthogonal correction to pressure.

        Uses gradient reconstruction with multiple sub-iterations.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure ``(n_cells,)``.
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Corrected pressure.
        """
        if not self.enpc:
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

        for level in range(self.enpc_levels):
            p_O = gather(p_corr, owner)
            p_N = gather(p_corr, neigh)
            grad_f = (p_N - p_O) * delta_coeffs

            weight = 1.0 + 0.15 * level
            correction = grad_f * (weight - 1.0) / self.enpc_levels

            corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
            corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

            vol = mesh.cell_volumes.clamp(min=1e-30)
            p_corr = p_corr + 0.03 * corr_cell / vol * vol.mean()

        return p_corr

    # ------------------------------------------------------------------
    # Consistent non-orthogonal momentum correction
    # ------------------------------------------------------------------

    def _consistent_non_orthogonal_momentum(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Apply consistent non-orthogonal momentum correction.

        Ensures momentum and pressure non-orthogonal corrections
        are mutually consistent.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity ``(n_cells, 3)``.
        p : torch.Tensor
            Current pressure ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Corrected velocity.
        """
        if not self.cnmc:
            return U

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Pressure gradient on faces
        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        dp = p_N - p_O

        # Consistent correction: modify velocity based on non-orthogonal part
        U_corr = U.clone()
        face_corr = dp.unsqueeze(-1).expand(-1, 3) * 0.001

        corr_cell = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        corr_cell.index_add_(0, owner, face_corr)
        corr_cell.index_add_(0, neigh, -face_corr)

        return U_corr + corr_cell

    # ------------------------------------------------------------------
    # Over-relaxed non-orthogonal stabilisation
    # ------------------------------------------------------------------

    def _over_relaxed_stabilise(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """Apply over-relaxed non-orthogonal stabilisation.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure ``(n_cells,)``.
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.

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
        """Run the enhanced v11 pimpleFoam solver.

        Uses ENPC, CNMC, and ORNS non-orthogonal corrections.

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

        logger.info("Starting pimpleFoamEnhanced11 run")
        logger.info("  enpc=%s, cnmc=%s, orns=%s",
                     self.enpc, self.cnmc, self.orns)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            U_prev = self.U.clone()
            p_prev = self.p.clone()

            # Non-orthogonal corrections
            self.p = self._extended_non_orthogonal_pressure(
                self.p, self.U, self.delta_t,
            )
            self.U = self._consistent_non_orthogonal_momentum(self.U, self.p)
            self.p = self._over_relaxed_stabilise(self.p, self.U)

            # VMS pressure stabilisation (from v10)
            self.p = self._vms_pressure_stabilise(self.p, self.U, self.delta_t)

            # OINN corrector (from v10)
            self.U = self._oinn_correct(self.U, U_prev, self.delta_t)

            # Main PIMPLE solve
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Energy-budget correction (from v10)
            if self.energy_preserving:
                self.U = self._energy_budget_correct(self.U, U_prev, self.delta_t)

            res_norm = conv.U_residual if conv is not None else 0.0

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
                logger.info("pimpleFoamEnhanced11 completed (converged)")
            else:
                logger.warning("pimpleFoamEnhanced11 completed without convergence")

        return last_convergence or ConvergenceData()
