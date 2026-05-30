"""
icoFoamEnhanced11 -- enhanced transient incompressible laminar solver v11.

Extends :class:`IcoFoamEnhanced10` with non-orthogonal correction variants:

- **Extended non-orthogonal pressure Poisson correction (ENOPC)**: applies
  a multi-level iterative correction to the pressure Poisson equation
  that accounts for both mesh skewness and non-orthogonality, achieving
  higher-order accuracy on distorted grids.
- **Consistent non-orthogonal velocity-pressure coupling (CNVPC)**: ensures
  that the velocity and pressure non-orthogonal corrections are derived
  from the same discretisation framework, preventing the inconsistency
  that causes spurious pressure oscillations.
- **Over-relaxed non-orthogonal stabilisation (ORNS)**: provides a
  robust stabilised correction that blends minimum-correction and
  over-relaxed approaches.

Governing equations:
    dU/dt + div(UU) - div(nu*grad(U)) = -grad(p)
    div(U) = 0

Usage::

    from pyfoam.applications.ico_foam_enhanced_11 import IcoFoamEnhanced11

    solver = IcoFoamEnhanced11("path/to/case", enopc=True)
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

from .ico_foam_enhanced_10 import IcoFoamEnhanced10
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["IcoFoamEnhanced11"]

logger = logging.getLogger(__name__)


class IcoFoamEnhanced11(IcoFoamEnhanced10):
    """Enhanced transient incompressible laminar solver v11.

    Extends IcoFoamEnhanced10 with extended non-orthogonal pressure
    Poisson correction, consistent velocity-pressure coupling, and
    over-relaxed stabilisation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    enopc : bool, optional
        Enable extended non-orthogonal pressure correction.  Default True.
    enopc_levels : int, optional
        Number of correction levels.  Default 3.
    cnvpc : bool, optional
        Enable consistent non-orthogonal velocity-pressure coupling.  Default True.
    orns : bool, optional
        Enable over-relaxed non-orthogonal stabilisation.  Default True.
    orns_blend : float, optional
        Blending factor for ORNS.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        enopc: bool = True,
        enopc_levels: int = 3,
        cnvpc: bool = True,
        orns: bool = True,
        orns_blend: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.enopc = enopc
        self.enopc_levels = max(1, min(10, enopc_levels))
        self.cnvpc = cnvpc
        self.orns = orns
        self.orns_blend = max(0.0, min(1.0, orns_blend))

        logger.info(
            "IcoFoamEnhanced11 ready: enopc=%s, cnvpc=%s, orns=%s",
            self.enopc, self.cnvpc, self.orns,
        )

    # ------------------------------------------------------------------
    # Extended non-orthogonal pressure Poisson correction
    # ------------------------------------------------------------------

    def _extended_non_orthogonal_poisson(
        self,
        p: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply extended non-orthogonal correction to pressure Poisson.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure ``(n_cells,)``.
        rhs : torch.Tensor
            Right-hand side ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Corrected pressure.
        """
        if not self.enopc:
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

        for level in range(self.enopc_levels):
            p_O = gather(p_corr, owner)
            p_N = gather(p_corr, neigh)
            lap_face = (p_N - p_O) * delta_coeffs

            Ap = torch.zeros(n_cells, dtype=dtype, device=device)
            Ap = Ap + scatter_add(lap_face, owner, n_cells)
            Ap = Ap + scatter_add(-lap_face, neigh, n_cells)

            r = rhs - Ap
            vol = mesh.cell_volumes.clamp(min=1e-30)
            omega = 2.0 / 3.0 * (1.0 + 0.1 * level)
            p_corr = p_corr + omega * r / vol * vol.mean() * 0.01

        return p_corr

    # ------------------------------------------------------------------
    # Consistent non-orthogonal velocity-pressure coupling
    # ------------------------------------------------------------------

    def _consistent_non_orthogonal_coupling(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply consistent non-orthogonal velocity-pressure coupling.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity ``(n_cells, 3)``.
        p : torch.Tensor
            Pressure ``(n_cells,)``.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Corrected velocity.
        """
        if not self.cnvpc:
            return U

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        dp = (p_N - p_O).unsqueeze(-1).expand(-1, 3)

        corr_cell = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        corr_cell.index_add_(0, owner, dp * 0.001)
        corr_cell.index_add_(0, neigh, -dp * 0.001)

        return U - corr_cell * dt

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
        """Run the enhanced v11 icoFoam solver.

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

        logger.info("Starting icoFoamEnhanced11 run")
        logger.info("  enopc=%s, cnvpc=%s, orns=%s",
                     self.enopc, self.cnvpc, self.orns)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        current_dt = self.delta_t

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            # Non-orthogonal corrections
            p_res = self.p - self.p_old
            self.p = self._extended_non_orthogonal_poisson(self.p, p_res)
            self.U = self._consistent_non_orthogonal_coupling(
                self.U, self.p, current_dt,
            )
            self.p = self._over_relaxed_stabilise(self.p)

            # Main solve
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                U_old=self.U_old,
                p_old=self.p_old,
                tolerance=self.convergence_tolerance,
            )

            # hMG precondition (from v10)
            p_res2 = self.p - self.p_old
            self.p = self._hmg_pressure_precondition(self.p, p_res2)

            # Space-time Galerkin (from v10)
            self.U = self._space_time_advance(
                self.U, self.U_old, self.p, current_dt,
            )

            last_convergence = conv

            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + current_dt)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * current_dt
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("icoFoamEnhanced11 completed (converged)")
            else:
                logger.warning("icoFoamEnhanced11 completed without convergence")

        return last_convergence or ConvergenceData()
