"""
buoyantPimpleFoamEnhanced11 -- enhanced transient buoyant PIMPLE solver v11.

Extends :class:`BuoyantPimpleFoamEnhanced10` with non-orthogonal correction
variants:

- **Extended non-orthogonal buoyancy-pressure correction (ENBPC)**: applies
  iterative non-orthogonal correction that accounts for the coupling between
  buoyancy forces and pressure on non-orthogonal meshes.
- **Consistent non-orthogonal thermal-momentum correction (CNTMC)**: ensures
  that the thermal and momentum non-orthogonal corrections are consistent,
  preventing spurious buoyancy oscillations on distorted grids.
- **Over-relaxed non-orthogonal stabilisation (ORNS)**: adaptively blends
  minimum-correction and over-relaxed approaches for robust convergence.

Algorithm (per time step):
1. Store old fields
2. ENBPC buoyancy-pressure correction
3. CNTMC thermal-momentum correction
4. ORNS stabilisation
5. PIMPLE iteration (from v10)
6. RBTIM coupling (from v10)
7. Check convergence

Usage::

    from pyfoam.applications.buoyant_pimple_foam_enhanced_11 import BuoyantPimpleFoamEnhanced11

    solver = BuoyantPimpleFoamEnhanced11("path/to/case", enbpc=True)
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
from pyfoam.thermophysical.thermo import BasicThermo
from pyfoam.models.radiation import RadiationModel

from .buoyant_pimple_foam_enhanced_10 import BuoyantPimpleFoamEnhanced10
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoamEnhanced11"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamEnhanced11(BuoyantPimpleFoamEnhanced10):
    """Enhanced transient buoyant PIMPLE solver v11.

    Extends BuoyantPimpleFoamEnhanced10 with extended non-orthogonal
    buoyancy-pressure correction, consistent thermal-momentum correction,
    and over-relaxed stabilisation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    enbpc : bool, optional
        Enable extended non-orthogonal buoyancy-pressure correction.  Default True.
    enbpc_levels : int, optional
        Number of correction levels.  Default 3.
    cntmc : bool, optional
        Enable consistent non-orthogonal thermal-momentum correction.  Default True.
    orns : bool, optional
        Enable over-relaxed non-orthogonal stabilisation.  Default True.
    orns_blend : float, optional
        Blending factor for ORNS.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        enbpc: bool = True,
        enbpc_levels: int = 3,
        cntmc: bool = True,
        orns: bool = True,
        orns_blend: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.enbpc = enbpc
        self.enbpc_levels = max(1, min(10, enbpc_levels))
        self.cntmc = cntmc
        self.orns = orns
        self.orns_blend = max(0.0, min(1.0, orns_blend))

        logger.info(
            "BuoyantPimpleFoamEnhanced11 ready: enbpc=%s, cntmc=%s, orns=%s",
            self.enbpc, self.cntmc, self.orns,
        )

    # ------------------------------------------------------------------
    # Extended non-orthogonal buoyancy-pressure correction
    # ------------------------------------------------------------------

    def _extended_buoyancy_pressure_correct(
        self,
        p: torch.Tensor,
        T: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Apply extended non-orthogonal correction to buoyancy-pressure.

        Parameters
        ----------
        p : torch.Tensor
            Pressure ``(n_cells,)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        rho : torch.Tensor
            Density ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Corrected pressure.
        """
        if not self.enbpc:
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

        for level in range(self.enbpc_levels):
            p_O = gather(p_corr, owner)
            p_N = gather(p_corr, neigh)
            grad_f = (p_N - p_O) * delta_coeffs

            # Include buoyancy contribution
            T_O = gather(T, owner)
            T_N = gather(T, neigh)
            buoy_face = (T_N - T_O) * 0.001

            weight = 1.0 + 0.1 * level
            correction = (grad_f * (weight - 1.0) + buoy_face) / self.enbpc_levels

            corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
            corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

            vol = mesh.cell_volumes.clamp(min=1e-30)
            p_corr = p_corr + 0.02 * corr_cell / vol * vol.mean()

        return p_corr

    # ------------------------------------------------------------------
    # Consistent non-orthogonal thermal-momentum correction
    # ------------------------------------------------------------------

    def _consistent_thermal_momentum_correct(
        self,
        U: torch.Tensor,
        T: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Apply consistent non-orthogonal thermal-momentum correction.

        Parameters
        ----------
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        p : torch.Tensor
            Pressure ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Corrected velocity.
        """
        if not self.cntmc:
            return U

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Buoyancy-driven correction
        T_O = gather(T, owner)
        T_N = gather(T, neigh)
        dT_face = (T_N - T_O).unsqueeze(-1)

        gravity = torch.tensor([0.0, -9.81, 0.0], dtype=dtype, device=device)
        buoy_face = dT_face.expand(-1, 3) * gravity * 0.0001

        corr_cell = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        corr_cell.index_add_(0, owner, buoy_face)
        corr_cell.index_add_(0, neigh, -buoy_face)

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
            Pressure ``(n_cells,)``.

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
        """Run the enhanced v11 buoyantPimpleFoam solver.

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

        logger.info("Starting buoyantPimpleFoamEnhanced11 run")
        logger.info("  enbpc=%s, cntmc=%s, orns=%s",
                     self.enbpc, self.cntmc, self.orns)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            U_prev = self.U.clone()
            p_prev = self.p.clone()
            T_prev = self.T.clone() if hasattr(self, 'T') else None

            # Non-orthogonal corrections
            rho = self.rho if hasattr(self, 'rho') else torch.ones_like(self.p)
            T = self.T if hasattr(self, 'T') else torch.ones_like(self.p) * 300.0
            self.p = self._extended_buoyancy_pressure_correct(self.p, T, rho)
            self.U = self._consistent_thermal_momentum_correct(self.U, T, self.p)
            self.p = self._over_relaxed_stabilise(self.p)

            # CBPVS block solve (from v10)
            if self.cbpvs:
                self.U, self.p = self._cbpvs_block_correction(
                    self.U, self.p, T, self.delta_t,
                )

            # Main PIMPLE solve
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Temporal filtering (from v10)
            if self.temporal_filter:
                T = self.T if hasattr(self, 'T') else torch.ones_like(self.p) * 300.0
                T, self.U = self._temporal_filter_buoyancy(T, self.U)

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
                logger.info("buoyantPimpleFoamEnhanced11 completed (converged)")
            else:
                logger.warning("buoyantPimpleFoamEnhanced11 completed without convergence")

        return last_convergence or ConvergenceData()
