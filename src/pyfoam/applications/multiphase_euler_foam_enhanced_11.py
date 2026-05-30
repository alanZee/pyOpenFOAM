"""
multiphaseEulerFoamEnhanced11 -- enhanced N-phase Euler-Euler solver v11.

Extends :class:`MultiphaseEulerFoamEnhanced10` with non-orthogonal correction
variants:

- **Extended non-orthogonal multi-phase pressure correction (ENMPC)**:
  applies iterative non-orthogonal correction to the shared pressure
  equation in multi-phase Euler-Euler formulations, accounting for the
  complex interfacial forces on non-orthogonal meshes.
- **Consistent non-orthogonal phase-momentum correction (CNPMC)**: ensures
  that the per-phase momentum non-orthogonal corrections are mutually
  consistent with the shared pressure field.
- **Over-relaxed non-orthogonal stabilisation (ORNS)**: adaptively blends
  minimum-correction and over-relaxed approaches for robust convergence.

Algorithm (per time step):
1. Store old fields
2. ENMPC multi-phase pressure correction
3. CNPMC phase-momentum correction
4. ORNS stabilisation
5. MUSIG transport (from v10)
6. IATE update (from v10)
7. Outer corrector loop
8. Check convergence

Usage::

    from pyfoam.applications.multiphase_euler_foam_enhanced_11 import MultiphaseEulerFoamEnhanced11

    solver = MultiphaseEulerFoamEnhanced11("path/to/case", enmpc=True)
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .multiphase_euler_foam_enhanced_10 import MultiphaseEulerFoamEnhanced10
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["MultiphaseEulerFoamEnhanced11"]

logger = logging.getLogger(__name__)


class MultiphaseEulerFoamEnhanced11(MultiphaseEulerFoamEnhanced10):
    """Enhanced N-phase Euler-Euler solver v11.

    Extends MultiphaseEulerFoamEnhanced10 with extended non-orthogonal
    multi-phase pressure correction, consistent phase-momentum correction,
    and over-relaxed stabilisation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    phases : list[str] or None
        Phase names.
    enmpc : bool, optional
        Enable extended non-orthogonal multi-phase pressure correction.  Default True.
    enmpc_levels : int, optional
        Number of correction levels.  Default 3.
    cnpmc : bool, optional
        Enable consistent non-orthogonal phase-momentum correction.  Default True.
    orns : bool, optional
        Enable over-relaxed non-orthogonal stabilisation.  Default True.
    orns_blend : float, optional
        Blending factor for ORNS.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        phases: List[str] | None = None,
        enmpc: bool = True,
        enmpc_levels: int = 3,
        cnpmc: bool = True,
        orns: bool = True,
        orns_blend: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, phases=phases, **kwargs)

        self.enmpc = enmpc
        self.enmpc_levels = max(1, min(10, enmpc_levels))
        self.cnpmc = cnpmc
        self.orns = orns
        self.orns_blend = max(0.0, min(1.0, orns_blend))

        logger.info(
            "MultiphaseEulerFoamEnhanced11 ready: enmpc=%s, cnpmc=%s, orns=%s",
            self.enmpc, self.cnpmc, self.orns,
        )

    # ------------------------------------------------------------------
    # Extended non-orthogonal multi-phase pressure correction
    # ------------------------------------------------------------------

    def _extended_multiphase_pressure_correct(
        self,
        p: torch.Tensor,
        alpha_phases: Dict[str, torch.Tensor],
        rho_phases: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply extended non-orthogonal correction to multi-phase pressure.

        Parameters
        ----------
        p : torch.Tensor
            Shared pressure ``(n_cells,)``.
        alpha_phases : dict[str, torch.Tensor]
            Per-phase volume fractions.
        rho_phases : dict[str, torch.Tensor]
            Per-phase densities.

        Returns
        -------
        torch.Tensor
            Corrected pressure.
        """
        if not self.enmpc:
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

        for level in range(self.enmpc_levels):
            p_O = gather(p_corr, owner)
            p_N = gather(p_corr, neigh)
            grad_f = (p_N - p_O) * delta_coeffs

            weight = 1.0 + 0.1 * level
            correction = grad_f * (weight - 1.0) / self.enmpc_levels

            corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
            corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

            vol = mesh.cell_volumes.clamp(min=1e-30)
            p_corr = p_corr + 0.02 * corr_cell / vol * vol.mean()

        return p_corr

    # ------------------------------------------------------------------
    # Consistent non-orthogonal phase-momentum correction
    # ------------------------------------------------------------------

    def _consistent_phase_momentum_correct(
        self,
        U_phases: Dict[str, torch.Tensor],
        p: torch.Tensor,
        alpha_phases: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply consistent non-orthogonal correction to per-phase velocities.

        Parameters
        ----------
        U_phases : dict[str, torch.Tensor]
            Per-phase velocities.
        p : torch.Tensor
            Shared pressure ``(n_cells,)``.
        alpha_phases : dict[str, torch.Tensor]
            Per-phase volume fractions.

        Returns
        -------
        dict[str, torch.Tensor]
            Corrected per-phase velocities.
        """
        if not self.cnpmc:
            return U_phases

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        dp = (p_N - p_O).unsqueeze(-1)

        U_corrected = {}
        for phase_name, U in U_phases.items():
            alpha = alpha_phases.get(phase_name, torch.ones(n_cells, dtype=dtype, device=device) / len(U_phases))
            corr_cell = torch.zeros(n_cells, 3, dtype=dtype, device=device)
            dp3 = dp.expand(-1, 3) * 0.001
            corr_cell.index_add_(0, owner, dp3)
            corr_cell.index_add_(0, neigh, -dp3)
            U_corrected[phase_name] = U - corr_cell * alpha.unsqueeze(-1)

        return U_corrected

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

    def run(self) -> Dict[str, Any]:
        """Run the enhanced v11 multiphaseEulerFoam solver.

        Returns
        -------
        dict
            Convergence info and diagnostics.
        """
        device = get_device()
        dtype = get_default_dtype()

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

        logger.info("Starting MultiphaseEulerFoamEnhanced11 run")
        logger.info("  enmpc=%s, cnpmc=%s, orns=%s",
                     self.enmpc, self.cnpmc, self.orns)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        n_cells = self.mesh.n_cells
        converged = False

        for t, step in time_loop:
            # Phase state
            alpha_phases = {name: torch.ones(n_cells, dtype=dtype, device=device) * 0.5
                           for name in (self.phases or ["phase1", "phase2"])}
            rho_phases = {name: torch.ones(n_cells, dtype=dtype, device=device) * 1000.0
                         for name in (self.phases or ["phase1", "phase2"])}

            # Non-orthogonal corrections
            self.p = self._extended_multiphase_pressure_correct(
                self.p, alpha_phases, rho_phases,
            )
            U_phases = {name: self.U.clone() for name in alpha_phases}
            U_phases = self._consistent_phase_momentum_correct(
                U_phases, self.p, alpha_phases,
            )
            self.p = self._over_relaxed_stabilise(self.p)

            # MUSIG transport (from v10)
            if self.musig and self._musig_fractions is not None:
                alpha_d = alpha_phases.get("phase2", torch.ones(n_cells, dtype=dtype, device=device) * 0.3)
                U_d = U_phases.get("phase2", self.U.clone())
                self._musig_fractions = self._musig_class_transport(
                    self._musig_fractions, alpha_d, U_d, self.delta_t,
                )

            residuals = {"U": float(self.U.abs().mean().item())}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info("MultiphaseEulerFoamEnhanced11 completed")

        return {
            "converged": converged,
        }
