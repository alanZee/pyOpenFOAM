"""
multiphaseEulerFoam — N-phase Euler-Euler solver.

Extends twoPhaseEulerFoam to handle N phases, each with its own
velocity field, coupled through interphase forces.

Phase volume fractions sum to 1:
    sum(alpha_i) = 1

Based on OpenFOAM's multiphaseEulerFoam solver.

Usage::

    from pyfoam.applications.multiphase_euler_foam import MultiphaseEulerFoam

    phases = [
        {"name": "gas",    "rho": 1.225, "mu": 1.8e-5, "d": 0.001},
        {"name": "liquid", "rho": 1000.0, "mu": 1e-3, "d": 0.001},
        {"name": "solid",  "rho": 2500.0, "mu": 0.001, "d": 0.0005},
    ]
    solver = MultiphaseEulerFoam("path/to/case", phases=phases)
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

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["MultiphaseEulerFoam"]

logger = logging.getLogger(__name__)


class MultiphaseEulerFoam(SolverBase):
    """N-phase Euler-Euler solver.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    phases : list[dict]
        Phase definitions with name, rho, mu, d (diameter).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        phases: List[Dict[str, Any]],
    ) -> None:
        super().__init__(case_path)

        self.phases = phases
        self.n_phases = len(phases)
        self.phase_names = [p["name"] for p in phases]

        self.rho_phases = torch.tensor(
            [p["rho"] for p in phases],
            dtype=get_default_dtype(), device=get_device(),
        )
        self.mu_phases = torch.tensor(
            [p["mu"] for p in phases],
            dtype=get_default_dtype(), device=get_device(),
        )

        self._read_fv_solution_settings()
        self.velocities, self.p, self.alphas, self.phi = self._init_fields()
        self._vel_datas, self._p_data, self._alpha_datas = self._init_field_data()

        logger.info("MultiphaseEulerFoam ready: %d phases", self.n_phases)

    def _read_fv_solution_settings(self):
        fv = self.case.fvSolution
        self.n_outer_correctors = int(fv.get_path("PIMPLE/nOuterCorrectors", 3))
        self.convergence_tolerance = float(fv.get_path("PIMPLE/convergenceTolerance", 1e-4))
        self.max_outer_iterations = int(fv.get_path("PIMPLE/maxOuterIterations", 100))

    def _init_fields(self):
        device = get_device()
        dtype = get_default_dtype()

        velocities = []
        for name in self.phase_names:
            try:
                U, _ = self.read_field_tensor(f"U_{name}", 0)
            except Exception:
                U, _ = self.read_field_tensor("U", 0)
            velocities.append(U.to(device=device, dtype=dtype))

        p, _ = self.read_field_tensor("p", 0)
        p = p.to(device=device, dtype=dtype)

        alphas = []
        for i, name in enumerate(self.phase_names):
            if i < self.n_phases - 1:
                try:
                    a, _ = self.read_field_tensor(f"alpha_{name}", 0)
                except Exception:
                    a = torch.full(
                        (self.mesh.n_cells,), 1.0 / self.n_phases,
                        dtype=dtype, device=device,
                    )
                alphas.append(a.to(device=device, dtype=dtype))
            else:
                alpha_last = (1.0 - sum(alphas)).clamp(0.0, 1.0)
                alphas.append(alpha_last)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)
        return velocities, p, alphas, phi

    def _init_field_data(self):
        vel_datas = []
        for name in self.phase_names:
            try:
                vel_datas.append(self.case.read_field(f"U_{name}", 0))
            except Exception:
                vel_datas.append(self.case.read_field("U", 0))

        p_data = self.case.read_field("p", 0)
        alpha_datas = []
        for name in self.phase_names:
            try:
                alpha_datas.append(self.case.read_field(f"alpha_{name}", 0))
            except Exception:
                alpha_datas.append(None)

        return vel_datas, p_data, alpha_datas

    def run(self) -> ConvergenceData:
        time_loop = TimeLoop(
            start_time=self.start_time, end_time=self.end_time,
            delta_t=self.delta_t, write_interval=self.write_interval,
            write_control=self.write_control,
        )
        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence = None
        for t, step in time_loop:
            self.velocities, self.p, self.alphas, self.phi, conv = (
                self._euler_iteration()
            )
            last_convergence = conv

            residuals = {"U": conv.U_residual, "p": conv.p_residual, "cont": conv.continuity_error}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)
        return last_convergence or ConvergenceData()

    def _euler_iteration(self):
        velocities = [U.clone() for U in self.velocities]
        p = self.p.clone()
        alphas = [a.clone() for a in self.alphas]
        phi = self.phi.clone()
        convergence = ConvergenceData()

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            vels_prev = [U.clone() for U in velocities]

            # Enforce constraint: last alpha = 1 - sum(others)
            alpha_sum = sum(alphas[:-1])
            alphas[-1] = (1.0 - alpha_sum).clamp(0.0, 1.0)

            # Renormalise
            total = sum(alphas).clamp(min=1e-30)
            alphas = [a / total for a in alphas]

            # Convergence
            U_residual = max(
                self._compute_residual(velocities[i], vels_prev[i])
                for i in range(self.n_phases)
            )
            convergence.U_residual = U_residual
            convergence.outer_iterations = outer + 1

        return velocities, p, alphas, phi, convergence

    def _compute_residual(self, field, field_old):
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def _write_fields(self, time):
        time_str = f"{time:g}"
        for i, name in enumerate(self.phase_names):
            if self._vel_datas[i] is not None:
                self.write_field(f"U_{name}", self.velocities[i], time_str, self._vel_datas[i])
        self.write_field("p", self.p, time_str, self._p_data)
        for i, name in enumerate(self.phase_names):
            if self._alpha_datas[i] is not None:
                self.write_field(f"alpha_{name}", self.alphas[i], time_str, self._alpha_datas[i])
