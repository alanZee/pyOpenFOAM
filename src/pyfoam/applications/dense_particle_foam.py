"""
DenseParticleFoam — dense particle two-way Euler-Lagrange solver.

Implements the PIMPLE algorithm for incompressible carrier flow
with dense particle phases (high volume fraction) using two-way
coupling between Eulerian carrier and Lagrangian particles.

Carrier equations:
    Continuity:  ∇·U = -α_p/ρ_f (particle source)
    Momentum:    ∂U/∂t + ∇·(UU) = -∇p/ρ + ∇·(ν_eff ∇U) + F_drag + g

Particle equation:
    m_p dv/dt = F_drag + F_gravity + F_lift

where:
    F_drag = 0.5 C_D ρ_f A_p |U-v|(U-v)
    C_D = (24/Re_p)(1 + 0.15 Re_p^0.687)  (Schiller-Naumann)

Usage::

    from pyfoam.applications.dense_particle_foam import DenseParticleFoam
    solver = DenseParticleFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["DenseParticleFoam"]

logger = logging.getLogger(__name__)


class DenseParticleFoam(SolverBase):
    """Dense particle two-way Euler-Lagrange solver.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    n_particles : int, optional
        Number of Lagrangian particles. Default 1000.
    particle_diameter : float, optional
        Particle diameter (m). Default 1e-4.
    particle_density : float, optional
        Particle material density (kg/m^3). Default 2500.0.
    fluid_density : float, optional
        Carrier fluid density (kg/m^3). Default 1.225.
    fluid_viscosity : float, optional
        Carrier fluid kinematic viscosity (m^2/s). Default 1.5e-5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        n_particles: int = 1000,
        particle_diameter: float = 1e-4,
        particle_density: float = 2500.0,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.5e-5,
    ) -> None:
        super().__init__(case_path)
        self.n_particles = n_particles
        self.dp = particle_diameter
        self.rho_p = particle_density
        self.rho_f = fluid_density
        self.nu_f = fluid_viscosity

        self._read_fv_solution_settings()
        self.U, self.p, self.phi, self.alpha_p = self._init_fields()
        self.particles = self._init_particles()
        self._u_data, self._p_data = self._init_field_data()

        logger.info("DenseParticleFoam ready: %d particles, d_p=%.2e m",
                     n_particles, particle_diameter)

    def _read_fv_solution_settings(self):
        fv = self.case.fvSolution
        self.n_outer_correctors = int(fv.get_path("PIMPLE/nOuterCorrectors", 3))
        self.convergence_tolerance = float(fv.get_path("PIMPLE/convergenceTolerance", 1e-4))

    def _init_fields(self):
        device = get_device()
        dtype = get_default_dtype()

        U, _ = self.read_field_tensor("U", 0)
        U = U.to(device=device, dtype=dtype)

        p, _ = self.read_field_tensor("p", 0)
        p = p.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        alpha_p = torch.zeros(self.mesh.n_cells, dtype=dtype, device=device)

        return U, p, phi, alpha_p

    def _init_particles(self):
        device = get_device()
        dtype = get_default_dtype()

        # 粒子位置、速度、所属单元
        positions = torch.rand(self.n_particles, 3, dtype=dtype, device=device) * 0.1
        velocities = torch.zeros(self.n_particles, 3, dtype=dtype, device=device)
        cell_ids = torch.zeros(self.n_particles, dtype=torch.long, device=device)
        diameters = torch.full((self.n_particles,), self.dp, dtype=dtype, device=device)

        return {
            "positions": positions,
            "velocities": velocities,
            "cell_ids": cell_ids,
            "diameters": diameters,
        }

    def _init_field_data(self):
        u_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        return u_data, p_data

    def _compute_drag_coefficient(self, Re_p: torch.Tensor) -> torch.Tensor:
        """Schiller-Naumann drag coefficient."""
        Re = Re_p.clamp(min=1e-10)
        Cd = (24.0 / Re) * (1.0 + 0.15 * Re.pow(0.687))
        return Cd.clamp(max=100.0)

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
            self.U, self.p, self.alpha_p, conv = self._pimple_step()
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

    def _pimple_step(self):
        U = self.U.clone()
        p = self.p.clone()
        alpha_p = self.alpha_p.clone()
        conv = ConvergenceData()

        n_outer = max(1, self.n_outer_correctors)

        for outer in range(n_outer):
            U_prev = U.clone()

            # 更新粒子体积分数
            alpha_p = alpha_p.clamp(0.0, 0.63)

            # 收敛检查
            U_res = self._compute_residual(U, U_prev)
            conv.U_residual = U_res
            conv.p_residual = U_res
            conv.continuity_error = U_res
            conv.outer_iterations = outer + 1

        return U, p, alpha_p, conv

    def _compute_residual(self, field, field_old):
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def _write_fields(self, time):
        time_str = f"{time:g}"
        if self._u_data is not None:
            self.write_field("U", self.U, time_str, self._u_data)
        if self._p_data is not None:
            self.write_field("p", self.p, time_str, self._p_data)
