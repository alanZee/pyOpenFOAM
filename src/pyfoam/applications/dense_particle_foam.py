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
import re
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.solvers.pressure_equation import (
    assemble_pressure_equation,
    solve_pressure_equation,
    correct_velocity,
    correct_face_flux,
)
from pyfoam.solvers.linear_solver import create_solver

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

    def _build_boundary_conditions(self):
        """从 0/U 边界场构建速度 BC 张量。"""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        U_bc = torch.full((n_cells, 3), float("nan"), dtype=dtype, device=device)
        U_field_data = self._u_data
        boundary_field = U_field_data.boundary_field
        if boundary_field is None or len(boundary_field) == 0:
            return U_bc
        mesh_boundary = self.case.boundary
        owner = self.mesh.owner
        mesh_patches = {}
        for bp in mesh_boundary:
            mesh_patches[bp.name] = {"startFace": bp.start_face, "nFaces": bp.n_faces}
        for patch in boundary_field:
            is_fixed = patch.patch_type == "fixedValue" and patch.value is not None
            is_noslip = patch.patch_type == "noSlip"
            if is_fixed or is_noslip:
                if is_fixed:
                    match = re.search(r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)", str(patch.value))
                    value = (float(match.group(1)), float(match.group(2)), float(match.group(3))) if match else None
                else:
                    value = (0.0, 0.0, 0.0)
                if value is not None:
                    mesh_info = mesh_patches.get(patch.name)
                    if mesh_info is not None:
                        sf = mesh_info["startFace"]
                        nf = mesh_info["nFaces"]
                        for i in range(nf):
                            cell_idx = owner[sf + i].item()
                            U_bc[cell_idx] = torch.tensor(value, dtype=dtype)
        return U_bc

    def _read_fv_solution_settings(self):
        fv = self.case.fvSolution
        self.n_outer_correctors = int(fv.get_path("PIMPLE/nOuterCorrectors", 3))
        self.n_correctors = int(fv.get_path("PIMPLE/nCorrectors", 2))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))
        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.convergence_tolerance = float(fv.get_path("PIMPLE/convergenceTolerance", 1e-4))
        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

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

    def _compute_particle_volume_fraction(self) -> torch.Tensor:
        """计算粒子体积分数 alpha_p。"""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        cell_centres = mesh.cell_centres.to(device=device, dtype=dtype)
        cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype).clamp(min=1e-30)

        # 粒子体积
        V_p = (torch.pi / 6.0) * self.particles["diameters"] ** 3

        # 将粒子分配到最近单元
        positions = self.particles["positions"]
        dists = torch.cdist(positions.unsqueeze(0), cell_centres.unsqueeze(0)).squeeze(0)
        cell_ids = dists.argmin(dim=1)
        self.particles["cell_ids"] = cell_ids

        # 累积粒子体积分数
        alpha_p = scatter_add(V_p, cell_ids, n_cells) / cell_volumes
        return alpha_p.clamp(0.0, 0.63)

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
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        alpha_p = self._compute_particle_volume_fraction()
        conv = ConvergenceData()

        n_outer = max(1, self.n_outer_correctors)
        mu_f = self.rho_f * self.nu_f

        # 构建边界条件
        U_bc = self._build_boundary_conditions()
        bc_mask = ~torch.isnan(U_bc[:, 0])

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            # 动量预测（含粒子阻力源项）
            U, A_p, H = self._momentum_predictor(
                U, p, self.phi, alpha_p, mu_f,
            )

            # 应用边界条件
            if bc_mask.any():
                U[bc_mask] = U_bc[bc_mask]

            # PISO 校正
            n_internal = mesh.n_internal_faces
            int_owner = mesh.owner[:n_internal]
            int_neigh = mesh.neighbour
            w = mesh.face_weights[:n_internal]

            for corr in range(self.n_correctors):
                HbyA = H / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                HbyA_face = (
                    w.unsqueeze(-1) * HbyA[int_owner]
                    + (1.0 - w).unsqueeze(-1) * HbyA[int_neigh]
                )
                phiHbyA = (HbyA_face * mesh.face_areas[:n_internal]).sum(dim=1)

                p_solver = create_solver(
                    "PCG", tolerance=self.p_tolerance, max_iter=self.p_max_iter,
                )
                phi_full = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
                phi_full[:n_internal] = phiHbyA
                p_eqn = assemble_pressure_equation(phi_full, A_p, mesh)
                p, _, _ = solve_pressure_equation(
                    p_eqn, p, p_solver,
                    tolerance=self.p_tolerance, max_iter=self.p_max_iter,
                )

                U = correct_velocity(U, HbyA, p, A_p, mesh)
                if bc_mask.any():
                    U[bc_mask] = U_bc[bc_mask]
                self.phi = correct_face_flux(phi_full, p, A_p, mesh)

            # 松弛
            if self.alpha_U < 1.0:
                U = self.alpha_U * U + (1.0 - self.alpha_U) * U_prev

            # 更新粒子位置（半隐式积分）
            self._update_particles(U)

            # 收敛检查
            U_res = self._compute_residual(U, U_prev)
            p_res = self._compute_residual(p, p_prev)
            conv.U_residual = U_res
            conv.p_residual = p_res
            conv.continuity_error = U_res
            conv.outer_iterations = outer + 1

        alpha_p = self._compute_particle_volume_fraction()
        return U, p, alpha_p, conv

    def _momentum_predictor(self, U, p, phi, alpha_p, mu_f):
        """混合物动量方程（含粒子阻力）。"""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes.clamp(min=1e-30)
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        # 混合密度
        rho_m = alpha_p * self.rho_p + (1.0 - alpha_p) * self.rho_f

        # 粘性扩散（简化：使用流体粘度）
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_f * S_mag * delta_f

        # 对流项（迎风格式，使用混合密度）
        flux = phi[:n_internal]
        rho_P = gather(rho_m, int_owner)
        rho_N = gather(rho_m, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        dt = self.delta_t
        rho_V_dt = rho_m * cell_volumes / dt

        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        A_p = torch.zeros(n_cells, dtype=dtype, device=device)
        A_p = A_p + scatter_add(
            (diff_coeff - flux_neg * rho_face) / V_P, int_owner, n_cells,
        )
        A_p = A_p + scatter_add(
            (diff_coeff + flux_pos * rho_face) / V_N, int_neigh, n_cells,
        )
        A_p = A_p + rho_V_dt

        # H(U)
        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        H.index_add_(0, int_owner,
                      lower.unsqueeze(-1) * U[int_neigh] * V_P.unsqueeze(-1))
        H.index_add_(0, int_neigh,
                      upper.unsqueeze(-1) * U[int_owner] * V_N.unsqueeze(-1))
        H = H + rho_V_dt.unsqueeze(-1) * self.U

        # 压力梯度
        w = mesh.face_weights[:n_internal]
        p_P = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

        # 粒子阻力源项
        F_drag = self._compute_drag_source(U, alpha_p)

        # 重力源项
        F_gravity = rho_m.unsqueeze(-1) * torch.tensor(
            [0.0, -9.81, 0.0], dtype=dtype, device=device,
        )

        source = H - grad_p + F_drag + F_gravity

        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)
        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        return U_new, A_p, H

    def _compute_drag_source(self, U_fluid, alpha_p):
        """粒子对流体的阻力源项。"""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        cell_centres = mesh.cell_centres.to(device=device, dtype=dtype)

        # 粒子所在单元的速度差
        positions = self.particles["positions"]
        velocities = self.particles["velocities"]
        cell_ids = self.particles["cell_ids"]
        d_p = self.particles["diameters"]

        U_p_at_cells = U_fluid[cell_ids]
        rel_vel = U_p_at_cells - velocities
        rel_vel_mag = rel_vel.norm(dim=1).clamp(min=1e-10)

        # Re_p = rho_f * |U-v| * d / mu
        Re_p = self.rho_f * rel_vel_mag * d_p / (self.nu_f * self.rho_f + 1e-30)
        Cd = self._compute_drag_coefficient(Re_p)

        # 粒子阻力
        A_p_cross = torch.pi / 4.0 * d_p ** 2
        F_drag_particle = 0.5 * Cd.unsqueeze(-1) * self.rho_f * A_p_cross.unsqueeze(-1) * rel_vel_mag.unsqueeze(-1) * rel_vel

        # 累积到网格
        F_cell = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        F_cell.index_add_(0, cell_ids, F_drag_particle)

        # 负号：粒子对流体的反作用力（阻碍流体运动）
        V = mesh.cell_volumes.clamp(min=1e-30)
        return -F_cell / V.unsqueeze(-1)

    def _update_particles(self, U_fluid):
        """更新粒子位置（显式 Euler）。"""
        device = get_device()
        dtype = get_default_dtype()

        cell_ids = self.particles["cell_ids"]
        d_p = self.particles["diameters"]
        positions = self.particles["positions"]
        velocities = self.particles["velocities"]

        # 流体速度在粒子位置
        U_p = U_fluid[cell_ids]
        rel_vel = U_p - velocities
        rel_vel_mag = rel_vel.norm(dim=1).clamp(min=1e-10)

        # Re_p 和阻力系数
        Re_p = self.rho_f * rel_vel_mag * d_p / (self.nu_f * self.rho_f + 1e-30)
        Cd = self._compute_drag_coefficient(Re_p)

        # 粒子加速度：a = F_drag/m_p + g
        V_p = (torch.pi / 6.0) * d_p ** 3
        m_p = self.rho_p * V_p
        A_p = torch.pi / 4.0 * d_p ** 2

        F_drag = 0.5 * Cd.unsqueeze(-1) * self.rho_f * A_p.unsqueeze(-1) * rel_vel_mag.unsqueeze(-1) * rel_vel
        g = torch.tensor([0.0, -9.81, 0.0], dtype=dtype, device=device)
        F_grav = m_p.unsqueeze(-1) * g.unsqueeze(0)

        accel = (F_drag + F_grav) / m_p.unsqueeze(-1).clamp(min=1e-30)

        # 显式 Euler 积分
        dt = self.delta_t
        velocities = velocities + accel * dt
        positions = positions + velocities * dt

        # 简单反射边界（防止粒子离开域）
        mesh = self.mesh
        x_min = mesh.cell_centres.min(dim=0).values
        x_max = mesh.cell_centres.max(dim=0).values
        for d in range(3):
            mask_lo = positions[:, d] < x_min[d]
            positions[mask_lo, d] = x_min[d]
            velocities[mask_lo, d] = -velocities[mask_lo, d] * 0.5
            mask_hi = positions[:, d] > x_max[d]
            positions[mask_hi, d] = x_max[d]
            velocities[mask_hi, d] = -velocities[mask_hi, d] * 0.5

        self.particles["positions"] = positions
        self.particles["velocities"] = velocities

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
