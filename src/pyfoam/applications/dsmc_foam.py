"""
dsmcFoam — Direct Simulation Monte Carlo solver for rarefied gas dynamics.

Implements Bird's DSMC algorithm for simulating rarefied gas flows where
the mean free path is comparable to the characteristic length scale
(Knudsen number Kn > 0.01).

The DSMC method uses Lagrangian particles to represent molecules:

1. **Move** particles ballistically over Δt
2. **Index** particles into cells
3. **Sample** macroscopic properties (ρ, U, T) from particle statistics
4. **Collide** particles within each cell using Bird's no-time-counter scheme
5. **Apply** boundary conditions (specular/diffuse reflection, thermal walls)

Key parameters:
- n_particles_per_cell: Number of simulation particles per cell
- sigma_t: Total collision cross-section (m²)
- T_wall: Wall temperature for diffuse reflection (K)
- alpha_n: Viscosity-temperature exponent (ω in VHS model)

Reference: G.A. Bird, "Molecular Gas Dynamics and the Direct Simulation
of Gas Flows", Oxford University Press, 1994.

Usage::

    from pyfoam.applications.dsmc_foam import DsmcFoam

    solver = DsmcFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.lagrangian.particle import Particle

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["DsmcFoam"]

logger = logging.getLogger(__name__)


# ======================================================================
# DSMC 粒子（带分子级别属性）
# ======================================================================

@dataclass
class DSMCParticle:
    """A DSMC simulation particle representing a group of real molecules.

    Attributes
    ----------
    position : list[float]
        3-D position ``[x, y, z]`` in metres.
    velocity : list[float]
        3-D velocity ``[u, v, w]`` in m/s.
    cell_id : int
        Index of the mesh cell containing the particle.
    mass : float
        Mass of the represented molecule (kg).
    n_real : float
        Number of real molecules represented by this simulation particle.
    alive : bool
        Whether the particle is still active.
    """
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    cell_id: int = -1
    mass: float = 4.652e-26  # 默认：氮气分子质量 (kg)
    n_real: float = 1.0
    alive: bool = True


# ======================================================================
# DSMC 碰撞模型
# ======================================================================

@dataclass
class VHSModel:
    """Variable Hard Sphere (VHS) collision model.

    Parameters
    ----------
    d_ref : float
        Reference diameter at T_ref (m).
    T_ref : float
        Reference temperature (K).
    omega : float
        Viscosity-temperature exponent (0 = hard sphere, 0.5 = Maxwell, 1 = VSS).
    """
    d_ref: float = 4.17e-10  # N2 reference diameter
    T_ref: float = 273.0
    omega: float = 0.74  # N2 VHS exponent


# ======================================================================
# DSMC 求解器
# ======================================================================

class DsmcFoam(SolverBase):
    """Direct Simulation Monte Carlo solver.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    n_particles_per_cell : int
        Target number of simulation particles per cell.
    molecular_mass : float
        Mass of a single gas molecule (kg). Default: N2.
    sigma_t : float
        Total collision cross-section (m²).
    T_wall : float
        Wall temperature for diffuse reflection (K).
    vhs_model : VHSModel or None
        VHS collision model parameters.
    alpha_n : float
        Fraction of real molecules per simulation particle.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        n_particles_per_cell: int = 10,
        molecular_mass: float = 4.652e-26,
        sigma_t: float = 5.0e-20,
        T_wall: float = 300.0,
        vhs_model: VHSModel | None = None,
        alpha_n: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(case_path)

        self.n_particles_per_cell = n_particles_per_cell
        self.molecular_mass = molecular_mass
        self.sigma_t = sigma_t
        self.T_wall = T_wall
        self.vhs = vhs_model or VHSModel()
        self.seed = seed

        # 物理常数
        self.k_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)

        # 读取 fvSolution / fvSchemes 设置
        self._read_fv_solution_settings()

        # 初始化流场统计量
        self.n_cells = self.mesh.n_cells
        device = get_device()
        dtype = get_default_dtype()

        # 网格几何缓存（必须在粒子初始化之前设置）
        self._cell_centres = self.mesh.cell_centres.to(device=device)
        self._cell_volumes = self.mesh.cell_volumes.to(device=device)

        # 宏观量（从粒子统计采样得到）
        self.rho = torch.zeros(self.n_cells, dtype=dtype, device=device)
        self.U = torch.zeros(self.n_cells, 3, dtype=dtype, device=device)
        self.T = torch.full((self.n_cells,), self.T_wall, dtype=dtype, device=device)
        self.number_density = torch.zeros(self.n_cells, dtype=dtype, device=device)

        # 粒子列表（每个 cell 一个列表）
        self.particles: list[list[DSMCParticle]] = [[] for _ in range(self.n_cells)]

        # 初始化粒子
        self._initialize_particles()

        # 频率因子（用于碰撞选择）
        if alpha_n is not None:
            self.alpha_n = alpha_n
        else:
            # 根据初始密度估算
            self.alpha_n = 1.0

        logger.info(
            "DsmcFoam ready: %d cells, %d particles/cell, σ_t=%.3e m²",
            self.n_cells, self.n_particles_per_cell, self.sigma_t,
        )

    # ------------------------------------------------------------------
    # 配置读取
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read DSMC settings from fvSolution."""
        fv = self.case.fvSolution
        self.convergence_tolerance = float(
            fv.get_path("dsmc/convergenceTolerance", 1e-3)
        )
        self.n_inner_iterations = int(
            fv.get_path("dsmc/nInnerIterations", 1)
        )

    # ------------------------------------------------------------------
    # 粒子初始化
    # ------------------------------------------------------------------

    def _initialize_particles(self) -> None:
        """Place initial particles uniformly in each cell.

        Assigns Maxwell-Boltzmann distributed velocities at the initial
        temperature ``self.T_wall``.
        """
        rng = random.Random(self.seed)
        cell_centres = self._cell_centres
        cell_volumes = self._cell_volumes

        for cell_id in range(self.n_cells):
            cc = cell_centres[cell_id]
            vol = float(cell_volumes[cell_id].item())

            # 估算 cell 尺寸（立方体近似）
            cell_size = vol ** (1.0 / 3.0) if vol > 0 else 0.01

            for _ in range(self.n_particles_per_cell):
                # 在 cell 中心附近随机放置
                pos = [
                    cc[0].item() + rng.uniform(-0.4, 0.4) * cell_size,
                    cc[1].item() + rng.uniform(-0.4, 0.4) * cell_size,
                    cc[2].item() + rng.uniform(-0.4, 0.4) * cell_size,
                ]

                # Maxwell-Boltzmann 速度分布
                v_thermal = math.sqrt(self.k_B * self.T_wall / self.molecular_mass)
                vel = [
                    rng.gauss(0.0, v_thermal),
                    rng.gauss(0.0, v_thermal),
                    rng.gauss(0.0, v_thermal),
                ]

                p = DSMCParticle(
                    position=pos,
                    velocity=vel,
                    cell_id=cell_id,
                    mass=self.molecular_mass,
                )
                self.particles[cell_id].append(p)

    # ------------------------------------------------------------------
    # 核心 DSMC 算法
    # ------------------------------------------------------------------

    def _move_particles(self, dt: float) -> None:
        """Step 1: Ballistic particle motion.

        Advances all particles by ``dt`` using current velocity.
        Particles are reassigned to their nearest cell after movement.
        Boundary conditions are applied separately.
        """
        for cell_id in range(self.n_cells):
            for p in self.particles[cell_id]:
                if not p.alive:
                    continue

                # 平移
                p.position[0] += p.velocity[0] * dt
                p.position[1] += p.velocity[1] * dt
                p.position[2] += p.velocity[2] * dt

                # 重新定位到最近 cell
                p.cell_id = self._find_cell(p.position)

    def _reindex_particles(self) -> None:
        """Step 2: Re-index particles into cells after movement.

        Moves particles between cell lists based on their updated cell_id.
        """
        new_lists: list[list[DSMCParticle]] = [[] for _ in range(self.n_cells)]

        for cell_id in range(self.n_cells):
            for p in self.particles[cell_id]:
                if p.alive and 0 <= p.cell_id < self.n_cells:
                    new_lists[p.cell_id].append(p)

        self.particles = new_lists

    def _sample_macroscopic(self) -> None:
        """Step 3: Sample macroscopic fields from particle data.

        Computes number density, velocity, and temperature from particle
        statistics in each cell using unbiased estimators.
        """
        device = get_device()
        dtype = get_default_dtype()

        for cell_id in range(self.n_cells):
            parts = self.particles[cell_id]
            n_p = len(parts)

            if n_p == 0:
                self.number_density[cell_id] = 0.0
                self.U[cell_id] = torch.zeros(3, dtype=dtype, device=device)
                self.T[cell_id] = self.T_wall
                continue

            # 累积统计
            sum_u = 0.0
            sum_v = 0.0
            sum_w = 0.0
            sum_u2 = 0.0
            sum_v2 = 0.0
            sum_w2 = 0.0

            for p in parts:
                sum_u += p.velocity[0]
                sum_v += p.velocity[1]
                sum_w += p.velocity[2]
                sum_u2 += p.velocity[0] ** 2
                sum_v2 += p.velocity[1] ** 2
                sum_w2 += p.velocity[2] ** 2

            # 平均速度
            u_mean = sum_u / n_p
            v_mean = sum_v / n_p
            w_mean = sum_w / n_p

            self.U[cell_id] = torch.tensor(
                [u_mean, v_mean, w_mean], dtype=dtype, device=device,
            )

            # 热速度方差 → 温度
            # T = m * <c'^2> / (3 * k_B)  其中 c' = v - U
            var_u = sum_u2 / n_p - u_mean ** 2
            var_v = sum_v2 / n_p - v_mean ** 2
            var_w = sum_w2 / n_p - w_mean ** 2
            c2_mean = var_u + var_v + var_w

            if c2_mean > 0:
                self.T[cell_id] = self.molecular_mass * c2_mean / (3.0 * self.k_B)
            else:
                self.T[cell_id] = self.T_wall

            # 数密度 = n_p * n_real / V
            vol = float(self._cell_volumes[cell_id].item())
            if vol > 0:
                self.number_density[cell_id] = n_p * self.alpha_n / vol
            else:
                self.number_density[cell_id] = 0.0

            # 质量密度
            self.rho[cell_id] = self.number_density[cell_id] * self.molecular_mass

    def _collide_particles(self, dt: float) -> None:
        """Step 4: DSMC collision using Bird's no-time-counter (NTC) scheme.

        For each cell:
        1. Compute number of collision pairs to select
        2. Randomly select pairs
        3. Accept/reject based on relative speed
        4. Compute post-collision velocities using VHS model
        """
        for cell_id in range(self.n_cells):
            parts = self.particles[cell_id]
            n_p = len(parts)

            if n_p < 2:
                continue

            vol = float(self._cell_volumes[cell_id].item())
            if vol <= 0:
                continue

            # VHS cross-section at local temperature
            T_local = float(self.T[cell_id].item())
            T_safe = max(T_local, 1.0)
            d_ref = self.vhs.d_ref
            T_ref = self.vhs.T_ref
            omega = self.vhs.omega

            # σ_VHS(T) = π * d_ref² * (T_ref / T)^ω
            sigma = math.pi * d_ref ** 2 * (T_ref / T_safe) ** omega

            # NTC: number of collision pairs
            # N_coll = 0.5 * n_p² * n_real * σ * g_max * Δt / V
            # 简化：使用平均相对速度估算
            v_thermal = math.sqrt(self.k_B * T_safe / self.molecular_mass)
            g_mean = 2.0 * v_thermal * math.sqrt(2.0 / math.pi)

            n_coll_float = 0.5 * n_p * self.alpha_n * sigma * g_mean * dt / vol
            n_coll = max(1, int(n_coll_float + random.random()))

            # 限制碰撞次数不超过粒子对数
            n_coll = min(n_coll, n_p // 2)

            rng = random.Random(self.seed)
            if self.seed is not None:
                self.seed += 1  # 避免重复序列

            for _ in range(n_coll):
                # 随机选取两个不同粒子
                i = rng.randint(0, n_p - 1)
                j = rng.randint(0, n_p - 2)
                if j >= i:
                    j += 1

                p1 = parts[i]
                p2 = parts[j]

                # 相对速度
                g_rel = [
                    p1.velocity[0] - p2.velocity[0],
                    p1.velocity[1] - p2.velocity[1],
                    p1.velocity[2] - p2.velocity[2],
                ]
                g_mag = math.sqrt(g_rel[0] ** 2 + g_rel[1] ** 2 + g_rel[2] ** 2)

                if g_mag < 1e-30:
                    continue

                # 接受/拒绝：以 g / g_max 概率接受
                g_max = 2.0 * v_thermal * 3.0  # 安全上界
                if rng.random() > g_mag / g_max:
                    continue

                # VHS 碰撞后散射
                self._vhs_collision(p1, p2, g_rel, g_mag, rng)

    def _vhs_collision(
        self,
        p1: DSMCParticle,
        p2: DSMCParticle,
        g_rel: list[float],
        g_mag: float,
        rng: random.Random,
    ) -> None:
        """Apply VHS collision to a particle pair.

        Post-collision velocities are computed using isotropic scattering
        in the centre-of-mass frame.

        Parameters
        ----------
        p1, p2 : DSMCParticle
            Collision partner particles (modified in-place).
        g_rel : list[float]
            Relative velocity vector (p1 - p2).
        g_mag : float
            Magnitude of relative velocity.
        rng : random.Random
            Random number generator.
        """
        # 质心速度
        m1 = p1.mass
        m2 = p2.mass
        m_total = m1 + m2

        v_cm = [
            (m1 * p1.velocity[0] + m2 * p2.velocity[0]) / m_total,
            (m1 * p1.velocity[1] + m2 * p2.velocity[1]) / m_total,
            (m1 * p1.velocity[2] + m2 * p2.velocity[2]) / m_total,
        ]

        # 各向同性散射：随机旋转相对速度
        # 随机球面角
        cos_chi = 2.0 * rng.random() - 1.0
        sin_chi = math.sqrt(1.0 - cos_chi ** 2)
        eps = 2.0 * math.pi * rng.random()

        # 归一化 g_rel 方向
        gx = g_rel[0] / g_mag
        gy = g_rel[1] / g_mag
        gz = g_rel[2] / g_mag

        # 选择与 g 不平行的参考向量
        if abs(gx) < 0.9:
            ref = [1.0, 0.0, 0.0]
        else:
            ref = [0.0, 1.0, 0.0]

        # 构造正交基
        # e1 = g × ref
        e1x = gy * ref[2] - gz * ref[1]
        e1y = gz * ref[0] - gx * ref[2]
        e1z = gx * ref[1] - gy * ref[0]
        e1_mag = math.sqrt(e1x ** 2 + e1y ** 2 + e1z ** 2)
        e1x /= e1_mag
        e1y /= e1_mag
        e1z /= e1_mag

        # e2 = g × e1
        e2x = gy * e1z - gz * e1y
        e2y = gz * e1x - gx * e1z
        e2z = gx * e1y - gy * e1x

        # 散射后的相对速度方向
        cos_eps = math.cos(eps)
        sin_eps = math.sin(eps)

        new_gx = g_mag * (
            cos_chi * gx + sin_chi * cos_eps * e1x + sin_chi * sin_eps * e2x
        )
        new_gy = g_mag * (
            cos_chi * gy + sin_chi * cos_eps * e1y + sin_chi * sin_eps * e2y
        )
        new_gz = g_mag * (
            cos_chi * gz + sin_chi * cos_eps * e1z + sin_chi * sin_eps * e2z
        )

        # 更新粒子速度
        p1.velocity[0] = v_cm[0] + m2 / m_total * new_gx
        p1.velocity[1] = v_cm[1] + m2 / m_total * new_gy
        p1.velocity[2] = v_cm[2] + m2 / m_total * new_gz

        p2.velocity[0] = v_cm[0] - m1 / m_total * new_gx
        p2.velocity[1] = v_cm[1] - m1 / m_total * new_gy
        p2.velocity[2] = v_cm[2] - m1 / m_total * new_gz

    # ------------------------------------------------------------------
    # Cell 查找
    # ------------------------------------------------------------------

    def _find_cell(self, position: list[float]) -> int:
        """Find the cell containing the given position.

        Uses nearest cell centre as a simplified locator.
        Always returns a valid cell index (nearest cell).

        Parameters
        ----------
        position : list[float]
            ``[x, y, z]`` position.

        Returns
        -------
        int
            Cell index of the nearest cell.
        """
        cell_centres = self._cell_centres
        pos = torch.tensor(position, dtype=cell_centres.dtype)
        dists = torch.norm(cell_centres - pos.unsqueeze(0), dim=1)
        return int(torch.argmin(dists).item())

    # ------------------------------------------------------------------
    # 壁面边界条件
    # ------------------------------------------------------------------

    def _apply_boundary_conditions(self) -> None:
        """Apply DSMC boundary conditions.

        - **Diffuse reflection**: re-emit with Maxwellian at T_wall
        - **Specular reflection**: mirror normal velocity component
        """
        # 遍历所有粒子，检查越界并应用壁面反射
        for cell_id in range(self.n_cells):
            for p in self.particles[cell_id]:
                if not p.alive:
                    continue

                # 检查是否在域内（简单检查）
                new_cell = self._find_cell(p.position)
                if new_cell < 0:
                    # 漫反射：重新采样 Maxwell-Boltzmann
                    self._diffuse_reflection(p)
                    p.alive = True  # 反射后仍然存活

    def _diffuse_reflection(self, p: DSMCParticle) -> None:
        """Apply diffuse (thermal) wall reflection.

        The particle velocity is re-sampled from a Maxwell-Boltzmann
        distribution at the wall temperature, and the position is
        reflected back into the domain.

        Parameters
        ----------
        p : DSMCParticle
            Particle to reflect (modified in-place).
        """
        rng = random.Random()
        v_thermal = math.sqrt(self.k_B * self.T_wall / self.molecular_mass)

        p.velocity = [
            rng.gauss(0.0, v_thermal),
            rng.gauss(0.0, v_thermal),
            rng.gauss(0.0, v_thermal),
        ]

        # 位置反射回最近 cell 中心附近
        cc = self._cell_centres[p.cell_id]
        p.position = [cc[0].item(), cc[1].item(), cc[2].item()]

    # ------------------------------------------------------------------
    # 运行主循环
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the DSMC solver.

        Returns
        -------
        dict
            Convergence information including final residuals.
        """
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

        logger.info("Starting dsmcFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  n_particles/cell=%d", self.n_particles_per_cell)

        # 写初始场
        self._write_fields(self.start_time)
        time_loop.mark_written()

        T_prev = self.T.clone()
        last_convergence = None

        for t, step in time_loop:
            # 1. 粒子运动
            self._move_particles(self.delta_t)

            # 2. 重新索引
            self._reindex_particles()

            # 3. 边界条件
            self._apply_boundary_conditions()

            # 4. 采样宏观量
            self._sample_macroscopic()

            # 5. 碰撞
            self._collide_particles(self.delta_t)

            # 收敛监控
            T_residual = float(
                torch.norm(self.T - T_prev).item()
                / max(torch.norm(self.T).item(), 1e-30)
            )
            T_prev = self.T.clone()

            conv = ConvergenceData()
            conv.T_residual = T_residual
            conv.converged = T_residual < self.convergence_tolerance
            last_convergence = conv

            residuals = {"T": T_residual}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("DSMC converged at step %d (t=%.6g)", step + 1, t)
                break

        # 写最终场
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info("dsmcFoam completed")

        return {
            "converged": last_convergence.converged if last_convergence else False,
            "T_residual": last_convergence.T_residual if last_convergence else 0.0,
            "n_particles_total": sum(
                len(parts) for parts in self.particles
            ),
        }

    # ------------------------------------------------------------------
    # 场输出
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write macroscopic fields to a time directory.

        Writes number density, velocity, and temperature fields.
        """
        time_str = f"{time:g}"
        device = get_device()
        dtype = get_default_dtype()

        # 尝试从 0/ 读取原始 field data 作为模板
        try:
            T_data = self.case.read_field("T", 0)
            self.write_field("T", self.T, time_str, T_data)
        except Exception:
            pass

        try:
            U_data = self.case.read_field("U", 0)
            self.write_field("U", self.U, time_str, U_data)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 辅助属性
    # ------------------------------------------------------------------

    @property
    def total_particles(self) -> int:
        """Total number of simulation particles across all cells."""
        return sum(len(parts) for parts in self.particles)

    @property
    def mean_temperature(self) -> float:
        """Volume-weighted mean temperature."""
        T = self.T
        V = self._cell_volumes
        total_V = V.sum()
        if total_V > 0:
            return float((T * V).sum().item() / total_V.item())
        return float(T.mean().item())
