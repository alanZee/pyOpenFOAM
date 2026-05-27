"""
mdFoam — Lennard-Jones molecular dynamics solver.

Implements a simple molecular dynamics simulation using the Lennard-Jones
potential and Velocity Verlet integration:

    U(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]

    F_ij = -dU/dr * r_hat_ij

Integration:
    r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt

Periodic boundary conditions are applied in all three directions.
Thermostat: velocity rescaling (simple Berendsen-like).

Reads:
- ``0/U`` — initial velocities (or zero)
- ``system/controlDict`` — endTime, deltaT
- ``system/fvSolution`` — LJ parameters, thermostat settings

Usage::

    from pyfoam.applications.md_foam import MdFoam

    solver = MdFoam("path/to/case", n_particles=100, T_init=1.0)
    result = solver.run()
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

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["MdFoam", "LJParticle"]

logger = logging.getLogger(__name__)


@dataclass
class LJParticle:
    """A Lennard-Jones simulation particle.

    Attributes
    ----------
    position : list[float]
        3-D position ``[x, y, z]`` in reduced units.
    velocity : list[float]
        3-D velocity ``[u, v, w]`` in reduced units.
    force : list[float]
        3-D force ``[fx, fy, fz]`` in reduced units.
    mass : float
        Particle mass (default: 1.0 in reduced units).
    """
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    force: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    mass: float = 1.0


class MdFoam(SolverBase):
    """Lennard-Jones molecular dynamics solver.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    n_particles : int
        Number of simulation particles.
    T_init : float
        Initial temperature in reduced units.
    rho : float
        Number density in reduced units.
    epsilon : float
        LJ energy parameter (default: 1.0).
    sigma_lj : float
        LJ length parameter (default: 1.0).
    r_cut : float
        Cutoff radius for LJ potential (default: 2.5).
    thermostat_tau : float
        Berendsen thermostat relaxation time (default: 0.1).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        n_particles: int = 64,
        T_init: float = 1.0,
        rho: float = 0.8,
        epsilon: float = 1.0,
        sigma_lj: float = 1.0,
        r_cut: float = 2.5,
        thermostat_tau: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(case_path)

        self.n_particles = n_particles
        self.T_init = T_init
        self.rho = rho
        self.epsilon = epsilon
        self.sigma_lj = sigma_lj
        self.r_cut = r_cut
        self.thermostat_tau = thermostat_tau
        self.seed = seed

        # 读取 fvSolution 设置
        self._read_fv_solution_settings()

        # 计算模拟盒子尺寸
        box_volume = self.n_particles / self.rho
        self.L = box_volume ** (1.0 / 3.0)  # 盒子边长

        # 初始化粒子
        self.particles: list[LJParticle] = []
        self._initialize_particles()

        # 初始化力
        self._compute_forces()

        # 初始温度
        self.T_current = self._compute_temperature()
        self.T_history: list[float] = [self.T_current]
        self.E_potential: float = 0.0

        logger.info(
            "MdFoam ready: %d particles, L=%.4g, T_init=%.4g, rho=%.4g",
            self.n_particles, self.L, self.T_init, self.rho,
        )

    # ------------------------------------------------------------------
    # 配置读取
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read MD settings from fvSolution."""
        fv = self.case.fvSolution
        self.convergence_tolerance = float(
            fv.get_path("mdFoam/convergenceTolerance", 1e-3)
        )

    # ------------------------------------------------------------------
    # 粒子初始化
    # ------------------------------------------------------------------

    def _initialize_particles(self) -> None:
        """Place particles on an FCC lattice with Maxwell-Boltzmann velocities."""
        rng = random.Random(self.seed)

        # FCC 晶格：每晶胞 4 个原子
        n_cells_per_side = max(1, int(math.ceil((self.n_particles / 4.0) ** (1.0 / 3.0))))
        a = self.L / n_cells_per_side  # 晶格常数

        # FCC 基矢偏移
        basis = [
            [0.0, 0.0, 0.0],
            [0.5 * a, 0.5 * a, 0.0],
            [0.5 * a, 0.0, 0.5 * a],
            [0.0, 0.5 * a, 0.5 * a],
        ]

        positions = []
        for ix in range(n_cells_per_side):
            for iy in range(n_cells_per_side):
                for iz in range(n_cells_per_side):
                    for b in basis:
                        pos = [
                            ix * a + b[0],
                            iy * a + b[1],
                            iz * a + b[2],
                        ]
                        positions.append(pos)
                        if len(positions) >= self.n_particles:
                            break
                    if len(positions) >= self.n_particles:
                        break
                if len(positions) >= self.n_particles:
                    break
            if len(positions) >= self.n_particles:
                break

        # Maxwell-Boltzmann 速度分布
        v_std = math.sqrt(self.T_init)  # 约化单位: k_B*T/m = T

        self.particles = []
        for i in range(self.n_particles):
            pos = positions[i] if i < len(positions) else [
                rng.uniform(0, self.L) for _ in range(3)
            ]
            vel = [rng.gauss(0.0, v_std) for _ in range(3)]

            self.particles.append(LJParticle(
                position=pos,
                velocity=vel,
                force=[0.0, 0.0, 0.0],
                mass=1.0,
            ))

        # 移除质心动量
        self._remove_com_velocity()

    def _remove_com_velocity(self) -> None:
        """移除质心速度，确保总动量为零。"""
        vx_cm = sum(p.velocity[0] for p in self.particles) / self.n_particles
        vy_cm = sum(p.velocity[1] for p in self.particles) / self.n_particles
        vz_cm = sum(p.velocity[2] for p in self.particles) / self.n_particles

        for p in self.particles:
            p.velocity[0] -= vx_cm
            p.velocity[1] -= vy_cm
            p.velocity[2] -= vz_cm

    # ------------------------------------------------------------------
    # 力计算
    # ------------------------------------------------------------------

    def _compute_forces(self) -> None:
        """计算所有粒子对之间的 Lennard-Jones 力。

        使用最小镜像约定和截断半径。
        同时计算势能。
        """
        L = self.L
        r_cut = self.r_cut
        r_cut2 = r_cut ** 2
        epsilon = self.epsilon
        sigma = self.sigma_lj

        # 重置力和势能
        for p in self.particles:
            p.force = [0.0, 0.0, 0.0]
        self.E_potential = 0.0

        # 截断处的势能修正
        r6_inv_cut = (sigma / r_cut) ** 6
        U_cut = 4.0 * epsilon * (r6_inv_cut ** 2 - r6_inv_cut)

        for i in range(self.n_particles):
            pi = self.particles[i]
            for j in range(i + 1, self.n_particles):
                pj = self.particles[j]

                # 位移向量（最小镜像）
                dx = pj.position[0] - pi.position[0]
                dy = pj.position[1] - pi.position[1]
                dz = pj.position[2] - pi.position[2]

                # 最小镜像约定
                dx -= L * round(dx / L)
                dy -= L * round(dy / L)
                dz -= L * round(dz / L)

                r2 = dx ** 2 + dy ** 2 + dz ** 2

                if r2 < r_cut2 and r2 > 1e-12:
                    r = math.sqrt(r2)
                    r6_inv = (sigma ** 2 / r2) ** 3
                    r12_inv = r6_inv ** 2

                    # 力的大小: -dU/dr = 4*epsilon*(12*sigma^12/r^13 - 6*sigma^6/r^7)
                    # 简化: F_mag = 24*epsilon*(2*r12_inv - r6_inv) / r2
                    f_mag = 24.0 * epsilon * (2.0 * r12_inv - r6_inv) / r2

                    fx = f_mag * dx
                    fy = f_mag * dy
                    fz = f_mag * dz

                    # 牛顿第三定律
                    pi.force[0] += fx
                    pi.force[1] += fy
                    pi.force[2] += fz
                    pj.force[0] -= fx
                    pj.force[1] -= fy
                    pj.force[2] -= fz

                    # 势能（带截断修正）
                    U = 4.0 * epsilon * (r12_inv - r6_inv) - U_cut
                    self.E_potential += U

    # ------------------------------------------------------------------
    # Velocity Verlet 积分
    # ------------------------------------------------------------------

    def _velocity_verlet_step(self, dt: float) -> None:
        """执行一步 Velocity Verlet 积分。

        1. r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        2. 计算 a(t+dt)
        3. v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt
        """
        dt2 = dt * dt

        # 保存旧加速度
        old_forces = [
            [p.force[0], p.force[1], p.force[2]]
            for p in self.particles
        ]

        # Step 1: 更新位置
        for i, p in enumerate(self.particles):
            a_old = [
                old_forces[i][0] / p.mass,
                old_forces[i][1] / p.mass,
                old_forces[i][2] / p.mass,
            ]

            p.position[0] += p.velocity[0] * dt + 0.5 * a_old[0] * dt2
            p.position[1] += p.velocity[1] * dt + 0.5 * a_old[1] * dt2
            p.position[2] += p.velocity[2] * dt + 0.5 * a_old[2] * dt2

            # 周期性边界条件
            p.position[0] %= self.L
            p.position[1] %= self.L
            p.position[2] %= self.L

        # Step 2: 计算新力
        self._compute_forces()

        # Step 3: 更新速度
        for i, p in enumerate(self.particles):
            a_old = [
                old_forces[i][0] / p.mass,
                old_forces[i][1] / p.mass,
                old_forces[i][2] / p.mass,
            ]
            a_new = [
                p.force[0] / p.mass,
                p.force[1] / p.mass,
                p.force[2] / p.mass,
            ]

            p.velocity[0] += 0.5 * (a_old[0] + a_new[0]) * dt
            p.velocity[1] += 0.5 * (a_old[1] + a_new[1]) * dt
            p.velocity[2] += 0.5 * (a_old[2] + a_new[2]) * dt

    # ------------------------------------------------------------------
    # 温度与动能
    # ------------------------------------------------------------------

    def _compute_kinetic_energy(self) -> float:
        """计算总动能。"""
        KE = 0.0
        for p in self.particles:
            v2 = p.velocity[0] ** 2 + p.velocity[1] ** 2 + p.velocity[2] ** 2
            KE += 0.5 * p.mass * v2
        return KE

    def _compute_temperature(self) -> float:
        """计算瞬时温度。

        T = 2 * KE / (3 * N * k_B)，约化单位 k_B=1。
        """
        KE = self._compute_kinetic_energy()
        if self.n_particles > 0:
            return 2.0 * KE / (3.0 * self.n_particles)
        return 0.0

    def _apply_thermostat(self, T_target: float) -> None:
        """应用 Berendsen 速度重标定温控。

        lambda = sqrt(1 + dt/tau * (T_target/T_current - 1))
        """
        T_current = self._compute_temperature()
        if T_current < 1e-12:
            return

        dt = self.delta_t
        tau = self.thermostat_tau

        lam_sq = 1.0 + (dt / tau) * (T_target / T_current - 1.0)
        if lam_sq <= 0:
            lam_sq = 0.01
        lam = math.sqrt(lam_sq)

        for p in self.particles:
            p.velocity[0] *= lam
            p.velocity[1] *= lam
            p.velocity[2] *= lam

    # ------------------------------------------------------------------
    # 热力学量
    # ------------------------------------------------------------------

    @property
    def kinetic_energy(self) -> float:
        """总动能。"""
        return self._compute_kinetic_energy()

    @property
    def potential_energy(self) -> float:
        """总势能。"""
        return self.E_potential

    @property
    def total_energy(self) -> float:
        """总能量（动能 + 势能）。"""
        return self.kinetic_energy + self.potential_energy

    @property
    def temperature(self) -> float:
        """瞬时温度。"""
        return self._compute_temperature()

    @property
    def pressure(self) -> float:
        """维里定理计算压强。

        P = N*k_B*T/V + virial/(3*V)
        简化版本：仅使用动能项。
        """
        V = self.L ** 3
        if V > 0:
            return self.n_particles * self.T_current / V
        return 0.0

    # ------------------------------------------------------------------
    # 主求解循环
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """运行 mdFoam 求解器。

        Returns
        -------
        dict
            求解结果，包含 ``converged``, ``steps``, ``T_final``,
            ``E_total``, ``E_potential``, ``E_kinetic``。
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

        logger.info("Starting mdFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  n_particles=%d, L=%.4g", self.n_particles, self.L)

        converged = False

        for t, step in time_loop:
            # Velocity Verlet 积分
            self._velocity_verlet_step(self.delta_t)

            # 温度计算与温控
            self.T_current = self._compute_temperature()
            self._apply_thermostat(self.T_init)
            self.T_current = self._compute_temperature()
            self.T_history.append(self.T_current)

            # 收敛检查（基于温度稳定性）
            if len(self.T_history) >= 2:
                T_residual = abs(self.T_history[-1] - self.T_history[-2])
                converged = convergence.update(step + 1, {"T": T_residual})

            if converged:
                logger.info("MD converged at step %d (t=%.6g)", step + 1, t)
                break

        logger.info("mdFoam completed")
        logger.info("  T_final=%.6g, E_total=%.6g", self.T_current, self.total_energy)

        return {
            "converged": converged,
            "steps": time_loop.step + 1,
            "T_final": self.T_current,
            "E_total": self.total_energy,
            "E_potential": self.potential_energy,
            "E_kinetic": self.kinetic_energy,
        }

    # ------------------------------------------------------------------
    # 场输出（简化：写粒子坐标到文件）
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write particle positions to a time directory."""
        time_str = f"{time:g}"
        time_dir = self.case_path / time_str
        time_dir.mkdir(parents=True, exist_ok=True)

        # 写入粒子位置作为简单文本
        positions_file = time_dir / "positions.xyz"
        with open(positions_file, "w") as f:
            f.write(f"{self.n_particles}\n")
            f.write(f"t={time}\n")
            for p in self.particles:
                f.write(f"Ar {p.position[0]:.6g} {p.position[1]:.6g} {p.position[2]:.6g}\n")
