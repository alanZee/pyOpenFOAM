"""
fluidFoam — unified compressible solver with full energy equation.

Implements the PIMPLE algorithm for transient compressible
Navier-Stokes equations coupled with the full energy equation,
supporting variable density, temperature-dependent transport,
and viscous dissipation.

Governing equations:
    Continuity:  ∂ρ/∂t + ∇·(ρU) = 0
    Momentum:    ∂(ρU)/∂t + ∇·(ρUU) = -∇p + ∇·(μ∇U) + S_mom
    Energy:      ∂(ρCpT)/∂t + ∇·(ρUCpT) = ∇·(κ∇T) + p∇·U + Φ
    EOS:         ρ = ρ(p, T)

Algorithm (PIMPLE per time step):
    1. Store old fields
    2. Outer corrector loop:
       a. Momentum predictor (upwind convection + viscous diffusion + time derivative)
       b. PISO inner pressure correction loop
       c. Under-relaxation
       d. Update density from EOS
       e. Solve energy equation
       f. Update density again
       g. Check convergence

Usage::

    from pyfoam.applications.fluid_foam import FluidFoam

    solver = FluidFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.thermophysical.thermo import BasicThermo, create_air_thermo

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["FluidFoam"]

logger = logging.getLogger(__name__)


class FluidFoam(SolverBase):
    """Unified compressible solver with full energy equation.

    Combines variable density, temperature-dependent transport,
    and viscous dissipation under the PIMPLE algorithm.
    Differs from RhoPimpleFoam by exposing a richer energy coupling
    (explicit viscous dissipation, variable Prandtl number) and
    a unified interface suitable for extension to multicomponent flows.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model. If ``None``, uses air defaults.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
    ) -> None:
        super().__init__(case_path)

        # 热物理模型
        self.thermo = thermo or create_air_thermo()

        # 读取求解器设置
        self._read_fv_solution_settings()

        # 初始化场
        self.U, self.p, self.T, self.phi, self.rho = self._init_fields()
        self._U_data, self._p_data, self._T_data = self._init_field_data()

        # 存储旧场（时间导数用）
        self.U_old = self.U.clone()
        self.p_old = self.p.clone()
        self.T_old = self.T.clone()
        self.rho_old = self.rho.clone()

        logger.info("FluidFoam ready: %s", self.thermo)

    # ------------------------------------------------------------------
    # fvSolution 设置读取
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read PIMPLE and solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        self.T_solver = str(fv.get_path("solvers/T/solver", "PCG"))
        self.T_tolerance = float(fv.get_path("solvers/T/tolerance", 1e-6))
        self.T_max_iter = int(fv.get_path("solvers/T/maxIter", 1000))

        self.n_outer_correctors = int(
            fv.get_path("PIMPLE/nOuterCorrectors", 3)
        )
        self.n_correctors = int(
            fv.get_path("PIMPLE/nCorrectors", 2)
        )
        self.n_non_orth_correctors = int(
            fv.get_path("PIMPLE/nNonOrthogonalCorrectors", 0)
        )

        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))
        self.alpha_T = float(fv.get_path("PIMPLE/relaxationFactors/T", 1.0))

        self.convergence_tolerance = float(
            fv.get_path("PIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("PIMPLE/maxOuterIterations", 100)
        )

    # ------------------------------------------------------------------
    # 场初始化
    # ------------------------------------------------------------------

    def _init_fields(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise U, p, T, phi, rho from ``0/`` directory."""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        T_tensor, _ = self.read_field_tensor("T", 0)
        T = T_tensor.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        # EOS 初始密度
        rho = self.thermo.rho(p, T)

        return U, p, T, phi, rho

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        T_data = self.case.read_field("T", 0)
        return U_data, p_data, T_data

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the fluidFoam solver.

        Returns:
            Final :class:`ConvergenceData`.
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

        logger.info("Starting fluidFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # 存储旧场
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            self.U, self.p, self.T, self.phi, self.rho, conv = (
                self._pimple_iteration()
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
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("fluidFoam completed (converged)")
            else:
                logger.warning("fluidFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # PIMPLE 迭代
    # ------------------------------------------------------------------

    def _pimple_iteration(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run one PIMPLE time step for compressible flow with full energy.

        Returns:
            ``(U, p, T, phi, rho, convergence)``
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        T = self.T.clone()
        phi = self.phi.clone()
        rho = self.rho.clone()

        convergence = ConvergenceData()
        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()
            T_prev = T.clone()

            # ---- 动量预测 ----
            U, A_p, H = self._momentum_predictor(U, p, phi, rho, self.U_old)

            # ---- PISO 内部校正 ----
            for corr in range(self.n_correctors):
                HbyA = H / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                n_internal = mesh.n_internal_faces
                int_owner = mesh.owner[:n_internal]
                int_neigh = mesh.neighbour
                w = mesh.face_weights[:n_internal]

                HbyA_face = (
                    w.unsqueeze(-1) * HbyA[int_owner]
                    + (1.0 - w).unsqueeze(-1) * HbyA[int_neigh]
                )
                phiHbyA = (HbyA_face * mesh.face_areas[:n_internal]).sum(dim=1)

                # 压力求解
                p = self._solve_pressure_equation(p, phiHbyA, A_p, rho, mesh)

                # 速度校正
                grad_p = self._compute_grad(p, mesh)
                U = HbyA - grad_p / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                # Limit velocity to prevent divergence
                U_mag = U.norm(dim=1, keepdim=True)
                U_max_allowed = 1000.0
                U = torch.where(U_mag > U_max_allowed, U * (U_max_allowed / U_mag.clamp(min=1e-30)), U)

                # 通量校正
                p_P = gather(p, int_owner)
                p_N = gather(p, int_neigh)
                A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
                A_p_inv_face = (
                    w * gather(A_p_inv, int_owner)
                    + (1.0 - w) * gather(A_p_inv, int_neigh)
                )
                phi_internal = phiHbyA - (p_N - p_P) * A_p_inv_face
                phi = phi.clone()
                phi[:n_internal] = phi_internal

                if corr < self.n_correctors - 1:
                    H = self._recompute_H(U, phi, rho)

            # ---- 欠松弛 ----
            if self.alpha_U < 1.0:
                U = self.alpha_U * U + (1.0 - self.alpha_U) * U_prev
            if self.alpha_p < 1.0:
                p = self.alpha_p * p + (1.0 - self.alpha_p) * p_prev

            # ---- EOS 更新密度 ----
            rho = self.thermo.rho(p, T)

            # ---- 能量方程 ----
            T = self._solve_energy_equation(T, U, phi, rho, p, T_prev)

            # 能量方程后再更新密度
            rho = self.thermo.rho(p, T)

            # ---- 收敛检查 ----
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            T_residual = self._compute_residual(T, T_prev)
            continuity_error = self._compute_continuity_error(phi, rho)

            convergence.p_residual = p_residual
            convergence.U_residual = U_residual
            convergence.continuity_error = continuity_error
            convergence.outer_iterations = outer + 1

            if outer % 5 == 0 or outer < 3:
                logger.info(
                    "fluidFoam outer %d: U_res=%.6e, p_res=%.6e, "
                    "T_res=%.6e, cont=%.6e",
                    outer, U_residual, p_residual, T_residual,
                    continuity_error,
                )

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return U, p, T, phi, rho, convergence

    # ------------------------------------------------------------------
    # 动量方程
    # ------------------------------------------------------------------

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        U_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with time derivative (compressible)."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        # 粘度（温度相关）
        mu = self.thermo.mu(T=self.T)
        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))

        # 扩散系数
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        # 对流项（迎风格式）
        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        # 时间导数项：ρ · V / Δt
        dt = self.delta_t
        rho_V_dt = rho * cell_volumes / dt

        # 矩阵系数
        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        A_p = torch.zeros(n_cells, dtype=dtype, device=device)
        A_p = A_p + scatter_add(
            (diff_coeff - flux_neg * rho_face) / V_P, int_owner, n_cells
        )
        A_p = A_p + scatter_add(
            (diff_coeff + flux_pos * rho_face) / V_N, int_neigh, n_cells
        )

        # 对角线加入时间导数
        A_p = A_p + rho_V_dt

        # H(U)
        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # 时间导数源项：ρ · V · U_old / Δt
        H = H + rho_V_dt.unsqueeze(-1) * U_old

        # 压力梯度
        grad_p = self._compute_grad(p, mesh)

        source = H - grad_p

        # 求解
        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)

        # 欠松弛
        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        return U_new, A_p, H

    def _recompute_H(
        self, U: torch.Tensor, phi: torch.Tensor, rho: torch.Tensor
    ) -> torch.Tensor:
        """Recompute H(U) from corrected velocity."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        mu = self.thermo.mu(T=self.T)
        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))

        S_mag = mesh.face_areas[:n_internal].norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        V_P = gather(mesh.cell_volumes.clamp(min=1e-30), int_owner)
        V_N = gather(mesh.cell_volumes.clamp(min=1e-30), int_neigh)

        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        H.index_add_(0, int_owner, lower.unsqueeze(-1) * U[int_neigh] * V_P.unsqueeze(-1))
        H.index_add_(0, int_neigh, upper.unsqueeze(-1) * U[int_owner] * V_N.unsqueeze(-1))

        # 时间导数源项
        dt = self.delta_t
        rho_V_dt = rho * mesh.cell_volumes / dt
        H = H + rho_V_dt.unsqueeze(-1) * self.U_old

        return H

    # ------------------------------------------------------------------
    # 压力方程
    # ------------------------------------------------------------------

    def _solve_pressure_equation(
        self,
        p: torch.Tensor,
        phiHbyA: torch.Tensor,
        A_p: torch.Tensor,
        rho: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """Solve compressible pressure equation (Jacobi iteration)."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        w = mesh.face_weights[:n_internal]

        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = 0.5 * (rho_P + rho_N)

        A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
        A_p_inv_face = (
            w * gather(A_p_inv, int_owner)
            + (1.0 - w) * gather(A_p_inv, int_neigh)
        )

        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        face_coeff = rho_face * A_p_inv_face * S_mag * delta_f

        V_P = gather(cell_volumes.clamp(min=1e-30), int_owner)
        V_N = gather(cell_volumes.clamp(min=1e-30), int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        source = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        source = source + scatter_add(phiHbyA, int_owner, n_cells)
        source = source + scatter_add(-phiHbyA, int_neigh, n_cells)

        diag_safe = diag.abs().clamp(min=1e-30)
        for _ in range(self.p_max_iter):
            off_diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
            p_P = gather(p, int_owner)
            p_N = gather(p, int_neigh)
            off_diag = off_diag + scatter_add(lower * p_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * p_P, int_neigh, n_cells)

            p_new = (source - off_diag) / diag_safe

            if (p_new - p).abs().max() < self.p_tolerance:
                break
            p = p_new

        return p

    # ------------------------------------------------------------------
    # 能量方程（完整能量耦合）
    # ------------------------------------------------------------------

    def _solve_energy_equation(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
        T_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Solve energy equation with viscous dissipation and compressibility.

        ∂(ρCpT)/∂t + ∇·(ρUCpT) = ∇·(κ∇T) + p∇·U + Φ

        其中 Φ = 2μ(S:S) - (2/3)μ(∇·U)^2 是粘性耗散。
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # 热导率（温度相关）
        kappa = self.thermo.kappa(T)
        kappa_face = 0.5 * (gather(kappa, int_owner) + gather(kappa, int_neigh))

        # 扩散系数
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        diff_coeff = kappa_face * S_mag * delta_f

        # 对流项（迎风格式）
        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cp = self.thermo.Cp()
        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        # 时间导数：ρ · V · Cp / Δt
        dt = self.delta_t
        rho_V_Cp_dt = rho * cell_volumes * cp / dt

        # 矩阵系数
        lower = (-diff_coeff + flux_neg * rho_face * cp) / V_P
        upper = (-diff_coeff - flux_pos * rho_face * cp) / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(
            (diff_coeff - flux_neg * rho_face * cp) / V_P, int_owner, n_cells
        )
        diag = diag + scatter_add(
            (diff_coeff + flux_pos * rho_face * cp) / V_N, int_neigh, n_cells
        )

        # 对角线加入时间导数
        diag = diag + rho_V_Cp_dt

        # 源项1：粘性耗散 Φ = 2μ(S:S) - (2/3)μ(∇·U)^2
        mu = self.thermo.mu(T)
        grad_U = self._compute_grad_vector(U, mesh)
        S = 0.5 * (grad_U + grad_U.transpose(-1, -2))
        S_double_dot = (S * S).sum(dim=(1, 2))
        div_U = self._compute_div(U, phi, mesh)
        phi_viscous = 2.0 * mu * S_double_dot - (2.0 / 3.0) * mu * div_U**2

        # 源项2：p * ∇·U（可压缩性功）
        pressure_work = p * div_U

        # 源项3：时间导数源项
        time_source = rho_V_Cp_dt * self.T_old

        source = phi_viscous + pressure_work + time_source

        # Jacobi 迭代求解
        diag_safe = diag.abs().clamp(min=1e-30)
        for _ in range(self.T_max_iter):
            off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
            T_P = gather(T, int_owner)
            T_N = gather(T, int_neigh)
            off_diag = off_diag + scatter_add(lower * T_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * T_P, int_neigh, n_cells)

            T_new = (source - off_diag) / diag_safe
            T_new = T_new.clamp(min=200.0, max=5000.0)

            if (T_new - T).abs().max() < self.T_tolerance:
                break
            T = T_new

        # 欠松弛（使用上一次外迭代的 T_prev）
        if self.alpha_T < 1.0:
            T = self.alpha_T * T + (1.0 - self.alpha_T) * T_prev

        return T

    # ------------------------------------------------------------------
    # 梯度与散度
    # ------------------------------------------------------------------

    def _compute_grad(self, phi: torch.Tensor, mesh: Any) -> torch.Tensor:
        """Compute gradient of scalar field using Gauss theorem."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        owner = mesh.owner
        face_areas = mesh.face_areas

        int_owner = owner[:n_internal]
        int_neigh = mesh.neighbour
        w = mesh.face_weights[:n_internal]

        phi_P = gather(phi, int_owner)
        phi_N = gather(phi, int_neigh)
        phi_face = w * phi_P + (1.0 - w) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * face_areas[:n_internal]

        grad = torch.zeros(n_cells, 3, dtype=phi.dtype, device=phi.device)
        grad.index_add_(0, int_owner, face_contrib)
        grad.index_add_(0, int_neigh, -face_contrib)

        # 边界面贡献（零梯度近似）
        if n_faces > n_internal:
            bnd_owner = owner[n_internal:]
            phi_bnd = gather(phi, bnd_owner)
            bnd_contrib = phi_bnd.unsqueeze(-1) * face_areas[n_internal:]
            grad.index_add_(0, bnd_owner, bnd_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    def _compute_grad_vector(self, U: torch.Tensor, mesh: Any) -> torch.Tensor:
        """Compute gradient of vector field using Gauss theorem."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        U_P = U[int_owner]
        U_N = U[int_neigh]
        U_face = w.unsqueeze(-1) * U_P + (1.0 - w).unsqueeze(-1) * U_N

        grad_U = torch.zeros(n_cells, 3, 3, dtype=U.dtype, device=U.device)
        for j in range(3):
            face_contrib = U_face[:, j].unsqueeze(-1) * face_areas
            grad_U[:, :, j].index_add_(0, int_owner, face_contrib)
            grad_U[:, :, j].index_add_(0, int_neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-30)
        return grad_U / V

    def _compute_div(
        self, U: torch.Tensor, phi: torch.Tensor, mesh: Any
    ) -> torch.Tensor:
        """Compute divergence of vector field."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        U_P = U[int_owner]
        U_N = U[int_neigh]
        U_face = 0.5 * (U_P + U_N)

        flux = (U_face * face_areas).sum(dim=1)

        div = torch.zeros(n_cells, dtype=U.dtype, device=U.device)
        div = div + scatter_add(flux, int_owner, n_cells)
        div = div + scatter_add(-flux, int_neigh, n_cells)

        V = mesh.cell_volumes.clamp(min=1e-30)
        return div / V

    # ------------------------------------------------------------------
    # 残差与连续性误差
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_residual(
        field: torch.Tensor,
        field_old: torch.Tensor,
    ) -> float:
        """Normalized L2 residual."""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def _compute_continuity_error(
        self, phi: torch.Tensor, rho: torch.Tensor
    ) -> float:
        """Compute continuity error (mean |div(rho*phi)|)."""
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        rho_face = 0.5 * (
            gather(rho, owner[:n_internal]) + gather(rho, neighbour)
        )
        mass_flux = phi[:n_internal] * rho_face

        div_rho_phi = torch.zeros(n_cells, dtype=phi.dtype, device=phi.device)
        div_rho_phi = div_rho_phi + scatter_add(
            mass_flux, owner[:n_internal], n_cells
        )
        div_rho_phi = div_rho_phi + scatter_add(
            -mass_flux, neighbour, n_cells
        )

        V = mesh.cell_volumes.clamp(min=1e-30)
        div_rho_phi = div_rho_phi / V

        return float(div_rho_phi.abs().mean().item())

    # ------------------------------------------------------------------
    # 场输出
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, T to a time directory."""
        if abs(time) < 0.001 and time != 0:
            time_str = f"{time:.10f}".rstrip("0").rstrip(".")
        else:
            time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("T", self.T, time_str, self._T_data)
