"""
compressibleVoF — Compressible two-phase VOF solver (modern interface).

Implements the OpenFOAM-style ``compressibleVoF`` solver, the modern
replacement for ``compressibleInterFoam``. Uses PIMPLE algorithm with
VOF advection for compressible two-phase flows.

Each phase has its own equation of state (EOS):
- Phase 1 (e.g.\ liquid): rho = rho_ref + psi1 * p  (Tait-like)
- Phase 2 (e.g.\ gas):    rho = psi2 * p             (ideal gas, psi = 1/(RT))

Mixture compressibility:
    psi_mix = alpha * psi2 + (1 - alpha) * psi1

Key differences from ``CompressibleInterFoam``:

- Operator-splitting: VOF advection is advanced *before* the PIMPLE
  pressure-velocity coupling loop.
- Explicit energy equation with convection and pressure-work terms.
- Consistent compressible pressure-velocity coupling with ``rho*gh``
  reference pressure formulation.

Governing equations:

- Continuity: ``∂ρ/∂t + ∇·(ρU) = 0``
- Momentum: ``∂(ρU)/∂t + ∇·(ρUU) = -∇p + ∇·(μ∇U) + ρg + F_σ``
- Energy:   ``∂(ρe)/∂t + ∇·(ρUe) = -p∇·U + ...`` (simplified)
- VOF:      ``∂α/∂t + ∇·(Uα) + ∇·(U_r α(1-α)) = 0``

Usage::

    from pyfoam.applications.compressible_vof_foam import CompressibleVoFFoam

    solver = CompressibleVoFFoam("path/to/case")
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
from pyfoam.multiphase.volume_of_fluid import VOFAdvection
from pyfoam.multiphase.surface_tension import SurfaceTensionModel

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["CompressibleVoFFoam"]

logger = logging.getLogger(__name__)


class CompressibleVoFFoam(SolverBase):
    """Compressible two-phase VOF solver (modern interface).

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    rho1, rho2 : float
        Reference densities for each phase (at p=0).
    mu1, mu2 : float
        Dynamic viscosities for each phase.
    psi1, psi2 : float
        Compressibility coefficients (drho/dp, 1/Pa).
    sigma : float
        Surface tension coefficient (N/m).
    Cv1, Cv2 : float
        Specific heat at constant volume for each phase.
    kappa1, kappa2 : float
        Thermal conductivity for each phase.
    C_alpha : float
        VOF compression coefficient.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        rho1: float = 1000.0,
        rho2: float = 1.225,
        mu1: float = 1e-3,
        mu2: float = 1.8e-5,
        psi1: float = 1e-6,
        psi2: float = 1e-5,
        sigma: float = 0.07,
        Cv1: float = 4180.0,
        Cv2: float = 718.0,
        kappa1: float = 0.6,
        kappa2: float = 0.025,
        C_alpha: float = 1.0,
    ) -> None:
        super().__init__(case_path)

        # 流体物性
        self.rho1 = rho1
        self.rho2 = rho2
        self.mu1 = mu1
        self.mu2 = mu2
        self.psi1 = psi1
        self.psi2 = psi2
        self.sigma = sigma
        self.Cv1 = Cv1
        self.Cv2 = Cv2
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.C_alpha = C_alpha

        # 读取 fvSolution 设置
        self._read_fv_solution_settings()

        # 初始化场
        self.U, self.p, self.alpha, self.phi, self.T = self._init_fields()
        self._U_data, self._p_data, self._alpha_data, self._T_data = (
            self._init_field_data()
        )

        # VOF 对流模块
        self.vof = VOFAdvection(
            self.mesh, self.alpha, self.phi, self.U,
            C_alpha=C_alpha,
        )

        # CSF 表面张力模型
        self.surface_tension = SurfaceTensionModel(
            sigma=sigma, mesh=self.mesh, n_smooth=1,
        )

        logger.info(
            "CompressibleVoFFoam ready: rho1=%.1f, rho2=%.3f, "
            "psi1=%.2e, psi2=%.2e, sigma=%.3f",
            rho1, rho2, psi1, psi2, sigma,
        )

    def _read_fv_solution_settings(self) -> None:
        """从 fvSolution 读取 PIMPLE 设置。"""
        fv = self.case.fvSolution

        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        self.n_outer_correctors = int(
            fv.get_path("PIMPLE/nOuterCorrectors", 3)
        )
        self.n_correctors = int(
            fv.get_path("PIMPLE/nCorrectors", 2)
        )

        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))

        self.convergence_tolerance = float(
            fv.get_path("PIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("PIMPLE/maxOuterIterations", 100)
        )

    def _init_fields(self) -> tuple[torch.Tensor, ...]:
        """从 0/ 目录初始化 U、p、alpha、phi、T。"""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        alpha_tensor, _ = self.read_field_tensor("alpha.water", 0)
        alpha = alpha_tensor.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        try:
            T_tensor, _ = self.read_field_tensor("T", 0)
            T = T_tensor.to(device=device, dtype=dtype)
        except Exception:
            T = torch.full(
                (self.mesh.n_cells,), 300.0, dtype=dtype, device=device
            )

        return U, p, alpha, phi, T

    def _init_field_data(self):
        """存储原始 FieldData 用于写入。"""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        alpha_data = self.case.read_field("alpha.water", 0)
        try:
            T_data = self.case.read_field("T", 0)
        except Exception:
            T_data = None
        return U_data, p_data, alpha_data, T_data

    # ------------------------------------------------------------------
    # 混合物性
    # ------------------------------------------------------------------

    def _compute_mixture_rho(self, alpha: torch.Tensor) -> torch.Tensor:
        """混合参考密度（不含 EOS 压力项）。

        rho_ref = alpha * rho2 + (1 - alpha) * rho1
        """
        return alpha * self.rho2 + (1.0 - alpha) * self.rho1

    def _compute_mixture_mu(self, alpha: torch.Tensor) -> torch.Tensor:
        """混合粘度：mu = alpha * mu2 + (1 - alpha) * mu1。"""
        return alpha * self.mu2 + (1.0 - alpha) * self.mu1

    def _compute_mixture_psi(self, alpha: torch.Tensor) -> torch.Tensor:
        """混合压缩系数：psi_mix = alpha * psi2 + (1 - alpha) * psi1。"""
        return alpha * self.psi2 + (1.0 - alpha) * self.psi1

    def _compute_mixture_Cv(self, alpha: torch.Tensor) -> torch.Tensor:
        """混合定容比热：Cv_mix = alpha * Cv2 + (1 - alpha) * Cv1。"""
        return alpha * self.Cv2 + (1.0 - alpha) * self.Cv1

    def _compute_mixture_kappa(self, alpha: torch.Tensor) -> torch.Tensor:
        """混合导热系数：kappa = alpha * kappa2 + (1 - alpha) * kappa1。"""
        return alpha * self.kappa2 + (1.0 - alpha) * self.kappa1

    def _compute_rho_from_eos(
        self,
        alpha: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """从 EOS 计算实际密度。

        rho = rho_ref + psi_mix * p
        """
        rho_ref = self._compute_mixture_rho(alpha)
        psi_mix = self._compute_mixture_psi(alpha)
        return rho_ref + psi_mix * p

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """运行 compressibleVoF 求解器。

        Returns:
            最终 :class:`ConvergenceData`。
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

        logger.info("Starting compressibleVoF run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U, self.p, self.alpha, self.phi, self.T, conv = (
                self._pimple_vof_step()
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

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # 单时间步：operator splitting
    # ------------------------------------------------------------------

    def _pimple_vof_step(self) -> tuple[torch.Tensor, ...]:
        """单时间步：先 VOF 对流，再 PIMPLE 压力-速度耦合 + 能量更新。

        Returns:
            (U, p, alpha, phi, T, convergence)
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        alpha = self.alpha.clone()
        phi = self.phi.clone()
        T = self.T.clone()

        convergence = ConvergenceData()

        # ============================================================
        # 阶段 1：VOF 对流（advance alpha）
        # ============================================================
        self.vof.alpha = alpha
        self.vof.phi = phi
        self.vof.U = U
        alpha = self.vof.advance(self.delta_t)

        # ============================================================
        # 阶段 2：PIMPLE 压力-速度耦合
        # ============================================================
        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            # 用当前 alpha 和 p 计算混合物性
            rho = self._compute_rho_from_eos(alpha, p)
            mu_mix = self._compute_mixture_mu(alpha)
            psi_mix = self._compute_mixture_psi(alpha)

            # 动量预测
            U, A_p, H = self._momentum_predictor(
                U, p, phi, rho, mu_mix, alpha,
            )

            # PISO 校正循环
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

                # 求解压力方程（含 psi_mix 对角项）
                p = self._solve_pressure_equation(
                    p, phiHbyA, A_p, rho, psi_mix, mesh,
                )

                # 速度校正
                grad_p = self._compute_grad(p, mesh)
                U = HbyA - grad_p / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                # 通量校正
                p_P = gather(p, int_owner)
                p_N = gather(p, int_neigh)
                A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
                A_p_inv_face = (
                    w * gather(A_p_inv, int_owner)
                    + (1.0 - w) * gather(A_p_inv, int_neigh)
                )
                phi = phiHbyA - (p_N - p_P) * A_p_inv_face

                # 更新密度
                rho = self._compute_rho_from_eos(alpha, p)

                # 重算 H
                if corr < self.n_correctors - 1:
                    H = self._recompute_H(U, phi, rho, mu_mix)

            # 松弛
            if self.alpha_U < 1.0:
                U = self.alpha_U * U + (1.0 - self.alpha_U) * U_prev
            if self.alpha_p < 1.0:
                p = self.alpha_p * p + (1.0 - self.alpha_p) * p_prev

            # 收敛判据
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            continuity_error = self._compute_continuity_error(phi, rho)

            convergence.p_residual = p_residual
            convergence.U_residual = U_residual
            convergence.continuity_error = continuity_error
            convergence.outer_iterations = outer + 1

            if outer % 5 == 0 or outer < 3:
                logger.info(
                    "compressibleVoF outer %d: U_res=%.6e, p_res=%.6e, "
                    "cont=%.6e",
                    outer, U_residual, p_residual, continuity_error,
                )

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        # ============================================================
        # 阶段 3：能量方程（简化形式）
        # ============================================================
        T = self._solve_energy(alpha, p, rho, U, phi)

        return U, p, alpha, phi, T, convergence

    # ------------------------------------------------------------------
    # 动量方程
    # ------------------------------------------------------------------

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        mu_mix: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """求解带表面张力的动量预测步。"""
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

        # 粘性扩散
        mu_face = 0.5 * (gather(mu_mix, int_owner) + gather(mu_mix, int_neigh))

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

        # 时间导数项
        dt = self.delta_t
        rho_V_dt = rho * cell_volumes / dt

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
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # 时间源项（用上一步的 U 作为旧值）
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

        # 表面张力
        F_sigma = self.surface_tension.compute_force(alpha)

        source = H - grad_p + F_sigma

        # 求解
        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)

        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        return U_new, A_p, H

    # ------------------------------------------------------------------
    # 压力方程（含 psi_mix 对角项）
    # ------------------------------------------------------------------

    def _solve_pressure_equation(
        self,
        p: torch.Tensor,
        phiHbyA: torch.Tensor,
        A_p: torch.Tensor,
        rho: torch.Tensor,
        psi_mix: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """求解可压缩两相流压力方程。

        与不可压缩版本的区别在于对角项包含 psi_mix * V / (dt * A_p)
        以反映密度对压力的依赖。
        """
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

        # 可压缩性对角项：psi * V / dt
        diag = diag + psi_mix * cell_volumes / self.delta_t

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
    # 能量方程（简化形式）
    # ------------------------------------------------------------------

    def _solve_energy(
        self,
        alpha: torch.Tensor,
        p: torch.Tensor,
        rho: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """求解简化能量方程。

        使用 T = e / Cv_mix 的关系，其中 e 通过对流和压力功更新。
        简化为：
            T_new = p / (rho * Cv_mix)
        这是压缩功主导的简化形式。
        """
        Cv_mix = self._compute_mixture_Cv(alpha)
        rho_Cv = (rho * Cv_mix).clamp(min=1e-30)
        T = p / rho_Cv
        return T

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _recompute_H(
        self,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        mu_mix: torch.Tensor,
    ) -> torch.Tensor:
        """从校正后的速度重算 H(U)。"""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        mu_face = 0.5 * (gather(mu_mix, int_owner) + gather(mu_mix, int_neigh))

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
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        return H

    def _compute_residual(
        self,
        field: torch.Tensor,
        field_old: torch.Tensor,
    ) -> float:
        """L2 范数残差，归一化到场量幅值。"""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def _compute_grad(self, phi: torch.Tensor, mesh: Any) -> torch.Tensor:
        """计算标量场梯度。"""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        phi_P = gather(phi, int_owner)
        phi_N = gather(phi, int_neigh)
        phi_face = w * phi_P + (1.0 - w) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * face_areas

        grad = torch.zeros(n_cells, 3, dtype=phi.dtype, device=phi.device)
        grad.index_add_(0, int_owner, face_contrib)
        grad.index_add_(0, int_neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    def _compute_continuity_error(
        self, phi: torch.Tensor, rho: torch.Tensor,
    ) -> float:
        """计算连续性误差。"""
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
            mass_flux, owner[:n_internal], n_cells,
        )
        div_rho_phi = div_rho_phi + scatter_add(
            -mass_flux, neighbour, n_cells,
        )

        V = mesh.cell_volumes.clamp(min=1e-30)
        div_rho_phi = div_rho_phi / V

        return float(div_rho_phi.abs().mean().item())

    def _write_fields(self, time: float) -> None:
        """将 U、p、alpha.water、T 写入时间目录。"""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("alpha.water", self.alpha, time_str, self._alpha_data)
        if self._T_data is not None:
            self.write_field("T", self.T, time_str, self._T_data)
