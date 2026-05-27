"""
pdrFoam — premixed combustion solver with b-Xi model.

Implements the PIMPLE algorithm for transient compressible
premixed combustion using the Bradley-Bradley-Xi (b-Xi) model.

The progress variable b represents the burnt-gas fraction:
    b = 0  (unburnt)
    b = 1  (fully burnt)

Governing equations:
    Continuity:  ∂ρ/∂t + ∇·(ρU) = 0
    Momentum:    ∂(ρU)/∂t + ∇·(ρUU) = -∇p + ∇·(μ∇U)
    Progress:    ∂(ρb)/∂t + ∇·(ρUb) = ∇·(ρDb∇b) + ρ·S_T·|∇b|
    Energy:      ∂(ρCpT)/∂t + ∇·(ρUCpT) = ∇·(κ∇T) + Q·ω_b
    EOS:         ρ(p, T, b) — mixture-dependent

Flame speed model:
    S_L  = S_L0 · (T/T_ref)^α · (p/p_ref)^β  (laminar flame speed)
    Xi   = 1 + (Xi_0 - 1) · (u'/S_L)^0.5      (flame wrinkling)
    S_T  = S_L · Xi                              (turbulent flame speed)

where:
    S_L0   = unstrained laminar flame speed
    Xi_0   = turbulent flame speed factor
    u'     = turbulent velocity fluctuation

Density coupling:
    ρ = 1 / (b/ρ_b + (1-b)/ρ_u)   (mixed regime)

Usage::

    from pyfoam.applications.pdr_foam import PDRFoam

    solver = PDRFoam("path/to/case")
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

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PDRFoam"]

logger = logging.getLogger(__name__)


class PDRFoam(SolverBase):
    """Premixed combustion solver with b-Xi model.

    Solves for the progress variable b (burnt-gas fraction),
    coupled with compressible momentum and energy equations.
    The turbulent flame speed S_T is computed from the Xi model.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    SL0 : float, optional
        Unstrained laminar flame speed (m/s). Default 0.4.
    Xi0 : float, optional
        Turbulent flame speed factor. Default 2.0.
    rho_unburnt : float, optional
        Unburnt gas density (kg/m^3). Default 1.2.
    rho_burnt : float, optional
        Burnt gas density (kg/m^3). Default 0.15.
    T_unburnt : float, optional
        Unburnt temperature (K). Default 300.0.
    T_adiabatic : float, optional
        Adiabatic flame temperature (K). Default 2000.0.
    Cp : float, optional
        Specific heat capacity (J/(kg·K)). Default 1005.0.
    mu : float, optional
        Dynamic viscosity (Pa·s). Default 1.8e-5.
    Db : float, optional
        Progress variable diffusivity (m^2/s). Default 2e-4.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        SL0: float = 0.4,
        Xi0: float = 2.0,
        rho_unburnt: float = 1.2,
        rho_burnt: float = 0.15,
        T_unburnt: float = 300.0,
        T_adiabatic: float = 2000.0,
        Cp: float = 1005.0,
        mu: float = 1.8e-5,
        Db: float = 2e-4,
    ) -> None:
        super().__init__(case_path)

        # 火焰参数
        self.SL0 = SL0
        self.Xi0 = Xi0
        self.rho_unburnt = rho_unburnt
        self.rho_burnt = rho_burnt
        self.T_unburnt = T_unburnt
        self.T_adiabatic = T_adiabatic
        self.Cp = Cp
        self.mu = mu
        self.Db = Db

        # 读取求解器设置
        self._read_fv_solution_settings()

        # 初始化场
        self.b, self.U, self.p, self.T, self.phi, self.rho = self._init_fields()
        self._b_data, self._U_data, self._p_data, self._T_data = self._init_field_data()

        # 存储旧场
        self.b_old = self.b.clone()
        self.U_old = self.U.clone()
        self.p_old = self.p.clone()
        self.T_old = self.T.clone()
        self.rho_old = self.rho.clone()

        logger.info(
            "PDRFoam ready: SL0=%.3f m/s, Xi0=%.2f, "
            "rho_u=%.2f, rho_b=%.2f",
            self.SL0, self.Xi0, self.rho_unburnt, self.rho_burnt,
        )

    # ------------------------------------------------------------------
    # fvSolution 设置
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

        self.b_solver = str(fv.get_path("solvers/b/solver", "PBiCGStab"))
        self.b_tolerance = float(fv.get_path("solvers/b/tolerance", 1e-6))
        self.b_max_iter = int(fv.get_path("solvers/b/maxIter", 1000))

        self.n_outer_correctors = int(
            fv.get_path("PIMPLE/nOuterCorrectors", 3)
        )
        self.n_correctors = int(
            fv.get_path("PIMPLE/nCorrectors", 2)
        )

        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))
        self.alpha_b = float(fv.get_path("PIMPLE/relaxationFactors/b", 0.5))

        self.convergence_tolerance = float(
            fv.get_path("PIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("PIMPLE/maxOuterIterations", 100)
        )

        # PDR 特有参数
        self.SL0 = float(fv.get_path("PDR/SL0", self.SL0))
        self.Xi0 = float(fv.get_path("PDR/Xi0", self.Xi0))
        self.rho_unburnt = float(fv.get_path("PDR/rhoUnburnt", self.rho_unburnt))
        self.rho_burnt = float(fv.get_path("PDR/rhoBurnt", self.rho_burnt))
        self.T_unburnt = float(fv.get_path("PDR/TUnburnt", self.T_unburnt))
        self.T_adiabatic = float(fv.get_path("PDR/TAdiabatic", self.T_adiabatic))

    # ------------------------------------------------------------------
    # 场初始化
    # ------------------------------------------------------------------

    def _init_fields(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise b, U, p, T, phi, rho."""
        device = get_device()
        dtype = get_default_dtype()

        try:
            b_tensor, _ = self.read_field_tensor("b", 0)
            b = b_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            b = torch.zeros(self.mesh.n_cells, dtype=dtype, device=device)

        try:
            U_tensor, _ = self.read_field_tensor("U", 0)
            U = U_tensor.to(device=device, dtype=dtype)
        except Exception:
            U = torch.zeros(self.mesh.n_cells, 3, dtype=dtype, device=device)

        try:
            p_tensor, _ = self.read_field_tensor("p", 0)
            p = p_tensor.to(device=device, dtype=dtype)
        except Exception:
            p = torch.full(
                (self.mesh.n_cells,), 101325.0, dtype=dtype, device=device
            )

        try:
            T_tensor, _ = self.read_field_tensor("T", 0)
            T = T_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            T = torch.full(
                (self.mesh.n_cells,), self.T_unburnt, dtype=dtype, device=device
            )

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        # 混合密度：ρ = 1 / (b/ρ_b + (1-b)/ρ_u)
        rho = self._rho_from_b(b)

        return b, U, p, T, phi, rho

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        try:
            b_data = self.case.read_field("b", 0)
        except Exception:
            b_data = None

        try:
            U_data = self.case.read_field("U", 0)
        except Exception:
            U_data = None

        try:
            p_data = self.case.read_field("p", 0)
        except Exception:
            p_data = None

        try:
            T_data = self.case.read_field("T", 0)
        except Exception:
            T_data = None

        return b_data, U_data, p_data, T_data

    # ------------------------------------------------------------------
    # 火焰模型
    # ------------------------------------------------------------------

    def _rho_from_b(self, b: torch.Tensor) -> torch.Tensor:
        """计算混合区密度：ρ = 1 / (b/ρ_b + (1-b)/ρ_u)。

        这是预混燃烧中常用的密度模型，保证在未燃 (b=0) 时 ρ=ρ_u，
        完全燃烧 (b=1) 时 ρ=ρ_b。
        """
        b_safe = b.clamp(0.0, 1.0)
        return 1.0 / (
            b_safe / self.rho_burnt
            + (1.0 - b_safe) / self.rho_unburnt
        )

    def _T_from_b(self, b: torch.Tensor) -> torch.Tensor:
        """计算温度：T = T_u + b * (T_adiabatic - T_u)。

        线性简化模型，假设比热恒定。
        """
        b_safe = b.clamp(0.0, 1.0)
        return self.T_unburnt + b_safe * (self.T_adiabatic - self.T_unburnt)

    def _laminar_flame_speed(self, T: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """计算层流火焰速度。

        S_L = S_L0 * (T/T_ref)^alpha * (p/p_ref)^beta

        简化为幂律关系。参考条件为未燃气体温度和大气压力。
        """
        T_ref = self.T_unburnt
        p_ref = 101325.0
        alpha = 1.5  # 温度指数
        beta = -0.1  # 压力指数

        T_ratio = (T / T_ref).clamp(min=0.1)
        p_ratio = (p / p_ref).clamp(min=0.01)

        return self.SL0 * T_ratio.pow(alpha) * p_ratio.pow(beta)

    def _turbulent_flame_speed(
        self,
        SL: torch.Tensor,
        u_prime: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算湍流火焰速度 S_T 和皱褶因子 Xi。

        Xi  = 1 + (Xi_0 - 1) * (u'/S_L)^0.5
        S_T = S_L * Xi

        Parameters
        ----------
        SL : torch.Tensor
            层流火焰速度。
        u_prime : torch.Tensor
            湍流脉动速度（从速度梯度估算）。

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(S_T, Xi)``
        """
        SL_safe = SL.clamp(min=1e-6)
        wrinkling = 1.0 + (self.Xi0 - 1.0) * (u_prime / SL_safe).clamp(min=0.0).sqrt()
        S_T = SL * wrinkling
        return S_T, wrinkling

    def _estimate_turbulent_fluctuation(self, U: torch.Tensor) -> torch.Tensor:
        """估算湍流脉动速度 u'。

        从速度梯度估算：u' ≈ C * |∇U| * Δx

        其中 C ≈ 0.1 是经验常数，Δx 是胞尺寸。
        """
        mesh = self.mesh
        grad_U = self._compute_grad_vector(U, mesh)
        # 应变率幅值
        strain_rate = (grad_U * grad_U).sum(dim=(1, 2)).sqrt()
        # 特征长度（立方根体积）
        dx = mesh.cell_volumes.pow(1.0 / 3.0)
        # u' = C * strain * dx
        C = 0.1
        return C * strain_rate * dx

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the pdrFoam solver.

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

        logger.info("Starting pdrFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  SL0=%.3f, Xi0=%.2f", self.SL0, self.Xi0)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # 存储旧场
            self.b_old = self.b.clone()
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            self.b, self.U, self.p, self.T, self.phi, self.rho, conv = (
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
                logger.info("pdrFoam completed (converged)")
            else:
                logger.warning("pdrFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # PIMPLE 迭代
    # ------------------------------------------------------------------

    def _pimple_iteration(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run one PIMPLE time step for premixed combustion.

        Returns:
            ``(b, U, p, T, phi, rho, convergence)``
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        b = self.b.clone()
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
            b_prev = b.clone()

            # ---- 动量预测 ----
            U, A_p, H = self._momentum_predictor(U, p, phi, rho)

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

                p = self._solve_pressure_equation(p, phiHbyA, A_p, rho, mesh)

                grad_p = self._compute_grad(p, mesh)
                U = HbyA - grad_p / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

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

            # ---- 进度变量输运 ----
            b = self._solve_progress_equation(b, U, phi, rho, self.delta_t)

            # ---- 更新密度与温度 ----
            rho = self._rho_from_b(b)
            T = self._T_from_b(b)

            # ---- 收敛检查 ----
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            b_residual = self._compute_residual(b, b_prev)
            continuity_error = self._compute_continuity_error(phi, rho)

            convergence.p_residual = p_residual
            convergence.U_residual = U_residual
            convergence.continuity_error = continuity_error
            convergence.outer_iterations = outer + 1

            if outer % 5 == 0 or outer < 3:
                logger.info(
                    "PDR outer %d: U_res=%.6e, p_res=%.6e, "
                    "b_res=%.6e, cont=%.6e",
                    outer, U_residual, p_residual, b_residual,
                    continuity_error,
                )

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return b, U, p, T, phi, rho, convergence

    # ------------------------------------------------------------------
    # 动量方程
    # ------------------------------------------------------------------

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with time derivative."""
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

        # 恒定粘度（可扩展为 Sutherland）
        mu_full = torch.full((n_cells,), self.mu, dtype=dtype, device=device)
        mu_face = 0.5 * (gather(mu_full, int_owner) + gather(mu_full, int_neigh))

        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        dt = self.delta_t
        rho_V_dt = rho * cell_volumes / dt

        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        A_p = torch.zeros(n_cells, dtype=dtype, device=device)
        A_p = A_p + scatter_add(
            (diff_coeff - flux_neg * rho_face) / V_P, int_owner, n_cells
        )
        A_p = A_p + scatter_add(
            (diff_coeff + flux_pos * rho_face) / V_N, int_neigh, n_cells
        )
        A_p = A_p + rho_V_dt

        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        H = H + rho_V_dt.unsqueeze(-1) * self.U_old

        grad_p = self._compute_grad(p, mesh)

        source = H - grad_p

        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)

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

        mu_full = torch.full((n_cells,), self.mu, dtype=dtype, device=device)
        mu_face = 0.5 * (gather(mu_full, int_owner) + gather(mu_full, int_neigh))

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

        dt = self.delta_t
        rho_V_dt = rho * mesh.cell_volumes / dt
        H = H + rho_V_dt.unsqueeze(-1) * self.U_old

        return H

    # ------------------------------------------------------------------
    # 进度变量输运方程（b-Xi 模型）
    # ------------------------------------------------------------------

    def _solve_progress_equation(
        self,
        b: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Solve progress variable transport equation with b-Xi model.

        ∂(ρb)/∂t + ∇·(ρUb) = ∇·(ρDb∇b) + ρ·S_T·|∇b|

        其中 S_T 是湍流火焰速度，|∇b| 是源项（火焰传播）。
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # 计算火焰速度
        T_local = self._T_from_b(b)
        SL = self._laminar_flame_speed(T_local, self.p)
        u_prime = self._estimate_turbulent_fluctuation(U)
        ST, Xi = self._turbulent_flame_speed(SL, u_prime)

        # 计算 |∇b|（用于反应源项）
        grad_b = self._compute_grad(b, mesh)
        grad_b_mag = grad_b.norm(dim=1)

        # 反应源项：ρ * S_T * |∇b|
        reaction_source = rho * ST * grad_b_mag

        # 扩散系数（含湍流增强）
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        rho_face_diff = 0.5 * (gather(rho, int_owner) + gather(rho, int_neigh))
        # 有效扩散率 = 分子 + 湍流
        Db_eff = self.Db
        diff_coeff = rho_face_diff * Db_eff * S_mag * delta_f

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

        rho_V_dt = rho * cell_volumes / dt

        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(
            (diff_coeff - flux_neg * rho_face) / V_P, int_owner, n_cells
        )
        diag = diag + scatter_add(
            (diff_coeff + flux_pos * rho_face) / V_N, int_neigh, n_cells
        )
        diag = diag + rho_V_dt

        # 源项：时间导数 + 燃烧反应
        source = rho_V_dt * self.b_old + reaction_source * cell_volumes

        # Jacobi 迭代
        diag_safe = diag.abs().clamp(min=1e-30)
        b_result = b.clone()
        for _ in range(self.b_max_iter):
            off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
            b_P = gather(b_result, int_owner)
            b_N = gather(b_result, int_neigh)
            off_diag = off_diag + scatter_add(lower * b_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * b_P, int_neigh, n_cells)

            b_new = (source - off_diag) / diag_safe

            if (b_new - b_result).abs().max() < self.b_tolerance:
                break
            b_result = b_new

        # 欠松弛
        if self.alpha_b < 1.0:
            b_result = self.alpha_b * b_result + (1.0 - self.alpha_b) * self.b_old

        # 限制到 [0, 1]
        return b_result.clamp(0.0, 1.0)

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
    # 梯度
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

        if n_faces > n_internal:
            bnd_owner = owner[n_internal:]
            phi_bnd = gather(phi, bnd_owner)
            bnd_contrib = phi_bnd.unsqueeze(-1) * face_areas[n_internal:]
            grad.index_add_(0, bnd_owner, bnd_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    def _compute_grad_vector(self, U: torch.Tensor, mesh: Any) -> torch.Tensor:
        """Compute gradient of vector field."""
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
        """Write b, U, p, T to a time directory."""
        if abs(time) < 0.001 and time != 0:
            time_str = f"{time:.10f}".rstrip("0").rstrip(".")
        else:
            time_str = f"{time:g}"

        if self._b_data is not None:
            self.write_field("b", self.b, time_str, self._b_data)
        if self._U_data is not None:
            self.write_field("U", self.U, time_str, self._U_data)
        if self._p_data is not None:
            self.write_field("p", self.p, time_str, self._p_data)
        if self._T_data is not None:
            self.write_field("T", self.T, time_str, self._T_data)
