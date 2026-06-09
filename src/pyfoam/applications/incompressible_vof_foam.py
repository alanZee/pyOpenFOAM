"""
incompressibleVoF — Modern VOF two-phase incompressible solver.

Implements the OpenFOAM-style ``incompressibleVoF`` solver, the modern
replacement for ``interFoam``. Uses PIMPLE algorithm with explicit
MULES-bounded VOF advection for interface tracking.

Key differences from ``InterFoam``:

- Alpha advection is performed as a separate step *before* the
  PIMPLE pressure-velocity coupling loop (operator splitting).
- Semi-implicit MULES limiter ensures boundedness of the volume
  fraction field (alpha in [0, 1]) while preserving conservation.
- Consistent formulation with ``rho*gh`` reference pressure.

Governing equations:

- Momentum (mixture):
  ``∂(ρU)/∂t + ∇·(ρUU) = -∇p_rgh + ∇·(μ∇U) + ρg + F_σ``
- Pressure (PIMPLE outer correctors)
- VOF advection: ``∂α/∂t + ∇·(Uα) + ∇·(U_r α(1-α)) = 0``

Mixture properties:
- ``ρ = α·ρ₂ + (1-α)·ρ₁``
- ``μ = α·μ₂ + (1-α)·μ₁``

Usage::

    from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam

    solver = IncompressibleVoFFoam("path/to/case")
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
from pyfoam.multiphase.volume_of_fluid import VOFAdvection
from pyfoam.multiphase.mules import MULESLimiter
from pyfoam.multiphase.surface_tension import SurfaceTensionModel

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["IncompressibleVoFFoam"]

logger = logging.getLogger(__name__)


class IncompressibleVoFFoam(SolverBase):
    """Modern VOF two-phase incompressible solver.

    Replaces ``InterFoam`` with a cleaner operator-splitting strategy:
    VOF advection is advanced first, then the PIMPLE loop solves the
    pressure-velocity coupling with updated mixture properties.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    rho1 : float
        Density of fluid 1 (default 1000.0, water).
    rho2 : float
        Density of fluid 2 (default 1.225, air).
    mu1 : float
        Dynamic viscosity of fluid 1 (default 1e-3, water).
    mu2 : float
        Dynamic viscosity of fluid 2 (default 1.8e-5, air).
    sigma : float
        Surface tension coefficient (default 0.07 N/m, water-air).
    C_alpha : float
        VOF compression coefficient (default 1.0).
    mules_iterations : int
        Number of MULES limiter iterations (default 3).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        rho1: float = 1000.0,
        rho2: float = 1.225,
        mu1: float = 1e-3,
        mu2: float = 1.8e-5,
        sigma: float = 0.07,
        C_alpha: float = 1.0,
        mules_iterations: int = 3,
    ) -> None:
        super().__init__(case_path)

        # 流体物性
        self.rho1 = rho1
        self.rho2 = rho2
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma = sigma
        self.C_alpha = C_alpha
        self.mules_iterations = mules_iterations

        # 读取 fvSolution 设置
        self._read_fv_solution_settings()

        # 初始化场
        self.U, self.p, self.alpha, self.phi = self._init_fields()
        self._U_data, self._p_data, self._alpha_data = self._init_field_data()

        # 混合物性
        self.rho = self._compute_mixture_rho(self.alpha)
        self.mu_mix = self._compute_mixture_mu(self.alpha)

        # VOF 对流模块
        self.vof = VOFAdvection(
            self.mesh, self.alpha, self.phi, self.U,
            C_alpha=C_alpha,
            use_mules=True,
            mules_iterations=mules_iterations,
        )

        # 显式 MULES 限制器（用于 PIMPLE 内的有界更新）
        self.mules = MULESLimiter(self.mesh, n_iterations=mules_iterations)

        # CSF 表面张力模型
        self.surface_tension = SurfaceTensionModel(
            sigma=sigma, mesh=self.mesh, n_smooth=1,
        )

        logger.info(
            "IncompressibleVoFFoam ready: rho1=%.1f, rho2=%.3f, "
            "mu1=%.2e, mu2=%.2e, sigma=%.3f, C_alpha=%.2f",
            rho1, rho2, mu1, mu2, sigma, C_alpha,
        )

    def _build_boundary_conditions(self):
        """从 0/U 边界场构建速度 BC 张量。"""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        U_bc = torch.full((n_cells, 3), float("nan"), dtype=dtype, device=device)
        U_field_data = self._U_data
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

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从 0/ 目录初始化 U、p、alpha、phi。"""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        alpha_tensor, _ = self.read_field_tensor("alpha.water", 0)
        alpha = alpha_tensor.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return U, p, alpha, phi

    def _init_field_data(self):
        """存储原始 FieldData 用于写入。"""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        alpha_data = self.case.read_field("alpha.water", 0)
        return U_data, p_data, alpha_data

    def _compute_mixture_rho(self, alpha: torch.Tensor) -> torch.Tensor:
        """混合密度：ρ = α·ρ₂ + (1-α)·ρ₁。"""
        return alpha * self.rho2 + (1.0 - alpha) * self.rho1

    def _compute_mixture_mu(self, alpha: torch.Tensor) -> torch.Tensor:
        """混合粘度：μ = α·μ₂ + (1-α)·μ₁。"""
        return alpha * self.mu2 + (1.0 - alpha) * self.mu1

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """运行 incompressibleVoF 求解器。

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

        logger.info("Starting incompressibleVoF run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U, self.p, self.alpha, self.phi, conv = (
                self._pimple_vof_step()
            )
            last_convergence = conv

            # 更新混合物性
            self.rho = self._compute_mixture_rho(self.alpha)
            self.mu_mix = self._compute_mixture_mu(self.alpha)

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

    def _pimple_vof_step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """单时间步：先 VOF 对流，再 PIMPLE 压力-速度耦合。

        Returns:
            (U, p, alpha, phi, convergence)
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        alpha = self.alpha.clone()
        phi = self.phi.clone()

        convergence = ConvergenceData()

        # ============================================================
        # 阶段 1：VOF 对流（advance alpha）
        # 与 interFoam 不同，alpha 更新在 PIMPLE 循环之外完成
        # ============================================================
        self.vof.alpha = alpha
        self.vof.phi = phi
        self.vof.U = U
        alpha = self.vof.advance(self.delta_t)

        # 用新 alpha 更新混合物性
        rho = self._compute_mixture_rho(alpha)
        mu_mix = self._compute_mixture_mu(alpha)

        # ============================================================
        # 阶段 2：PIMPLE 压力-速度耦合
        # ============================================================
        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        # 构建边界条件
        U_bc = self._build_boundary_conditions()
        bc_mask = ~torch.isnan(U_bc[:, 0])

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            # 动量预测
            U, A_p, H = self._momentum_predictor(
                U, p, phi, rho, mu_mix, alpha,
            )

            # 应用边界条件
            if bc_mask.any():
                U[bc_mask] = U_bc[bc_mask]

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

                # 求解压力方程
                p = self._solve_pressure_equation(
                    p, phiHbyA, A_p, rho, mesh,
                )

                # 速度校正
                grad_p = self._compute_grad(p, mesh)
                U = HbyA - grad_p / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                # 应用边界条件
                if bc_mask.any():
                    U[bc_mask] = U_bc[bc_mask]

                # 通量校正
                p_P = gather(p, int_owner)
                p_N = gather(p, int_neigh)
                A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
                A_p_inv_face = (
                    w * gather(A_p_inv, int_owner)
                    + (1.0 - w) * gather(A_p_inv, int_neigh)
                )
                phi = phiHbyA - (p_N - p_P) * A_p_inv_face

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
                    "incompressibleVoF outer %d: U_res=%.6e, p_res=%.6e, "
                    "cont=%.6e",
                    outer, U_residual, p_residual, continuity_error,
                )

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return U, p, alpha, phi, convergence

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
        """求解带表面张力的动量方程。"""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        int_owner = owner[:n_internal]
        int_neigh = neighbour

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

        # 时间源项
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
        F_sigma = self._compute_surface_tension()

        source = H - grad_p + F_sigma

        # 求解
        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)

        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        return U_new, A_p, H

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _compute_surface_tension(self) -> torch.Tensor:
        """CSF 模型计算表面张力：F = σ·κ·∇α。"""
        return self.surface_tension.compute_force(self.alpha)

    def _recompute_H(
        self, U: torch.Tensor, phi: torch.Tensor,
        rho: torch.Tensor, mu_mix: torch.Tensor,
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

    def _solve_pressure_equation(
        self,
        p: torch.Tensor,
        phiHbyA: torch.Tensor,
        A_p: torch.Tensor,
        rho: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """求解两相流压力方程。"""
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
        """将 U、p、alpha.water 写入时间目录。"""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("alpha.water", self.alpha, time_str, self._alpha_data)
