"""
cavitatingFoam — Cavitation solver for incompressible two-phase flow.

Solves the two-phase incompressible Navier-Stokes equations with
cavitation mass transfer between liquid and vapor phases.

The vapor volume fraction is transported:
    ∂α/∂t + ∇·(Uα) + ∇·(U_r α(1-α)) = m_dot / rho_v

where m_dot is the cavitation mass transfer rate.

Based on OpenFOAM's cavitatingFoam solver.

Usage::

    from pyfoam.applications.cavitating_foam import CavitatingFoam

    solver = CavitatingFoam("path/to/case")
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
from pyfoam.multiphase.volume_of_fluid import VOFAdvection
from pyfoam.multiphase.cavitation import SchnerrSauer

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["CavitatingFoam"]

logger = logging.getLogger(__name__)


class CavitatingFoam(SolverBase):
    """Cavitation solver for incompressible two-phase flow.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    rho_l : float
        Liquid density (default 1000.0 kg/m³).
    rho_v : float
        Vapor density (default 0.02 kg/m³).
    mu_l : float
        Liquid viscosity (default 1e-3 Pa·s).
    mu_v : float
        Vapor viscosity (default 1e-5 Pa·s).
    p_v : float
        Vapor pressure (default 2300.0 Pa).
    n_b : float
        Bubble number density for Schnerr-Sauer (default 1e13 m^-3).
    C_alpha : float
        VOF compression coefficient (default 1.0).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        rho_l: float = 1000.0,
        rho_v: float = 0.02,
        mu_l: float = 1e-3,
        mu_v: float = 1e-5,
        p_v: float = 2300.0,
        n_b: float = 1e13,
        C_alpha: float = 1.0,
    ) -> None:
        super().__init__(case_path)

        self.rho_l = rho_l
        self.rho_v = rho_v
        self.mu_l = mu_l
        self.mu_v = mu_v
        self.p_v = p_v
        self.C_alpha = C_alpha

        self._read_fv_solution_settings()
        self.U, self.p, self.alpha, self.phi = self._init_fields()
        self._U_data, self._p_data, self._alpha_data = self._init_field_data()

        self.vof = VOFAdvection(
            self.mesh, self.alpha, self.phi, self.U,
            C_alpha=C_alpha,
        )

        self.cavitation_model = SchnerrSauer(n_b=n_b, p_v=p_v)

        logger.info("CavitatingFoam ready")

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

    def _read_fv_solution_settings(self):
        fv = self.case.fvSolution
        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))
        self.n_outer_correctors = int(fv.get_path("PIMPLE/nOuterCorrectors", 3))
        self.n_correctors = int(fv.get_path("PIMPLE/nCorrectors", 2))
        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))
        self.convergence_tolerance = float(fv.get_path("PIMPLE/convergenceTolerance", 1e-4))
        self.max_outer_iterations = int(fv.get_path("PIMPLE/maxOuterIterations", 100))

    def _init_fields(self):
        device = get_device()
        dtype = get_default_dtype()

        U, _ = self.read_field_tensor("U", 0)
        U = U.to(device=device, dtype=dtype)

        p, _ = self.read_field_tensor("p_rgh", 0)
        p = p.to(device=device, dtype=dtype)

        alpha, _ = self.read_field_tensor("alpha.vapor", 0)
        alpha = alpha.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)
        return U, p, alpha, phi

    def _init_field_data(self):
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p_rgh", 0)
        alpha_data = self.case.read_field("alpha.vapor", 0)
        return U_data, p_data, alpha_data

    def _compute_mixture_rho(self, alpha):
        return alpha * self.rho_v + (1.0 - alpha) * self.rho_l

    def _compute_mixture_mu(self, alpha):
        return alpha * self.mu_v + (1.0 - alpha) * self.mu_l

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
            self.U, self.p, self.alpha, self.phi, conv = (
                self._pimple_cavitation_iteration()
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

    def _pimple_cavitation_iteration(self):
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        alpha = self.alpha.clone()
        phi = self.phi.clone()
        convergence = ConvergenceData()

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        # 构建边界条件
        U_bc = self._build_boundary_conditions()
        bc_mask = ~torch.isnan(U_bc[:, 0])

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            # Compute cavitation mass transfer
            m_dot = self.cavitation_model.compute_mass_transfer(
                alpha, p, self.rho_l, self.rho_v,
            )

            # VOF advection with cavitation source
            self.vof.alpha = alpha
            self.vof.phi = phi
            self.vof.U = U
            alpha = self.vof.advance(self.delta_t)

            # Apply cavitation source term
            V = mesh.cell_volumes.clamp(min=1e-30)
            alpha = alpha + self.delta_t * m_dot / self.rho_v / V
            alpha = alpha.clamp(0.0, 1.0)

            # Mixture properties
            rho = self._compute_mixture_rho(alpha)
            mu_mix = self._compute_mixture_mu(alpha)

            # 动量预测（完整的对流-扩散-时间项组装）
            U, A_p, H = self._momentum_predictor(
                U, p, phi, rho, mu_mix, alpha,
            )

            # 应用边界条件
            if bc_mask.any():
                U[bc_mask] = U_bc[bc_mask]

            # 速度限制器（防止发散）
            U_mag = U.norm(dim=1, keepdim=True).clamp(min=1e-30)
            U_limit = 100.0
            U = torch.where(U_mag > U_limit, U * (U_limit / U_mag), U)
            n_internal = mesh.n_internal_faces
            int_owner = mesh.owner[:n_internal]
            int_neigh = mesh.neighbour
            w = mesh.face_weights[:n_internal]

            for corr in range(self.n_correctors):
                HbyA = H / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                # Face flux from HbyA interpolation
                HbyA_face = (
                    w.unsqueeze(-1) * HbyA[int_owner]
                    + (1.0 - w).unsqueeze(-1) * HbyA[int_neigh]
                )
                phiHbyA = (HbyA_face * mesh.face_areas[:n_internal]).sum(dim=1)

                # Assemble and solve pressure equation
                phi_full = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
                phi_full[:n_internal] = phiHbyA
                p_solver = create_solver(
                    "PCG", tolerance=self.p_tolerance, max_iter=self.p_max_iter,
                )
                p_eqn = assemble_pressure_equation(phi_full, A_p, mesh)
                p, _, _ = solve_pressure_equation(
                    p_eqn, p, p_solver,
                    tolerance=self.p_tolerance, max_iter=self.p_max_iter,
                )

                # Correct velocity and face flux
                U = correct_velocity(U, HbyA, p, A_p, mesh)
                if bc_mask.any():
                    U[bc_mask] = U_bc[bc_mask]
                phi = correct_face_flux(phi_full, p, A_p, mesh)

                # Recompute H for next correction
                if corr < self.n_correctors - 1:
                    H = self._recompute_H(U, phi, rho, mu_mix)

            # Under-relaxation
            if self.alpha_U < 1.0:
                U = self.alpha_U * U + (1.0 - self.alpha_U) * U_prev
            if self.alpha_p < 1.0:
                p = self.alpha_p * p + (1.0 - self.alpha_p) * p_prev

            # 速度限制器
            U_mag = U.norm(dim=1, keepdim=True).clamp(min=1e-30)
            U = torch.where(U_mag > U_limit, U * (U_limit / U_mag), U)

            # Convergence
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            convergence.U_residual = U_residual
            convergence.p_residual = p_residual
            convergence.outer_iterations = outer + 1

        return U, p, alpha, phi, convergence

    def _momentum_predictor(
        self, U, p, phi, rho, mu_mix, alpha,
    ):
        """求解混合物动量方程。"""
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

        source = H - grad_p

        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)
        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        return U_new, A_p, H

    def _recompute_H(self, U, phi, rho, mu_mix):
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
        H.index_add_(0, int_owner,
                      lower.unsqueeze(-1) * U[int_neigh] * V_P.unsqueeze(-1))
        H.index_add_(0, int_neigh,
                      upper.unsqueeze(-1) * U[int_owner] * V_N.unsqueeze(-1))
        return H

    def _compute_residual(self, field, field_old):
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def _write_fields(self, time):
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p_rgh", self.p, time_str, self._p_data)
        self.write_field("alpha.vapor", self.alpha, time_str, self._alpha_data)
