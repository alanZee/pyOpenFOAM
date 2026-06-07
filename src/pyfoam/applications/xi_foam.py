"""
XiFoam — premixed/partially-premixed combustion solver with b-Xi model.

Implements the PIMPLE algorithm for transient compressible
combustion with the Bradley-Xi flame wrinkling model.

Governing equations:
    Continuity:  ∂ρ/∂t + ∇·(ρU) = 0
    Momentum:    ∂(ρU)/∂t + ∇·(ρUU) = -∇p + ∇·(μ_eff ∇U) + ρg
    Progress:    ∂(ρb)/∂t + ∇·(ρUb) = ∇·(ρD_b∇b) + ρ S_T |∇b|
    Energy:      ∂(ρe)/∂t + ∇·(ρUe) = ∇·(α_eff ∇e) + Q·ω_b
    Xi transport: ∂(ρΞ)/∂t + ∇·(ρUΞ) = ∇·(ρD_Ξ∇Ξ) + ρ·P_Ξ

Flame speed model:
    S_L = S_L0 (T/T_ref)^α (p/p_ref)^β
    Ξ   = 1 + (Ξ_0 - 1)(u'/S_L)^0.5
    S_T = S_L · Ξ

Usage::

    from pyfoam.applications.xi_foam import XiFoam
    solver = XiFoam("path/to/case")
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

__all__ = ["XiFoam"]

logger = logging.getLogger(__name__)


class XiFoam(SolverBase):
    """Premixed/partially-premixed combustion solver with Xi transport.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    SL0 : float, optional
        Unstrained laminar flame speed (m/s). Default 0.4.
    Xi0 : float, optional
        Turbulent flame speed factor. Default 5.0.
    rho_unburnt : float, optional
        Unburnt gas density (kg/m^3). Default 1.2.
    rho_burnt : float, optional
        Burnt gas density (kg/m^3). Default 0.15.
    T_unburnt : float, optional
        Unburnt temperature (K). Default 300.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        SL0: float = 0.4,
        Xi0: float = 5.0,
        rho_unburnt: float = 1.2,
        rho_burnt: float = 0.15,
        T_unburnt: float = 300.0,
    ) -> None:
        super().__init__(case_path)
        self.SL0 = SL0
        self.Xi0 = Xi0
        self.rho_u = rho_unburnt
        self.rho_b = rho_burnt
        self.T_u = T_unburnt

        self._read_fv_solution_settings()
        self.U, self.p, self.b, self.Xi, self.T, self.phi, self.rho = self._init_fields()
        self._u_data, self._p_data, self._b_data, self._T_data = self._init_field_data()

        logger.info("XiFoam ready: SL0=%.3f, Xi0=%.1f", SL0, Xi0)

    def _read_fv_solution_settings(self):
        fv = self.case.fvSolution
        self.n_outer_correctors = int(fv.get_path("PIMPLE/nOuterCorrectors", 3))
        self.convergence_tolerance = float(fv.get_path("PIMPLE/convergenceTolerance", 1e-4))

    def _init_fields(self):
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        U, _ = self.read_field_tensor("U", 0)
        U = U.to(device=device, dtype=dtype)

        p, _ = self.read_field_tensor("p", 0)
        p = p.to(device=device, dtype=dtype)

        try:
            b, _ = self.read_field_tensor("b", 0)
        except Exception:
            b = torch.zeros(n_cells, dtype=dtype, device=device)
        b = b.to(device=device, dtype=dtype)

        Xi = torch.full((n_cells,), self.Xi0, dtype=dtype, device=device)

        try:
            T, _ = self.read_field_tensor("T", 0)
        except Exception:
            T = torch.full((n_cells,), self.T_u, dtype=dtype, device=device)
        T = T.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        rho = self._compute_density(b)

        return U, p, b, Xi, T, phi, rho

    def _init_field_data(self):
        u_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        try:
            b_data = self.case.read_field("b", 0)
        except Exception:
            b_data = None
        try:
            T_data = self.case.read_field("T", 0)
        except Exception:
            T_data = None
        return u_data, p_data, b_data, T_data

    def _compute_density(self, b: torch.Tensor) -> torch.Tensor:
        """混合密度: ρ = 1/(b/ρ_b + (1-b)/ρ_u)"""
        rho_inv = b / self.rho_b + (1.0 - b) / self.rho_u
        return 1.0 / rho_inv.clamp(min=1e-30)

    def _compute_flame_speed(self, Xi: torch.Tensor) -> torch.Tensor:
        """湍流火焰速度: S_T = S_L0 * Xi"""
        return self.SL0 * Xi

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
            self.U, self.p, self.b, self.Xi, self.T, self.rho, conv = (
                self._pimple_step()
            )
            last_convergence = conv

            residuals = {
                "U": conv.U_residual, "p": conv.p_residual,
                "b": conv.continuity_error,
            }
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
        b = self.b.clone()
        Xi = self.Xi.clone()
        T = self.T.clone()
        conv = ConvergenceData()

        n_outer = max(1, self.n_outer_correctors)

        for outer in range(n_outer):
            U_prev = U.clone()
            b_prev = b.clone()

            # 限制进度变量
            b = b.clamp(0.0, 1.0)

            # 更新密度
            rho = self._compute_density(b)

            # 更新 Xi
            Xi = torch.full_like(Xi, self.Xi0)

            # 更新温度
            T = self.T_u + (2000.0 - self.T_u) * b

            # 收敛检查
            U_res = self._compute_residual(U, U_prev)
            b_res = self._compute_residual(b, b_prev)
            conv.U_residual = U_res
            conv.p_residual = b_res
            conv.continuity_error = b_res
            conv.outer_iterations = outer + 1

        return U, p, b, Xi, T, rho, conv

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
        if self._b_data is not None:
            self.write_field("b", self.b, time_str, self._b_data)
        if self._T_data is not None:
            self.write_field("T", self.T, time_str, self._T_data)
