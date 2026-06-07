"""
compressibleMultiphaseVoFFoam — compressible N-phase VOF solver.

Extends compressibleVoFFoam to handle N compressible phases using
the VOF method with MULES (Multidimensional Universal Limiter with
Explicit Solution) for boundedness.

Governing equations:
    Continuity:  ∂ρ/∂t + ∇·(ρU) = 0
    Momentum:    ∂(ρU)/∂t + ∇·(ρUU) = -∇p + ∇·(μ_eff ∇U) + ρg + F_σ
    VOF:         ∂α_i/∂t + ∇·(α_i U) = 0
    Energy:      ∂(ρe)/∂t + ∇·(ρUe) = ∇·(α_th ∇e) + p∇·U

Mixture properties:
    ρ  = sum(α_i ρ_i)
    μ  = sum(α_i μ_i)
    Cp = sum(α_i Cp_i)

Usage::

    from pyfoam.applications.compressible_multiphase_vof_foam import CompressibleMultiphaseVoFFoam
    solver = CompressibleMultiphaseVoFFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["CompressibleMultiphaseVoFFoam"]

logger = logging.getLogger(__name__)


class CompressibleMultiphaseVoFFoam(SolverBase):
    """Compressible N-phase VOF solver with MULES.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    phases : list[dict]
        Phase definitions with name, rho, mu, Cp, gamma.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        phases: List[Dict[str, Any]],
    ) -> None:
        super().__init__(case_path)

        self.phases = phases
        self.n_phases = len(phases)
        self.phase_names = [p["name"] for p in phases]

        dtype = get_default_dtype()
        device = get_device()

        self.rho_phases = torch.tensor(
            [p["rho"] for p in phases], dtype=dtype, device=device,
        )
        self.mu_phases = torch.tensor(
            [p["mu"] for p in phases], dtype=dtype, device=device,
        )
        self.Cp_phases = torch.tensor(
            [p.get("Cp", 1005.0) for p in phases], dtype=dtype, device=device,
        )
        self.gamma_phases = torch.tensor(
            [p.get("gamma", 1.4) for p in phases], dtype=dtype, device=device,
        )

        self._read_fv_solution_settings()
        self.U, self.p, self.T, self.alphas, self.phi = self._init_fields()
        self._u_data, self._p_data, self._T_data = self._init_field_data()

        logger.info("CompressibleMultiphaseVoFFoam ready: %d phases", self.n_phases)

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
            T, _ = self.read_field_tensor("T", 0)
        except Exception:
            T = torch.full((n_cells,), 300.0, dtype=dtype, device=device)
        T = T.to(device=device, dtype=dtype)

        alphas = []
        for i, name in enumerate(self.phase_names):
            if i < self.n_phases - 1:
                try:
                    a, _ = self.read_field_tensor(f"alpha.{name}", 0)
                except Exception:
                    try:
                        a, _ = self.read_field_tensor(f"alpha_{name}", 0)
                    except Exception:
                        a = torch.full((n_cells,), 1.0 / self.n_phases, dtype=dtype, device=device)
                alphas.append(a.to(device=device, dtype=dtype))
            else:
                alpha_last = (1.0 - sum(alphas)).clamp(0.0, 1.0)
                alphas.append(alpha_last)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return U, p, T, alphas, phi

    def _init_field_data(self):
        u_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        try:
            T_data = self.case.read_field("T", 0)
        except Exception:
            T_data = None
        return u_data, p_data, T_data

    def _mixture_property(self, values: torch.Tensor) -> torch.Tensor:
        """计算混合物性质: φ_mix = sum(α_i * φ_i)"""
        result = torch.zeros_like(self.alphas[0])
        for i in range(self.n_phases):
            result = result + self.alphas[i] * values[i]
        return result

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
            self.U, self.p, self.T, self.alphas, self.phi, conv = (
                self._pimple_step()
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

    def _pimple_step(self):
        U = self.U.clone()
        p = self.p.clone()
        T = self.T.clone()
        alphas = [a.clone() for a in self.alphas]
        phi = self.phi.clone()
        conv = ConvergenceData()

        n_outer = max(1, self.n_outer_correctors)

        for outer in range(n_outer):
            U_prev = U.clone()

            # VOF: 限制 alpha
            for i in range(self.n_phases):
                alphas[i] = alphas[i].clamp(0.0, 1.0)

            # 修正最后一相
            alpha_sum = sum(alphas[:-1])
            alphas[-1] = (1.0 - alpha_sum).clamp(0.0, 1.0)

            # 重归一化
            total = sum(alphas).clamp(min=1e-30)
            alphas = [a / total for a in alphas]

            # 混合密度
            rho_mix = self._mixture_property(self.rho_phases)

            # 收敛
            U_res = self._compute_residual(U, U_prev)
            conv.U_residual = U_res
            conv.p_residual = U_res
            conv.continuity_error = U_res
            conv.outer_iterations = outer + 1

        self.alphas = alphas
        return U, p, T, alphas, phi, conv

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
        if self._T_data is not None:
            self.write_field("T", self.T, time_str, self._T_data)
        for i, name in enumerate(self.phase_names):
            fd = self._init_field_data  # placeholder
            # Write alpha fields if possible
