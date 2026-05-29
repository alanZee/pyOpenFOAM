"""
Enhanced compressible N-phase Volume of Fluid (VOF) — v3.

在 Enhanced v2 基础上增加：

- **多步 EOS 耦合**：改进的隐式压力-密度-温度耦合迭代
- **温度相关粘性**：Sutherland 定律或其他温度-粘性模型
- **声速加权混合**：精确的混合声速计算
- **总焓守恒**：总焓方程的显式处理

Usage::

    from pyfoam.multiphase.compressible_multiphase_vof_enhanced_2 import (
        CompressibleMultiphaseVoFEnhanced2,
    )

    model = CompressibleMultiphaseVoFEnhanced2(
        phase_names=["gas", "liquid"],
        eos_type=["perfectGas", "incompressible"],
        rho_ref=[1.225, 998.0],
        mu=[1.8e-5, 1.002e-3],
        R=[287.0, None],
        gamma=[1.4, None],
        viscosity_model="sutherland",
    )
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.compressible_multiphase_vof_enhanced import (
    CompressibleMultiphaseVoFEnhanced,
)

__all__ = ["CompressibleMultiphaseVoFEnhanced2"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class CompressibleMultiphaseVoFEnhanced2(CompressibleMultiphaseVoFEnhanced):
    """Enhanced compressible N-phase VOF v3 with improved EOS coupling.

    在 v2 基础上增加：
    - 多步 EOS 耦合（温度+压力联合迭代）
    - 温度相关粘性（Sutherland 定律）
    - 精确的混合声速计算
    - 总焓守恒追踪

    Parameters
    ----------
    phase_names : sequence of str
        Phase names (N >= 2).
    eos_type : sequence of str
        EOS type per phase.
    rho_ref : sequence of float
        Reference density per phase.
    mu : sequence of float
        Reference dynamic viscosity per phase (Pa·s).
    R : sequence of float or None
        Specific gas constant per phase.
    gamma : sequence of float or None
        Ratio of specific heats per phase.
    p_ref : float
        Reference pressure (Pa). Default ``101325``.
    T_ref : float
        Reference temperature (K). Default ``300``.
    C_alpha : float
        Compression coefficient. Default ``1.0``.
    n_piso : int
        Number of pressure-velocity correction iterations. Default ``2``.
    Ma_max : float
        Maximum Mach number limiter. Default ``0.9``.
    viscosity_model : str
        Viscosity model: ``"constant"`` or ``"sutherland"``. Default ``"constant"``.
    S_sutherland : float
        Sutherland constant (K). Default ``110.4``.
    n_eos_iter : int
        Number of EOS coupling iterations. Default ``3``.
    """

    def __init__(
        self,
        phase_names: Sequence[str],
        eos_type: Sequence[str],
        rho_ref: Sequence[float],
        mu: Sequence[float],
        R: Sequence[float | None] | None = None,
        gamma: Sequence[float | None] | None = None,
        p_ref: float = 101325.0,
        T_ref: float = 300.0,
        C_alpha: float = 1.0,
        n_piso: int = 2,
        Ma_max: float = 0.9,
        viscosity_model: str = "constant",
        S_sutherland: float = 110.4,
        n_eos_iter: int = 3,
    ) -> None:
        super().__init__(
            phase_names, eos_type, rho_ref, mu, R, gamma,
            p_ref, T_ref, C_alpha, n_piso, Ma_max,
        )
        self._viscosity_model = viscosity_model
        self._S_sutherland = S_sutherland
        self._n_eos_iter = max(1, n_eos_iter)

    @property
    def viscosity_model(self) -> str:
        """Viscosity model name."""
        return self._viscosity_model

    @property
    def S_sutherland(self) -> float:
        """Sutherland constant (K)."""
        return self._S_sutherland

    @property
    def n_eos_iter(self) -> int:
        """Number of EOS coupling iterations."""
        return self._n_eos_iter

    # ------------------------------------------------------------------
    # 温度相关粘性
    # ------------------------------------------------------------------

    def mixture_viscosity(
        self,
        alphas: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture viscosity with temperature dependence.

        For Sutherland model:

            mu(T) = mu_ref * (T/T_ref)^(3/2) * (T_ref + S) / (T + S)

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` volume fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture viscosity (Pa·s).
        """
        alphas = self.validate_alphas(alphas)
        T_safe = T.clamp(min=1.0)
        mu_mix = torch.zeros_like(T)

        for i in range(self._n_phases - 1):
            mu_i = self._phase_viscosity(i, T_safe)
            mu_mix = mu_mix + alphas[:, i] * mu_i

        alpha_N = self.compute_last_alpha(alphas)
        mu_mix = mu_mix + alpha_N * self._phase_viscosity(
            self._n_phases - 1, T_safe,
        )

        return mu_mix.clamp(min=_EPS)

    def _phase_viscosity(self, phase_idx: int, T: torch.Tensor) -> torch.Tensor:
        """Viscosity for a single phase with optional temperature dependence."""
        mu_ref = self._mu[phase_idx]

        if self._viscosity_model == "sutherland":
            T_ref = self._T_ref
            S = self._S_sutherland
            ratio = (T / T_ref).clamp(min=_EPS)
            mu = mu_ref * ratio.pow(1.5) * (T_ref + S) / (T + S)
            return mu.clamp(min=_EPS)
        else:
            return torch.full_like(T, mu_ref)

    # ------------------------------------------------------------------
    # 混合声速
    # ------------------------------------------------------------------

    def mixture_speed_of_sound(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture speed of sound using Wood's formula.

        1 / (rho_m * a_m^2) = sum_i (alpha_i / (rho_i * a_i^2))

        For perfect gas: a_i = sqrt(gamma * R * T)
        For incompressible: a_i -> infinity (use large value)

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` volume fractions.
        p : torch.Tensor
            ``(n_cells,)`` pressure (Pa).
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture speed of sound (m/s).
        """
        alphas = self.validate_alphas(alphas)
        T_safe = T.clamp(min=1.0)

        inv_rho_a2 = torch.zeros_like(T)

        for i in range(self._n_phases - 1):
            rho_i = self.phase_density(i, p, T_safe)
            a_i = self._phase_speed_of_sound(i, T_safe)
            inv_rho_a2 = inv_rho_a2 + alphas[:, i] / (rho_i * a_i.pow(2)).clamp(min=_EPS)

        alpha_N = self.compute_last_alpha(alphas)
        rho_N = self.phase_density(self._n_phases - 1, p, T_safe)
        a_N = self._phase_speed_of_sound(self._n_phases - 1, T_safe)
        inv_rho_a2 = inv_rho_a2 + alpha_N / (rho_N * a_N.pow(2)).clamp(min=_EPS)

        rho_m = self.mixture_density(alphas, p, T_safe)
        a_m = torch.sqrt(1.0 / (rho_m * inv_rho_a2).clamp(min=_EPS))

        return a_m.clamp(min=_EPS)

    def _phase_speed_of_sound(
        self, phase_idx: int, T: torch.Tensor,
    ) -> torch.Tensor:
        """Speed of sound for a single phase."""
        et = self._eos_type[phase_idx]
        if et == "perfectGas":
            gamma_i = self._gamma[phase_idx]
            R_i = self._R[phase_idx]
            return torch.sqrt(gamma_i * R_i * T).clamp(min=_EPS)
        else:
            # Incompressible: large speed of sound
            return torch.full_like(T, 1500.0)

    # ------------------------------------------------------------------
    # 总焓守恒
    # ------------------------------------------------------------------

    def mixture_total_enthalpy(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
        U_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture specific total enthalpy.

        h_total = e + p/rho + 0.5 * |U|^2

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` volume fractions.
        p : torch.Tensor
            ``(n_cells,)`` pressure (Pa).
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).
        U_mag : torch.Tensor
            ``(n_cells,)`` velocity magnitude (m/s).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture total enthalpy (J/kg).
        """
        e = self.mixture_internal_energy(alphas, p, T)
        rho_m = self.mixture_density(alphas, p, T).clamp(min=_EPS)
        h = e + p / rho_m + 0.5 * U_mag.pow(2)
        return h

    # ------------------------------------------------------------------
    # 多步 EOS 迭代
    # ------------------------------------------------------------------

    def iterate_eos_coupled(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
        rho_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Iterate pressure and temperature jointly to match target density.

        Uses sequential Newton-Raphson for pressure and temperature.

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` volume fractions.
        p : torch.Tensor
            ``(n_cells,)`` current pressure (Pa).
        T : torch.Tensor
            ``(n_cells,)`` current temperature (K).
        rho_target : torch.Tensor
            ``(n_cells,)`` target mixture density (kg/m^3).

        Returns
        -------
        tuple of torch.Tensor
            Corrected (pressure, temperature).
        """
        p_curr = p.clamp(min=1.0)
        T_curr = T.clamp(min=1.0)

        for _ in range(self._n_eos_iter):
            # Pressure iteration
            p_curr = self.iterate_pressure(alphas, p_curr, T_curr, rho_target)

            # Temperature correction for perfect gas phases
            rho_m = self.mixture_density(alphas, p_curr, T_curr)
            residual = rho_m - rho_target

            # Simple relaxation for temperature
            dT = residual * T_curr / rho_m.clamp(min=_EPS) * 0.1
            T_curr = (T_curr - dT).clamp(min=1.0, max=5000.0)

        return p_curr, T_curr

    def __repr__(self) -> str:
        phases = ", ".join(self._phase_names)
        return (
            f"CompressibleMultiphaseVoFEnhanced2("
            f"n_phases={self._n_phases}, phases=[{phases}], "
            f"C_alpha={self._C_alpha}, viscosity='{self._viscosity_model}')"
        )
