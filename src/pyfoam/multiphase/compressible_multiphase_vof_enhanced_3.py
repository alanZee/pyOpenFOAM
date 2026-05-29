"""
Enhanced compressible N-phase Volume of Fluid (VOF) — v4.

在 Enhanced v3 基础上增加：

- **隐式 EOS 松弛**：改进的温度-压力隐式松弛迭代
- **混合规则修正**：混合密度和粘性的加权修正
- **声速限制器**：跨声速条件下的马赫数限制器增强

Usage::

    from pyfoam.multiphase.compressible_multiphase_vof_enhanced_3 import (
        CompressibleMultiphaseVoFEnhanced3,
    )

    model = CompressibleMultiphaseVoFEnhanced3(
        phase_names=["gas", "liquid"],
        eos_type=["perfectGas", "incompressible"],
        rho_ref=[1.225, 998.0],
        mu=[1.8e-5, 1.002e-3],
        R=[287.0, None],
        gamma=[1.4, None],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_2 import (
    CompressibleMultiphaseVoFEnhanced2,
)

__all__ = ["CompressibleMultiphaseVoFEnhanced3"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class CompressibleMultiphaseVoFEnhanced3(CompressibleMultiphaseVoFEnhanced2):
    """Enhanced compressible N-phase VOF v4 with improved EOS coupling.

    在 v3 基础上增加：
    - 隐式 EOS 松弛（implicit relaxation）
    - 混合规则修正（weighted mixing correction）
    - 声速限制器增强（transonic limiter）

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
        Number of PISO iterations. Default ``2``.
    Ma_max : float
        Maximum Mach number. Default ``0.9``.
    viscosity_model : str
        Viscosity model. Default ``"constant"``.
    S_sutherland : float
        Sutherland constant (K). Default ``110.4``.
    n_eos_iter : int
        Number of EOS coupling iterations. Default ``3``.
    relaxation_factor : float
        EOS relaxation factor (0-1). Default ``0.8``.
    mixing_correction : bool
        Enable weighted mixing correction. Default ``True``.
    transonic_limiter : bool
        Enable enhanced transonic limiter. Default ``True``.
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
        relaxation_factor: float = 0.8,
        mixing_correction: bool = True,
        transonic_limiter: bool = True,
    ) -> None:
        super().__init__(
            phase_names, eos_type, rho_ref, mu, R, gamma,
            p_ref, T_ref, C_alpha, n_piso, Ma_max,
            viscosity_model, S_sutherland, n_eos_iter,
        )
        self._relaxation_factor = max(0.01, min(relaxation_factor, 1.0))
        self._mixing_correction = mixing_correction
        self._transonic_limiter = transonic_limiter

    @property
    def relaxation_factor(self) -> float:
        """EOS relaxation factor."""
        return self._relaxation_factor

    @property
    def mixing_correction(self) -> bool:
        """Whether weighted mixing correction is enabled."""
        return self._mixing_correction

    @property
    def transonic_limiter(self) -> bool:
        """Whether transonic limiter is enabled."""
        return self._transonic_limiter

    # ------------------------------------------------------------------
    # 隐式 EOS 松弛
    # ------------------------------------------------------------------

    def relaxed_eos_update(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
        rho_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implicitly relaxed EOS update.

        Uses under-relaxation to stabilize pressure-temperature coupling:

            p_new = p_old + relaxation * (p_target - p_old)
            T_new = T_old + relaxation * (T_target - T_old)

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
            Relaxed (pressure, temperature).
        """
        p_curr = p.clamp(min=1.0)
        T_curr = T.clamp(min=1.0)

        for _ in range(self._n_eos_iter):
            # Target pressure from density
            rho_m = self.mixture_density(alphas, p_curr, T_curr)
            residual = rho_m - rho_target

            # Pressure correction (Newton-like)
            # dp/drho ~ a^2 for each phase, use mixture
            a_m = self.mixture_speed_of_sound(alphas, p_curr, T_curr)
            dp = residual * a_m.pow(2)

            # Temperature correction
            dT = residual * T_curr / rho_m.clamp(min=_EPS)

            # Apply under-relaxation
            p_curr = (p_curr - self._relaxation_factor * dp).clamp(min=1.0)
            T_curr = (T_curr - self._relaxation_factor * dT).clamp(min=1.0, max=5000.0)

        return p_curr, T_curr

    # ------------------------------------------------------------------
    # 混合规则修正
    # ------------------------------------------------------------------

    def corrected_mixture_density(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture density with weighted mixing correction.

        Uses harmonic mean for compressible-incompressible blends:

            rho_m = (sum_i alpha_i / rho_i)^(-1)

        when mixing_correction is enabled, otherwise falls back to
        standard arithmetic mean.

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
            ``(n_cells,)`` corrected mixture density (kg/m^3).
        """
        if not self._mixing_correction:
            return self.mixture_density(alphas, p, T)

        alphas = self.validate_alphas(alphas)
        T_safe = T.clamp(min=1.0)

        inv_rho = torch.zeros_like(T)

        for i in range(self._n_phases - 1):
            rho_i = self.phase_density(i, p, T_safe)
            inv_rho = inv_rho + alphas[:, i] / rho_i.clamp(min=_EPS)

        alpha_N = self.compute_last_alpha(alphas)
        rho_N = self.phase_density(self._n_phases - 1, p, T_safe)
        inv_rho = inv_rho + alpha_N / rho_N.clamp(min=_EPS)

        # Harmonic mean density
        rho_harmonic = 1.0 / inv_rho.clamp(min=_EPS)

        # Blend with arithmetic mean (avoids singularities)
        rho_arithmetic = self.mixture_density(alphas, p, T_safe)

        # Blend: 70% harmonic + 30% arithmetic for stability
        return (0.7 * rho_harmonic + 0.3 * rho_arithmetic).clamp(min=_EPS)

    # ------------------------------------------------------------------
    # 声速限制器
    # ------------------------------------------------------------------

    def transonic_mach_limiter(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
        U_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Enhanced Mach number limiter for transonic conditions.

        Applies a smooth limiter when Ma approaches Ma_max:

            U_limited = U * tanh(Ma_max / Ma) / tanh(1)

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
            ``(n_cells,)`` limited velocity magnitude (m/s).
        """
        if not self._transonic_limiter:
            return U_mag

        a_m = self.mixture_speed_of_sound(alphas, p, T)
        Ma = U_mag / a_m.clamp(min=_EPS)

        # Smooth limiter: reduces velocity as Ma -> Ma_max
        limit_factor = torch.tanh(
            torch.tensor(self._Ma_max, device=U_mag.device, dtype=U_mag.dtype)
        ) / torch.tanh(Ma.clamp(min=_EPS))

        limit_factor = limit_factor.clamp(max=1.0)

        return U_mag * limit_factor

    # ------------------------------------------------------------------
    # EOS 迭代（覆写父类）
    # ------------------------------------------------------------------

    def iterate_eos_coupled(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
        rho_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Iterate EOS with relaxation and mixing correction (v4 override).

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
        return self.relaxed_eos_update(alphas, p, T, rho_target)

    def __repr__(self) -> str:
        phases = ", ".join(self._phase_names)
        return (
            f"CompressibleMultiphaseVoFEnhanced3("
            f"n_phases={self._n_phases}, phases=[{phases}], "
            f"relaxation={self._relaxation_factor}, "
            f"viscosity='{self._viscosity_model}')"
        )
