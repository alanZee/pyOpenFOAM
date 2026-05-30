"""
Enhanced compressible N-phase Volume of Fluid (VOF) — v5.

在 Enhanced v4 基础上增加：

- **EOS 一致性检查**：验证混合密度与组分密度的一致性
- **压力-速度耦合增强**：改进 PISO 迭代的松弛策略
- **声速混合规则**：基于 Wood 公式的混合声速计算

Usage::

    from pyfoam.multiphase.compressible_multiphase_vof_enhanced_4 import (
        CompressibleMultiphaseVoFEnhanced4,
    )

    model = CompressibleMultiphaseVoFEnhanced4(
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
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_3 import (
    CompressibleMultiphaseVoFEnhanced3,
)

__all__ = ["CompressibleMultiphaseVoFEnhanced4"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class CompressibleMultiphaseVoFEnhanced4(CompressibleMultiphaseVoFEnhanced3):
    """Enhanced compressible N-phase VOF v5 with EOS consistency and Wood speed of sound.

    在 v4 基础上增加：
    - EOS 一致性检查（consistency check）
    - 压力-速度耦合增强（enhanced PISO relaxation）
    - Wood 公式混合声速（Wood's mixture speed of sound）

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
    eos_consistency_tol : float
        Tolerance for EOS consistency check. Default ``1e-4``.
    piso_relax : float
        PISO pressure under-relaxation factor. Default ``0.3``.
    use_wood_speed : bool
        Use Wood's formula for mixture speed of sound. Default ``True``.
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
        eos_consistency_tol: float = 1e-4,
        piso_relax: float = 0.3,
        use_wood_speed: bool = True,
    ) -> None:
        super().__init__(
            phase_names, eos_type, rho_ref, mu, R, gamma,
            p_ref, T_ref, C_alpha, n_piso, Ma_max,
            viscosity_model, S_sutherland, n_eos_iter,
            relaxation_factor, mixing_correction, transonic_limiter,
        )
        self._eos_consistency_tol = max(_EPS, eos_consistency_tol)
        self._piso_relax = max(0.01, min(piso_relax, 1.0))
        self._use_wood_speed = use_wood_speed

    @property
    def eos_consistency_tol(self) -> float:
        """EOS consistency tolerance."""
        return self._eos_consistency_tol

    @property
    def piso_relax(self) -> float:
        """PISO pressure relaxation factor."""
        return self._piso_relax

    @property
    def use_wood_speed(self) -> bool:
        """Whether to use Wood's mixture speed of sound."""
        return self._use_wood_speed

    # ------------------------------------------------------------------
    # EOS 一致性检查
    # ------------------------------------------------------------------

    def check_eos_consistency(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
        rho_expected: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Check EOS consistency between computed and expected densities.

        Returns a residual tensor and a boolean indicating convergence.

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` volume fractions.
        p : torch.Tensor
            ``(n_cells,)`` pressure (Pa).
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).
        rho_expected : torch.Tensor
            ``(n_cells,)`` expected mixture density (kg/m^3).

        Returns
        -------
        tuple of (torch.Tensor, bool)
            Residual tensor and convergence flag.
        """
        rho_computed = self.mixture_density(alphas, p, T)
        residual = (rho_computed - rho_expected).abs() / rho_expected.clamp(min=_EPS)
        max_residual = float(residual.max().item())
        converged = max_residual < self._eos_consistency_tol

        if not converged:
            logger.debug(
                "EOS consistency check: max_residual=%.6e (tol=%.6e)",
                max_residual, self._eos_consistency_tol,
            )

        return residual, converged

    # ------------------------------------------------------------------
    # Wood 公式混合声速
    # ------------------------------------------------------------------

    def wood_mixture_speed_of_sound(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture speed of sound using Wood's formula.

        Wood's formula for a bubbly/liquid mixture:

            1 / (rho_m * a_m^2) = sum_i (alpha_i / (rho_i * a_i^2))

        This is valid for low-frequency acoustics in dispersed mixtures.

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

        inv_rho_a2 = torch.zeros_like(p)

        for i in range(self._n_phases - 1):
            rho_i = self.phase_density(i, p, T_safe)
            a_i = self.phase_speed_of_sound(i, p, T_safe)
            inv_rho_a2 = inv_rho_a2 + alphas[:, i] / (rho_i * a_i.pow(2)).clamp(min=_EPS)

        alpha_N = self.compute_last_alpha(alphas)
        rho_N = self.phase_density(self._n_phases - 1, p, T_safe)
        a_N = self.phase_speed_of_sound(self._n_phases - 1, p, T_safe)
        inv_rho_a2 = inv_rho_a2 + alpha_N / (rho_N * a_N.pow(2)).clamp(min=_EPS)

        rho_m = self.mixture_density(alphas, p, T_safe)
        a_m_wood = torch.sqrt(1.0 / (rho_m * inv_rho_a2).clamp(min=_EPS))

        return a_m_wood.clamp(min=1.0)

    # ------------------------------------------------------------------
    # 增强 PISO 迭代
    # ------------------------------------------------------------------

    def enhanced_piso_pressure_correction(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
        rho_target: torch.Tensor,
        n_correctors: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enhanced PISO pressure-velocity coupling with multiple correctors.

        Applies successive pressure corrections with under-relaxation:

            p^(k+1) = p^(k) + relax * (rho_target - rho_m^(k)) * a_m^2

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` volume fractions.
        p : torch.Tensor
            ``(n_cells,)`` pressure (Pa).
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).
        rho_target : torch.Tensor
            ``(n_cells,)`` target mixture density (kg/m^3).
        n_correctors : int
            Number of PISO correction steps. Default ``2``.

        Returns
        -------
        tuple of torch.Tensor
            Corrected (pressure, temperature).
        """
        p_curr = p.clamp(min=1.0)
        T_curr = T.clamp(min=1.0)

        for k in range(n_correctors):
            rho_m = self.mixture_density(alphas, p_curr, T_curr)

            if self._use_wood_speed:
                a_m = self.wood_mixture_speed_of_sound(alphas, p_curr, T_curr)
            else:
                a_m = self.mixture_speed_of_sound(alphas, p_curr, T_curr)

            residual = rho_target - rho_m
            dp = residual * a_m.pow(2) * self._piso_relax
            p_curr = (p_curr + dp).clamp(min=1.0)

            # Temperature correction via ideal gas law consistency
            dT = residual * T_curr / rho_m.clamp(min=_EPS) * self._piso_relax * 0.1
            T_curr = (T_curr + dT).clamp(min=1.0, max=5000.0)

        return p_curr, T_curr

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
        """Iterate EOS with enhanced PISO coupling (v5 override).

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
        return self.enhanced_piso_pressure_correction(
            alphas, p, T, rho_target, n_correctors=self._n_piso,
        )

    def __repr__(self) -> str:
        phases = ", ".join(self._phase_names)
        return (
            f"CompressibleMultiphaseVoFEnhanced4("
            f"n_phases={self._n_phases}, phases=[{phases}], "
            f"piso_relax={self._piso_relax}, "
            f"viscosity='{self._viscosity_model}')"
        )
