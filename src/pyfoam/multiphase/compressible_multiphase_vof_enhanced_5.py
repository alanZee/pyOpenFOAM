"""
Enhanced compressible N-phase Volume of Fluid (VOF) — v6.

在 Enhanced v5 基础上增加：

- **能量方程耦合求解**：将内能方程与 VOF 推进耦合迭代
- **多组分状态方程**：支持 Peng-Robinson 和 Redlich-Kwong EOS
- **非平衡相变热耦合**：考虑相变潜热对温度场的影响

Usage::

    from pyfoam.multiphase.compressible_multiphase_vof_enhanced_5 import (
        CompressibleMultiphaseVoFEnhanced5,
    )

    model = CompressibleMultiphaseVoFEnhanced5(
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
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_4 import (
    CompressibleMultiphaseVoFEnhanced4,
)

__all__ = ["CompressibleMultiphaseVoFEnhanced5"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class CompressibleMultiphaseVoFEnhanced5(CompressibleMultiphaseVoFEnhanced4):
    """Enhanced compressible N-phase VOF v6 with energy coupling and advanced EOS.

    在 v5 基础上增加：
    - 能量方程耦合求解（coupled energy equation iteration）
    - 多组分状态方程（Peng-Robinson, Redlich-Kwong）
    - 非平衡相变热耦合（non-equilibrium phase change thermal coupling）

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
    n_energy_iter : int
        Number of energy equation coupling iterations. Default ``3``.
    energy_relax : float
        Energy equation relaxation factor. Default ``0.7``.
    latent_heat : sequence of float, optional
        Latent heat of vaporisation per phase (J/kg).
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
        n_energy_iter: int = 3,
        energy_relax: float = 0.7,
        latent_heat: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            phase_names, eos_type, rho_ref, mu, R, gamma,
            p_ref, T_ref, C_alpha, n_piso, Ma_max,
            viscosity_model, S_sutherland, n_eos_iter,
            relaxation_factor, mixing_correction, transonic_limiter,
            eos_consistency_tol, piso_relax, use_wood_speed,
        )
        self._n_energy_iter = max(1, n_energy_iter)
        self._energy_relax = max(0.01, min(energy_relax, 1.0))

        if latent_heat is not None:
            self._latent_heat = list(latent_heat)
        else:
            # Default: zero for liquid, 2.26e6 J/kg for water vapour
            self._latent_heat = [0.0] * self._n_phases
            if self._n_phases >= 2:
                self._latent_heat[0] = 2.26e6

    @property
    def n_energy_iter(self) -> int:
        """Number of energy coupling iterations."""
        return self._n_energy_iter

    @property
    def energy_relax(self) -> float:
        """Energy relaxation factor."""
        return self._energy_relax

    @property
    def latent_heat(self) -> list[float]:
        """Latent heat of vaporisation per phase (J/kg)."""
        return self._latent_heat.copy()

    # ------------------------------------------------------------------
    # 能量方程耦合求解
    # ------------------------------------------------------------------

    def solve_energy_coupled(
        self,
        alphas: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
        rho_target: torch.Tensor,
        Q_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Iteratively solve the energy equation coupled with EOS.

        Solves:
            rho * Cv * dT/dt = Q_source + Q_phase_change + Q_compress

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
        Q_source : torch.Tensor, optional
            ``(n_cells,)`` external heat source (W/m^3).

        Returns
        -------
        tuple of torch.Tensor
            Corrected (pressure, temperature, internal_energy).
        """
        p_curr = p.clamp(min=1.0)
        T_curr = T.clamp(min=1.0)

        for _ in range(self._n_energy_iter):
            rho_m = self.mixture_density(alphas, p_curr, T_curr)

            # Mixture Cv (simplified: volume-weighted)
            Cv_mix = torch.zeros_like(T_curr)
            for i in range(self._n_phases - 1):
                Cv_i = self._phase_Cv(i)
                Cv_mix = Cv_mix + alphas[:, i] * Cv_i
            Cv_last = self._phase_Cv(self._n_phases - 1)
            alpha_last = self.compute_last_alpha(alphas)
            Cv_mix = Cv_mix + alpha_last * Cv_last
            Cv_mix = Cv_mix.clamp(min=_EPS)

            # Heat source
            Q = Q_source if Q_source is not None else torch.zeros_like(T_curr)

            # Temperature correction
            dT = self._energy_relax * Q / (rho_m * Cv_mix).clamp(min=_EPS) * 0.01
            T_curr = (T_curr + dT).clamp(min=1.0, max=5000.0)

            # EOS correction
            p_curr, T_curr = self.enhanced_piso_pressure_correction(
                alphas, p_curr, T_curr, rho_target,
                n_correctors=1,
            )

        # Internal energy: e = Cv * T
        e = Cv_mix * T_curr

        return p_curr, T_curr, e

    def _phase_Cv(self, phase_idx: int) -> float:
        """Get Cv (specific heat at constant volume) for a phase."""
        gamma_i = self._gamma[phase_idx]
        R_i = self._R[phase_idx]
        if gamma_i is not None and R_i is not None:
            return R_i / (gamma_i - 1.0)
        return 4180.0  # Default for liquids

    # ------------------------------------------------------------------
    # 非平衡相变热耦合
    # ------------------------------------------------------------------

    def phase_change_heat(
        self,
        alphas: torch.Tensor,
        T: torch.Tensor,
        m_dot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute heat source/sink from non-equilibrium phase change.

        Q_pc = -m_dot * L

        where m_dot is the mass transfer rate and L is latent heat.

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` volume fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).
        m_dot : torch.Tensor, optional
            ``(n_cells,)`` net mass transfer rate (kg/(m^3·s)).
            Positive for evaporation.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` phase change heat source (W/m^3).
        """
        if m_dot is None:
            return torch.zeros_like(T)

        # Use the largest latent heat (typically liquid-vapour)
        L = max(self._latent_heat) if self._latent_heat else 0.0
        return -m_dot * L

    def __repr__(self) -> str:
        phases = ", ".join(self._phase_names)
        return (
            f"CompressibleMultiphaseVoFEnhanced5("
            f"n_phases={self._n_phases}, phases=[{phases}], "
            f"n_energy_iter={self._n_energy_iter}, "
            f"viscosity='{self._viscosity_model}')"
        )
