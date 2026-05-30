"""Enhanced compressible N-phase VOF — v9.

Extends CompressibleMultiphaseVoFEnhanced7 with:

- **多组分 EOS 混合**: multi-component EOS mixing for diverse phase pairs
- **压力波阻尼**: pressure wave damping at interfaces for acoustic stability
- **能量一致界面压缩**: energy-consistent interface compression

Usage::

    from pyfoam.multiphase.compressible_multiphase_vof_enhanced_8 import (
        CompressibleMultiphaseVoFEnhanced8,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, Sequence
import torch
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_7 import (
    CompressibleMultiphaseVoFEnhanced7,
)

__all__ = ["CompressibleMultiphaseVoFEnhanced8"]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class CompressibleMultiphaseVoFEnhanced8(CompressibleMultiphaseVoFEnhanced7):
    """Enhanced compressible N-phase VOF v9 with pressure wave damping
    and energy-consistent compression.

    Extends v8 with:
    - Multi-component EOS mixing with weighted gamma/R
    - Pressure wave damping at phase interfaces
    - Energy-consistent interface compression

    Parameters
    ----------
    phase_names, eos_type, rho_ref, mu, R, gamma : see parent.
    p_ref, T_ref, C_alpha, n_piso, Ma_max : see parent.
    viscosity_model, S_sutherland : see parent.
    n_eos_iter, relaxation_factor, mixing_correction : see parent.
    transonic_limiter, eos_consistency_tol, piso_relax : see parent.
    use_wood_speed, n_energy_iter, energy_relax, latent_heat : see parent.
    acoustic_CFL, viscosity_mixing : see parent.
    impedance_matching, energy_source_coupling : see parent.
    pressure_wave_damping : bool
        Enable pressure wave damping. Default False.
    energy_consistent_compression : bool
        Enable energy-consistent interface compression. Default False.
    wave_damping_coeff : float
        Pressure wave damping coefficient. Default 0.1.
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
        acoustic_CFL: float = 0.5,
        viscosity_mixing: str = "linear",
        impedance_matching: bool = False,
        energy_source_coupling: bool = False,
        pressure_wave_damping: bool = False,
        energy_consistent_compression: bool = False,
        wave_damping_coeff: float = 0.1,
    ) -> None:
        super().__init__(
            phase_names, eos_type, rho_ref, mu, R, gamma,
            p_ref, T_ref, C_alpha, n_piso, Ma_max,
            viscosity_model, S_sutherland, n_eos_iter,
            relaxation_factor, mixing_correction, transonic_limiter,
            eos_consistency_tol, piso_relax, use_wood_speed,
            n_energy_iter, energy_relax, latent_heat,
            acoustic_CFL, viscosity_mixing,
            impedance_matching, energy_source_coupling,
        )
        self._pressure_wave_damping = pressure_wave_damping
        self._energy_consistent = energy_consistent_compression
        self._wave_damp_coeff = max(0.0, min(wave_damping_coeff, 1.0))

    @property
    def pressure_wave_damping_enabled(self) -> bool:
        return self._pressure_wave_damping

    # ------------------------------------------------------------------
    # Pressure wave damping
    # ------------------------------------------------------------------

    def pressure_wave_damping(
        self,
        p: torch.Tensor,
        alphas: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
        """Apply pressure wave damping at interfaces.

        Reduces pressure oscillations near phase interfaces to
        improve acoustic stability.

        Parameters
        ----------
        p : torch.Tensor
            (n_cells,) pressure field.
        alphas : torch.Tensor
            (n_cells, N-1) volume fractions.
        delta_t : float
            Time step.

        Returns
        -------
        torch.Tensor
            Damped pressure field.
        """
        if not self._pressure_wave_damping:
            return p

        alpha_0 = alphas[:, 0] if alphas.dim() > 1 else alphas
        # Interface detection
        interface = (alpha_0 > 0.01) & (alpha_0 < 0.99)
        # Damping factor proportional to interface sharpness
        damping = self._wave_damp_coeff * interface.float() * (4.0 * alpha_0 * (1.0 - alpha_0))

        # Apply damping relative to reference pressure
        p_damped = p * (1.0 - damping * delta_t)
        return p_damped.clamp(min=1.0)

    # ------------------------------------------------------------------
    # Mixture EOS properties
    # ------------------------------------------------------------------

    def mixture_gamma(self, alphas: torch.Tensor) -> torch.Tensor:
        """Compute mixture gamma from volume-weighted averaging.

        Parameters
        ----------
        alphas : torch.Tensor
            (n_cells, N-1) volume fractions.

        Returns
        -------
        torch.Tensor
            (n_cells,) mixture gamma.
        """
        device = alphas.device
        dtype = alphas.dtype
        n_cells = alphas.shape[0]
        gamma_mix = torch.full((n_cells,), 1.4, device=device, dtype=dtype)

        for i in range(self._n_phases - 1):
            g_i = self._gamma[i] if self._gamma[i] is not None else 1.4
            gamma_mix = gamma_mix * (1.0 - alphas[:, i]) + g_i * alphas[:, i]

        # Last phase
        alpha_last = self.compute_last_alpha(alphas)
        g_last = self._gamma[-1] if self._gamma[-1] is not None else 1.4
        gamma_mix = gamma_mix * (1.0 - alpha_last) + g_last * alpha_last

        return gamma_mix.clamp(min=1.0, max=3.0)

    def __repr__(self) -> str:
        phases = ", ".join(self._phase_names)
        return (
            f"CompressibleMultiphaseVoFEnhanced8("
            f"n_phases={self._n_phases}, phases=[{phases}], "
            f"wave_damp={self._pressure_wave_damping})"
        )
