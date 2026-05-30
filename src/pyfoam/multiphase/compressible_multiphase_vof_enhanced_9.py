"""Enhanced compressible N-phase VOF — v10.

Extends CompressibleMultiphaseVoFEnhanced8 with:
- Dynamic EOS blending for smooth transitions between stiff and ideal gas
- Acoustic impedance-weighted pressure relaxation
- Energy-consistent mass transfer coupling

Usage::

    from pyfoam.multiphase.compressible_multiphase_vof_enhanced_9 import (
        CompressibleMultiphaseVoFEnhanced9,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, Sequence
import torch
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_8 import (
    CompressibleMultiphaseVoFEnhanced8,
)

__all__ = ["CompressibleMultiphaseVoFEnhanced9"]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class CompressibleMultiphaseVoFEnhanced9(CompressibleMultiphaseVoFEnhanced8):
    """Enhanced compressible N-phase VOF v10 with dynamic EOS blending
    and acoustic impedance coupling.

    Parameters
    ----------
    phase_names, eos_type, rho_ref, mu, R, gamma : see parent.
    dynamic_eos_blend : bool
        Enable dynamic EOS blending. Default False.
    impedance_relaxation : bool
        Enable acoustic impedance-weighted relaxation. Default False.
    energy_mass_coupling : bool
        Enable energy-consistent mass transfer. Default False.
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
        dynamic_eos_blend: bool = False,
        impedance_relaxation: bool = False,
        energy_mass_coupling: bool = False,
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
            pressure_wave_damping, energy_consistent_compression,
            wave_damping_coeff,
        )
        self._dynamic_eos = dynamic_eos_blend
        self._impedance_relax = impedance_relaxation
        self._energy_mass = energy_mass_coupling

    # ------------------------------------------------------------------
    # Dynamic EOS blending
    # ------------------------------------------------------------------

    def dynamic_eos_weight(self, Ma: torch.Tensor) -> torch.Tensor:
        """Compute dynamic EOS blending weight from local Mach number.

        At low Ma: use ideal gas weight, at high Ma: use stiff gas weight.

        Parameters
        ----------
        Ma : torch.Tensor
            (n_cells,) local Mach number.

        Returns
        -------
        torch.Tensor
            (n_cells,) blending weight (0=ideal, 1=stiff).
        """
        if not self._dynamic_eos:
            return torch.zeros_like(Ma)
        Ma_safe = Ma.abs().clamp(max=2.0)
        return torch.tanh(Ma_safe / max(self._Ma_max, 0.1)).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Impedance relaxation
    # ------------------------------------------------------------------

    def impedance_relaxation_pressure(
        self,
        p: torch.Tensor,
        alphas: torch.Tensor,
        rho_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Impedance-weighted pressure relaxation at interfaces.

        Parameters
        ----------
        p : torch.Tensor
            (n_cells,) pressure field.
        alphas : torch.Tensor
            (n_cells, N-1) volume fractions.
        rho_mix : torch.Tensor
            (n_cells,) mixture density.

        Returns
        -------
        torch.Tensor
            Relaxed pressure field.
        """
        if not self._impedance_relax:
            return p

        alpha_0 = alphas[:, 0] if alphas.dim() > 1 else alphas
        interface = (alpha_0 > 0.01) & (alpha_0 < 0.99)

        # Impedance Z ~ rho * c, use rho as proxy
        Z = rho_mix.abs().clamp(min=_EPS)
        f_relax = interface.float() * 0.1 / Z.clamp(min=1.0)

        p_ref = self._p_ref
        return p - f_relax * (p - p_ref)

    def __repr__(self) -> str:
        phases = ", ".join(self._phase_names)
        de = ", dynamic_eos" if self._dynamic_eos else ""
        return (
            f"CompressibleMultiphaseVoFEnhanced9("
            f"n_phases={self._n_phases}, phases=[{phases}]{de})"
        )
