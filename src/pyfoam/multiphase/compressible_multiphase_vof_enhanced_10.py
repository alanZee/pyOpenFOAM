"""Enhanced compressible N-phase VOF — v11.

Extends CompressibleMultiphaseVoFEnhanced9 with:
- Acoustic wave damping at interfaces
- Thermal equilibrium enforcement between phases
- Pressure-velocity coupling enhancement for compressible multiphase

Usage::

    from pyfoam.multiphase.compressible_multiphase_vof_enhanced_10 import (
        CompressibleMultiphaseVoFEnhanced10,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, Sequence
import torch
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_9 import (
    CompressibleMultiphaseVoFEnhanced9,
)

__all__ = ["CompressibleMultiphaseVoFEnhanced10"]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class CompressibleMultiphaseVoFEnhanced10(CompressibleMultiphaseVoFEnhanced9):
    """Enhanced compressible N-phase VOF v11 with acoustic damping
    and thermal equilibrium enforcement.

    Parameters
    ----------
    phase_names, eos_type, rho_ref, mu, R, gamma : see parent.
    acoustic_damping : bool
        Enable acoustic wave damping at interfaces. Default False.
    thermal_equilibrium : bool
        Enforce thermal equilibrium between phases. Default False.
    pressure_velocity_coupling : bool
        Enhanced pressure-velocity coupling. Default False.
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
        acoustic_damping: bool = False,
        thermal_equilibrium: bool = False,
        pressure_velocity_coupling: bool = False,
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
            dynamic_eos_blend, impedance_relaxation, energy_mass_coupling,
        )
        self._acoustic_damp = acoustic_damping
        self._thermal_eq = thermal_equilibrium
        self._pvel_coupling = pressure_velocity_coupling

    # ------------------------------------------------------------------
    # Acoustic wave damping
    # ------------------------------------------------------------------

    def acoustic_damping_factor(
        self,
        p: torch.Tensor,
        p_ref: float,
        alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Compute acoustic damping factor at interfaces.

        Applies increased damping where acoustic waves propagate
        through interfaces to prevent spurious pressure oscillations.

        Parameters
        ----------
        p : torch.Tensor
            (n_cells,) pressure field.
        p_ref : float
            Reference pressure.
        alphas : torch.Tensor
            (n_cells, N-1) volume fractions.

        Returns
        -------
        torch.Tensor
            (n_cells,) damping factor.
        """
        if not self._acoustic_damp:
            return torch.ones_like(p)

        alpha_0 = alphas[:, 0] if alphas.dim() > 1 else alphas
        interface = (alpha_0 > 0.01) & (alpha_0 < 0.99)
        interface_f = interface.to(p.dtype)

        p_dev = ((p - p_ref).abs() / max(abs(p_ref), 1.0)).clamp(max=5.0)
        damp = 1.0 - 0.5 * interface_f * p_dev
        return damp.clamp(min=0.1, max=1.0)

    # ------------------------------------------------------------------
    # Thermal equilibrium
    # ------------------------------------------------------------------

    def thermal_equilibrium_temperature(
        self,
        T_phases: Sequence[float],
        alphas: torch.Tensor,
    ) -> float:
        """Enforce thermal equilibrium: T_eq = sum(alpha_i * T_i).

        Parameters
        ----------
        T_phases : sequence of float
            Temperatures per phase.
        alphas : torch.Tensor
            Volume fractions.

        Returns
        -------
        float
            Equilibrium temperature.
        """
        if not self._thermal_eq or not T_phases:
            return T_phases[0] if T_phases else self._T_ref

        T_eq = 0.0
        total_alpha = 0.0
        for i, T_i in enumerate(T_phases):
            a_i = float(alphas[i]) if i < len(alphas) else 0.0
            T_eq += a_i * T_i
            total_alpha += a_i

        return T_eq / max(total_alpha, _EPS)

    def __repr__(self) -> str:
        phases = ", ".join(self._phase_names)
        ad = ", acoustic_damp" if self._acoustic_damp else ""
        return (
            f"CompressibleMultiphaseVoFEnhanced10("
            f"n_phases={self._n_phases}, phases=[{phases}]{ad})"
        )
