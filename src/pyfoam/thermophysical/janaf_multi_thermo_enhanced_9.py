"""
Enhanced JANAF thermodynamic model v9 — radiation coupling, multi-phase flash, and activity coefficients.

Extends :class:`~pyfoam.thermophysical.janaf_multi_thermo_enhanced_8.JanafMultiThermoEnhanced8`
with:

- Radiation equilibrium coupling for high-temperature thermo
- Multi-phase flash calculation (isothermal-isobaric equilibrium)
- Species activity coefficients from JANAF data

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo_enhanced_9 import JanafMultiThermoEnhanced9
    from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

    phases = [
        JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
        JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermoEnhanced9(R=461.5, phases=phases, blend_width=5.0)
"""

from __future__ import annotations

import math
import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_8 import JanafMultiThermoEnhanced8
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

__all__ = ["JanafMultiThermoEnhanced9"]

logger = logging.getLogger(__name__)

_R_UNIV = 8.314462618  # J/(mol*K)
_STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2*K^4)


class JanafMultiThermoEnhanced9(JanafMultiThermoEnhanced8):
    """Enhanced multi-phase JANAF v9 with radiation coupling and flash calculation.

    Extends :class:`JanafMultiThermoEnhanced8` with:

    - **Radiation coupling**: adds Stefan-Boltzmann radiation heat loss/gain
      to the energy balance at high temperatures.
    - **Multi-phase flash**: simplified isothermal-isobaric flash to determine
      equilibrium phase fractions.
    - **Activity coefficients**: computes species activity from partial pressure
      and reference fugacity.

    Parameters
    ----------
    R, phases, Hf, blend_width, steepness, beta_coeffs, latent_sensible_fraction :
        See parent.
    S_ref, T_ref, S_ref_table, Cp_extrapolate_power, T_extrapolate_start :
        See parent.
    gibbs_max_iter, gibbs_tol : See parent.
    radiation_emissivity : float
        Surface emissivity for radiation coupling. Default 0.0 (disabled).
    radiation_area_factor : float
        Area-to-volume ratio for radiation (1/m). Default 1.0.
    flash_max_iter : int
        Maximum iterations for flash calculation. Default 20.
    flash_tol : float
        Convergence tolerance for flash. Default 1e-6.
    """

    def __init__(
        self,
        R: float,
        phases: Sequence[JanafPhase],
        Hf: float = 0.0,
        blend_width: float = 0.0,
        steepness: float = 2.0,
        beta_coeffs: Sequence[float] | None = None,
        latent_sensible_fraction: float = 0.1,
        S_ref: float = 0.0,
        T_ref: float = 298.15,
        S_ref_table: dict[float, float] | None = None,
        Cp_extrapolate_power: float = 0.5,
        T_extrapolate_start: float = 6000.0,
        gibbs_max_iter: int = 50,
        gibbs_tol: float = 1e-6,
        radiation_emissivity: float = 0.0,
        radiation_area_factor: float = 1.0,
        flash_max_iter: int = 20,
        flash_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            R=R, phases=phases, Hf=Hf, blend_width=blend_width,
            steepness=steepness, beta_coeffs=beta_coeffs,
            latent_sensible_fraction=latent_sensible_fraction,
            S_ref=S_ref, T_ref=T_ref, S_ref_table=S_ref_table,
            Cp_extrapolate_power=Cp_extrapolate_power,
            T_extrapolate_start=T_extrapolate_start,
            gibbs_max_iter=gibbs_max_iter, gibbs_tol=gibbs_tol,
        )
        self._radiation_emissivity = max(0.0, min(radiation_emissivity, 1.0))
        self._radiation_area_factor = max(0.0, radiation_area_factor)
        self._flash_max_iter = max(1, flash_max_iter)
        self._flash_tol = max(1e-12, flash_tol)

    @property
    def radiation_emissivity(self) -> float:
        """Surface emissivity for radiation."""
        return self._radiation_emissivity

    # ------------------------------------------------------------------
    # Radiation coupling
    # ------------------------------------------------------------------

    def radiation_heat_loss(self, T: float) -> float:
        """Radiation heat loss per unit volume.

        Q_rad = emissivity * sigma * area_factor * (T^4 - T_ambient^4)

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Radiation heat loss (W/m^3). Negative if T < T_ambient.
        """
        if self._radiation_emissivity < 1e-15:
            return 0.0
        T_amb = max(self._T_ref, 1.0)
        T_safe = max(T, 1.0)
        return (
            self._radiation_emissivity
            * _STEFAN_BOLTZMANN
            * self._radiation_area_factor
            * (T_safe ** 4 - T_amb ** 4)
        )

    # ------------------------------------------------------------------
    # Multi-phase flash
    # ------------------------------------------------------------------

    def flash_isothermal(
        self,
        T: float,
        p: float,
        initial_Y: Sequence[float],
        thermo_models: Sequence["JanafMultiThermoEnhanced9"],
    ) -> list[float]:
        """Simplified isothermal-isobaric flash calculation.

        Uses successive substitution to find equilibrium phase fractions.

        Parameters
        ----------
        T : float
            Temperature (K).
        p : float
            Pressure (Pa).
        initial_Y : sequence of float
            Initial mass fractions.
        thermo_models : sequence of JanafMultiThermoEnhanced9
            One thermo model per species.

        Returns
        -------
        list of float
            Equilibrium mass fractions.
        """
        Y = list(initial_Y)
        n = len(Y)

        for _ in range(self._flash_max_iter):
            # Compute species Gibbs energies
            g = []
            for i, model in enumerate(thermo_models):
                h_i = float(model.Cp(T).item()) if hasattr(model.Cp(T), "item") else float(model.Cp(T))
                s_i = model.S_ref_at_T(T)
                g.append(h_i - T * s_i)

            # Activity-based correction
            g_avg = sum(Y[i] * g[i] for i in range(n))
            max_change = 0.0
            for i in range(n):
                correction = -0.02 * (g[i] - g_avg)
                Y[i] = max(1e-10, Y[i] + correction)
                max_change = max(max_change, abs(correction))

            total = sum(Y)
            if total > 0:
                Y = [y / total for y in Y]

            if max_change < self._flash_tol:
                break

        return Y

    # ------------------------------------------------------------------
    # Activity coefficients
    # ------------------------------------------------------------------

    def activity_coefficient(
        self,
        T: float,
        species_idx: int,
        thermo_models: Sequence["JanafMultiThermoEnhanced9"],
    ) -> float:
        """Compute species activity coefficient.

        gamma_i = exp(G_i / (R * T))

        Parameters
        ----------
        T : float
            Temperature (K).
        species_idx : int
            Species index.
        thermo_models : sequence of JanafMultiThermoEnhanced9
            One thermo model per species.

        Returns
        -------
        float
            Activity coefficient (dimensionless).
        """
        if species_idx < 0 or species_idx >= len(thermo_models):
            return 1.0
        model = thermo_models[species_idx]
        T_safe = max(T, 1.0)
        h_i = float(model.Cp(T).item()) if hasattr(model.Cp(T), "item") else float(model.Cp(T))
        s_i = model.S_ref_at_T(T)
        g_i = h_i - T_safe * s_i
        gamma = math.exp(g_i / max(_R_UNIV * T_safe, 1e-10))
        return max(min(gamma, 1e3), 1e-3)

    def __repr__(self) -> str:
        total_L = self.total_latent_heat()
        rad = f", eps={self._radiation_emissivity}" if self._radiation_emissivity > 0 else ""
        return (
            f"JanafMultiThermoEnhanced9(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf}, L_total={total_L:.0f}{rad}, "
            f"gibbs_max_iter={self._gibbs_max_iter})"
        )
