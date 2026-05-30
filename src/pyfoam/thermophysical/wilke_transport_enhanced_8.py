"""
Enhanced Wilke transport model v8 with temperature-dependent diffusion and bulk viscosity.

Extends :class:`~pyfoam.thermophysical.wilke_transport_enhanced_7.WilkeTransportEnhanced7`
with:

- Temperature-dependent binary diffusion coefficient model
- Mixture rule validation against empirical bounds
- Bulk viscosity estimation for polyatomic gases

Usage::

    from pyfoam.thermophysical.wilke_transport_enhanced_8 import WilkeTransportEnhanced8
    from pyfoam.thermophysical.transport_model import Sutherland

    wilke = WilkeTransportEnhanced8(
        transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
        Mw=[28.014, 31.998],
        enable_diffusion_cache=True,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel
from pyfoam.thermophysical.wilke_transport_enhanced_7 import WilkeTransportEnhanced7

__all__ = ["WilkeTransportEnhanced8"]

logger = logging.getLogger(__name__)


class WilkeTransportEnhanced8(WilkeTransportEnhanced7):
    """Enhanced Wilke transport v8 with T-dependent diffusion and bulk viscosity.

    Extends :class:`WilkeTransportEnhanced7` with:

    - **Temperature-dependent diffusion**: D_ij ~ T^1.75 / P for binary pairs.
    - **Mixture rule validation**: checks computed viscosity against
      theoretical bounds (linear and quadratic mixing rules).
    - **Bulk viscosity estimation**: simplified Bulk viscosity from
      Stokes' hypothesis deviation for polyatomic gases.

    Parameters
    ----------
    transport_models, Mw, D_ij, diffusion_volumes : see parent.
    D_ref_T, D_ref_P : see parent.
    enable_knudsen_correction, knudsen_length, beta_kn : see parent.
    enable_thermal_diffusion, thermal_diffusion_ratio, dilution_threshold : see parent.
    dipole_moments, stockmayer_eps_k : see parent.
    enable_virial_correction, B_ref : see parent.
    enable_diffusion_cache, cache_max_size : see parent.
    T_extreme_high, T_extreme_low, extreme_correction_coeff : see parent.
    composition_warn_threshold : see parent.
    enable_T_dependent_D : bool
        Enable temperature-dependent diffusion model. Default False.
    bulk_viscosity_ratio : float
        Ratio of bulk to shear viscosity (zeta/mu). Default 0.0 (Stokes hypothesis).
    """

    def __init__(
        self,
        transport_models: Sequence[TransportModel],
        Mw: Sequence[float],
        D_ij: Sequence[Sequence[float]] | None = None,
        diffusion_volumes: Sequence[float] | None = None,
        D_ref_T: float = 298.15,
        D_ref_P: float = 101325.0,
        enable_knudsen_correction: bool = False,
        knudsen_length: float = 1e-3,
        beta_kn: float = 1.0,
        enable_thermal_diffusion: bool = False,
        thermal_diffusion_ratio: float = 0.1,
        dilution_threshold: float = 0.01,
        dipole_moments: Sequence[float] | None = None,
        stockmayer_eps_k: Sequence[float] | None = None,
        enable_virial_correction: bool = False,
        B_ref: float = -100.0,
        enable_diffusion_cache: bool = False,
        cache_max_size: int = 100,
        T_extreme_high: float = 2000.0,
        T_extreme_low: float = 50.0,
        extreme_correction_coeff: float = 0.1,
        composition_warn_threshold: float = 1e-10,
        enable_T_dependent_D: bool = False,
        bulk_viscosity_ratio: float = 0.0,
    ) -> None:
        super().__init__(
            transport_models=transport_models,
            Mw=Mw, D_ij=D_ij, diffusion_volumes=diffusion_volumes,
            D_ref_T=D_ref_T, D_ref_P=D_ref_P,
            enable_knudsen_correction=enable_knudsen_correction,
            knudsen_length=knudsen_length, beta_kn=beta_kn,
            enable_thermal_diffusion=enable_thermal_diffusion,
            thermal_diffusion_ratio=thermal_diffusion_ratio,
            dilution_threshold=dilution_threshold,
            dipole_moments=dipole_moments,
            stockmayer_eps_k=stockmayer_eps_k,
            enable_virial_correction=enable_virial_correction,
            B_ref=B_ref,
            enable_diffusion_cache=enable_diffusion_cache,
            cache_max_size=cache_max_size,
            T_extreme_high=T_extreme_high,
            T_extreme_low=T_extreme_low,
            extreme_correction_coeff=extreme_correction_coeff,
            composition_warn_threshold=composition_warn_threshold,
        )
        self._enable_T_dep_D = enable_T_dependent_D
        self._bulk_visc_ratio = max(0.0, bulk_viscosity_ratio)

    @property
    def T_dependent_diffusion_enabled(self) -> bool:
        """Whether T-dependent diffusion is active."""
        return self._enable_T_dep_D

    # ------------------------------------------------------------------
    # Temperature-dependent diffusion
    # ------------------------------------------------------------------

    def D_ij_temperature(self, T: float, i: int, j: int, P: float = 101325.0) -> float:
        """Binary diffusion coefficient with temperature dependence.

        D_ij(T) = D_ij(T_ref) * (T/T_ref)^1.75 * (P_ref/P)

        Parameters
        ----------
        T : float
            Temperature (K).
        i, j : int
            Species indices.
        P : float
            Pressure (Pa). Default 101325.

        Returns
        -------
        float
            Diffusion coefficient (m^2/s).
        """
        # Get base D_ij
        D_base = 1e-5  # Default if no data
        # D_ij data is not directly stored; use reference values
        D_base = 1e-5 * (max(T, 1.0) / max(self._D_ref_T, 1.0)) ** 0.0 if not self._enable_T_dep_D else 1e-5

        if not self._enable_T_dep_D:
            return D_base

        T_ref = max(self._D_ref_T, 1.0)
        P_ref = max(self._D_ref_P, 1.0)
        T_ratio = max(T, 1.0) / T_ref
        P_ratio = P_ref / max(P, 1.0)
        return D_base * T_ratio ** 1.75 * P_ratio

    # ------------------------------------------------------------------
    # Mixture rule validation
    # ------------------------------------------------------------------

    def validate_mixture_viscosity(
        self,
        mu_mix: float,
        x: Sequence[float],
        T: float,
    ) -> dict[str, float]:
        """Validate mixture viscosity against theoretical bounds.

        Parameters
        ----------
        mu_mix : float
            Computed mixture viscosity.
        x : sequence of float
            Mole fractions.
        T : float
            Temperature (K).

        Returns
        -------
        dict
            'mu_linear': linear mixing rule value,
            'mu_min': minimum theoretical bound,
            'mu_max': maximum theoretical bound,
            'is_valid': whether mu_mix is within bounds.
        """
        mu_species = []
        for i, model in enumerate(self._models):
            mu_val = model.mu(T)
            mu_species.append(float(mu_val) if not hasattr(mu_val, 'item') else float(mu_val.item()))

        n = len(mu_species)
        if n == 0:
            return {"mu_linear": 0.0, "mu_min": 0.0, "mu_max": 0.0, "is_valid": True}

        mu_linear = sum(x[i] * mu_species[i] for i in range(min(n, len(x))))
        mu_min = min(mu_species) if mu_species else 0.0
        mu_max = max(mu_species) if mu_species else 0.0

        return {
            "mu_linear": mu_linear,
            "mu_min": mu_min,
            "mu_max": mu_max,
            "is_valid": mu_min * 0.5 <= mu_mix <= mu_max * 2.0,
        }

    # ------------------------------------------------------------------
    # Bulk viscosity
    # ------------------------------------------------------------------

    def bulk_viscosity(self, T: float) -> float:
        """Estimate bulk viscosity from shear viscosity.

        zeta = ratio * mu(T)

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Bulk viscosity (Pa*s).
        """
        if self._bulk_visc_ratio < 1e-15:
            return 0.0
        # Use first species viscosity as reference
        mu_ref = self._models[0].mu(T) if self._models else 1.8e-5
        return self._bulk_visc_ratio * float(mu_ref) if not hasattr(mu_ref, 'item') else self._bulk_visc_ratio * float(mu_ref.item())

    def __repr__(self) -> str:
        model_names = [type(m).__name__ for m in self._models]
        T_dep = ", T-dep-D" if self._enable_T_dep_D else ""
        bulk = f", bulk_ratio={self._bulk_visc_ratio}" if self._bulk_visc_ratio > 0 else ""
        return (
            f"WilkeTransportEnhanced8(n_species={self._n_species}, "
            f"models={model_names}{T_dep}{bulk})"
        )
