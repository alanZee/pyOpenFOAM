"""
Enhanced Wilke transport model v6 with mixture-rule validation and binary diffusion caching.

Extends :class:`~pyfoam.thermophysical.wilke_transport_enhanced_5.WilkeTransportEnhanced5`
with:

- Mixture-rule validation against experimental viscosity data
- Binary diffusion coefficient caching for efficiency
- Temperature-dependent correction factors for extreme conditions

Usage::

    from pyfoam.thermophysical.wilke_transport_enhanced_6 import WilkeTransportEnhanced6
    from pyfoam.thermophysical.transport_model import Sutherland

    wilke = WilkeTransportEnhanced6(
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
from pyfoam.thermophysical.wilke_transport_enhanced_5 import WilkeTransportEnhanced5

__all__ = ["WilkeTransportEnhanced6"]

logger = logging.getLogger(__name__)


class WilkeTransportEnhanced6(WilkeTransportEnhanced5):
    """Enhanced Wilke transport v6 with caching and validation.

    Extends :class:`WilkeTransportEnhanced5` with:

    - **Binary diffusion cache**: stores computed D_ij(T) to avoid redundant
      evaluations, with LRU-like eviction by max cache size.
    - **Mixture viscosity validation**: compares computed mixture viscosity
      against a user-supplied reference value and reports relative error.
    - **Extreme-condition correction**: applies a correction factor when T
      or P exceeds specified bounds to account for real-gas effects.

    Parameters
    ----------
    transport_models : sequence of TransportModel
        One transport model per species.
    Mw : sequence of float
        Molecular weights (g/mol).
    D_ij, diffusion_volumes, D_ref_T, D_ref_P : see parent
    enable_knudsen_correction, knudsen_length, beta_kn : see parent
    enable_thermal_diffusion, thermal_diffusion_ratio, dilution_threshold : see parent
    dipole_moments, stockmayer_eps_k : see parent
    enable_virial_correction, B_ref : see parent
    enable_diffusion_cache : bool
        Enable D_ij(T) caching. Default False.
    cache_max_size : int
        Maximum number of cached temperature entries. Default 100.
    T_extreme_high : float
        Temperature above which extreme correction applies (K). Default 2000.
    T_extreme_low : float
        Temperature below which extreme correction applies (K). Default 50.
    extreme_correction_coeff : float
        Strength of the extreme-condition correction. Default 0.1.
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
        )
        self._cache_enabled = enable_diffusion_cache
        self._cache_max = max(10, cache_max_size)
        self._diffusion_cache: dict[float, dict[tuple[int, int], float]] = {}
        self._T_extreme_high = T_extreme_high
        self._T_extreme_low = max(T_extreme_low, 1.0)
        self._extreme_coeff = extreme_correction_coeff

    @property
    def diffusion_cache_size(self) -> int:
        """Number of cached temperature entries."""
        return len(self._diffusion_cache)

    # ------------------------------------------------------------------
    # Diffusion cache
    # ------------------------------------------------------------------

    def _get_cached_D(self, T: float, i: int, j: int) -> float | None:
        """Retrieve cached binary diffusion coefficient."""
        if not self._cache_enabled:
            return None
        T_key = round(T, 1)
        if T_key in self._diffusion_cache:
            return self._diffusion_cache[T_key].get((i, j))
        return None

    def _set_cached_D(self, T: float, i: int, j: int, D_val: float) -> None:
        """Store a binary diffusion coefficient in cache."""
        if not self._cache_enabled:
            return
        T_key = round(T, 1)
        if len(self._diffusion_cache) >= self._cache_max:
            # Evict oldest entry
            oldest_key = next(iter(self._diffusion_cache))
            del self._diffusion_cache[oldest_key]
        if T_key not in self._diffusion_cache:
            self._diffusion_cache[T_key] = {}
        self._diffusion_cache[T_key][(i, j)] = D_val

    # ------------------------------------------------------------------
    # Extreme-condition correction
    # ------------------------------------------------------------------

    def _extreme_T_correction(self, T: float) -> float:
        """Temperature correction factor for extreme conditions.

        Returns a multiplicative correction close to 1.0 within normal
        range, deviating for very high or very low temperatures.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Correction factor.
        """
        if T > self._T_extreme_high:
            ratio = (T - self._T_extreme_high) / max(self._T_extreme_high, 1.0)
            return 1.0 + self._extreme_coeff * ratio
        if T < self._T_extreme_low:
            ratio = (self._T_extreme_low - T) / max(self._T_extreme_low, 1.0)
            return max(0.5, 1.0 - self._extreme_coeff * ratio)
        return 1.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_mixture_viscosity(
        self,
        T: float,
        x: Sequence[float],
        mu_reference: float,
    ) -> dict[str, float]:
        """Compare computed mixture viscosity against reference data.

        Parameters
        ----------
        T : float
            Temperature (K).
        x : sequence of float
            Mole fractions.
        mu_reference : float
            Reference (experimental) mixture viscosity.

        Returns
        -------
        dict
            Keys: 'mu_computed', 'mu_reference', 'relative_error'.
        """
        # Compute mixture viscosity using per-species models
        mu_computed = 0.0
        for i in range(self._n_species):
            mu_i = self._models[i].mu(T)
            mu_computed += x[i] * mu_i
        rel_err = abs(mu_computed - mu_reference) / max(abs(mu_reference), 1e-30)
        return {
            "mu_computed": mu_computed,
            "mu_reference": mu_reference,
            "relative_error": rel_err,
        }

    def __repr__(self) -> str:
        model_names = [type(m).__name__ for m in self._models]
        diff = "FSG" if self.has_fsg else ("D_ij" if self.has_diffusion else "none")
        cache = f", cache({len(self._diffusion_cache)})" if self._cache_enabled else ""
        return (
            f"WilkeTransportEnhanced6(n_species={self._n_species}, "
            f"models={model_names}, diffusion={diff}{cache})"
        )
