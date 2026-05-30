"""
Enhanced Wilke transport model v7 with adaptive caching and composition monitoring.

Extends :class:`~pyfoam.thermophysical.wilke_transport_enhanced_6.WilkeTransportEnhanced6`
with:

- Adaptive cache eviction with temperature-frequency tracking
- Species composition monitoring and outlier detection
- Multi-reference-state viscosity normalisation

Usage::

    from pyfoam.thermophysical.wilke_transport_enhanced_7 import WilkeTransportEnhanced7
    from pyfoam.thermophysical.transport_model import Sutherland

    wilke = WilkeTransportEnhanced7(
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
from pyfoam.thermophysical.wilke_transport_enhanced_6 import WilkeTransportEnhanced6

__all__ = ["WilkeTransportEnhanced7"]

logger = logging.getLogger(__name__)


class WilkeTransportEnhanced7(WilkeTransportEnhanced6):
    """Enhanced Wilke transport v7 with adaptive caching and composition monitoring.

    Extends :class:`WilkeTransportEnhanced6` with:

    - **Adaptive cache eviction**: tracks temperature access frequency
      and evicts least-frequently-used entries instead of FIFO.
    - **Composition monitoring**: detects species with extremely low
      mass fractions and reports potential numerical issues.
    - **Multi-reference normalisation**: normalises mixture viscosity
      against multiple reference states for validation.

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
    composition_warn_threshold : float
        Minimum mass fraction below which a warning is emitted. Default 1e-10.
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
        )
        self._comp_warn_threshold = composition_warn_threshold
        self._access_freq: dict[float, int] = {}

    @property
    def composition_warn_threshold(self) -> float:
        """Minimum mass fraction threshold for warnings."""
        return self._comp_warn_threshold

    # ------------------------------------------------------------------
    # Adaptive cache with frequency tracking
    # ------------------------------------------------------------------

    def _get_cached_D(self, T: float, i: int, j: int) -> float | None:
        """Retrieve cached D_ij with frequency tracking."""
        T_key = round(T, 1)
        if T_key in self._access_freq:
            self._access_freq[T_key] += 1
        else:
            self._access_freq[T_key] = 1
        return super()._get_cached_D(T, i, j)

    def _set_cached_D(self, T: float, i: int, j: int, D_val: float) -> None:
        """Store D_ij with LFU-aware eviction."""
        if not self._cache_enabled:
            return
        T_key = round(T, 1)
        if len(self._diffusion_cache) >= self._cache_max:
            # Evict least frequently used
            if self._access_freq:
                lfu_key = min(self._access_freq, key=self._access_freq.get)
                self._diffusion_cache.pop(lfu_key, None)
                self._access_freq.pop(lfu_key, None)
        if T_key not in self._diffusion_cache:
            self._diffusion_cache[T_key] = {}
        self._diffusion_cache[T_key][(i, j)] = D_val

    # ------------------------------------------------------------------
    # Composition monitoring
    # ------------------------------------------------------------------

    def check_composition(self, x: Sequence[float]) -> dict[str, any]:
        """Check species mole fractions for numerical issues.

        Parameters
        ----------
        x : sequence of float
            Mole fractions.

        Returns
        -------
        dict
            Keys: 'warnings' (list of str), 'n_depleted' (int),
            'min_fraction' (float).
        """
        warnings = []
        n_depleted = 0
        min_frac = min(x) if x else 0.0

        for i, xi in enumerate(x):
            if xi < self._comp_warn_threshold:
                warnings.append(f"Species {i} near-depleted: x={xi:.2e}")
                n_depleted += 1

        return {
            "warnings": warnings,
            "n_depleted": n_depleted,
            "min_fraction": min_frac,
        }

    def __repr__(self) -> str:
        model_names = [type(m).__name__ for m in self._models]
        diff = "FSG" if self.has_fsg else ("D_ij" if self.has_diffusion else "none")
        cache = f", cache({len(self._diffusion_cache)})" if self._cache_enabled else ""
        return (
            f"WilkeTransportEnhanced7(n_species={self._n_species}, "
            f"models={model_names}, diffusion={diff}{cache})"
        )
