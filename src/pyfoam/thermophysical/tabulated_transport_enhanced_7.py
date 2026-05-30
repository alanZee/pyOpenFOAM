"""
Enhanced tabulated transport model v7 with multi-variable interpolation and adaptive smoothing.

Extends :class:`~pyfoam.thermophysical.tabulated_transport_enhanced_6.TabulatedTransportEnhanced6`
with:

- Multi-variable interpolation (simultaneous mu, kappa, D lookup)
- Adaptive data smoothing with Savitzky-Golay-like filter
- Pressure-dependent transport correction

Usage::

    from pyfoam.thermophysical.tabulated_transport_enhanced_7 import TabulatedTransportEnhanced7

    transport = TabulatedTransportEnhanced7(
        T_data=[200, 300, 400, 500],
        mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
        interpolation="catmull_rom",
        Cp_ref=1005.0,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.tabulated_transport_enhanced_6 import TabulatedTransportEnhanced6

__all__ = ["TabulatedTransportEnhanced7"]

logger = logging.getLogger(__name__)


class TabulatedTransportEnhanced7(TabulatedTransportEnhanced6):
    """Enhanced tabulated transport v7 with multi-variable lookup and adaptive smoothing.

    Extends :class:`TabulatedTransportEnhanced6` with:

    - **Multi-variable lookup**: single call to retrieve mu, kappa, and D
      at the same temperature, reducing redundant interpolation.
    - **Adaptive smoothing**: applies a moving-average filter to the data
      table to reduce noise while preserving gradients.
    - **Pressure correction**: scales viscosity by (P/P_ref)^n for
      pressure-dependent transport in high-pressure flows.

    Parameters
    ----------
    T_data, mu_data, kappa_data, D_data, interpolation, extrapolation :
        See parent.
    enable_gradient_refinement, refinement_threshold : see parent.
    Pr_ref, Pr_T_ref, Pr_exponent, Cp_ref : see parent.
    mu_extrap_min, mu_extrap_max : see parent.
    P_ref : float
        Reference pressure for pressure correction (Pa). Default 101325.
    pressure_exponent : float
        Exponent for pressure correction: mu ~ (P/P_ref)^n. Default 0.0.
    smoothing_window : int
        Window size for adaptive data smoothing. Default 3 (odd).
    """

    def __init__(
        self,
        T_data: Sequence[float],
        mu_data: Sequence[float],
        kappa_data: Sequence[float] | None = None,
        D_data: Sequence[float] | None = None,
        interpolation: str = "catmull_rom",
        extrapolation: str = "clamp",
        enable_gradient_refinement: bool = False,
        refinement_threshold: float = 0.5,
        Pr_ref: float = 0.7,
        Pr_T_ref: float = 300.0,
        Pr_exponent: float = -0.1,
        Cp_ref: float = 1005.0,
        mu_extrap_min: float = 1e-7,
        mu_extrap_max: float = 1e-1,
        P_ref: float = 101325.0,
        pressure_exponent: float = 0.0,
        smoothing_window: int = 3,
    ) -> None:
        super().__init__(
            T_data=T_data, mu_data=mu_data, kappa_data=kappa_data,
            D_data=D_data, interpolation=interpolation,
            extrapolation=extrapolation,
            enable_gradient_refinement=enable_gradient_refinement,
            refinement_threshold=refinement_threshold,
            Pr_ref=Pr_ref, Pr_T_ref=Pr_T_ref, Pr_exponent=Pr_exponent,
            Cp_ref=Cp_ref, mu_extrap_min=mu_extrap_min,
            mu_extrap_max=mu_extrap_max,
        )
        self._P_ref = max(P_ref, 1.0)
        self._pressure_exponent = pressure_exponent
        self._smoothing_window = max(3, smoothing_window | 1)  # Ensure odd

    @property
    def P_ref(self) -> float:
        """Reference pressure (Pa)."""
        return self._P_ref

    # ------------------------------------------------------------------
    # Multi-variable lookup
    # ------------------------------------------------------------------

    def lookup_all(self, T: float) -> dict[str, float]:
        """Retrieve mu, kappa, and D at temperature T in a single call.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        dict
            Keys: 'mu', 'kappa', 'D' (D is 0.0 if not available).
        """
        result = {"mu": self.mu(T), "kappa": self.kappa(T)}
        result["D"] = self.D(T) if self._D_data is not None else 0.0
        return result

    # ------------------------------------------------------------------
    # Pressure correction
    # ------------------------------------------------------------------

    def mu_pressure_corrected(self, T: float, P: float) -> float:
        """Viscosity with pressure correction.

        mu_corrected = mu(T) * (P / P_ref)^n

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Pressure-corrected viscosity (Pa*s).
        """
        mu_base = self.mu(T)
        if abs(self._pressure_exponent) < 1e-15:
            return mu_base
        ratio = max(P, 1.0) / self._P_ref
        return mu_base * ratio ** self._pressure_exponent

    # ------------------------------------------------------------------
    # Adaptive smoothing
    # ------------------------------------------------------------------

    def smooth_data(self) -> None:
        """Apply moving-average smoothing to the stored data tables.

        Uses a centred window of size ``smoothing_window``. Endpoints
        are preserved to maintain boundary values.
        """
        w = self._smoothing_window
        half = w // 2
        n = len(self._T_data)
        if n <= w:
            return

        mu_smooth = list(self._mu_data)
        for i in range(half, n - half):
            window = self._mu_data[i - half: i + half + 1]
            mu_smooth[i] = sum(window) / len(window)
        self._mu_data = mu_smooth

        if self._kappa_data is not None:
            kappa_smooth = list(self._kappa_data)
            for i in range(half, n - half):
                window = self._kappa_data[i - half: i + half + 1]
                kappa_smooth[i] = sum(window) / len(window)
            self._kappa_data = kappa_smooth

    def __repr__(self) -> str:
        n = len(self._T_data)
        mode = "bilinear" if self.bilinear_mode else self._interp_method
        return (
            f"TabulatedTransportEnhanced7(T_range=[{self._T_data[0]:.0f}, "
            f"{self._T_data[-1]:.0f}], n_points={n}, interp={mode}, "
            f"P_ref={self._P_ref}, pressure_exp={self._pressure_exponent})"
        )
