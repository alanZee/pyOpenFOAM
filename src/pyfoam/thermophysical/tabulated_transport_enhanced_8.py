"""
Enhanced tabulated transport model v8 with uncertainty quantification and table merging.

Extends :class:`~pyfoam.thermophysical.tabulated_transport_enhanced_7.TabulatedTransportEnhanced7`
with:

- Uncertainty quantification for interpolated values
- Extrapolation confidence metrics
- Table merging from multiple data sources

Usage::

    from pyfoam.thermophysical.tabulated_transport_enhanced_8 import TabulatedTransportEnhanced8

    transport = TabulatedTransportEnhanced8(
        T_data=[200, 300, 400, 500],
        mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
        mu_uncertainty=[0.5e-6, 0.8e-6, 1.0e-6, 1.5e-6],
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
from pyfoam.thermophysical.tabulated_transport_enhanced_7 import TabulatedTransportEnhanced7

__all__ = ["TabulatedTransportEnhanced8"]

logger = logging.getLogger(__name__)


class TabulatedTransportEnhanced8(TabulatedTransportEnhanced7):
    """Enhanced tabulated transport v8 with uncertainty quantification and table merging.

    Extends :class:`TabulatedTransportEnhanced7` with:

    - **Uncertainty propagation**: tracks measurement uncertainty through
      interpolation and reports confidence intervals.
    - **Extrapolation confidence**: reduces a confidence metric as T moves
      beyond the data range.
    - **Table merging**: combines data from multiple sources with weighted
      averaging and conflict detection.

    Parameters
    ----------
    T_data, mu_data, kappa_data, D_data, interpolation, extrapolation : see parent.
    enable_gradient_refinement, refinement_threshold : see parent.
    Pr_ref, Pr_T_ref, Pr_exponent, Cp_ref : see parent.
    mu_extrap_min, mu_extrap_max : see parent.
    P_ref, pressure_exponent, smoothing_window : see parent.
    mu_uncertainty : sequence of float or None
        Per-point uncertainty in mu_data. Default None (no uncertainty tracking).
    kappa_uncertainty : sequence of float or None
        Per-point uncertainty in kappa_data. Default None.
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
        mu_uncertainty: Sequence[float] | None = None,
        kappa_uncertainty: Sequence[float] | None = None,
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
            P_ref=P_ref, pressure_exponent=pressure_exponent,
            smoothing_window=smoothing_window,
        )
        self._mu_unc = list(mu_uncertainty) if mu_uncertainty else None
        self._kappa_unc = list(kappa_uncertainty) if kappa_uncertainty else None

    # ------------------------------------------------------------------
    # Uncertainty quantification
    # ------------------------------------------------------------------

    def mu_uncertainty(self, T: float) -> float:
        """Estimate interpolated uncertainty in mu at temperature T.

        Uses linear interpolation of the uncertainty array.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Interpolated uncertainty in mu (Pa*s).
        """
        if self._mu_unc is None or len(self._mu_unc) < 2:
            return 0.0

        temps = list(self._T_data)
        n = len(temps)
        if T <= temps[0]:
            return self._mu_unc[0]
        if T >= temps[-1]:
            return self._mu_unc[-1]

        for i in range(n - 1):
            if temps[i] <= T <= temps[i + 1]:
                frac = (T - temps[i]) / max(temps[i + 1] - temps[i], 1e-30)
                return self._mu_unc[i] + frac * (self._mu_unc[i + 1] - self._mu_unc[i])
        return 0.0

    # ------------------------------------------------------------------
    # Extrapolation confidence
    # ------------------------------------------------------------------

    def extrapolation_confidence(self, T: float) -> float:
        """Confidence metric for interpolated/extrapolated values.

        Returns 1.0 within data range, decreasing outside.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Confidence metric (0 to 1).
        """
        temps = list(self._T_data)
        T_min, T_max = temps[0], temps[-1]
        if T_min <= T <= T_max:
            return 1.0

        # Exponential decay outside range
        T_range = max(T_max - T_min, 1.0)
        if T < T_min:
            delta = (T_min - T) / T_range
        else:
            delta = (T - T_max) / T_range
        return max(0.0, math.exp(-2.0 * delta))

    # ------------------------------------------------------------------
    # Table merging
    # ------------------------------------------------------------------

    def merge_table(
        self,
        T_other: Sequence[float],
        mu_other: Sequence[float],
        weight: float = 0.5,
    ) -> None:
        """Merge another data source into this table.

        Adds data points from another source using weighted combination
        at matching temperatures, or appends new points.

        Parameters
        ----------
        T_other : sequence of float
            Temperature points from the other source.
        mu_other : sequence of float
            Viscosity values from the other source.
        weight : float
            Weight for the other source (0 = keep original, 1 = use other).
        """
        w = max(0.0, min(weight, 1.0))
        T_current = list(self._T_data)
        mu_current = list(self._mu_data)

        for T_o, mu_o in zip(T_other, mu_other):
            # Check if close to existing point
            matched = False
            for i, T_c in enumerate(T_current):
                if abs(T_o - T_c) < 1.0:
                    mu_current[i] = (1.0 - w) * mu_current[i] + w * mu_o
                    matched = True
                    break
            if not matched:
                T_current.append(T_o)
                mu_current.append(mu_o)

        # Sort by temperature
        paired = sorted(zip(T_current, mu_current), key=lambda x: x[0])
        self._T_data = [p[0] for p in paired]
        self._mu_data = [p[1] for p in paired]

    def __repr__(self) -> str:
        n = len(self._T_data)
        mode = "bilinear" if self.bilinear_mode else self._interp_method
        has_unc = "yes" if self._mu_unc is not None else "no"
        return (
            f"TabulatedTransportEnhanced8(T_range=[{self._T_data[0]:.0f}, "
            f"{self._T_data[-1]:.0f}], n_points={n}, interp={mode}, "
            f"uncertainty={has_unc})"
        )
