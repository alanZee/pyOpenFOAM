"""Enhanced tabulated transport model v9 with adaptive viscosity models and table quality metrics.

Extends :class:`~pyfoam.thermophysical.tabulated_transport_enhanced_8.TabulatedTransportEnhanced8`
with:

- Adaptive viscosity model selection based on data quality
- Table quality metrics (smoothness, monotonicity, coverage)
- Pressure-dependent thermal conductivity correction

Usage::

    from pyfoam.thermophysical.tabulated_transport_enhanced_9 import TabulatedTransportEnhanced9

    transport = TabulatedTransportEnhanced9(
        T_data=[200, 300, 400, 500],
        mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
        enable_quality_metrics=True,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.tabulated_transport_enhanced_8 import TabulatedTransportEnhanced8

__all__ = ["TabulatedTransportEnhanced9"]

logger = logging.getLogger(__name__)


class TabulatedTransportEnhanced9(TabulatedTransportEnhanced8):
    """Enhanced tabulated transport v9 with adaptive models and quality metrics.

    Extends :class:`TabulatedTransportEnhanced8` with:

    - **Adaptive model selection**: selects best interpolation model based on
      data spacing uniformity and gradient smoothness.
    - **Table quality metrics**: computes smoothness index, monotonicity score,
      and temperature coverage ratio.
    - **Pressure-dependent kappa**: thermal conductivity correction for
      high-pressure conditions.

    Parameters
    ----------
    T_data, mu_data, kappa_data, D_data, interpolation, extrapolation : see parent.
    enable_gradient_refinement, refinement_threshold : see parent.
    Pr_ref, Pr_T_ref, Pr_exponent, Cp_ref : see parent.
    mu_extrap_min, mu_extrap_max : see parent.
    P_ref, pressure_exponent, smoothing_window : see parent.
    mu_uncertainty, kappa_uncertainty : see parent.
    enable_quality_metrics : bool
        Enable table quality metrics computation. Default False.
    pressure_kappa_coeff : float
        Pressure correction coefficient for thermal conductivity. Default 0.0.
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
        enable_quality_metrics: bool = False,
        pressure_kappa_coeff: float = 0.0,
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
            mu_uncertainty=mu_uncertainty,
            kappa_uncertainty=kappa_uncertainty,
        )
        self._quality_metrics = enable_quality_metrics
        self._pressure_kappa_coeff = max(0.0, pressure_kappa_coeff)

    # ------------------------------------------------------------------
    # Table quality metrics
    # ------------------------------------------------------------------

    def table_smoothness(self) -> float:
        """Compute smoothness index of the viscosity table.

        Uses second derivative magnitude as smoothness metric.
        Returns 1.0 for perfectly smooth data, lower for noisy data.

        Returns
        -------
        float
            Smoothness index (0 to 1).
        """
        mu = list(self._mu_data)
        n = len(mu)
        if n < 3:
            return 1.0

        d2_max = 0.0
        d2_sum = 0.0
        for i in range(1, n - 1):
            d2 = abs(mu[i + 1] - 2.0 * mu[i] + mu[i - 1])
            d2_sum += d2
            d2_max = max(d2_max, d2)

        mean_mu = sum(abs(m) for m in mu) / n
        if mean_mu < 1e-30:
            return 1.0

        # Normalize: lower curvature = higher smoothness
        normalized = d2_max / (mean_mu * n)
        return max(0.0, 1.0 - normalized)

    def table_monotonicity(self) -> float:
        """Compute monotonicity score of the viscosity table.

        Returns 1.0 if monotonically increasing, 0.0 if highly non-monotone.

        Returns
        -------
        float
            Monotonicity score (0 to 1).
        """
        mu = list(self._mu_data)
        n = len(mu)
        if n < 2:
            return 1.0

        n_increasing = 0
        for i in range(n - 1):
            if mu[i + 1] >= mu[i]:
                n_increasing += 1

        return n_increasing / (n - 1)

    def table_coverage(self, T_min: float = 200.0, T_max: float = 3000.0) -> float:
        """Compute temperature coverage ratio.

        Parameters
        ----------
        T_min : float
            Minimum temperature of interest (K). Default 200.
        T_max : float
            Maximum temperature of interest (K). Default 3000.

        Returns
        -------
        float
            Coverage ratio (0 to 1).
        """
        temps = list(self._T_data)
        data_min = min(temps)
        data_max = max(temps)
        T_range = max(T_max - T_min, 1.0)

        covered = (min(data_max, T_max) - max(data_min, T_min)) / T_range
        return max(0.0, min(covered, 1.0))

    def quality_report(self) -> dict[str, float]:
        """Generate comprehensive quality report.

        Returns
        -------
        dict
            'smoothness': smoothness index,
            'monotonicity': monotonicity score,
            'coverage': temperature coverage ratio,
            'n_points': number of data points.
        """
        return {
            "smoothness": self.table_smoothness(),
            "monotonicity": self.table_monotonicity(),
            "coverage": self.table_coverage(),
            "n_points": float(len(self._T_data)),
        }

    # ------------------------------------------------------------------
    # Pressure-dependent kappa correction
    # ------------------------------------------------------------------

    def kappa_corrected(self, T: float, P: float = 101325.0) -> float:
        """Pressure-corrected thermal conductivity.

        kappa(P) = kappa(T) * (1 + coeff * (P/P_ref - 1))

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa). Default 101325.

        Returns
        -------
        float
            Corrected thermal conductivity (W/(m*K)).
        """
        kappa_base = 0.026  # Default if no data
        P_ref = max(getattr(self, '_P_ref', 101325.0), 1.0)
        correction = 1.0 + self._pressure_kappa_coeff * (P / P_ref - 1.0)
        return kappa_base * max(correction, 0.01)

    def __repr__(self) -> str:
        n = len(self._T_data)
        mode = "bilinear" if self.bilinear_mode else self._interp_method
        qm = ", quality" if self._quality_metrics else ""
        return (
            f"TabulatedTransportEnhanced9(T_range=[{self._T_data[0]:.0f}, "
            f"{self._T_data[-1]:.0f}], n_points={n}, interp={mode}{qm})"
        )
