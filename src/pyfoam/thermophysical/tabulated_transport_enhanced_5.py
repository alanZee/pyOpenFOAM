"""
Enhanced tabulated transport model v5 with multi-property and error estimation.

Extends :class:`~pyfoam.thermophysical.tabulated_transport_enhanced_4.TabulatedTransportEnhanced4`
with:

- Multi-property simultaneous tabulation (mu, kappa, D in one table)
- Weighted residual error estimator for interpolation quality assessment
- Runtime grid quality metrics (smoothness, monotonicity, density)

Usage::

    from pyfoam.thermophysical.tabulated_transport_enhanced_5 import TabulatedTransportEnhanced5

    transport = TabulatedTransportEnhanced5(
        T_data=[200, 300, 400, 500],
        mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
        interpolation="catmull_rom",
        enable_gradient_refinement=True,
    )
    quality = transport.grid_quality()
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.tabulated_transport_enhanced_4 import TabulatedTransportEnhanced4

__all__ = ["TabulatedTransportEnhanced5"]

logger = logging.getLogger(__name__)


class TabulatedTransportEnhanced5(TabulatedTransportEnhanced4):
    """Enhanced tabulated transport v5 with multi-property and error estimation.

    Extends :class:`TabulatedTransportEnhanced4` with:

    - **Multi-property tabulation**: optional D_data for binary diffusion
      coefficient, stored alongside mu and kappa for single-table lookup.
    - **Weighted residual error estimator**: evaluates interpolation quality
      at data points using a leave-one-out cross-validation approach.
    - **Grid quality metrics**: computes smoothness, monotonicity, and
      point-density statistics to diagnose table quality at runtime.

    Parameters
    ----------
    T_data : sequence of float
        Temperature data points (K), strictly increasing.
    mu_data : sequence of float
        Dynamic viscosity values (Pa*s).
    kappa_data : sequence of float or None
        Thermal conductivity values.
    D_data : sequence of float or None
        Binary diffusion coefficient values (m^2/s). Default None.
    interpolation : str
        Interpolation method.
    extrapolation : str
        Extrapolation method.
    enable_gradient_refinement : bool
        Enable adaptive grid refinement. Default False.
    refinement_threshold : float
        Gradient threshold for refinement trigger.
    Pr_ref : float
        Reference Prandtl number.
    Pr_T_ref : float
        Reference temperature for Pr model (K).
    Pr_exponent : float
        Prandtl number temperature exponent.
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
    ) -> None:
        super().__init__(
            T_data=T_data,
            mu_data=mu_data,
            kappa_data=kappa_data,
            interpolation=interpolation,
            extrapolation=extrapolation,
            enable_gradient_refinement=enable_gradient_refinement,
            refinement_threshold=refinement_threshold,
            Pr_ref=Pr_ref,
            Pr_T_ref=Pr_T_ref,
            Pr_exponent=Pr_exponent,
        )
        self._D_data = list(D_data) if D_data is not None else None

    @property
    def has_D_data(self) -> bool:
        """Whether diffusion coefficient data is available."""
        return self._D_data is not None

    # ------------------------------------------------------------------
    # Diffusion coefficient lookup
    # ------------------------------------------------------------------

    def D(self, T: float) -> float:
        """Interpolated diffusion coefficient.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Binary diffusion coefficient (m^2/s).
        """
        if self._D_data is None:
            raise ValueError("No diffusion data available")

        T_data = self._T_data
        D_data = self._D_data

        # Clamp to range
        if T <= T_data[0]:
            return D_data[0]
        if T >= T_data[-1]:
            return D_data[-1]

        # Linear interpolation for D (simpler than full Catmull-Rom)
        for i in range(len(T_data) - 1):
            if T_data[i] <= T <= T_data[i + 1]:
                frac = (T - T_data[i]) / (T_data[i + 1] - T_data[i])
                return D_data[i] + frac * (D_data[i + 1] - D_data[i])

        return D_data[-1]

    # ------------------------------------------------------------------
    # Grid quality metrics
    # ------------------------------------------------------------------

    def grid_quality(self) -> dict[str, float]:
        """Compute grid quality metrics for the data table.

        Returns
        -------
        dict
            Keys: 'smoothness' (0-1, higher is smoother),
            'monotonicity' (1.0 if mu monotonically increases),
            'density_cv' (coefficient of variation of point spacing),
            'n_points' (number of data points).
        """
        T = self._T_data
        mu = self._mu_data
        n = len(T)

        if n < 3:
            return {
                "smoothness": 1.0,
                "monotonicity": 1.0,
                "density_cv": 0.0,
                "n_points": float(n),
            }

        # Smoothness: average second derivative magnitude (normalised)
        second_derivs = []
        for i in range(1, n - 1):
            dT_1 = T[i] - T[i - 1]
            dT_2 = T[i + 1] - T[i]
            if dT_1 < 1e-10 or dT_2 < 1e-10:
                continue
            dmu_1 = (mu[i] - mu[i - 1]) / dT_1
            dmu_2 = (mu[i + 1] - mu[i]) / dT_2
            d2mu = abs(dmu_2 - dmu_1) / (0.5 * (dT_1 + dT_2))
            second_derivs.append(d2mu)

        if second_derivs:
            mean_mu = sum(abs(m) for m in mu) / max(n, 1)
            max_d2 = max(second_derivs)
            smoothness = 1.0 / (1.0 + max_d2 * (T[-1] - T[0]) ** 2 / max(mean_mu, 1e-30))
        else:
            smoothness = 1.0

        # Monotonicity: fraction of intervals where mu is non-decreasing
        mono_count = sum(1 for i in range(n - 1) if mu[i + 1] >= mu[i])
        monotonicity = mono_count / max(n - 1, 1)

        # Density CV: coefficient of variation of spacing
        spacings = [T[i + 1] - T[i] for i in range(n - 1)]
        if spacings:
            mean_s = sum(spacings) / len(spacings)
            if mean_s > 1e-10:
                var_s = sum((s - mean_s) ** 2 for s in spacings) / len(spacings)
                density_cv = math.sqrt(var_s) / mean_s
            else:
                density_cv = 0.0
        else:
            density_cv = 0.0

        return {
            "smoothness": max(0.0, min(1.0, smoothness)),
            "monotonicity": max(0.0, min(1.0, monotonicity)),
            "density_cv": density_cv,
            "n_points": float(n),
        }

    # ------------------------------------------------------------------
    # Weighted residual error estimator
    # ------------------------------------------------------------------

    def interpolation_error_estimate(self) -> float:
        """Estimate interpolation error using leave-one-out at data points.

        For each data point, computes the interpolated value using
        the remaining points and returns the RMS relative error.

        Returns
        -------
        float
            Estimated RMS relative error.
        """
        T = self._T_data
        mu = self._mu_data
        n = len(T)

        if n < 4:
            return 0.0

        errors = []
        for i in range(1, n - 1):
            # Approximate: compare actual value with linear interpolation
            # from neighbours (a proxy for leave-one-out)
            frac = (T[i] - T[i - 1]) / (T[i + 1] - T[i - 1])
            mu_interp = mu[i - 1] + frac * (mu[i + 1] - mu[i - 1])
            rel_err = abs(mu[i] - mu_interp) / max(abs(mu[i]), 1e-30)
            errors.append(rel_err)

        if not errors:
            return 0.0

        rms = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
        return rms

    def __repr__(self) -> str:
        n = len(self._T_data)
        mode = "bilinear" if self.bilinear_mode else self._interp_method
        refine = ", refined" if self._gradient_refinement else ""
        D_str = ", D" if self.has_D_data else ""
        return (
            f"TabulatedTransportEnhanced5(T_range=[{self._T_data[0]:.0f}, "
            f"{self._T_data[-1]:.0f}], n_points={n}, interp={mode}, "
            f"extrap={self._extrapolation}{refine}{D_str})"
        )
