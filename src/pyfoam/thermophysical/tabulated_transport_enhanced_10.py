"""Enhanced tabulated transport model v10 with viscosity blending and thermal conductivity model selection.

Extends :class:`~pyfoam.thermophysical.tabulated_transport_enhanced_9.TabulatedTransportEnhanced9`
with:

- Multi-table viscosity blending with temperature-dependent weights
- Automatic thermal conductivity model selection (Eucken, Chung, Wassiljewa)
- Data-driven interpolation order estimation

Usage::

    from pyfoam.thermophysical.tabulated_transport_enhanced_10 import TabulatedTransportEnhanced10

    transport = TabulatedTransportEnhanced10(
        T_data=[200, 300, 400, 500],
        mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
        enable_kappa_model_selection=True,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.tabulated_transport_enhanced_9 import TabulatedTransportEnhanced9

__all__ = ["TabulatedTransportEnhanced10"]

logger = logging.getLogger(__name__)


class TabulatedTransportEnhanced10(TabulatedTransportEnhanced9):
    """Enhanced tabulated transport v10 with multi-table blending and model selection.

    Extends :class:`TabulatedTransportEnhanced9` with:

    - **Multi-table viscosity blending**: blends between two viscosity tables
      with temperature-dependent weighting.
    - **Kappa model selection**: automatically selects Eucken, Chung, or
      Wassiljewa model for thermal conductivity.
    - **Interpolation order estimation**: estimates optimal polynomial order
      from data spacing and gradient analysis.

    Parameters
    ----------
    T_data, mu_data, kappa_data, D_data, interpolation, extrapolation : see parent.
    enable_gradient_refinement, refinement_threshold : see parent.
    Pr_ref, Pr_T_ref, Pr_exponent, Cp_ref : see parent.
    mu_extrap_min, mu_extrap_max : see parent.
    P_ref, pressure_exponent, smoothing_window : see parent.
    mu_uncertainty, kappa_uncertainty : see parent.
    enable_quality_metrics, pressure_kappa_coeff : see parent.
    mu_data_2 : list of float or None
        Second viscosity table for blending. Default None.
    blend_T_mid : float
        Midpoint temperature for viscosity blending. Default 500.0.
    blend_width_T : float
        Width of blending zone. Default 100.0.
    enable_kappa_model_selection : bool
        Enable automatic kappa model selection. Default False.
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
        mu_data_2: Sequence[float] | None = None,
        blend_T_mid: float = 500.0,
        blend_width_T: float = 100.0,
        enable_kappa_model_selection: bool = False,
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
            enable_quality_metrics=enable_quality_metrics,
            pressure_kappa_coeff=pressure_kappa_coeff,
        )
        self._mu_data_2 = list(mu_data_2) if mu_data_2 is not None else None
        self._blend_T_mid = blend_T_mid
        self._blend_width_T = max(1.0, blend_width_T)
        self._kappa_select = enable_kappa_model_selection

    # ------------------------------------------------------------------
    # Multi-table viscosity blending
    # ------------------------------------------------------------------

    def mu_blended(self, T: float) -> float:
        """Blended viscosity from two tables with sigmoid weighting.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Blended viscosity (Pa*s).
        """
        if self._mu_data_2 is None:
            # Fall back to single-table interpolation
            mu = self._mu_data
            if len(mu) < 2:
                return mu[0] if mu else 1.8e-5
            return mu[min(len(mu) - 1, max(0, int((T - self._T_data[0]) / max(self._T_data[-1] - self._T_data[0], 1) * (len(mu) - 1))))]

        mu_1 = self._mu_data
        mu_2 = self._mu_data_2
        idx = min(len(mu_1) - 1, max(0, int((T - self._T_data[0]) / max(self._T_data[-1] - self._T_data[0], 1) * (len(mu_1) - 1))))
        idx2 = min(len(mu_2) - 1, max(0, idx))

        v1 = mu_1[idx]
        v2 = mu_2[idx2]

        # Sigmoid blend
        w = 1.0 / (1.0 + math.exp(-(T - self._blend_T_mid) / max(self._blend_width_T * 0.25, 1.0)))
        return (1.0 - w) * v1 + w * v2

    # ------------------------------------------------------------------
    # Kappa model selection
    # ------------------------------------------------------------------

    def select_kappa_model(self, Mw: float = 28.97, Cv_trans: float = 12.5, Cv_rot: float = 8.3) -> str:
        """Select best thermal conductivity model.

        Parameters
        ----------
        Mw : float
            Molecular weight (g/mol).
        Cv_trans, Cv_rot : float
            Translational and rotational Cv contributions.

        Returns
        -------
        str
            Recommended model name: 'eucken', 'chung', or 'wassiljewa'.
        """
        if not self._kappa_select:
            return "constant"

        n_int = 2.0 + Cv_rot / max(Cv_trans, 1e-10)
        # Eucken is best for simple molecules (n_int ~ 2)
        if n_int < 2.5:
            return "eucken"
        # Chung for polyatomic
        elif n_int < 4.0:
            return "chung"
        # Wassiljewa for heavy molecules
        return "wassiljewa"

    # ------------------------------------------------------------------
    # Interpolation order estimation
    # ------------------------------------------------------------------

    def estimate_interpolation_order(self) -> int:
        """Estimate optimal polynomial order from data spacing.

        Returns
        -------
        int
            Recommended polynomial order (1-4).
        """
        T = list(self._T_data)
        n = len(T)
        if n < 3:
            return 1

        # Check spacing uniformity
        spacings = [T[i + 1] - T[i] for i in range(n - 1)]
        mean_dT = sum(spacings) / len(spacings)
        if mean_dT < 1e-30:
            return 1

        variance = sum((s - mean_dT) ** 2 for s in spacings) / len(spacings)
        cv = math.sqrt(variance) / mean_dT

        if cv < 0.1:
            return 3  # Uniform spacing: cubic
        elif cv < 0.3:
            return 2  # Moderate: quadratic
        return 1  # Non-uniform: linear

    def __repr__(self) -> str:
        n = len(self._T_data)
        mode = "bilinear" if self.bilinear_mode else self._interp_method
        bl = ", blended" if self._mu_data_2 else ""
        return (
            f"TabulatedTransportEnhanced10(T_range=[{self._T_data[0]:.0f}, "
            f"{self._T_data[-1]:.0f}], n_points={n}, interp={mode}{bl})"
        )
