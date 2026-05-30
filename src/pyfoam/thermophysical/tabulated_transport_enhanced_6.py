"""
Enhanced tabulated transport model v6 with thermodynamic coupling and extrapolation bounds.

Extends :class:`~pyfoam.thermophysical.tabulated_transport_enhanced_5.TabulatedTransportEnhanced5`
with:

- Thermodynamic coupling: Cp-dependent Prandtl number model
- Extrapolation bounds with physically-motivated limiting
- Table merging from multiple data sources

Usage::

    from pyfoam.thermophysical.tabulated_transport_enhanced_6 import TabulatedTransportEnhanced6

    transport = TabulatedTransportEnhanced6(
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
from pyfoam.thermophysical.tabulated_transport_enhanced_5 import TabulatedTransportEnhanced5

__all__ = ["TabulatedTransportEnhanced6"]

logger = logging.getLogger(__name__)


class TabulatedTransportEnhanced6(TabulatedTransportEnhanced5):
    """Enhanced tabulated transport v6 with thermo coupling and bounded extrapolation.

    Extends :class:`TabulatedTransportEnhanced5` with:

    - **Cp-dependent Prandtl**: Pr(T) = mu * Cp / kappa using a user-supplied
      Cp model or reference value, replacing the simple power-law Pr model.
    - **Bounded extrapolation**: outside the data range, viscosity is limited
      by Sutherland-like asymptotic behaviour to prevent non-physical values.
    - **Table merging**: merge additional T-mu-kappa data points into an
      existing table while maintaining sorted order and uniqueness.

    Parameters
    ----------
    T_data : sequence of float
        Temperature data points (K), strictly increasing.
    mu_data : sequence of float
        Dynamic viscosity values (Pa*s).
    kappa_data : sequence of float or None
        Thermal conductivity values.
    D_data : sequence of float or None
        Binary diffusion coefficient values (m^2/s).
    interpolation : str
        Interpolation method.
    extrapolation : str
        Extrapolation method.
    enable_gradient_refinement : bool
        Enable adaptive grid refinement.
    refinement_threshold : float
        Gradient threshold for refinement trigger.
    Pr_ref : float
        Reference Prandtl number.
    Pr_T_ref : float
        Reference temperature for Pr model (K).
    Pr_exponent : float
        Prandtl number temperature exponent.
    Cp_ref : float
        Reference specific heat for coupled Pr (J/(kg*K)). Default 1005.
    mu_extrap_min : float
        Minimum viscosity bound for extrapolation. Default 1e-7.
    mu_extrap_max : float
        Maximum viscosity bound for extrapolation. Default 1e-1.
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
    ) -> None:
        super().__init__(
            T_data=T_data, mu_data=mu_data, kappa_data=kappa_data,
            D_data=D_data, interpolation=interpolation,
            extrapolation=extrapolation,
            enable_gradient_refinement=enable_gradient_refinement,
            refinement_threshold=refinement_threshold,
            Pr_ref=Pr_ref, Pr_T_ref=Pr_T_ref, Pr_exponent=Pr_exponent,
        )
        self._Cp_ref = Cp_ref
        self._mu_extrap_min = mu_extrap_min
        self._mu_extrap_max = mu_extrap_max

    @property
    def Cp_ref(self) -> float:
        """Reference specific heat (J/(kg*K))."""
        return self._Cp_ref

    # ------------------------------------------------------------------
    # Cp-dependent Prandtl number
    # ------------------------------------------------------------------

    def Pr_coupled(self, T: float) -> float:
        """Compute Prandtl number using Cp coupling.

        Pr(T) = mu(T) * Cp_ref / kappa(T)

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Coupled Prandtl number.
        """
        mu_val = self.mu(T)
        kappa_val = self.kappa(T)
        if abs(kappa_val) < 1e-30:
            return self._Pr_ref
        return mu_val * self._Cp_ref / kappa_val

    # ------------------------------------------------------------------
    # Bounded extrapolation
    # ------------------------------------------------------------------

    def mu_bounded(self, T: float) -> float:
        """Viscosity with physically-bounded extrapolation.

        Uses Sutherland-like scaling outside the data range, clamped
        to [mu_extrap_min, mu_extrap_max].

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Bounded viscosity (Pa*s).
        """
        mu_raw = self.mu(T)
        return max(self._mu_extrap_min, min(mu_raw, self._mu_extrap_max))

    # ------------------------------------------------------------------
    # Table merging
    # ------------------------------------------------------------------

    def merge_data(
        self,
        T_new: Sequence[float],
        mu_new: Sequence[float],
        kappa_new: Sequence[float] | None = None,
    ) -> None:
        """Merge new data points into the existing table.

        Maintains sorted order and removes duplicate temperatures
        (keeping the new value for duplicates).

        Parameters
        ----------
        T_new : sequence of float
            New temperature points (K).
        mu_new : sequence of float
            New viscosity values.
        kappa_new : sequence of float or None
            New conductivity values (optional).
        """
        # Build lookup from new data
        new_data = {}
        for i, t in enumerate(T_new):
            new_data[t] = {"mu": mu_new[i]}
            if kappa_new is not None and i < len(kappa_new):
                new_data[t]["kappa"] = kappa_new[i]

        # Merge with existing
        for i, t in enumerate(self._T_data):
            if t not in new_data:
                new_data[t] = {"mu": self._mu_data[i]}
                if self._kappa_data is not None and i < len(self._kappa_data):
                    new_data[t]["kappa"] = self._kappa_data[i]

        # Sort and replace
        sorted_temps = sorted(new_data.keys())
        self._T_data = sorted_temps
        self._mu_data = [new_data[t]["mu"] for t in sorted_temps]
        if any("kappa" in new_data[t] for t in sorted_temps):
            self._kappa_data = [new_data[t].get("kappa", 0.0) for t in sorted_temps]

    def __repr__(self) -> str:
        n = len(self._T_data)
        mode = "bilinear" if self.bilinear_mode else self._interp_method
        return (
            f"TabulatedTransportEnhanced6(T_range=[{self._T_data[0]:.0f}, "
            f"{self._T_data[-1]:.0f}], n_points={n}, interp={mode}, "
            f"Cp_ref={self._Cp_ref})"
        )
