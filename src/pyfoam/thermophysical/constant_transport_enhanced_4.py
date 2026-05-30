"""
Enhanced constant transport model v4 with pressure-dependent corrections.

Extends :class:`~pyfoam.thermophysical.constant_transport_enhanced_3.ConstantTransportEnhanced3`
with:

- Barus pressure-viscosity model: mu(P) = mu_0 * exp(alpha_p * P)
- Free-volume viscosity model for polymer melts
- Combined T+P correction with selectable coupling strategy

Usage::

    from pyfoam.thermophysical.constant_transport_enhanced_4 import ConstantTransportEnhanced4

    transport = ConstantTransportEnhanced4(
        mu=0.1,
        T_ref=300.0,
        correction_model="barus",
        barus_alpha=1e-8,
    )
    mu = transport.mu_P(T=300.0, P=1e7)
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.constant_transport_enhanced_3 import ConstantTransportEnhanced3

__all__ = ["ConstantTransportEnhanced4"]

logger = logging.getLogger(__name__)


class ConstantTransportEnhanced4(ConstantTransportEnhanced3):
    """Enhanced constant transport v4 with pressure and free-volume models.

    Extends :class:`ConstantTransportEnhanced3` with:

    - **Barus model**: mu(P) = mu_0 * exp(alpha_p * P)
      Captures exponential increase of lubricant viscosity with pressure.
    - **Free-volume model**: mu(T, P) = mu_0 * exp(B * V_inf / (V_free(T, P)))
      where V_free = V_ref * (alpha_f*(T-T_ref) - beta_f*(P-P_ref)).
    - **Combined T+P coupling**: mu_total = mu_T(T) * mu_P(P) (multiplicative)
      or mu_total = mu_T(T) + mu_P(P) - mu_0 (additive).

    Parameters
    ----------
    mu : float
        Base dynamic viscosity (Pa*s).
    kappa : float or None
        Base thermal conductivity.
    T_ref : float
        Reference temperature (K). Default 300.0.
    correction_model : str
        Temperature correction model.
    barus_alpha : float
        Barus pressure-viscosity coefficient (1/Pa). Default 1e-8.
    P_ref : float
        Reference pressure (Pa). Default 101325.0.
    fv_B : float
        Free-volume B parameter. Default 1.0.
    fv_alpha_f : float
        Free-volume thermal expansion coefficient (1/K). Default 1e-3.
    fv_beta_f : float
        Free-volume pressure contraction coefficient (1/Pa). Default 1e-9.
    pressure_coupling : str
        "multiplicative" or "additive". Default "multiplicative".
    """

    def __init__(
        self,
        mu: float = 1.8e-5,
        kappa: float | None = None,
        T_ref: float = 300.0,
        correction_model: str = "polynomial",
        barus_alpha: float = 1e-8,
        P_ref: float = 101325.0,
        fv_B: float = 1.0,
        fv_alpha_f: float = 1e-3,
        fv_beta_f: float = 1e-9,
        pressure_coupling: str = "multiplicative",
        **kwargs,
    ) -> None:
        super().__init__(mu=mu, kappa=kappa, T_ref=T_ref, correction_model=correction_model, **kwargs)
        self._barus_alpha = barus_alpha
        self._P_ref = P_ref
        self._fv_B = fv_B
        self._fv_alpha_f = fv_alpha_f
        self._fv_beta_f = fv_beta_f
        self._pressure_coupling = pressure_coupling

    @property
    def barus_alpha(self) -> float:
        """Barus pressure-viscosity coefficient (1/Pa)."""
        return self._barus_alpha

    @property
    def P_ref(self) -> float:
        """Reference pressure (Pa)."""
        return self._P_ref

    # ------------------------------------------------------------------
    # Barus pressure model
    # ------------------------------------------------------------------

    def _barus_factor(self, P: float) -> float:
        """Barus pressure correction factor.

        factor = exp(alpha_p * P)

        Parameters
        ----------
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Barus correction factor.
        """
        exponent = self._barus_alpha * P
        exponent = max(min(exponent, 50.0), -50.0)
        return math.exp(exponent)

    # ------------------------------------------------------------------
    # Free-volume model
    # ------------------------------------------------------------------

    def _free_volume_factor(self, T: float, P: float) -> float:
        """Free-volume viscosity correction factor.

        V_free = alpha_f * (T - T_ref) - beta_f * (P - P_ref)
        factor = exp(B / max(V_free, V_min))

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Free-volume correction factor.
        """
        dT = T - self._T_ref
        dP = P - self._P_ref
        V_free = self._fv_alpha_f * dT - self._fv_beta_f * dP
        V_free = max(V_free, 1e-10)
        exponent = self._fv_B / V_free
        exponent = max(min(exponent, 50.0), -50.0)
        return math.exp(exponent)

    # ------------------------------------------------------------------
    # Pressure-dependent viscosity
    # ------------------------------------------------------------------

    def mu_P(
        self,
        T: float,
        P: float,
        model: str = "barus",
    ) -> float:
        """Compute viscosity with T and P corrections.

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).
        model : str
            Pressure model: "barus" or "free_volume".

        Returns
        -------
        float
            Pressure-corrected viscosity (Pa*s).
        """
        device = get_device()
        dtype = get_default_dtype()

        T_t = torch.tensor(T, dtype=dtype, device=device)
        mu_T = float(self.mu(T_t).item())

        if model == "barus":
            mu_P_factor = self._barus_factor(P)
        elif model == "free_volume":
            mu_P_factor = self._free_volume_factor(T, P)
        else:
            raise ValueError(f"Unknown pressure model '{model}'")

        if self._pressure_coupling == "multiplicative":
            return mu_T * mu_P_factor
        else:
            # Additive: mu_total = mu_T + (mu_P_factor - 1) * mu_base
            return mu_T + (mu_P_factor - 1.0) * self._mu

    def __repr__(self) -> str:
        return (
            f"ConstantTransportEnhanced4(mu={self._mu}, kappa={self._kappa}, "
            f"T_ref={self._T_ref}, model={self._correction_model}, "
            f"barus_alpha={self._barus_alpha})"
        )
