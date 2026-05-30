"""
Enhanced constant transport model v5 with Ree-Eyring and viscosity index.

Extends :class:`~pyfoam.thermophysical.constant_transport_enhanced_4.ConstantTransportEnhanced4`
with:

- Ree-Eyring non-Newtonian shear-thinning model
- Viscosity index (VI) correction for lubricant applications
- Combined T+P+shear-rate multiphysics coupling

Usage::

    from pyfoam.thermophysical.constant_transport_enhanced_5 import ConstantTransportEnhanced5

    transport = ConstantTransportEnhanced5(
        mu=0.1,
        enable_shear_thinning=True,
        ree_yring_tau_star=1e4,
        ree_yring_mu_inf=0.01,
    )
    mu = transport.mu_sheared(T=300.0, P=1e7, shear_rate=1e5)
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.constant_transport_enhanced_4 import ConstantTransportEnhanced4

__all__ = ["ConstantTransportEnhanced5"]

logger = logging.getLogger(__name__)


class ConstantTransportEnhanced5(ConstantTransportEnhanced4):
    """Enhanced constant transport v5 with Ree-Eyring and viscosity index.

    Extends :class:`ConstantTransportEnhanced4` with:

    - **Ree-Eyring model**: mu(gamma_dot) = mu_inf + (mu_0 - mu_inf)
      * arcsinh(tau_star * gamma_dot / tau_star) / (tau_star * gamma_dot)
      Captures shear-thinning behaviour for polymer solutions and lubricants.
    - **Viscosity index (VI) correction**: mu(T) adjusted by the ASTM D2270
      viscosity index to account for temperature sensitivity differences
      between lubricant grades.
    - **Coupled T+P+shear**: multiplicative combination of all corrections.

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
        Barus pressure-viscosity coefficient (1/Pa).
    P_ref : float
        Reference pressure (Pa).
    fv_B : float
        Free-volume B parameter.
    fv_alpha_f : float
        Free-volume thermal expansion coefficient (1/K).
    fv_beta_f : float
        Free-volume pressure contraction coefficient (1/Pa).
    pressure_coupling : str
        "multiplicative" or "additive".
    enable_shear_thinning : bool
        Enable Ree-Eyring shear-thinning. Default False.
    ree_yring_tau_star : float
        Ree-Eyring characteristic shear stress (Pa). Default 1e4.
    ree_yring_mu_inf : float
        Infinite-shear-rate viscosity (Pa*s). Default 0.01.
    viscosity_index : float
        ASTM D2270 viscosity index. Default 100.
    VI_T_low : float
        Low temperature for VI calculation (K). Default 313.15.
    VI_T_high : float
        High temperature for VI calculation (K). Default 373.15.
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
        enable_shear_thinning: bool = False,
        ree_yring_tau_star: float = 1e4,
        ree_yring_mu_inf: float = 0.01,
        viscosity_index: float = 100.0,
        VI_T_low: float = 313.15,
        VI_T_high: float = 373.15,
        **kwargs,
    ) -> None:
        super().__init__(
            mu=mu, kappa=kappa, T_ref=T_ref,
            correction_model=correction_model,
            barus_alpha=barus_alpha, P_ref=P_ref,
            fv_B=fv_B, fv_alpha_f=fv_alpha_f, fv_beta_f=fv_beta_f,
            pressure_coupling=pressure_coupling,
            **kwargs,
        )
        self._shear_thinning = enable_shear_thinning
        self._tau_star = ree_yring_tau_star
        self._mu_inf = ree_yring_mu_inf
        self._VI = viscosity_index
        self._VI_T_low = VI_T_low
        self._VI_T_high = VI_T_high

    @property
    def shear_thinning_enabled(self) -> bool:
        """Whether Ree-Eyring shear-thinning is active."""
        return self._shear_thinning

    @property
    def viscosity_index(self) -> float:
        """ASTM D2270 viscosity index."""
        return self._VI

    # ------------------------------------------------------------------
    # Ree-Eyring shear-thinning
    # ------------------------------------------------------------------

    def _ree_eyring_factor(self, shear_rate: float) -> float:
        """Ree-Eyring shear-thinning correction factor.

        factor = arcsinh(tau_star * gamma_dot / tau_star) / (tau_star * gamma_dot / tau_star)

        For small gamma_dot: factor -> 1.0 (Newtonian)
        For large gamma_dot: factor -> 0 (mu_inf-dominated)

        Parameters
        ----------
        shear_rate : float
            Shear rate (1/s).

        Returns
        -------
        float
            Ree-Eyring correction factor.
        """
        if not self._shear_thinning or abs(shear_rate) < 1e-30:
            return 1.0

        x = self._tau_star * shear_rate / max(self._tau_star, 1e-10)
        # arcsinh(x) / x, with limiting value 1 at x -> 0
        if abs(x) < 1e-6:
            return 1.0

        return math.asinh(x) / x

    # ------------------------------------------------------------------
    # Viscosity index correction
    # ------------------------------------------------------------------

    def _vi_correction(self, T: float) -> float:
        """Viscosity index temperature correction.

        Higher VI means less temperature sensitivity.
        Approximated as: factor = 1 + (VI - 100) * (T_ref - T) / (VI_scale * T_ref)

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            VI correction factor.
        """
        VI_scale = 10000.0  # Empirical scaling
        deviation = (self._VI - 100.0) * (self._T_ref - T) / (VI_scale * self._T_ref)
        factor = 1.0 + deviation
        return max(factor, 0.1)

    # ------------------------------------------------------------------
    # Combined shear + T + P viscosity
    # ------------------------------------------------------------------

    def mu_sheared(
        self,
        T: float,
        P: float = 101325.0,
        shear_rate: float = 0.0,
        pressure_model: str = "barus",
    ) -> float:
        """Compute viscosity with T + P + shear corrections.

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).
        shear_rate : float
            Shear rate (1/s). Default 0 (Newtonian).
        pressure_model : str
            Pressure model: "barus" or "free_volume".

        Returns
        -------
        float
            Fully corrected viscosity (Pa*s).
        """
        mu_tp = self.mu_P(T=T, P=P, model=pressure_model)
        shear_factor = self._ree_eyring_factor(shear_rate)

        if self._shear_thinning:
            # Blend between Newtonian and infinite-shear
            mu = self._mu_inf + (mu_tp - self._mu_inf) * shear_factor
        else:
            mu = mu_tp

        # Apply VI correction
        vi = self._vi_correction(T)
        return mu * vi

    def __repr__(self) -> str:
        return (
            f"ConstantTransportEnhanced5(mu={self._mu}, kappa={self._kappa}, "
            f"T_ref={self._T_ref}, model={self._correction_model}, "
            f"VI={self._VI}, shear_thinning={self._shear_thinning})"
        )
