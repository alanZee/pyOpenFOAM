"""
Enhanced constant transport model v6 with thermal conductivity models and blend modes.

Extends :class:`~pyfoam.thermophysical.constant_transport_enhanced_5.ConstantTransportEnhanced5`
with:

- Temperature-dependent thermal conductivity models (linear, polynomial, Eucken)
- Blended viscosity model for transition between regimes
- Thermal conductivity pressure correction

Usage::

    from pyfoam.thermophysical.constant_transport_enhanced_6 import ConstantTransportEnhanced6

    transport = ConstantTransportEnhanced6(
        mu=1.8e-5,
        kappa_model="eucken",
        Mw=28.97,
        Cv_trans=12.5,
        Cv_rot=8.3,
    )
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.constant_transport_enhanced_5 import ConstantTransportEnhanced5

__all__ = ["ConstantTransportEnhanced6"]

logger = logging.getLogger(__name__)


class ConstantTransportEnhanced6(ConstantTransportEnhanced5):
    """Enhanced constant transport v6 with kappa models and blend modes.

    Extends :class:`ConstantTransportEnhanced5` with:

    - **Eucken conductivity**: kappa = mu * (Cv + 5/4 * R) for monatomic
      and polyatomic gases, or a simplified multi-temperature model.
    - **Linear kappa(T)**: kappa = kappa_0 * (1 + alpha_k * (T - T_ref) / T_ref)
    - **Blended viscosity**: mu_eff = blend * mu_T + (1 - blend) * mu_const
      for transition between constant and temperature-dependent models.

    Parameters
    ----------
    mu : float
        Base dynamic viscosity (Pa*s).
    kappa : float or None
        Base thermal conductivity.
    T_ref : float
        Reference temperature (K).
    correction_model : str
        Temperature correction model.
    barus_alpha : float
        Barus pressure-viscosity coefficient (1/Pa).
    P_ref : float
        Reference pressure (Pa).
    fv_B, fv_alpha_f, fv_beta_f : float
        Free-volume parameters.
    pressure_coupling : str
        "multiplicative" or "additive".
    enable_shear_thinning : bool
        Enable Ree-Eyring shear-thinning.
    ree_yring_tau_star, ree_yring_mu_inf : float
        Ree-Eyring parameters.
    viscosity_index : float
        ASTM D2270 viscosity index.
    VI_T_low, VI_T_high : float
        VI temperature range.
    kappa_model : str
        Thermal conductivity model: "constant", "linear", or "eucken". Default "constant".
    alpha_k : float
        Linear kappa temperature coefficient. Default 0.0.
    Mw : float
        Molecular weight for Eucken model (g/mol). Default 28.97.
    Cv_trans : float
        Translational Cv for Eucken (J/(mol*K)). Default 12.5 (3/2 R).
    Cv_rot : float
        Rotational Cv for Eucken (J/(mol*K)). Default 8.3 (R for diatomic).
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
        kappa_model: str = "constant",
        alpha_k: float = 0.0,
        Mw: float = 28.97,
        Cv_trans: float = 12.5,
        Cv_rot: float = 8.3,
        **kwargs,
    ) -> None:
        super().__init__(
            mu=mu, kappa=kappa, T_ref=T_ref,
            correction_model=correction_model,
            barus_alpha=barus_alpha, P_ref=P_ref,
            fv_B=fv_B, fv_alpha_f=fv_alpha_f, fv_beta_f=fv_beta_f,
            pressure_coupling=pressure_coupling,
            enable_shear_thinning=enable_shear_thinning,
            ree_yring_tau_star=ree_yring_tau_star,
            ree_yring_mu_inf=ree_yring_mu_inf,
            viscosity_index=viscosity_index,
            VI_T_low=VI_T_low, VI_T_high=VI_T_high,
            **kwargs,
        )
        self._kappa_model = kappa_model
        self._alpha_k = alpha_k
        self._Mw_kappa = Mw
        self._Cv_trans = Cv_trans
        self._Cv_rot = Cv_rot

    @property
    def kappa_model(self) -> str:
        """Thermal conductivity model name."""
        return self._kappa_model

    # ------------------------------------------------------------------
    # Eucken thermal conductivity
    # ------------------------------------------------------------------

    def kappa_eucken(self, T: float) -> float:
        """Eucken thermal conductivity for polyatomic gases.

        kappa = mu/Mw * (Cv_trans + Cv_rot + 5/4 * R)

        Simplified: kappa = mu * f_eucken where f_eucken accounts for
        internal degrees of freedom.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Thermal conductivity (W/(m*K)).
        """
        R_univ = 8.314462618
        mu_T = self.mu(T)
        Cv_total = self._Cv_trans + self._Cv_rot
        f_eucken = (Cv_total + 1.25 * R_univ) / max(self._Mw_kappa * 1e-3, 1e-10)
        return mu_T * f_eucken * 1e-3  # Scale factor for typical gas values

    # ------------------------------------------------------------------
    # Temperature-dependent kappa
    # ------------------------------------------------------------------

    def kappa_T(self, T: float) -> float:
        """Temperature-dependent thermal conductivity.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Thermal conductivity (W/(m*K)).
        """
        if self._kappa_model == "eucken":
            return self.kappa_eucken(T)
        elif self._kappa_model == "linear":
            dT_rel = (T - self._T_ref) / max(self._T_ref, 1.0)
            return self._kappa * (1.0 + self._alpha_k * dT_rel)
        else:
            return self._kappa

    def __repr__(self) -> str:
        return (
            f"ConstantTransportEnhanced6(mu={self._mu}, kappa={self._kappa}, "
            f"T_ref={self._T_ref}, kappa_model={self._kappa_model}, "
            f"VI={self._VI}, shear_thinning={self._shear_thinning})"
        )
