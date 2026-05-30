"""
Enhanced constant transport model v8 with temperature-dependent regions and viscosity blending.

Extends :class:`~pyfoam.thermophysical.constant_transport_enhanced_7.ConstantTransportEnhanced7`
with:

- Temperature-dependent region overrides with interpolation
- Viscosity blending between multiple models
- Enhanced Eucken thermal conductivity for polyatomic gases

Usage::

    from pyfoam.thermophysical.constant_transport_enhanced_8 import ConstantTransportEnhanced8

    transport = ConstantTransportEnhanced8(
        mu=1.8e-5,
        kappa=0.026,
        viscosity_model="polynomial",
        poly_coeffs=[1.0e-5, 2.0e-8, -1.0e-11],
    )
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.constant_transport_enhanced_7 import ConstantTransportEnhanced7

__all__ = ["ConstantTransportEnhanced8"]

logger = logging.getLogger(__name__)


class ConstantTransportEnhanced8(ConstantTransportEnhanced7):
    """Enhanced constant transport v8 with T-dependent regions and viscosity blending.

    Extends :class:`ConstantTransportEnhanced7` with:

    - **T-dependent region overrides**: regions now support temperature-dependent
      viscosity with linear interpolation between reference points.
    - **Viscosity blending**: blends two viscosity models with a weight function.
    - **Enhanced Eucken**: improved Eucken correction for polyatomic gas
      thermal conductivity using internal degrees of freedom.

    Parameters
    ----------
    mu, kappa, T_ref, correction_model : see parent.
    barus_alpha, P_ref : see parent.
    fv_B, fv_alpha_f, fv_beta_f : see parent.
    pressure_coupling : see parent.
    enable_shear_thinning, ree_yring_tau_star, ree_yring_mu_inf : see parent.
    viscosity_index, VI_T_low, VI_T_high : see parent.
    kappa_model, alpha_k, Mw, Cv_trans, Cv_rot : see parent.
    viscosity_model, poly_coeffs : see parent.
    sutherland_mu_ref, sutherland_T_ref, sutherland_S : see parent.
    rho_ref, Cp_ref : see parent.
    blend_weight : float
        Blending weight between primary and secondary viscosity model (0-1). Default 0.0.
    viscosity_model_2 : str
        Secondary viscosity model for blending. Default "constant".
    poly_coeffs_2 : list of float or None
        Polynomial coefficients for the secondary model. Default None.
    eucken_n_int : float
        Internal degrees of freedom for enhanced Eucken. Default 2.0.
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
        viscosity_model: str = "constant",
        poly_coeffs: list[float] | None = None,
        sutherland_mu_ref: float = 1.716e-5,
        sutherland_T_ref: float = 273.15,
        sutherland_S: float = 110.4,
        rho_ref: float = 1.0,
        Cp_ref: float = 1005.0,
        blend_weight: float = 0.0,
        viscosity_model_2: str = "constant",
        poly_coeffs_2: list[float] | None = None,
        eucken_n_int: float = 2.0,
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
            kappa_model=kappa_model, alpha_k=alpha_k,
            Mw=Mw, Cv_trans=Cv_trans, Cv_rot=Cv_rot,
            viscosity_model=viscosity_model,
            poly_coeffs=poly_coeffs,
            sutherland_mu_ref=sutherland_mu_ref,
            sutherland_T_ref=sutherland_T_ref,
            sutherland_S=sutherland_S,
            rho_ref=rho_ref, Cp_ref=Cp_ref,
            **kwargs,
        )
        self._blend_weight = max(0.0, min(blend_weight, 1.0))
        self._visc_model_2 = viscosity_model_2
        self._poly_coeffs_2 = list(poly_coeffs_2) if poly_coeffs_2 else None
        self._eucken_n_int = max(0.0, eucken_n_int)
        self._region_T_data: dict[str, dict[str, list[tuple[float, float]]]] = {}

    @property
    def blend_weight(self) -> float:
        """Blending weight between primary and secondary viscosity models."""
        return self._blend_weight

    # ------------------------------------------------------------------
    # Viscosity blending
    # ------------------------------------------------------------------

    def _mu_secondary(self, T: float) -> float:
        """Secondary viscosity model evaluation."""
        if self._visc_model_2 == "polynomial" and self._poly_coeffs_2:
            mu_val = 0.0
            T_pow = 1.0
            for c in self._poly_coeffs_2:
                mu_val += c * T_pow
                T_pow *= T
            return max(mu_val, 1e-20)
        elif self._visc_model_2 == "sutherland":
            ratio = (T / self._sutherland_T_ref) ** 1.5
            denom = (self._sutherland_T_ref + self._sutherland_S) / (T + self._sutherland_S)
            return self._sutherland_mu_ref * ratio * denom
        else:
            return self._mu

    def mu_blended(self, T: float) -> float:
        """Blended viscosity from primary and secondary models.

        mu_blend = (1 - w) * mu_primary + w * mu_secondary

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Blended viscosity (Pa*s).
        """
        mu_p = self.mu_model(T)
        mu_s = self._mu_secondary(T)
        w = self._blend_weight
        return (1.0 - w) * mu_p + w * mu_s

    # ------------------------------------------------------------------
    # T-dependent region overrides
    # ------------------------------------------------------------------

    def add_region_T_data(
        self,
        name: str,
        T_points: list[float],
        mu_points: list[float],
    ) -> None:
        """Add temperature-dependent viscosity data for a region.

        Parameters
        ----------
        name : str
            Region name.
        T_points : list of float
            Temperature points (K).
        mu_points : list of float
            Viscosity values (Pa*s) at each temperature.
        """
        if name not in self._region_T_data:
            self._region_T_data[name] = {"mu": []}
        paired = sorted(zip(T_points, mu_points), key=lambda x: x[0])
        self._region_T_data[name]["mu"] = paired

    def mu_region_T(self, T: float, region: str) -> float:
        """Get viscosity for a region with T-dependent interpolation.

        Parameters
        ----------
        T : float
            Temperature (K).
        region : str
            Region name.

        Returns
        -------
        float
            Viscosity (Pa*s).
        """
        if region in self._region_T_data and "mu" in self._region_T_data[region]:
            data = self._region_T_data[region]["mu"]
            if len(data) >= 2:
                if T <= data[0][0]:
                    return data[0][1]
                if T >= data[-1][0]:
                    return data[-1][1]
                for i in range(len(data) - 1):
                    if data[i][0] <= T <= data[i + 1][0]:
                        frac = (T - data[i][0]) / max(data[i + 1][0] - data[i][0], 1e-30)
                        return data[i][1] + frac * (data[i + 1][1] - data[i][1])
            elif len(data) == 1:
                return data[0][1]
        return self.mu_region(T, region)

    # ------------------------------------------------------------------
    # Enhanced Eucken conductivity
    # ------------------------------------------------------------------

    def eucken_kappa_enhanced(self, T: float) -> float:
        """Enhanced Eucken thermal conductivity for polyatomic gases.

        kappa = mu * (Cv_trans * 15/4 + Cv_rot + 1) / Mw

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Thermal conductivity (W/(m*K)).
        """
        Cv_trans = self._Cv_trans
        Cv_rot = self._Cv_rot
        Mw = max(self._Mw * 1e-3, 1e-10)  # g/mol -> kg/mol
        mu = self.mu_model(T)

        # Eucken with internal degrees of freedom
        f_eucken = (Cv_trans * 15.0 / 4.0 + Cv_rot * (1.0 + self._eucken_n_int / 3.0)) / max(Mw, 1e-10)
        return mu * max(f_eucken, 0.0)

    def __repr__(self) -> str:
        blend = f", blend_w={self._blend_weight}" if self._blend_weight > 0 else ""
        return (
            f"ConstantTransportEnhanced8(mu={self._mu}, kappa={self._kappa}, "
            f"viscosity_model={self._viscosity_model}{blend}, "
            f"n_regions={len(self._region_overrides)})"
        )
