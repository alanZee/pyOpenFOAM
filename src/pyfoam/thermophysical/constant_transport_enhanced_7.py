"""
Enhanced constant transport model v7 with multi-region support and runtime viscosity switching.

Extends :class:`~pyfoam.thermophysical.constant_transport_enhanced_6.ConstantTransportEnhanced6`
with:

- Runtime viscosity model switching (constant / polynomial / Sutherland)
- Multi-region transport with zone-dependent properties
- Thermal diffusivity computation

Usage::

    from pyfoam.thermophysical.constant_transport_enhanced_7 import ConstantTransportEnhanced7

    transport = ConstantTransportEnhanced7(
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
from pyfoam.thermophysical.constant_transport_enhanced_6 import ConstantTransportEnhanced6

__all__ = ["ConstantTransportEnhanced7"]

logger = logging.getLogger(__name__)


class ConstantTransportEnhanced7(ConstantTransportEnhanced6):
    """Enhanced constant transport v7 with runtime switching and multi-region support.

    Extends :class:`ConstantTransportEnhanced6` with:

    - **Runtime viscosity switching**: select between constant, polynomial,
      and Sutherland viscosity models at runtime.
    - **Multi-region support**: store per-region viscosity/conductivity
      overrides with zone name indexing.
    - **Thermal diffusivity**: alpha = kappa / (rho * Cp) computation.

    Parameters
    ----------
    mu, kappa, T_ref, correction_model : see parent.
    barus_alpha, P_ref : see parent.
    fv_B, fv_alpha_f, fv_beta_f : see parent.
    pressure_coupling : see parent.
    enable_shear_thinning, ree_yring_tau_star, ree_yring_mu_inf : see parent.
    viscosity_index, VI_T_low, VI_T_high : see parent.
    kappa_model, alpha_k, Mw, Cv_trans, Cv_rot : see parent.
    viscosity_model : str
        Active viscosity model: "constant", "polynomial", or "sutherland".
        Default "constant".
    poly_coeffs : sequence of float or None
        Polynomial viscosity coefficients [a0, a1, a2, ...] such that
        mu(T) = a0 + a1*T + a2*T^2 + ... Default None.
    sutherland_mu_ref, sutherland_T_ref, sutherland_S : float
        Sutherland parameters for "sutherland" viscosity model.
    rho_ref : float
        Reference density for thermal diffusivity (kg/m^3). Default 1.0.
    Cp_ref : float
        Reference specific heat for thermal diffusivity (J/(kg*K)). Default 1005.0.
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
            **kwargs,
        )
        self._viscosity_model = viscosity_model
        self._poly_coeffs = list(poly_coeffs) if poly_coeffs else None
        self._sutherland_mu_ref = sutherland_mu_ref
        self._sutherland_T_ref = max(sutherland_T_ref, 1.0)
        self._sutherland_S = max(sutherland_S, 1.0)
        self._rho_ref = max(rho_ref, 1e-10)
        self._Cp_ref = max(Cp_ref, 1.0)
        self._region_overrides: dict[str, dict[str, float]] = {}

    @property
    def viscosity_model(self) -> str:
        """Active viscosity model name."""
        return self._viscosity_model

    @property
    def region_names(self) -> list[str]:
        """List of registered region names."""
        return list(self._region_overrides.keys())

    # ------------------------------------------------------------------
    # Runtime viscosity switching
    # ------------------------------------------------------------------

    def mu_model(self, T: float) -> float:
        """Viscosity using the selected runtime model.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Viscosity (Pa*s).
        """
        if self._viscosity_model == "polynomial" and self._poly_coeffs:
            mu_val = 0.0
            T_pow = 1.0
            for c in self._poly_coeffs:
                mu_val += c * T_pow
                T_pow *= T
            return max(mu_val, 1e-20)
        elif self._viscosity_model == "sutherland":
            ratio = (T / self._sutherland_T_ref) ** 1.5
            denom = (self._sutherland_T_ref + self._sutherland_S) / (T + self._sutherland_S)
            return self._sutherland_mu_ref * ratio * denom
        else:
            return self._mu

    # ------------------------------------------------------------------
    # Multi-region support
    # ------------------------------------------------------------------

    def add_region(self, name: str, mu: float | None = None, kappa: float | None = None) -> None:
        """Add a region override for viscosity and/or conductivity.

        Parameters
        ----------
        name : str
            Region name.
        mu : float or None
            Region viscosity override. None keeps default.
        kappa : float or None
            Region conductivity override. None keeps default.
        """
        self._region_overrides[name] = {}
        if mu is not None:
            self._region_overrides[name]["mu"] = mu
        if kappa is not None:
            self._region_overrides[name]["kappa"] = kappa

    def mu_region(self, T: float, region: str) -> float:
        """Get viscosity for a specific region.

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
        if region in self._region_overrides and "mu" in self._region_overrides[region]:
            return self._region_overrides[region]["mu"]
        return self.mu_model(T)

    # ------------------------------------------------------------------
    # Thermal diffusivity
    # ------------------------------------------------------------------

    def thermal_diffusivity(self, T: float) -> float:
        """Compute thermal diffusivity.

        alpha = kappa(T) / (rho_ref * Cp_ref)

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Thermal diffusivity (m^2/s).
        """
        k = self.kappa_T(T)
        return k / (self._rho_ref * self._Cp_ref)

    def __repr__(self) -> str:
        return (
            f"ConstantTransportEnhanced7(mu={self._mu}, kappa={self._kappa}, "
            f"T_ref={self._T_ref}, viscosity_model={self._viscosity_model}, "
            f"kappa_model={self._kappa_model}, n_regions={len(self._region_overrides)})"
        )
