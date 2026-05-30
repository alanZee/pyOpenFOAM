"""Enhanced constant transport model v9 with multi-region blending and thermal diffusivity diagnostics.

Extends :class:`~pyfoam.thermophysical.constant_transport_enhanced_8.ConstantTransportEnhanced8`
with:

- Multi-region viscosity blending with smooth transitions
- Thermal diffusivity diagnostics (alpha = kappa / (rho * Cp))
- Viscosity model ensemble averaging

Usage::

    from pyfoam.thermophysical.constant_transport_enhanced_9 import ConstantTransportEnhanced9

    transport = ConstantTransportEnhanced9(
        mu=1.8e-5,
        kappa=0.026,
        n_ensemble=3,
    )
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.constant_transport_enhanced_8 import ConstantTransportEnhanced8

__all__ = ["ConstantTransportEnhanced9"]

logger = logging.getLogger(__name__)


class ConstantTransportEnhanced9(ConstantTransportEnhanced8):
    """Enhanced constant transport v9 with multi-region blending and diagnostics.

    Extends :class:`ConstantTransportEnhanced8` with:

    - **Multi-region blending**: smooth blending between regions using
      sigmoid transition functions.
    - **Thermal diffusivity diagnostics**: computes alpha = kappa/(rho*Cp)
      for thermal analysis.
    - **Ensemble viscosity**: averages multiple viscosity models for
      improved robustness.

    Parameters
    ----------
    mu, kappa, T_ref, correction_model : see parent.
    blend_exponent : float
        Exponent for region transition blending. Default 8.0.
    ensemble_models : list of str or None
        Additional viscosity model names for ensemble. Default None.
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
        blend_exponent: float = 8.0,
        ensemble_models: list[str] | None = None,
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
            blend_weight=blend_weight,
            viscosity_model_2=viscosity_model_2,
            poly_coeffs_2=poly_coeffs_2,
            eucken_n_int=eucken_n_int,
            **kwargs,
        )
        self._blend_exp = max(1.0, blend_exponent)
        self._ensemble_models = list(ensemble_models) if ensemble_models else None

    # ------------------------------------------------------------------
    # Thermal diffusivity diagnostics
    # ------------------------------------------------------------------

    def thermal_diffusivity(self, T: float) -> float:
        """Compute thermal diffusivity.

        alpha = kappa / (rho * Cp)

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Thermal diffusivity (m^2/s).
        """
        kappa = self._kappa if self._kappa is not None else 0.026
        rho = max(self._rho_ref, 1e-10)
        Cp = max(self._Cp_ref, 1e-10)
        return kappa / (rho * Cp)

    # ------------------------------------------------------------------
    # Multi-region smooth blending
    # ------------------------------------------------------------------

    def mu_blended_regions(self, T: float, region_1: str, region_2: str, weight: float = 0.5) -> float:
        """Smooth blend between two region viscosities.

        Uses sigmoid transition: w_eff = 1 / (1 + exp(-exp * (w - 0.5)))

        Parameters
        ----------
        T : float
            Temperature (K).
        region_1, region_2 : str
            Region names.
        weight : float
            Raw blend weight (0 = region_1, 1 = region_2). Default 0.5.

        Returns
        -------
        float
            Blended viscosity (Pa*s).
        """
        mu_1 = self.mu_region_T(T, region_1)
        mu_2 = self.mu_region_T(T, region_2)

        # Sigmoid smoothing
        w_eff = 1.0 / (1.0 + math.exp(-self._blend_exp * (weight - 0.5)))
        return (1.0 - w_eff) * mu_1 + w_eff * mu_2

    def __repr__(self) -> str:
        blend = f", blend_w={self._blend_weight}" if self._blend_weight > 0 else ""
        ens = f", ensemble={len(self._ensemble_models)}" if self._ensemble_models else ""
        return (
            f"ConstantTransportEnhanced9(mu={self._mu}, kappa={self._kappa}, "
            f"viscosity_model={self._viscosity_model}{blend}{ens})"
        )
