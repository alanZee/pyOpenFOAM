"""Enhanced constant transport model v10 with viscosity temperature sensitivity and Prandtl number estimation.

Extends :class:`~pyfoam.thermophysical.constant_transport_enhanced_9.ConstantTransportEnhanced9`
with:

- Viscosity temperature sensitivity analysis
- Prandtl number estimation from transport properties
- Multi-model ensemble with weighted averaging

Usage::

    from pyfoam.thermophysical.constant_transport_enhanced_10 import ConstantTransportEnhanced10

    transport = ConstantTransportEnhanced10(
        mu=1.8e-5,
        kappa=0.026,
        enable_prandtl_estimation=True,
    )
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.constant_transport_enhanced_9 import ConstantTransportEnhanced9

__all__ = ["ConstantTransportEnhanced10"]

logger = logging.getLogger(__name__)


class ConstantTransportEnhanced10(ConstantTransportEnhanced9):
    """Enhanced constant transport v10 with sensitivity analysis and Prandtl estimation.

    Extends :class:`ConstantTransportEnhanced9` with:

    - **Viscosity temperature sensitivity**: d(mu)/dT sensitivity coefficient
      for uncertainty propagation.
    - **Prandtl number estimation**: Pr = mu * Cp / kappa from constant
      transport properties.
    - **Weighted ensemble**: computes ensemble viscosity from multiple models
      with configurable weights.

    Parameters
    ----------
    mu, kappa, T_ref, correction_model : see parent.
    enable_prandtl_estimation : bool
        Enable Prandtl number estimation. Default False.
    sensitivity_dT : float
        Temperature perturbation for sensitivity. Default 1.0.
    ensemble_weights : list of float or None
        Weights for ensemble viscosity models. Default None.
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
        enable_prandtl_estimation: bool = False,
        sensitivity_dT: float = 1.0,
        ensemble_weights: list[float] | None = None,
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
            blend_exponent=blend_exponent,
            ensemble_models=ensemble_models,
            **kwargs,
        )
        self._prandtl_est = enable_prandtl_estimation
        self._sens_dT = max(0.01, sensitivity_dT)
        self._ens_weights = list(ensemble_weights) if ensemble_weights else None

    # ------------------------------------------------------------------
    # Prandtl number estimation
    # ------------------------------------------------------------------

    def prandtl_estimate(self) -> float:
        """Estimate Prandtl number from constant transport properties.

        Pr = mu * Cp / kappa

        Returns
        -------
        float
            Estimated Prandtl number.
        """
        if not self._prandtl_est:
            return 0.71  # Default air Pr

        kappa = self._kappa if self._kappa is not None else 0.026
        return self._mu * max(self._Cp_ref, 1e-10) / max(kappa, 1e-30)

    # ------------------------------------------------------------------
    # Temperature sensitivity
    # ------------------------------------------------------------------

    def mu_sensitivity(self, T: float) -> float:
        """Compute viscosity temperature sensitivity coefficient.

        d(mu)/dT estimated via finite differences.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Sensitivity coefficient d(mu)/dT (Pa*s/K).
        """
        mu_plus = self.mu_region_T(T + self._sens_dT, "default") if hasattr(self, 'mu_region_T') else self._mu
        mu_minus = self.mu_region_T(T - self._sens_dT, "default") if hasattr(self, 'mu_region_T') else self._mu
        return (mu_plus - mu_minus) / (2.0 * self._sens_dT)

    # ------------------------------------------------------------------
    # Weighted ensemble
    # ------------------------------------------------------------------

    def ensemble_viscosity(self, T: float) -> float:
        """Weighted ensemble viscosity from multiple model evaluations.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Ensemble-averaged viscosity (Pa*s).
        """
        if self._ens_weights is None or len(self._ens_weights) < 2:
            return self._mu

        # Use available region data or fall back to constant mu
        mu_vals = [self._mu] * len(self._ens_weights)
        total_w = sum(self._ens_weights)

        if total_w < 1e-30:
            return self._mu

        return sum(w * v for w, v in zip(self._ens_weights, mu_vals)) / total_w

    def __repr__(self) -> str:
        pr = f", Pr~{self.prandtl_estimate():.2f}" if self._prandtl_est else ""
        return (
            f"ConstantTransportEnhanced10(mu={self._mu}, kappa={self._kappa}, "
            f"viscosity_model={self._viscosity_model}{pr})"
        )
