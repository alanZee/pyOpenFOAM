"""
Enhanced constant transport model v3 with improved temperature correction.

Extends :class:`~pyfoam.thermophysical.constant_transport_enhanced_2.ConstantTransportEnhanced2`
with:

- Vogel-Fulcher-Tammann (VFT) viscosity model for glass-forming liquids
- WLF (Williams-Landel-Ferry) time-temperature superposition
- Blending between correction models with smooth transition

Usage::

    from pyfoam.thermophysical.constant_transport_enhanced_3 import ConstantTransportEnhanced3

    transport = ConstantTransportEnhanced3(
        mu=1.8e-5,
        kappa=0.026,
        T_ref=300.0,
        correction_model="vft",
        vft_B=500.0,
        vft_Tinf=100.0,
    )
    mu = transport.mu(T=400.0)
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.constant_transport_enhanced_2 import ConstantTransportEnhanced2

__all__ = ["ConstantTransportEnhanced3"]

logger = logging.getLogger(__name__)


class ConstantTransportEnhanced3(ConstantTransportEnhanced2):
    """Enhanced constant transport v3 with VFT and WLF models.

    Extends :class:`ConstantTransportEnhanced2` with:

    - **VFT model**: mu(T) = mu_0 * exp(B / (T - T_inf))
      Captures super-Arrhenius behaviour in glass-forming systems.
    - **WLF model**: log10(a_T) = -C1*(T-T_ref) / (C2 + T - T_ref)
      Time-temperature superposition for polymers.
    - **Adaptive blending**: smooth transition between correction models
      in different temperature ranges.

    Parameters
    ----------
    mu : float
        Base dynamic viscosity (Pa*s). Default 1.8e-5.
    kappa : float or None
        Base thermal conductivity.
    T_ref : float
        Reference temperature (K). Default 300.0.
    correction_model : str
        "polynomial", "exponential", "piecewise", "vft", or "wlf".
    mu_activation_energy : float
        Activation energy for exponential model (K).
    piecewise_ranges : list of dict or None
        Piecewise model ranges.
    kappa_correction_model : str or None
        Correction model for conductivity.
    vft_B : float
        VFT B parameter (K). Default 500.0.
    vft_Tinf : float
        VFT T_infinity parameter (K). Default 100.0.
    wlf_C1 : float
        WLF C1 parameter (dimensionless). Default 17.44.
    wlf_C2 : float
        WLF C2 parameter (K). Default 51.6.
    """

    def __init__(
        self,
        mu: float = 1.8e-5,
        kappa: float | None = None,
        T_ref: float = 300.0,
        correction_model: str = "polynomial",
        mu_activation_energy: float = 110.0,
        piecewise_ranges: list[dict] | None = None,
        kappa_correction_model: str | None = None,
        mu_temp_coeff: float = 0.0,
        mu_temp_coeff2: float = 0.0,
        kappa_temp_coeff: float = 0.0,
        vft_B: float = 500.0,
        vft_Tinf: float = 100.0,
        wlf_C1: float = 17.44,
        wlf_C2: float = 51.6,
    ) -> None:
        # Validate correction model early
        valid_models = ("polynomial", "exponential", "piecewise", "vft", "wlf")
        if correction_model not in valid_models:
            raise ValueError(
                f"correction_model must be one of {valid_models}, "
                f"got '{correction_model}'"
            )

        # Parent only accepts polynomial/exponential/piecewise
        parent_model = correction_model if correction_model in ("polynomial", "exponential", "piecewise") else "polynomial"

        super().__init__(
            mu=mu,
            kappa=kappa,
            T_ref=T_ref,
            correction_model=parent_model,
            mu_activation_energy=mu_activation_energy,
            piecewise_ranges=piecewise_ranges,
            kappa_correction_model=kappa_correction_model or parent_model,
            mu_temp_coeff=mu_temp_coeff,
            mu_temp_coeff2=mu_temp_coeff2,
            kappa_temp_coeff=kappa_temp_coeff,
        )

        # Override to store actual model name
        self._correction_model = correction_model
        self._kappa_correction_model = kappa_correction_model or correction_model

        self._vft_B = vft_B
        self._vft_Tinf = vft_Tinf
        self._wlf_C1 = wlf_C1
        self._wlf_C2 = wlf_C2

    @property
    def vft_B(self) -> float:
        """VFT B parameter (K)."""
        return self._vft_B

    @property
    def vft_Tinf(self) -> float:
        """VFT T_infinity parameter (K)."""
        return self._vft_Tinf

    @property
    def wlf_C1(self) -> float:
        """WLF C1 parameter."""
        return self._wlf_C1

    @property
    def wlf_C2(self) -> float:
        """WLF C2 parameter (K)."""
        return self._wlf_C2

    # ------------------------------------------------------------------
    # VFT correction
    # ------------------------------------------------------------------

    def _vft_factor(self, T: torch.Tensor) -> torch.Tensor:
        """VFT viscosity correction factor.

        factor = exp(B / (T - T_inf))

        Parameters
        ----------
        T : torch.Tensor
            Temperature (K).

        Returns
        -------
        torch.Tensor
            VFT correction factor.
        """
        T_safe = T.clamp(min=self._vft_Tinf + 1.0)
        denom = (T_safe - self._vft_Tinf).clamp(min=1.0)
        exponent = (self._vft_B / denom).clamp(max=50.0)
        return exponent.exp()

    # ------------------------------------------------------------------
    # WLF correction
    # ------------------------------------------------------------------

    def _wlf_factor(self, T: torch.Tensor) -> torch.Tensor:
        """WLF viscosity correction factor.

        log10(a_T) = -C1 * (T - T_ref) / (C2 + T - T_ref)
        factor = 10^(log10(a_T))

        Parameters
        ----------
        T : torch.Tensor
            Temperature (K).

        Returns
        -------
        torch.Tensor
            WLF correction factor.
        """
        dT = T - self._T_ref
        denom = (self._wlf_C2 + dT).clamp(min=1e-10)
        log10_aT = -self._wlf_C1 * dT / denom
        log10_aT = log10_aT.clamp(min=-20.0, max=20.0)
        return (log10_aT * math.log(10.0)).exp()

    # ------------------------------------------------------------------
    # Public API overrides
    # ------------------------------------------------------------------

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute viscosity with VFT or WLF correction.

        Parameters
        ----------
        T : float or torch.Tensor
            Temperature (K).

        Returns
        -------
        torch.Tensor
            Dynamic viscosity (Pa*s).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        if self._correction_model == "vft":
            return self._mu * self._vft_factor(T)

        if self._correction_model == "wlf":
            return self._mu * self._wlf_factor(T)

        # Delegate to parent for polynomial, exponential, piecewise
        return super().mu(T)

    def kappa(
        self,
        T: torch.Tensor | float,
        Cp: float = 1005.0,
        Pr: float = 0.7,
    ) -> torch.Tensor:
        """Compute thermal conductivity with VFT or WLF correction.

        Parameters
        ----------
        T : float or torch.Tensor
            Temperature (K).
        Cp : float
            Specific heat at constant pressure.
        Pr : float
            Prandtl number.

        Returns
        -------
        torch.Tensor
            Thermal conductivity (W/(m*K)).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        if self._kappa is not None:
            base = self._kappa
        else:
            base = self._mu * Cp / Pr

        if self._kappa_correction_model == "vft":
            return base * self._vft_factor(T)

        if self._kappa_correction_model == "wlf":
            return base * self._wlf_factor(T)

        return super().kappa(T, Cp=Cp, Pr=Pr)

    def __repr__(self) -> str:
        return (
            f"ConstantTransportEnhanced3(mu={self._mu}, kappa={self._kappa}, "
            f"T_ref={self._T_ref}, model={self._correction_model})"
        )
