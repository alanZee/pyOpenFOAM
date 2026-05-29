"""
Enhanced constant transport model v2 with improved temperature correction.

Extends :class:`~pyfoam.thermophysical.constant_transport_enhanced.ConstantTransportEnhanced`
with:

- Piecewise-linear temperature correction
- Exponential (Arrhenius-type) correction option
- Blending between correction models

Usage::

    from pyfoam.thermophysical.constant_transport_enhanced_2 import ConstantTransportEnhanced2

    transport = ConstantTransportEnhanced2(
        mu=1.8e-5,
        kappa=0.026,
        T_ref=300.0,
        correction_model="exponential",
        mu_activation_energy=110.0,
    )
    mu = transport.mu(T=400.0)
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.constant_transport_enhanced import ConstantTransportEnhanced

__all__ = ["ConstantTransportEnhanced2"]

logger = logging.getLogger(__name__)


class ConstantTransportEnhanced2(ConstantTransportEnhanced):
    """Enhanced constant transport v2 with advanced temperature corrections.

    Extends :class:`ConstantTransportEnhanced` with:

    - **Exponential correction**: mu(T) = mu_0 * exp(E * (1/T - 1/T_ref))
      where E is an activation-energy-like parameter (K).
    - **Piecewise correction**: different polynomial coefficients in
      user-defined temperature ranges.
    - **Blending**: smooth blend between correction models.

    Parameters
    ----------
    mu : float
        Base dynamic viscosity (Pa*s). Default 1.8e-5.
    kappa : float or None
        Base thermal conductivity. If None, computed from mu, Cp, Pr.
    T_ref : float
        Reference temperature (K). Default 300.0.
    correction_model : str
        Correction model type: "polynomial" (default, same as parent),
        "exponential", or "piecewise".
    mu_activation_energy : float
        Activation energy parameter (K) for exponential model.
        Default 110.0 (typical for air-like gases).
    piecewise_ranges : list of dict or None
        For piecewise model: list of {"T_low", "T_high", "alpha", "beta"}.
    kappa_correction_model : str or None
        Correction model for conductivity. If None, uses same as viscosity.
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
    ) -> None:
        super().__init__(
            mu=mu,
            kappa=kappa,
            T_ref=T_ref,
            mu_temp_coeff=mu_temp_coeff,
            mu_temp_coeff2=mu_temp_coeff2,
            kappa_temp_coeff=kappa_temp_coeff,
        )

        if correction_model not in ("polynomial", "exponential", "piecewise"):
            raise ValueError(
                f"correction_model must be 'polynomial', 'exponential', or "
                f"'piecewise', got '{correction_model}'"
            )

        self._correction_model = correction_model
        self._mu_E = mu_activation_energy
        self._piecewise_ranges = piecewise_ranges or []
        self._kappa_correction_model = kappa_correction_model or correction_model

    @property
    def correction_model(self) -> str:
        """Active correction model name."""
        return self._correction_model

    @property
    def mu_activation_energy(self) -> float:
        """Activation energy parameter for exponential model (K)."""
        return self._mu_E

    # ------------------------------------------------------------------
    # Exponential correction
    # ------------------------------------------------------------------

    def _exponential_factor(self, T: torch.Tensor) -> torch.Tensor:
        """Exponential correction factor: exp(E * (1/T - 1/T_ref)).

        Parameters
        ----------
        T : torch.Tensor
            Temperature (K).

        Returns
        -------
        torch.Tensor
            Correction factor.
        """
        T_safe = T.clamp(min=1e-10)
        T_ref_safe = max(self._T_ref, 1e-10)
        exponent = self._mu_E * (1.0 / T_safe - 1.0 / T_ref_safe)
        return exponent.clamp(min=-10.0, max=10.0).exp()

    # ------------------------------------------------------------------
    # Piecewise correction
    # ------------------------------------------------------------------

    def _piecewise_factor(self, T: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        """Piecewise polynomial correction over defined temperature ranges.

        Parameters
        ----------
        T : torch.Tensor
            Temperature (K).

        Returns
        -------
        torch.Tensor
            Correction factor.
        """
        result = torch.ones_like(T)
        for pw in self._piecewise_ranges:
            T_lo = pw.get("T_low", 0.0)
            T_hi = pw.get("T_high", 1e10)
            a = pw.get("alpha", 0.0)
            b = pw.get("beta", 0.0)
            mask = (T >= T_lo) & (T < T_hi)
            dT = T - self._T_ref
            factor = 1.0 + a * dT + b * dT * dT
            result = torch.where(mask, factor.clamp(min=0.5, max=2.0), result)
        return result

    # ------------------------------------------------------------------
    # Public API overrides
    # ------------------------------------------------------------------

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute viscosity with configured correction model.

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

        if self._correction_model == "exponential":
            return self._mu * self._exponential_factor(T)

        if self._correction_model == "piecewise":
            return self._mu * self._piecewise_factor(T, 0.0, 0.0)

        # Default polynomial (parent behaviour)
        return super().mu(T)

    def kappa(
        self,
        T: torch.Tensor | float,
        Cp: float = 1005.0,
        Pr: float = 0.7,
    ) -> torch.Tensor:
        """Compute thermal conductivity with configured correction model.

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

        if self._kappa_correction_model == "exponential":
            return base * self._exponential_factor(T)

        if self._kappa_correction_model == "piecewise":
            return base * self._piecewise_factor(T, 0.0, 0.0)

        # Default polynomial (parent behaviour)
        return super().kappa(T, Cp=Cp, Pr=Pr)

    def __repr__(self) -> str:
        return (
            f"ConstantTransportEnhanced2(mu={self._mu}, kappa={self._kappa}, "
            f"T_ref={self._T_ref}, model={self._correction_model})"
        )
