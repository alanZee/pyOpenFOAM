"""
Enhanced constant transport model with optional temperature correction.

Extends :class:`~pyfoam.thermophysical.constant_transport.ConstantTransport`
with:

- Optional linear temperature correction: mu(T) = mu_0 * (1 + alpha * (T - T_ref))
- Optional quadratic temperature correction
- Configurable reference temperature

Usage::

    from pyfoam.thermophysical.constant_transport_enhanced import ConstantTransportEnhanced

    transport = ConstantTransportEnhanced(
        mu=1.8e-5,
        kappa=0.026,
        T_ref=300.0,
        mu_temp_coeff=1e-7,  # linear correction
    )
    mu = transport.mu(T=400.0)  # 1.8e-5 + 1e-7 * (400 - 300)
"""

from __future__ import annotations

import logging

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.constant_transport import ConstantTransport

__all__ = ["ConstantTransportEnhanced"]

logger = logging.getLogger(__name__)


class ConstantTransportEnhanced(ConstantTransport):
    """Enhanced constant transport with optional temperature correction.

    Extends :class:`ConstantTransport` with:

    - **Linear correction**: mu(T) = mu_0 * (1 + alpha * (T - T_ref))
    - **Quadratic correction**: mu(T) = mu_0 * (1 + alpha * dT + beta * dT^2)
    - **Kappa temperature correction**: same polynomial form for kappa

    When all correction coefficients are zero, behaves identically to
    the parent class.

    Parameters
    ----------
    mu : float
        Base dynamic viscosity (Pa·s). Default 1.8e-5.
    kappa : float or None
        Base thermal conductivity (W/(m·K)). If None, computed from
        mu, Cp, and Pr.
    T_ref : float
        Reference temperature (K). Default 300.0.
    mu_temp_coeff : float
        Linear temperature correction coefficient for viscosity (1/K).
        Default 0.0 (no correction).
    mu_temp_coeff2 : float
        Quadratic temperature correction coefficient for viscosity (1/K^2).
        Default 0.0.
    kappa_temp_coeff : float
        Linear temperature correction coefficient for thermal conductivity.
        Default 0.0.

    Examples::

        # Nearly constant with slight T-dependence
        transport = ConstantTransportEnhanced(
            mu=1.8e-5, kappa=0.026, T_ref=300.0, mu_temp_coeff=1e-7
        )
        mu = transport.mu(T=400.0)
    """

    def __init__(
        self,
        mu: float = 1.8e-5,
        kappa: float | None = None,
        T_ref: float = 300.0,
        mu_temp_coeff: float = 0.0,
        mu_temp_coeff2: float = 0.0,
        kappa_temp_coeff: float = 0.0,
    ) -> None:
        super().__init__(mu=mu, kappa=kappa)

        if T_ref <= 0:
            raise ValueError(f"T_ref must be positive, got {T_ref}")

        self._T_ref = T_ref
        self._mu_alpha = mu_temp_coeff
        self._mu_beta = mu_temp_coeff2
        self._kappa_alpha = kappa_temp_coeff

    @property
    def T_ref(self) -> float:
        """Reference temperature (K)."""
        return self._T_ref

    @property
    def mu_temp_coeff(self) -> float:
        """Linear temperature correction coefficient for viscosity."""
        return self._mu_alpha

    @property
    def mu_temp_coeff2(self) -> float:
        """Quadratic temperature correction coefficient for viscosity."""
        return self._mu_beta

    @property
    def kappa_temp_coeff(self) -> float:
        """Temperature correction coefficient for thermal conductivity."""
        return self._kappa_alpha

    def _correction_factor(
        self,
        T: torch.Tensor,
        alpha: float,
        beta: float = 0.0,
    ) -> torch.Tensor:
        """Compute polynomial correction factor: 1 + alpha*dT + beta*dT^2.

        Args:
            T: Temperature tensor (K).
            alpha: Linear coefficient (1/K).
            beta: Quadratic coefficient (1/K^2).

        Returns:
            Correction factor tensor.
        """
        dT = T - self._T_ref
        factor = 1.0 + alpha * dT
        if beta != 0.0:
            factor = factor + beta * dT * dT
        return factor.clamp(min=0.5, max=2.0)  # prevent unphysical values

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute viscosity with optional temperature correction.

        mu(T) = mu_0 * (1 + alpha * (T - T_ref) + beta * (T - T_ref)^2)

        Args:
            T: Temperature (K).

        Returns:
            Dynamic viscosity (Pa·s).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        if self._mu_alpha == 0.0 and self._mu_beta == 0.0:
            # No correction: behave like parent
            return torch.full_like(T, self._mu)

        factor = self._correction_factor(T, self._mu_alpha, self._mu_beta)
        return self._mu * factor

    def kappa(
        self,
        T: torch.Tensor | float,
        Cp: float = 1005.0,
        Pr: float = 0.7,
    ) -> torch.Tensor:
        """Compute thermal conductivity with optional temperature correction.

        Args:
            T: Temperature (K).
            Cp: Specific heat at constant pressure (J/(kg·K)).
            Pr: Prandtl number.

        Returns:
            Thermal conductivity (W/(m·K)).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        if self._kappa is not None:
            base = self._kappa
        else:
            base = self._mu * Cp / Pr

        if self._kappa_alpha == 0.0:
            return torch.full_like(T, base)

        factor = self._correction_factor(T, self._kappa_alpha)
        return base * factor

    def __repr__(self) -> str:
        return (
            f"ConstantTransportEnhanced(mu={self._mu}, kappa={self._kappa}, "
            f"T_ref={self._T_ref}, alpha={self._mu_alpha})"
        )
