"""
Sutherland transport model with thermal conductivity support.

Implements temperature-dependent viscosity using Sutherland's law
and optionally temperature-dependent thermal conductivity using a
separate polynomial or the classic :math:`\\kappa = \\mu C_p / Pr`
relation.

This is an enhanced version of the basic Sutherland model in
``transport_model.py``, adding thermal conductivity computation.

Usage::

    from pyfoam.thermophysical.sutherland_transport import SutherlandTransport

    transport = SutherlandTransport()
    mu = transport.mu(T=300.0)
    kappa = transport.kappa(T=300.0, Cp=1005.0, Pr=0.7)
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel

__all__ = ["SutherlandTransport"]

logger = logging.getLogger(__name__)


class SutherlandTransport(TransportModel):
    """Sutherland viscosity with optional polynomial thermal conductivity.

    Dynamic viscosity uses Sutherland's law:

    .. math::

        \\mu(T) = \\mu_{\\text{ref}} \\cdot (T/T_{\\text{ref}})^{3/2}
                  \\cdot (T_{\\text{ref}} + S) / (T + S)

    Thermal conductivity can be specified as:
    - A polynomial in temperature (if ``kappa_coeffs`` is provided)
    - Computed from :math:`\\kappa = \\mu C_p / Pr` (default)

    Parameters
    ----------
    mu_ref : float
        Reference dynamic viscosity (Pa·s) at T_ref.
        Default 1.716e-5 (air at 273.15 K).
    T_ref : float
        Reference temperature (K). Default 273.15.
    S : float
        Sutherland constant (K). Default 110.4 for air.
    kappa_coeffs : sequence of float or None
        Polynomial coefficients for thermal conductivity.
        If None, thermal conductivity is computed from mu, Cp, and Pr.

    Examples::

        # Standard air
        transport = SutherlandTransport()
        mu = transport.mu(T=300.0)

        # With explicit kappa polynomial
        transport = SutherlandTransport(kappa_coeffs=[0.024, 7e-5])
    """

    def __init__(
        self,
        mu_ref: float = 1.716e-5,
        T_ref: float = 273.15,
        S: float = 110.4,
        kappa_coeffs: Sequence[float] | None = None,
    ) -> None:
        if mu_ref <= 0:
            raise ValueError(f"mu_ref must be positive, got {mu_ref}")
        if T_ref <= 0:
            raise ValueError(f"T_ref must be positive, got {T_ref}")
        if S <= 0:
            raise ValueError(f"S must be positive, got {S}")

        self._mu_ref = mu_ref
        self._T_ref = T_ref
        self._S = S
        self._kappa_coeffs = list(kappa_coeffs) if kappa_coeffs is not None else None

    def _eval_poly(self, coeffs: list[float], T: torch.Tensor) -> torch.Tensor:
        """Evaluate polynomial using Horner's method."""
        result = torch.zeros_like(T)
        for c in reversed(coeffs):
            result = result * T + c
        return result

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute dynamic viscosity using Sutherland's law.

        Args:
            T: Temperature (K) — scalar or ``(n_cells,)`` tensor.

        Returns:
            Dynamic viscosity (Pa·s).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1.0)
        T_ratio = T_safe / self._T_ref
        return (
            self._mu_ref
            * T_ratio.pow(1.5)
            * (self._T_ref + self._S)
            / (T_safe + self._S)
        )

    def kappa(
        self,
        T: torch.Tensor | float,
        Cp: float = 1005.0,
        Pr: float = 0.7,
    ) -> torch.Tensor:
        """Compute thermal conductivity.

        If ``kappa_coeffs`` were provided, uses the polynomial.
        Otherwise:

        .. math::

            \\kappa = \\frac{\\mu \\cdot C_p}{Pr}

        Args:
            T: Temperature (K).
            Cp: Specific heat at constant pressure (J/(kg·K)).
            Pr: Prandtl number.

        Returns:
            Thermal conductivity (W/(m·K)).
        """
        if self._kappa_coeffs is not None:
            device = get_device()
            dtype = get_default_dtype()
            if not isinstance(T, torch.Tensor):
                T = torch.tensor(T, dtype=dtype, device=device)
            T_safe = T.clamp(min=1.0)
            return self._eval_poly(self._kappa_coeffs, T_safe)

        return self.mu(T) * Cp / Pr

    @property
    def mu_ref(self) -> float:
        """Reference dynamic viscosity."""
        return self._mu_ref

    @property
    def T_ref(self) -> float:
        """Reference temperature."""
        return self._T_ref

    @property
    def S(self) -> float:
        """Sutherland constant."""
        return self._S

    @property
    def kappa_coeffs(self) -> list[float] | None:
        """Thermal conductivity polynomial coefficients (or None)."""
        return self._kappa_coeffs.copy() if self._kappa_coeffs is not None else None

    def __repr__(self) -> str:
        return (
            f"SutherlandTransport(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S})"
        )
