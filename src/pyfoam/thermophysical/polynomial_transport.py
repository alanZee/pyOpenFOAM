"""
Polynomial transport model for viscosity and thermal conductivity.

Implements temperature-dependent viscosity and thermal conductivity
using polynomial expressions, as used in OpenFOAM's
``polynomialTransport`` class.

.. math::

    \\mu(T) = \\sum_{i=0}^{n} a_i T^i

    \\kappa(T) = \\sum_{i=0}^{n} b_i T^i

Usage::

    from pyfoam.thermophysical.polynomial_transport import PolynomialTransport

    # μ = 1e-5 + 4e-8*T  (Pa·s)
    transport = PolynomialTransport(mu_coeffs=[1e-5, 4e-8])
    mu = transport.mu(T=300.0)  # ~2.2e-5 Pa·s
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel

__all__ = ["PolynomialTransport"]

logger = logging.getLogger(__name__)


class PolynomialTransport(TransportModel):
    """Polynomial viscosity and thermal conductivity model.

    Dynamic viscosity is computed as a polynomial in temperature:

    .. math::

        \\mu(T) = a_0 + a_1 T + a_2 T^2 + \\ldots + a_n T^n

    Optionally, thermal conductivity can also be specified as a polynomial.

    Parameters
    ----------
    mu_coeffs : sequence of float
        Polynomial coefficients for viscosity ``[a0, a1, ..., an]``.
    kappa_coeffs : sequence of float or None
        Polynomial coefficients for thermal conductivity.
        If None, thermal conductivity is computed from μ, Cp, and Pr.

    Examples::

        # Linear viscosity: μ = 1e-5 + 4e-8 * T
        transport = PolynomialTransport(mu_coeffs=[1e-5, 4e-8])
        mu = transport.mu(T=300.0)

        # Cubic viscosity
        transport = PolynomialTransport(mu_coeffs=[1e-5, 3e-8, 1e-11, -1e-15])
    """

    def __init__(
        self,
        mu_coeffs: Sequence[float],
        kappa_coeffs: Sequence[float] | None = None,
    ) -> None:
        if len(mu_coeffs) == 0:
            raise ValueError("mu_coeffs must not be empty")

        self._mu_coeffs = list(mu_coeffs)
        self._kappa_coeffs = list(kappa_coeffs) if kappa_coeffs is not None else None

    def _eval_poly(
        self,
        coeffs: list[float],
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate polynomial using Horner's method.

        Args:
            coeffs: Polynomial coefficients [a0, a1, ..., an].
            T: Temperature tensor.

        Returns:
            Polynomial value.
        """
        # Horner's method: a0 + T*(a1 + T*(a2 + ...))
        result = torch.zeros_like(T)
        for c in reversed(coeffs):
            result = result * T + c
        return result

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute dynamic viscosity using polynomial.

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
        return self._eval_poly(self._mu_coeffs, T_safe)

    def kappa(
        self,
        T: torch.Tensor | float,
        Cp: float = 1005.0,
        Pr: float = 0.7,
    ) -> torch.Tensor:
        """Compute thermal conductivity.

        If ``kappa_coeffs`` were provided at construction, uses the
        polynomial. Otherwise computes from viscosity:

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
    def mu_coeffs(self) -> list[float]:
        """Viscosity polynomial coefficients."""
        return self._mu_coeffs.copy()

    @property
    def kappa_coeffs(self) -> list[float] | None:
        """Thermal conductivity polynomial coefficients (or None)."""
        return self._kappa_coeffs.copy() if self._kappa_coeffs is not None else None

    def __repr__(self) -> str:
        return (
            f"PolynomialTransport(mu_coeffs={self._mu_coeffs}, "
            f"kappa_coeffs={self._kappa_coeffs})"
        )
