"""
Constant transport model with thermal conductivity support.

Implements a transport model with constant viscosity and constant
thermal conductivity, as used in OpenFOAM's ``constantTransport``
class.

This is an enhanced version of the basic ConstantViscosity model in
``transport_model.py``, adding explicit thermal conductivity.

Usage::

    from pyfoam.thermophysical.constant_transport import ConstantTransport

    transport = ConstantTransport(mu=1.8e-5, kappa=0.026)
    mu = transport.mu(T=300.0)    # always 1.8e-5
    kappa = transport.kappa(T=300.0)  # always 0.026
"""

from __future__ import annotations

import logging

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel

__all__ = ["ConstantTransport"]

logger = logging.getLogger(__name__)


class ConstantTransport(TransportModel):
    """Constant viscosity and constant thermal conductivity model.

    Both dynamic viscosity and thermal conductivity are independent
    of temperature.

    Parameters
    ----------
    mu : float
        Constant dynamic viscosity (Pa·s). Default 1.8e-5 (air at STP).
    kappa : float or None
        Constant thermal conductivity (W/(m·K)).
        If None, thermal conductivity is computed from mu, Cp, and Pr.
        Default None.

    Examples::

        transport = ConstantTransport(mu=1.8e-5, kappa=0.026)
        assert transport.mu(300.0) == 1.8e-5
        assert transport.kappa(300.0) == 0.026
    """

    def __init__(
        self,
        mu: float = 1.8e-5,
        kappa: float | None = None,
    ) -> None:
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        if kappa is not None and kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")

        self._mu = mu
        self._kappa = kappa

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Return constant dynamic viscosity.

        Args:
            T: Temperature (K) — ignored.

        Returns:
            Dynamic viscosity (Pa·s).
        """
        device = get_device()
        dtype = get_default_dtype()

        if isinstance(T, torch.Tensor):
            return torch.full_like(T, self._mu)
        return torch.tensor(self._mu, dtype=dtype, device=device)

    def kappa(
        self,
        T: torch.Tensor | float,
        Cp: float = 1005.0,
        Pr: float = 0.7,
    ) -> torch.Tensor:
        """Compute thermal conductivity.

        If ``kappa`` was set at construction, returns that constant.
        Otherwise:

        .. math::

            \\kappa = \\frac{\\mu \\cdot C_p}{Pr}

        Args:
            T: Temperature (K) — ignored for constant kappa.
            Cp: Specific heat at constant pressure (J/(kg·K)).
            Pr: Prandtl number.

        Returns:
            Thermal conductivity (W/(m·K)).
        """
        device = get_device()
        dtype = get_default_dtype()

        if self._kappa is not None:
            if isinstance(T, torch.Tensor):
                return torch.full_like(T, self._kappa)
            return torch.tensor(self._kappa, dtype=dtype, device=device)

        val = self._mu * Cp / Pr
        if isinstance(T, torch.Tensor):
            return torch.full_like(T, val)
        return torch.tensor(val, dtype=dtype, device=device)

    @property
    def mu_value(self) -> float:
        """The constant viscosity value."""
        return self._mu

    @property
    def kappa_value(self) -> float | None:
        """The constant thermal conductivity value (or None)."""
        return self._kappa

    def __repr__(self) -> str:
        return f"ConstantTransport(mu={self._mu}, kappa={self._kappa})"
