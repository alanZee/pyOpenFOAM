"""
Thermophysical transport models for heat and mass transfer.

Provides:

- :class:`ThermophysicalTransportModel` — abstract base
- :class:`FourierTransport` — Fourier law: q = -k * grad(T)
- :class:`FickianTransport` — Fick's law: j = -D * grad(Y)

Usage::

    from pyfoam.thermophysical.thermophysical_transport import FourierTransport

    model = FourierTransport(kappa=0.026)
    q = model.flux(grad_X=[100.0, 0.0, 0.0])  # heat flux (W/m²)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from pyfoam.core.device import get_device, get_default_dtype


__all__ = [
    "ThermophysicalTransportModel",
    "FourierTransport",
    "FickianTransport",
]


# ======================================================================
# 抽象基类
# ======================================================================

class ThermophysicalTransportModel(ABC):
    """Abstract base for thermophysical transport models.

    Subclasses implement :meth:`flux` to compute the diffusive flux
    (heat flux or species mass flux) from a gradient vector.
    """

    @abstractmethod
    def flux(
        self,
        grad_X: torch.Tensor | list[float],
    ) -> torch.Tensor:
        """Compute diffusive flux from a gradient.

        Parameters
        ----------
        grad_X : torch.Tensor or list[float]
            Gradient vector.  For Fourier transport this is ``grad(T)``
            in K/m; for Fickian transport this is ``grad(Y)`` in 1/m.
            Shape ``(n, 3)`` or ``(3,)``.

        Returns
        -------
        torch.Tensor
            Diffusive flux vector, same shape as input.
            For Fourier: q = -k * grad(T) in W/m².
            For Fickian: j = -D * grad(Y) in kg/(m²·s).
        """


# ======================================================================
# Fourier 热传导
# ======================================================================

class FourierTransport(ThermophysicalTransportModel):
    """Fourier law for heat conduction.

    Computes heat flux as:

    .. math::

        \\mathbf{q} = -\\kappa \\, \\nabla T

    Parameters
    ----------
    kappa : float
        Thermal conductivity (W/(m·K)).  Default ``0.026`` (air at ~300 K).

    Examples::

        model = FourierTransport(kappa=0.026)
        q = model.flux(grad_X=[100.0, 0.0, 0.0])
        # q = [-2.6, 0.0, 0.0]  W/m²
    """

    def __init__(self, kappa: float = 0.026) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        self._kappa = kappa

    @property
    def kappa(self) -> float:
        """Thermal conductivity (W/(m·K))."""
        return self._kappa

    def flux(
        self,
        grad_X: torch.Tensor | list[float],
    ) -> torch.Tensor:
        """Compute heat flux q = -kappa * grad(T).

        Parameters
        ----------
        grad_X : torch.Tensor or list[float]
            Temperature gradient ``grad(T)`` (K/m).
            Shape ``(3,)`` or ``(n, 3)``.

        Returns
        -------
        torch.Tensor
            Heat flux (W/m²), same shape as input.
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(grad_X, torch.Tensor):
            grad_X = torch.tensor(grad_X, dtype=dtype, device=device)

        return -self._kappa * grad_X

    def __repr__(self) -> str:
        return f"FourierTransport(kappa={self._kappa})"


# ======================================================================
# Fick 扩散
# ======================================================================

class FickianTransport(ThermophysicalTransportModel):
    """Fick's law for species mass diffusion.

    Computes species mass flux as:

    .. math::

        \\mathbf{j} = -D \\, \\nabla Y

    Parameters
    ----------
    D : float
        Mass diffusivity (m²/s).  Default ``2.1e-5`` (air-water vapour
        at ~300 K).
    rho : float
        Mixture density (kg/m³).  Default ``1.225`` (air at STP).
        The mass flux is ``j = -rho * D * grad(Y)``.

    Examples::

        model = FickianTransport(D=2.1e-5, rho=1.225)
        j = model.flux(grad_X=[0.01, 0.0, 0.0])
        # j = [-2.5725e-7, 0.0, 0.0]  kg/(m²·s)
    """

    def __init__(self, D: float = 2.1e-5, rho: float = 1.225) -> None:
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")
        self._D = D
        self._rho = rho

    @property
    def D(self) -> float:
        """Mass diffusivity (m²/s)."""
        return self._D

    @property
    def rho(self) -> float:
        """Mixture density (kg/m³)."""
        return self._rho

    def flux(
        self,
        grad_X: torch.Tensor | list[float],
    ) -> torch.Tensor:
        """Compute species mass flux j = -rho * D * grad(Y).

        Parameters
        ----------
        grad_X : torch.Tensor or list[float]
            Species mass fraction gradient ``grad(Y)`` (1/m).
            Shape ``(3,)`` or ``(n, 3)``.

        Returns
        -------
        torch.Tensor
            Species mass flux (kg/(m²·s)), same shape as input.
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(grad_X, torch.Tensor):
            grad_X = torch.tensor(grad_X, dtype=dtype, device=device)

        return -self._rho * self._D * grad_X

    def __repr__(self) -> str:
        return f"FickianTransport(D={self._D}, rho={self._rho})"
