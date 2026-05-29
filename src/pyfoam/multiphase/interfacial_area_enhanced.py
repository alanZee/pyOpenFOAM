"""
Enhanced interfacial area density models for multiphase simulations.

Extends the basic interfacial area models with:

- :class:`SauterMeanInterfacialArea` — population-based Sauter mean diameter
- :class:`BreakupCoalescenceInterfacialArea` — dynamic interfacial area from
  breakup/coalescence equilibrium
- :class:`BlendedInterfacialArea` — blends between two models based on
  a transition criterion (e.g., volume fraction)

These models are important for bubble column, fluidised bed, and
spray simulations where the interfacial area dynamically evolves.

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced import (
        SauterMeanInterfacialArea,
        BreakupCoalescenceInterfacialArea,
    )

    model = SauterMeanInterfacialArea(d32_0=3e-3, alpha_min=1e-4)
    a_i = model.compute(alpha, n_cells=100)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Sequence, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area import InterfacialAreaModel

__all__ = [
    "SauterMeanInterfacialArea",
    "BreakupCoalescenceInterfacialArea",
    "BlendedInterfacialArea",
]

logger = logging.getLogger(__name__)


@InterfacialAreaModel.register("sauterMean")
class SauterMeanInterfacialArea(InterfacialAreaModel):
    """Population-based Sauter mean diameter interfacial area model.

    Computes interfacial area density from the Sauter mean diameter d32
    with alpha-dependent correction for dense systems:

        a_i = 6 * alpha / d32 * C(alpha)

    where C(alpha) is a correction factor for dense systems:

        C(alpha) = (1 - alpha)^n   (Richardson-Zaki type)

    At dilute conditions (alpha -> 0), this reduces to the standard
    dilute sphere correlation: a_i = 6 * alpha / d32.

    Parameters
    ----------
    d32_0 : float
        Reference Sauter mean diameter (m). Default: 3e-3.
    alpha_min : float
        Minimum volume fraction for non-zero area. Default: 1e-4.
    richardson_zaki_n : float
        Exponent for dense correction. Default: 2.0.
    """

    def __init__(
        self,
        d32_0: float = 3e-3,
        alpha_min: float = 1e-4,
        richardson_zaki_n: float = 2.0,
    ) -> None:
        self._d32_0 = d32_0
        self._alpha_min = alpha_min
        self._n_rz = richardson_zaki_n

    @property
    def d32_0(self) -> float:
        """Reference Sauter mean diameter (m)."""
        return self._d32_0

    @property
    def alpha_min(self) -> float:
        """Minimum alpha for non-zero area."""
        return self._alpha_min

    @property
    def richardson_zaki_n(self) -> float:
        """Richardson-Zaki exponent."""
        return self._n_rz

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Sauter mean interfacial area density.

        a_i = 6 * alpha * (1 - alpha)^(n-1) / d32_0

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` interfacial area density (1/m).
        """
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)

        mask = alpha_dev >= self._alpha_min

        # a_i = 6 * alpha * (1 - alpha)^(n-1) / d32
        dense_correction = (1.0 - alpha_dev).pow(self._n_rz - 1.0)
        a_i = 6.0 * alpha_dev * dense_correction / max(self._d32_0, 1e-20)
        a_i = torch.where(mask, a_i, torch.zeros_like(a_i))

        return a_i.clamp(min=0.0)


@InterfacialAreaModel.register("breakupCoalescence")
class BreakupCoalescenceInterfacialArea(InterfacialAreaModel):
    """Dynamic interfacial area from breakup/coalescence equilibrium.

    Models the interfacial area as a balance between breakup
    (increasing area) and coalescence (decreasing area):

        a_i = a_eq * (1 + C_dev * (alpha - alpha_eq)^2)

    where a_eq is the equilibrium interfacial area and C_dev is a
    deviation coefficient. The equilibrium value depends on the
    Weber number and volume fraction:

        a_eq = 6 * alpha / d_eq

    with d_eq from the Hinze-Topycal breakup criterion:

        d_eq = C_h * sigma^(3/5) * epsilon^(-2/5) * rho_c^(-3/5)

    where sigma is surface tension, epsilon is turbulent dissipation rate,
    and rho_c is continuous-phase density.

    Parameters
    ----------
    d_eq_0 : float
        Reference equilibrium diameter (m). Default: 1e-3.
    C_dev : float
        Deviation coefficient. Default: 0.5.
    alpha_eq : float
        Equilibrium volume fraction. Default: 0.1.
    alpha_min : float
        Minimum alpha for non-zero area. Default: 1e-4.
    """

    def __init__(
        self,
        d_eq_0: float = 1e-3,
        C_dev: float = 0.5,
        alpha_eq: float = 0.1,
        alpha_min: float = 1e-4,
    ) -> None:
        self._d_eq_0 = d_eq_0
        self._C_dev = C_dev
        self._alpha_eq = alpha_eq
        self._alpha_min = alpha_min

    @property
    def d_eq_0(self) -> float:
        """Reference equilibrium diameter (m)."""
        return self._d_eq_0

    @property
    def C_dev(self) -> float:
        """Deviation coefficient."""
        return self._C_dev

    @property
    def alpha_eq(self) -> float:
        """Equilibrium volume fraction."""
        return self._alpha_eq

    @property
    def alpha_min(self) -> float:
        """Minimum alpha for non-zero area."""
        return self._alpha_min

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute breakup/coalescence equilibrium interfacial area.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        n_cells : int
            Number of cells.
        **kwargs
            Optional: ``epsilon`` (turbulent dissipation rate),
            ``sigma`` (surface tension), ``rho_c`` (continuous density).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` interfacial area density (1/m).
        """
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)

        mask = alpha_dev >= self._alpha_min

        # Equilibrium diameter (optionally modified by epsilon)
        d_eq = self._d_eq_0
        epsilon = kwargs.get("epsilon", None)
        sigma = kwargs.get("sigma", None)
        rho_c = kwargs.get("rho_c", None)

        if epsilon is not None and sigma is not None and rho_c is not None:
            eps_t = epsilon.to(device=device, dtype=dtype).clamp(min=1e-30)
            # Hinze-Topycal: d_eq = C_h * sigma^(3/5) * epsilon^(-2/5) * rho_c^(-3/5)
            C_h = 0.5  # model constant
            d_eq = C_h * sigma.pow(0.6) * eps_t.pow(-0.4) * rho_c.pow(-0.6)
            d_eq = d_eq.clamp(min=1e-10, max=1.0)

        # Equilibrium area
        a_eq = 6.0 * alpha_dev / d_eq.clamp(min=1e-20)

        # Deviation from equilibrium
        deviation = self._C_dev * (alpha_dev - self._alpha_eq).pow(2)
        a_i = a_eq * (1.0 + deviation)

        a_i = torch.where(mask, a_i, torch.zeros_like(a_i))
        return a_i.clamp(min=0.0)


class BlendedInterfacialArea(InterfacialAreaModel):
    """Blends two interfacial area models based on volume fraction.

    Smoothly transitions between two models using a sigmoid:

        w = sigmoid((alpha - alpha_blend) / width)
        a_i = (1 - w) * model_1.compute(alpha) + w * model_2.compute(alpha)

    This is useful for flows where different physical mechanisms dominate
    at different volume fractions (e.g., dilute vs dense regimes).

    Parameters
    ----------
    model_1 : InterfacialAreaModel
        Model for dilute regime (low alpha).
    model_2 : InterfacialAreaModel
        Model for dense regime (high alpha).
    alpha_blend : float
        Volume fraction at which blending is 50/50. Default: 0.3.
    blend_width : float
        Width of the blending sigmoid. Default: 0.1.
    """

    def __init__(
        self,
        model_1: InterfacialAreaModel,
        model_2: InterfacialAreaModel,
        alpha_blend: float = 0.3,
        blend_width: float = 0.1,
    ) -> None:
        self._model_1 = model_1
        self._model_2 = model_2
        self._alpha_blend = alpha_blend
        self._blend_width = max(blend_width, 1e-10)

    @property
    def model_1(self) -> InterfacialAreaModel:
        """Dilute-regime model."""
        return self._model_1

    @property
    def model_2(self) -> InterfacialAreaModel:
        """Dense-regime model."""
        return self._model_2

    @property
    def alpha_blend(self) -> float:
        """Blending midpoint volume fraction."""
        return self._alpha_blend

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute blended interfacial area density.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` blended interfacial area density (1/m).
        """
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)

        a_1 = self._model_1.compute(alpha, n_cells, **kwargs)
        a_2 = self._model_2.compute(alpha, n_cells, **kwargs)

        # Sigmoid blending weight
        x = (alpha_dev - self._alpha_blend) / self._blend_width
        w = torch.sigmoid(x)

        return (1.0 - w) * a_1 + w * a_2
