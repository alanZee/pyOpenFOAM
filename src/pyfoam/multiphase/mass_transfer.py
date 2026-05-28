"""
Mass transfer models for interfacial phase change.

Provides abstract base class and concrete implementations for
interfacial mass transfer between phases in multiphase simulations.

Models:
- :class:`MassTransferModel` — abstract base with RTS registry
- :class:`LeeMassTransfer` — Lee evaporation/condensation model
- :class:`ThermalPhaseChange` — thermal phase change based on
  heat diffusion across the interface

Usage::

    from pyfoam.multiphase.mass_transfer import MassTransferModel

    model = MassTransferModel.create("Lee", T_sat=373.15)
    m_dot = model.compute(alpha, T, p, rho_l, rho_v)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "MassTransferModel",
    "LeeMassTransfer",
    "ThermalPhaseChange",
]


class MassTransferModel(ABC):
    """Abstract base class for interfacial mass transfer models.

    Subclasses must implement :meth:`compute`.

    RTS (Run-Time Selection) registry allows string-based lookup::

        model = MassTransferModel.create("Lee", T_sat=373.15)
    """

    _registry: ClassVar[dict[str, Type[MassTransferModel]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a mass transfer model under *name*."""

        def decorator(model_cls: Type[MassTransferModel]) -> Type[MassTransferModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Mass transfer model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> MassTransferModel:
        """Create a mass transfer model by name.

        Parameters
        ----------
        name : str
            Registered model name.
        **kwargs
            Arguments forwarded to the model constructor.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Unknown mass transfer model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return list of registered model names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute(
        self,
        alpha: torch.Tensor,
        T: torch.Tensor,
        p: torch.Tensor,
        rho_l: float,
        rho_v: float,
    ) -> torch.Tensor:
        """Compute interfacial mass transfer rate.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction of the dispersed phase.
        T : torch.Tensor
            ``(n_cells,)`` temperature field.
        p : torch.Tensor
            ``(n_cells,)`` pressure field.
        rho_l : float
            Liquid density (kg/m3).
        rho_v : float
            Vapour density (kg/m3).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mass transfer rate (kg/(m3 s)).
            Positive = evaporation (liquid -> vapour).
        """


# ======================================================================
# Lee mass transfer model
# ======================================================================


@MassTransferModel.register("Lee")
class LeeMassTransfer(MassTransferModel):
    """Lee evaporation/condensation mass transfer model.

    The Lee model is an empirical model widely used for boiling
    and condensation:

        When T > T_sat (evaporation):
            m_dot = r_evap * alpha_l * rho_l * (T - T_sat) / T_sat

        When T < T_sat (condensation):
            m_dot = -r_cond * alpha_v * rho_v * (T_sat - T) / T_sat

    This is similar to :class:`pyfoam.multiphase.phase_change.LeeModel`
    but under the mass transfer abstraction for use with Euler-Euler
    multiphase solvers.

    Parameters
    ----------
    T_sat : float
        Saturation temperature (K).  Default: 373.15.
    r_evap : float
        Evaporation rate coefficient (1/s).  Default: 0.1.
    r_cond : float
        Condensation rate coefficient (1/s).  Default: 0.1.
    """

    def __init__(
        self,
        T_sat: float = 373.15,
        r_evap: float = 0.1,
        r_cond: float = 0.1,
    ) -> None:
        self.T_sat = T_sat
        self.r_evap = r_evap
        self.r_cond = r_cond

    def compute(
        self,
        alpha: torch.Tensor,
        T: torch.Tensor,
        p: torch.Tensor,
        rho_l: float,
        rho_v: float,
    ) -> torch.Tensor:
        """Compute Lee mass transfer rate.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` vapour volume fraction.
        T : torch.Tensor
            ``(n_cells,)`` temperature.
        p : torch.Tensor
            ``(n_cells,)`` pressure (unused by Lee model).
        rho_l : float
            Liquid density.
        rho_v : float
            Vapour density.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mass transfer rate (positive = evaporation).
        """
        T_sat = self.T_sat
        alpha_l = (1.0 - alpha).clamp(min=0.0)
        alpha_v = alpha.clamp(min=0.0)

        # Evaporation: T > T_sat
        evap = self.r_evap * alpha_l * rho_l * (T - T_sat) / T_sat
        evap = evap.clamp(min=0.0)

        # Condensation: T < T_sat
        cond = -self.r_cond * alpha_v * rho_v * (T_sat - T) / T_sat
        cond = cond.clamp(max=0.0)

        return evap + cond

    def __repr__(self) -> str:
        return (
            f"LeeMassTransfer(T_sat={self.T_sat}, "
            f"r_evap={self.r_evap}, r_cond={self.r_cond})"
        )


# ======================================================================
# Thermal phase change model
# ======================================================================


@MassTransferModel.register("ThermalPhaseChange")
class ThermalPhaseChange(MassTransferModel):
    """Thermal phase change model based on interface heat diffusion.

    Computes mass transfer from the temperature gradient across the
    interface using Fourier's law:

        q_interface = k_eff * (T - T_sat) / delta_x
        m_dot = q_interface / h_lv

    where:
    - k_eff is the effective thermal conductivity
    - delta_x is the cell size (characteristic length)
    - h_lv is the latent heat of vaporisation

    This model captures the physics of conductive heat transfer at
    the interface and is more physically consistent than empirical
    models for problems where the thermal boundary layer resolves
    the interface temperature gradient.

    Parameters
    ----------
    T_sat : float
        Saturation temperature (K).  Default: 373.15.
    h_lv : float
        Latent heat of vaporisation (J/kg).  Default: 2257e3 (water).
    k_eff : float
        Effective thermal conductivity (W/(m K)).  Default: 0.6.
    delta_x : float
        Characteristic cell size (m).  Default: 1e-3.
    relaxation : float
        Under-relaxation factor in (0, 1].  Default: 1.0.
    """

    def __init__(
        self,
        T_sat: float = 373.15,
        h_lv: float = 2257e3,
        k_eff: float = 0.6,
        delta_x: float = 1e-3,
        relaxation: float = 1.0,
    ) -> None:
        self.T_sat = T_sat
        self.h_lv = h_lv
        self.k_eff = k_eff
        self.delta_x = delta_x
        self.relaxation = relaxation
        self._m_dot_old: torch.Tensor | None = None

    def compute(
        self,
        alpha: torch.Tensor,
        T: torch.Tensor,
        p: torch.Tensor,
        rho_l: float,
        rho_v: float,
    ) -> torch.Tensor:
        """Compute thermal phase change mass transfer rate.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction of the dispersed phase.
        T : torch.Tensor
            ``(n_cells,)`` temperature.
        p : torch.Tensor
            ``(n_cells,)`` pressure (unused, kept for interface).
        rho_l : float
            Liquid density (unused directly, kept for interface).
        rho_v : float
            Vapour density (unused directly, kept for interface).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mass transfer rate (positive = evaporation).
        """
        # Temperature difference from saturation
        dT = T - self.T_sat

        # Interface heat flux: q = k_eff * dT / delta_x
        q = self.k_eff * dT / self.delta_x

        # Mass transfer: m_dot = q / h_lv
        # Only at cells where interface exists (0 < alpha < 1)
        interface_mask = (alpha > 1e-6) & (alpha < 1.0 - 1e-6)

        m_dot = q / self.h_lv

        # Mask to interface cells only
        m_dot = m_dot * interface_mask.to(dtype=m_dot.dtype)

        # Apply under-relaxation if needed
        if self._m_dot_old is not None and self.relaxation < 1.0:
            m_dot = self.relaxation * m_dot + (1.0 - self.relaxation) * self._m_dot_old

        self._m_dot_old = m_dot.clone()

        return m_dot

    def reset(self) -> None:
        """Reset stored old mass transfer rate."""
        self._m_dot_old = None

    def __repr__(self) -> str:
        return (
            f"ThermalPhaseChange(T_sat={self.T_sat}, h_lv={self.h_lv}, "
            f"k_eff={self.k_eff}, delta_x={self.delta_x})"
        )
