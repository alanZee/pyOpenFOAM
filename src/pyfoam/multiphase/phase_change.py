"""
Phase change models for boiling and condensation.

Provides the abstract base class and concrete implementations for
interfacial mass transfer due to phase change (boiling / condensation).

Models:
- :class:`PhaseChangeModel` — abstract base with RTS registry
- :class:`LeeModel` — Lee empirical phase change model
- :class:`SchnerrSauerEnhanced` — enhanced Schnerr-Sauer with improved
  convergence (under-relaxation, pressure limiting, alpha clipping)

Usage::

    from pyfoam.multiphase.phase_change import PhaseChangeModel

    model = PhaseChangeModel.create("Lee", T_sat=373.15)
    m_dot = model.compute_mass_transfer(alpha, T, p, rho_l, rho_v)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "PhaseChangeModel",
    "LeeModel",
    "SchnerrSauerEnhanced",
]

logger = logging.getLogger(__name__)


class PhaseChangeModel(ABC):
    """Abstract base class for phase change models.

    Subclasses must implement :meth:`compute_mass_transfer`.

    RTS (Run-Time Selection) registry allows string-based lookup::

        model = PhaseChangeModel.create("Lee", T_sat=373.15)
    """

    _registry: ClassVar[dict[str, Type[PhaseChangeModel]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a phase change model under *name*."""

        def decorator(model_cls: Type[PhaseChangeModel]) -> Type[PhaseChangeModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Phase change model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> PhaseChangeModel:
        """Create a phase change model by name.

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
                f"Unknown phase change model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return list of registered model names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute_mass_transfer(
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
            Positive = evaporation, negative = condensation.
        """


# ======================================================================
# Lee model
# ======================================================================


@PhaseChangeModel.register("Lee")
class LeeModel(PhaseChangeModel):
    """Lee empirical phase change model.

    A simple and widely used model for evaporation / condensation:

        When T > T_sat:
            m_dot = r_evap * alpha_l * rho_l * (T - T_sat) / T_sat
        When T < T_sat:
            m_dot = -r_cond * alpha_v * rho_v * (T_sat - T) / T_sat

    where r_evap and r_cond are empirical relaxation rate coefficients.

    Parameters
    ----------
    T_sat : float
        Saturation temperature (K). Default: 373.15 (water at 1 atm).
    r_evap : float
        Evaporation rate coefficient (1/s). Default: 0.1.
    r_cond : float
        Condensation rate coefficient (1/s). Default: 0.1.
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

    def compute_mass_transfer(
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
            ``(n_cells,)`` pressure (unused by Lee model, kept for interface).
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
            f"LeeModel(T_sat={self.T_sat}, r_evap={self.r_evap}, "
            f"r_cond={self.r_cond})"
        )


# ======================================================================
# Enhanced Schnerr-Sauer model
# ======================================================================


@PhaseChangeModel.register("SchnerrSauerEnhanced")
class SchnerrSauerEnhanced(PhaseChangeModel):
    """Enhanced Schnerr-Sauer phase change model.

    Based on the Rayleigh-Plesset equation for bubble dynamics with
    convergence enhancements:

    - **Alpha clipping**: bounds volume fraction to ``[alpha_min, 1 - alpha_min]``
      to prevent singularities.
    - **Pressure limiting**: caps |p - p_v| to ``p_max`` to bound source terms.
    - **Under-relaxation**: provides :meth:`relax` for inter-iteration smoothing.

    Core physics (identical to standard Schnerr-Sauer):

        R_b = (3 alpha / (4 pi n_b))^(1/3)
        m_dot = 3 rho_v alpha (1 - alpha) / R_b
                * sign(p_v - p) * sqrt(2/3 |p - p_v| / rho_l)

    Parameters
    ----------
    n_b : float
        Bubble number density (m^-3). Default: 1e13.
    p_v : float
        Vapour pressure (Pa). Default: 2300.0.
    alpha_min : float
        Minimum volume fraction clip. Default: 1e-6.
    p_max : float
        Maximum |p - p_v| (Pa). Default: 1e5.
    relaxation : float
        Under-relaxation factor in (0, 1]. Default: 1.0 (no relaxation).
    """

    def __init__(
        self,
        n_b: float = 1e13,
        p_v: float = 2300.0,
        alpha_min: float = 1e-6,
        p_max: float = 1e5,
        relaxation: float = 1.0,
    ) -> None:
        self.n_b = n_b
        self.p_v = p_v
        self.alpha_min = alpha_min
        self.p_max = p_max
        self.relaxation = relaxation
        self._m_dot_old: torch.Tensor | None = None

    def compute_mass_transfer(
        self,
        alpha: torch.Tensor,
        T: torch.Tensor,
        p: torch.Tensor,
        rho_l: float,
        rho_v: float,
    ) -> torch.Tensor:
        """Compute enhanced Schnerr-Sauer mass transfer rate.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` vapour volume fraction.
        T : torch.Tensor
            ``(n_cells,)`` temperature (unused, kept for interface).
        p : torch.Tensor
            ``(n_cells,)`` pressure.
        rho_l : float
            Liquid density.
        rho_v : float
            Vapour density.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mass transfer rate (positive = evaporation).
        """
        # Alpha clipping
        alpha = alpha.clamp(min=self.alpha_min, max=1.0 - self.alpha_min)

        # Bubble radius
        R_b = (3.0 * alpha / (4.0 * math.pi * self.n_b)).pow(1.0 / 3.0)
        R_b = R_b.clamp(min=1e-10)

        # Pressure difference with limiting
        p_diff = (p - self.p_v).clamp(-self.p_max, self.p_max)

        # Mass transfer
        # p < p_v: evaporation (m_dot > 0)
        # p > p_v: condensation (m_dot < 0)
        sign = -torch.sign(p_diff)
        m_dot = (
            3.0 * rho_v * alpha * (1.0 - alpha) / R_b
            * sign * (2.0 / 3.0 * p_diff.abs() / rho_l).sqrt()
        )

        return m_dot

    def relax(
        self,
        m_dot_new: torch.Tensor,
        m_dot_old: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply under-relaxation to the mass transfer rate.

        m_dot = relaxation * m_dot_new + (1 - relaxation) * m_dot_old

        Parameters
        ----------
        m_dot_new : torch.Tensor
            Newly computed mass transfer rate.
        m_dot_old : torch.Tensor or None
            Previous mass transfer rate. If None, uses the stored value.

        Returns
        -------
        torch.Tensor
            Relaxed mass transfer rate.
        """
        if m_dot_old is None:
            m_dot_old = self._m_dot_old

        if m_dot_old is None or self.relaxation >= 1.0:
            result = m_dot_new
        else:
            result = self.relaxation * m_dot_new + (1.0 - self.relaxation) * m_dot_old

        self._m_dot_old = result.clone()
        return result

    def reset(self) -> None:
        """Reset stored old mass transfer rate."""
        self._m_dot_old = None

    def __repr__(self) -> str:
        return (
            f"SchnerrSauerEnhanced(n_b={self.n_b}, p_v={self.p_v}, "
            f"alpha_min={self.alpha_min}, relaxation={self.relaxation})"
        )
