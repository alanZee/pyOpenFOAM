"""
Interphase turbulence interaction models for multiphase flows.

Models the turbulence coupling between phases in Euler-Euler multiphase
frameworks.  The interaction modifies the turbulent kinetic energy (TKE)
and dissipation rate of each phase based on the interphase momentum
exchange.

Models:

- :class:`TurbulenceInteractionModel` — abstract base with RTS registry
- :class:`StandardInteraction` — standard interphase turbulence
  interaction model based on the Lopez de Bertodano (1994) approach

The standard model computes the interphase TKE transfer as:

    S_k = C_ti * K_drag * |U_slip|² * alpha_d * alpha_c

where ``K_drag`` is the interphase drag coefficient, ``U_slip`` is the
slip velocity, ``alpha_d`` is the dispersed phase fraction, and
``alpha_c = 1 - alpha_d`` is the continuous phase fraction.

Usage::

    from pyfoam.multiphase.turbulence_interaction import StandardInteraction

    model = StandardInteraction(C_ti=1.0, sigma_k=0.75)
    S_k = model.compute_k_source(alpha_d, k_c, k_d, U_slip, K_drag)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TurbulenceInteractionModel",
    "StandardInteraction",
]

logger = logging.getLogger(__name__)


class TurbulenceInteractionModel(ABC):
    """Abstract base class for interphase turbulence interaction.

    Subclasses compute the TKE and dissipation rate source/sink terms
    due to the interaction between phases in multiphase flows.

    In OpenFOAM, this corresponds to the ``turbulenceInteraction``
    model in multiphaseEulerFoam / twoPhaseEulerFoam.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceInteractionModel"]]] = {}

    def __init__(
        self,
        C_ti: float = 1.0,
        sigma_k: float = 0.75,
    ) -> None:
        """
        Parameters
        ----------
        C_ti : float
            Turbulence interaction coefficient (default 1.0).
        sigma_k : float
            Turbulent Schmidt number for TKE (default 0.75).
        """
        self.C_ti = C_ti
        self.sigma_k = sigma_k

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register an interaction model under *name*."""

        def decorator(model_cls: Type[TurbulenceInteractionModel]) -> Type[TurbulenceInteractionModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Turbulence interaction model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceInteractionModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown turbulence interaction model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute_k_source(
        self,
        alpha_d: torch.Tensor,
        k_c: torch.Tensor,
        k_d: torch.Tensor,
        U_slip: torch.Tensor,
        K_drag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the TKE source term due to interphase interaction.

        Parameters
        ----------
        alpha_d : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        k_c : torch.Tensor
            Continuous phase TKE ``(n_cells,)``.
        k_d : torch.Tensor
            Dispersed phase TKE ``(n_cells,)``.
        U_slip : torch.Tensor
            Slip velocity magnitude |U_d - U_c| ``(n_cells,)``.
        K_drag : torch.Tensor
            Interphase drag coefficient ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            TKE source term (W/m^3), ``(n_cells,)``.
            Positive = energy transferred TO the continuous phase.
        """

    @abstractmethod
    def compute_epsilon_source(
        self,
        alpha_d: torch.Tensor,
        epsilon_c: torch.Tensor,
        k_c: torch.Tensor,
        k_d: torch.Tensor,
        U_slip: torch.Tensor,
        K_drag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the dissipation rate source term.

        Parameters
        ----------
        alpha_d : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        epsilon_c : torch.Tensor
            Continuous phase dissipation rate ``(n_cells,)``.
        k_c : torch.Tensor
            Continuous phase TKE ``(n_cells,)``.
        k_d : torch.Tensor
            Dispersed phase TKE ``(n_cells,)``.
        U_slip : torch.Tensor
            Slip velocity magnitude ``(n_cells,)``.
        K_drag : torch.Tensor
            Interphase drag coefficient ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Dissipation rate source (W/(m^3*s)), ``(n_cells,)``.
        """


@TurbulenceInteractionModel.register("standardInteraction")
class StandardInteraction(TurbulenceInteractionModel):
    """Standard interphase turbulence interaction model.

    Based on Lopez de Bertodano (1994) for Euler-Euler multiphase.
    The TKE source term is:

        S_k = C_ti * K_drag * |U_slip|^2 * alpha_d * alpha_c

    where alpha_c = 1 - alpha_d.

    The dissipation source is proportional to the TKE source:

        S_eps = C_ti * S_k * epsilon_c / max(k_c, k_min)

    This model is suitable for bubbly flows and particle-laden flows
    where the dispersed phase turbulence is driven by the continuous
    phase through interphase drag.
    """

    def compute_k_source(
        self,
        alpha_d: torch.Tensor,
        k_c: torch.Tensor,
        k_d: torch.Tensor,
        U_slip: torch.Tensor,
        K_drag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TKE source from interphase interaction.

        S_k = C_ti * K_drag * U_slip^2 * alpha_d * alpha_c
        """
        alpha_d_c = alpha_d.clamp(min=0.0, max=1.0)
        alpha_c = (1.0 - alpha_d).clamp(min=0.0, max=1.0)

        S_k = self.C_ti * K_drag * U_slip.pow(2) * alpha_d_c * alpha_c
        return S_k

    def compute_epsilon_source(
        self,
        alpha_d: torch.Tensor,
        epsilon_c: torch.Tensor,
        k_c: torch.Tensor,
        k_d: torch.Tensor,
        U_slip: torch.Tensor,
        K_drag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dissipation source from interphase interaction.

        S_eps = C_ti * S_k * epsilon_c / max(k_c, k_min)
        """
        k_min = torch.tensor(1e-16, dtype=k_c.dtype, device=k_c.device)
        S_k = self.compute_k_source(alpha_d, k_c, k_d, U_slip, K_drag)
        return self.C_ti * S_k * epsilon_c / k_c.clamp(min=k_min)
