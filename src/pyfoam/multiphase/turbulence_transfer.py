"""
Turbulence transfer models for multiphase Euler-Euler flows.

Models the turbulent momentum exchange between phases in multiphase
simulations. In Euler-Euler framework, each phase has its own turbulence
model, and the interphase turbulence transfer couples them.

Models:

- :class:`TurbulenceTransferModel` — abstract base with RTS registry
- :class:`ContinuousTurbulenceTransfer` — turbulence transfer for the
  continuous (carrier) phase
- :class:`DispersedTurbulenceTransfer` — turbulence transfer for the
  dispersed (particle/bubble/droplet) phase

The continuous phase turbulence is modified by the presence of dispersed
phase via additional production and dissipation terms. The dispersed
phase turbulence is driven by the continuous phase via drag-induced
turbulent kinetic energy transfer.

Usage::

    from pyfoam.multiphase.turbulence_transfer import (
        ContinuousTurbulenceTransfer,
        DispersedTurbulenceTransfer,
    )

    ct = ContinuousTurbulenceTransfer()
    k_transfer = ct.compute_k_transfer(alpha_d, k_c, k_d, U_slip)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TurbulenceTransferModel",
    "ContinuousTurbulenceTransfer",
    "DispersedTurbulenceTransfer",
]

logger = logging.getLogger(__name__)


class TurbulenceTransferModel(ABC):
    """Abstract base class for interphase turbulence transfer.

    Subclasses compute the turbulent kinetic energy (TKE) exchange
    rate between phases in Euler-Euler multiphase flows.

    In OpenFOAM, this corresponds to the ``phaseTransferTurbulence``
    model in multiphaseEulerFoam.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceTransferModel"]]] = {}

    def __init__(
        self,
        C_t: float = 1.0,
        sigma_t: float = 0.9,
    ) -> None:
        """
        Parameters
        ----------
        C_t : float
            Turbulence transfer coefficient (default 1.0).
        sigma_t : float
            Turbulent Schmidt/Prandtl number for TKE diffusion (default 0.9).
        """
        self.C_t = C_t
        self.sigma_t = sigma_t

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a transfer model under *name*."""

        def decorator(model_cls: Type[TurbulenceTransferModel]) -> Type[TurbulenceTransferModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Turbulence transfer model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceTransferModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown turbulence transfer model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute_k_transfer(
        self,
        alpha_d: torch.Tensor,
        k_c: torch.Tensor,
        k_d: torch.Tensor,
        U_slip: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the TKE transfer rate between phases.

        Parameters
        ----------
        alpha_d : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        k_c : torch.Tensor
            Continuous phase turbulent kinetic energy ``(n_cells,)``.
        k_d : torch.Tensor
            Dispersed phase turbulent kinetic energy ``(n_cells,)``.
        U_slip : torch.Tensor
            Slip velocity magnitude |U_d - U_c| ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            TKE transfer rate (W/m³), ``(n_cells,)``.
            Positive = energy transferred TO the dispersed phase.
        """

    @abstractmethod
    def compute_epsilon_transfer(
        self,
        alpha_d: torch.Tensor,
        epsilon_c: torch.Tensor,
        k_c: torch.Tensor,
        k_d: torch.Tensor,
        U_slip: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the dissipation rate transfer between phases.

        Parameters
        ----------
        alpha_d : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        epsilon_c : torch.Tensor
            Continuous phase dissipation rate ``(n_cells,)``.
        k_c : torch.Tensor
            Continuous phase turbulent kinetic energy ``(n_cells,)``.
        k_d : torch.Tensor
            Dispersed phase turbulent kinetic energy ``(n_cells,)``.
        U_slip : torch.Tensor
            Slip velocity magnitude ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Dissipation rate transfer (W/(m³·s)), ``(n_cells,)``.
        """


@TurbulenceTransferModel.register("continuousTurbulenceTransfer")
class ContinuousTurbulenceTransfer(TurbulenceTransferModel):
    """Turbulence transfer model for the continuous (carrier) phase.

    The continuous phase receives turbulence modification from the
    dispersed phase via:

    - Additional TKE production from drag-induced velocity fluctuations:
        P_transfer = C_t * alpha_d * K_drag * |U_slip|²

    - Additional dissipation from interphase interaction:
        epsilon_transfer = C_t * alpha_d * epsilon_c / k_c * k_transfer

    where K_drag is the drag coefficient and |U_slip| is the slip velocity.
    """

    def compute_k_transfer(
        self,
        alpha_d: torch.Tensor,
        k_c: torch.Tensor,
        k_d: torch.Tensor,
        U_slip: torch.Tensor,
    ) -> torch.Tensor:
        """TKE transfer for continuous phase.

        The transfer rate represents the TKE production due to the
        dispersed phase drag interaction:

            S_k = C_t * alpha_d * (1 - alpha_d) * |U_slip|² / tau_t

        where tau_t = max(k_c, eps_min) / max(epsilon_c, eps_min)
        is the turbulent time scale. For simplicity we use:

            S_k = C_t * alpha_d * (1 - alpha_d) * |U_slip|³ / L_t

        A simplified model: S_k = C_t * alpha_d * (1 - alpha_d) * U_slip^2
        """
        alpha_c = (1.0 - alpha_d).clamp(min=0.0, max=1.0)
        alpha_d_c = alpha_d.clamp(min=0.0, max=1.0)

        # TKE production from slip
        k_transfer = self.C_t * alpha_d_c * alpha_c * U_slip.pow(2)

        return k_transfer

    def compute_epsilon_transfer(
        self,
        alpha_d: torch.Tensor,
        epsilon_c: torch.Tensor,
        k_c: torch.Tensor,
        k_d: torch.Tensor,
        U_slip: torch.Tensor,
    ) -> torch.Tensor:
        """Dissipation transfer for continuous phase.

        Proportional to the TKE transfer and local dissipation rate:

            S_eps = C_t * S_k * epsilon_c / max(k_c, k_min)
        """
        k_min = torch.tensor(1e-16, dtype=k_c.dtype, device=k_c.device)
        k_transfer = self.compute_k_transfer(alpha_d, k_c, k_d, U_slip)

        eps_transfer = self.C_t * k_transfer * epsilon_c / k_c.clamp(min=k_min)

        return eps_transfer


@TurbulenceTransferModel.register("dispersedTurbulenceTransfer")
class DispersedTurbulenceTransfer(TurbulenceTransferModel):
    """Turbulence transfer model for the dispersed phase.

    The dispersed phase turbulence is driven by:

    - TKE transfer from the continuous phase via drag:
        S_k,d = C_t * K_drag * (k_c/alpha_c - k_d/alpha_d) * alpha_d

    - Dissipation transfer proportional to the k transfer:
        S_eps,d = C_t * S_k,d * epsilon_d / max(k_d, k_min)

    This is based on the Simonin model for bubble/droplet turbulence.
    """

    def compute_k_transfer(
        self,
        alpha_d: torch.Tensor,
        k_c: torch.Tensor,
        k_d: torch.Tensor,
        U_slip: torch.Tensor,
    ) -> torch.Tensor:
        """TKE transfer for dispersed phase.

        Based on Simonin (1991): the dispersed phase extracts energy
        from the continuous phase turbulence via drag:

            S_k,d = C_t * alpha_d * (1 - alpha_d) * (k_c - k_d * (1-alpha_d)/alpha_d_safe)
                / tau_t_drag

        Simplified:
            S_k,d = C_t * alpha_d * (1 - alpha_d) * (k_c * alpha_d - k_d * (1 - alpha_d))
                / (alpha_d_safe * max(alpha_c, 0.01))
        """
        alpha_c = (1.0 - alpha_d).clamp(min=0.01, max=1.0)
        alpha_d_c = alpha_d.clamp(min=0.0, max=1.0)

        # Energy exchange: continuous-to-dispersed when k_c > k_d
        k_exchange = (k_c - k_d).clamp(min=0.0)
        k_transfer = self.C_t * alpha_d_c * alpha_c * k_exchange

        return k_transfer

    def compute_epsilon_transfer(
        self,
        alpha_d: torch.Tensor,
        epsilon_c: torch.Tensor,
        k_c: torch.Tensor,
        k_d: torch.Tensor,
        U_slip: torch.Tensor,
    ) -> torch.Tensor:
        """Dissipation transfer for dispersed phase.

        Proportional to the TKE transfer and the local turbulent
        time scale:

            S_eps,d = S_k,d / tau_t

        where tau_t = max(k_d, eps_min) / max(epsilon_c, eps_min).
        """
        k_min = torch.tensor(1e-16, dtype=k_c.dtype, device=k_c.device)
        eps_min = torch.tensor(1e-16, dtype=epsilon_c.dtype, device=epsilon_c.device)

        k_transfer = self.compute_k_transfer(alpha_d, k_c, k_d, U_slip)

        tau_t = k_d.clamp(min=k_min) / epsilon_c.clamp(min=eps_min)
        eps_transfer = k_transfer / tau_t.clamp(min=1e-10)

        return eps_transfer
