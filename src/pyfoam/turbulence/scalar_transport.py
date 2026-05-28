"""
Scalar transport models for turbulence-quantity diffusion.

Provides gradient-diffusion models for the turbulent transport of
scalar quantities (temperature, species, kinetic energy) within the
RANS framework.  These models close the turbulent scalar flux term:

    <u'φ'> = -Γ_t * ∇φ

where Γ_t is the turbulent diffusivity.

Models:
    - :class:`ScalarTransportModel` — abstract base with RTS registry
    - :class:`SGDH` — Simple Gradient Diffusion Hypothesis
    - :class:`GGDH` — Generalized Gradient Diffusion Hypothesis

In OpenFOAM, scalar transport models are used within the
``turbulentTransportModel`` framework to close the turbulent
diffusion terms in transport equations::

    turbulentDiffusivityModel  SGDH;

    SGDHCoeffs
    {
        sigmaT  0.85;    // turbulent Prandtl/Schmidt number
    }

References:
    - SGDH: Reynolds (1975), "Modeling the turbulent transport..."
    - GGDH: Daly & Harlow (1970), "Transport equations in turbulence"
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "ScalarTransportModel",
    "SGDH",
    "GGDH",
]


class ScalarTransportModel(ABC):
    """Abstract base class for turbulent scalar transport models.

    Subclasses implement :meth:`compute_flux` to return the turbulent
    scalar flux vector for each cell.

    Provides an RTS (Run-Time Selection) registry consistent with
    :class:`~pyfoam.turbulence.turbulence_model.TurbulenceModel`.
    """

    _registry: ClassVar[dict[str, Type["ScalarTransportModel"]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a scalar transport model under *name*."""

        def decorator(model_cls: Type[ScalarTransportModel]) -> Type[ScalarTransportModel]:
            if name in cls._registry:
                raise ValueError(
                    f"ScalarTransportModel '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "ScalarTransportModel":
        """Factory: create a scalar transport model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown ScalarTransportModel '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute_flux(
        self,
        grad_phi: torch.Tensor,
        nut: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute turbulent scalar flux.

        Parameters
        ----------
        grad_phi : torch.Tensor
            Scalar gradient ``(n_cells, 3)``.
        nut : torch.Tensor
            Turbulent viscosity ``(n_cells,)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` turbulent scalar flux.
        """
        ...

    @abstractmethod
    def compute_diffusivity(
        self,
        nut: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute turbulent scalar diffusivity.

        Parameters
        ----------
        nut : torch.Tensor
            Turbulent viscosity ``(n_cells,)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` turbulent diffusivity.
        """
        ...


@ScalarTransportModel.register("SGDH")
class SGDH(ScalarTransportModel):
    """Simple Gradient Diffusion Hypothesis.

    The simplest turbulent scalar flux model:

        <u'φ'> = -(nut / sigmaT) * ∇φ

    where sigmaT is the turbulent Prandtl/Schmidt number.

    This is equivalent to the standard eddy diffusivity model where
    the turbulent diffusivity is:

        Γ_t = nut / sigmaT

    Parameters
    ----------
    sigmaT : float
        Turbulent Prandtl/Schmidt number.  Default: 0.85 (typical for heat).
        Lower values (0.7) are used for species transport.
    """

    def __init__(self, sigmaT: float = 0.85) -> None:
        self._sigmaT = sigmaT

    @property
    def sigmaT(self) -> float:
        """Turbulent Prandtl/Schmidt number."""
        return self._sigmaT

    def compute_flux(
        self,
        grad_phi: torch.Tensor,
        nut: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute SGDH turbulent scalar flux.

        <u'φ'> = -(nut / sigmaT) * ∇φ

        Parameters
        ----------
        grad_phi : torch.Tensor
            Scalar gradient ``(n_cells, 3)``.
        nut : torch.Tensor
            Turbulent viscosity ``(n_cells,)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` turbulent scalar flux.
        """
        device = get_device()
        dtype = get_default_dtype()
        grad_phi = grad_phi.to(device=device, dtype=dtype)
        nut = nut.to(device=device, dtype=dtype)

        gamma_t = nut / max(self._sigmaT, 1e-20)
        return -gamma_t.unsqueeze(-1) * grad_phi

    def compute_diffusivity(
        self,
        nut: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute SGDH turbulent diffusivity: Gamma_t = nut / sigmaT.

        Parameters
        ----------
        nut : torch.Tensor
            Turbulent viscosity ``(n_cells,)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` turbulent diffusivity.
        """
        device = get_device()
        dtype = get_default_dtype()
        nut = nut.to(device=device, dtype=dtype)
        return nut / max(self._sigmaT, 1e-20)


@ScalarTransportModel.register("GGDH")
class GGDH(ScalarTransportModel):
    """Generalized Gradient Diffusion Hypothesis.

    An improved scalar flux model that accounts for the anisotropy
    of the turbulent scalar flux:

        <u'φ'> = -C_T * k/epsilon * <u'u'> * ∇φ

    where <u'u'> is the Reynolds stress tensor, k is the turbulent
    kinetic energy, and epsilon is the dissipation rate.

    For the isotropic approximation (Boussinesq):

        <u'u'> ≈ (2/3) k I - nut * S

    where S is the mean strain rate tensor.  This reduces to:

        <u'φ'> = -C_T * (k/epsilon) * [(2/3)k * I - nut * S] * ∇φ

    The default coefficient C_T = 0.3 is from Daly & Harlow (1970).

    Parameters
    ----------
    C_T : float
        Model coefficient.  Default: 0.3 (Daly & Harlow 1970).
    sigmaT : float
        Fallback Prandtl/Schmidt number for when k/epsilon is not
        available.  Default: 0.85.
    """

    def __init__(
        self,
        C_T: float = 0.3,
        sigmaT: float = 0.85,
    ) -> None:
        self._C_T = C_T
        self._sigmaT = sigmaT

    @property
    def C_T(self) -> float:
        """Model coefficient."""
        return self._C_T

    @property
    def sigmaT(self) -> float:
        """Fallback turbulent Prandtl/Schmidt number."""
        return self._sigmaT

    def compute_flux(
        self,
        grad_phi: torch.Tensor,
        nut: torch.Tensor,
        n_cells: int,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute GGDH turbulent scalar flux.

        <u'φ'> = -C_T * (k/epsilon) * nut * ∇φ

        (simplified isotropic form; full tensor form requires the
        Reynolds stress tensor)

        Parameters
        ----------
        grad_phi : torch.Tensor
            Scalar gradient ``(n_cells, 3)``.
        nut : torch.Tensor
            Turbulent viscosity ``(n_cells,)``.
        n_cells : int
            Number of cells.
        k : torch.Tensor, optional
            Turbulent kinetic energy ``(n_cells,)``. Required for the
            full GGDH model; if missing, falls back to SGDH.
        epsilon : torch.Tensor, optional
            Turbulent dissipation rate ``(n_cells,)``. Required for
            the full GGDH model.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` turbulent scalar flux.
        """
        device = get_device()
        dtype = get_default_dtype()
        grad_phi = grad_phi.to(device=device, dtype=dtype)
        nut = nut.to(device=device, dtype=dtype)

        # Fall back to SGDH if k or epsilon not provided
        if k is None or epsilon is None:
            gamma_t = nut / max(self._sigmaT, 1e-20)
            return -gamma_t.unsqueeze(-1) * grad_phi

        k = k.to(device=device, dtype=dtype)
        epsilon = epsilon.to(device=device, dtype=dtype)

        # GGDH: Gamma_t = C_T * k^2 / epsilon (enhanced diffusivity)
        k_safe = k.clamp(min=1e-10)
        eps_safe = epsilon.clamp(min=1e-10)

        gamma_ggdh = self._C_T * k_safe.pow(2) / eps_safe

        # Blend with nut-based diffusivity for robustness
        gamma_sgdh = nut / max(self._sigmaT, 1e-20)
        gamma_t = torch.max(gamma_ggdh, gamma_sgdh)

        return -gamma_t.unsqueeze(-1) * grad_phi

    def compute_diffusivity(
        self,
        nut: torch.Tensor,
        n_cells: int,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute GGDH turbulent diffusivity.

        Gamma_t = max(C_T * k^2 / epsilon, nut / sigmaT)

        Parameters
        ----------
        nut : torch.Tensor
            Turbulent viscosity ``(n_cells,)``.
        n_cells : int
            Number of cells.
        k : torch.Tensor, optional
            Turbulent kinetic energy ``(n_cells,)``.
        epsilon : torch.Tensor, optional
            Turbulent dissipation rate ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` turbulent diffusivity.
        """
        device = get_device()
        dtype = get_default_dtype()
        nut = nut.to(device=device, dtype=dtype)

        # Fallback: SGDH
        gamma_sgdh = nut / max(self._sigmaT, 1e-20)

        if k is None or epsilon is None:
            return gamma_sgdh

        k = k.to(device=device, dtype=dtype)
        epsilon = epsilon.to(device=device, dtype=dtype)

        k_safe = k.clamp(min=1e-10)
        eps_safe = epsilon.clamp(min=1e-10)

        gamma_ggdh = self._C_T * k_safe.pow(2) / eps_safe

        return torch.max(gamma_ggdh, gamma_sgdh)
