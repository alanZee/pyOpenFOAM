"""
Lift force models for multiphase Euler-Euler flows.

Provides an abstract base class and two standard lift-force correlations:
Tomiyama lift for bubbles and Saffman lift for wall-bounded particulate flows.

Models:

- :class:`LiftModel` — abstract base with RTS registry
- :class:`TomiyamaLift` — Tomiyama lift force for bubbles
- :class:`SaffmanLift` — Saffman lift force (wall-bounded shear-induced)

All models register themselves via ``@LiftModel.register(name)``
and can be instantiated at run-time via ``LiftModel.create(name, ...)``.

Usage::

    from pyfoam.multiphase.lift_models import LiftModel

    lift = LiftModel.create("tomiyama", d=1e-3, rho_c=998.0, rho_d=1.225, sigma=0.072)
    F_L = lift.compute(alpha, U_rel, vorticity)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "LiftModel",
    "TomiyamaLift",
    "SaffmanLift",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class LiftModel(ABC):
    """Abstract base class for lift force models.

    Subclasses must implement :meth:`compute` which returns the lift
    force per unit volume.

    RTS (Run-Time Selection) registry allows string-based lookup::

        @LiftModel.register("tomiyama")
        class TomiyamaLift(LiftModel):
            ...

        lift = LiftModel.create("tomiyama", d=1e-3, rho_c=998.0, ...)
    """

    _registry: ClassVar[dict[str, Type[LiftModel]]] = {}

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a lift model class under *name*."""

        def decorator(model_cls: Type[LiftModel]) -> Type[LiftModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Lift model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> LiftModel:
        """Factory: create a lift model instance by registered *name*.

        Parameters
        ----------
        name : str
            Registered model type name.
        **kwargs
            Constructor arguments forwarded to the model class.

        Returns
        -------
        LiftModel
            Instantiated lift model.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown lift model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered lift model type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(
        self,
        alpha: torch.Tensor,
        U_rel: torch.Tensor,
        vorticity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute lift force per unit volume.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_rel : torch.Tensor
            Relative velocity ``(n_cells, 3)``.
        vorticity : torch.Tensor
            Continuous-phase vorticity ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Lift force per unit volume ``(n_cells, 3)``.
        """


# ---------------------------------------------------------------------------
# Concrete models
# ---------------------------------------------------------------------------


@LiftModel.register("tomiyama")
class TomiyamaLift(LiftModel):
    """Tomiyama lift force for bubbles in liquid.

    Based on the Tomiyama et al. (2002) correlation for the lift
    coefficient as a function of the Eotvos number::

        C_L = 0.12 * f(Eo)

    where f(Eo) depends on the bubble shape regime:
        - Eo < 4:      f = 1.0           (spherical)
        - 4 <= Eo < 10: f = 1 - 0.001*(Eo-4)^2  (ellipsoidal)
        - Eo >= 10:    f = 0.3           (cap/slug)

    The lift force per unit volume is::

        F_L = C_L * rho_c * alpha * (U_rel x omega)

    Parameters
    ----------
    d : float
        Bubble diameter (m).
    rho_c : float
        Continuous phase (liquid) density (kg/m^3).
    rho_d : float
        Dispersed phase (gas) density (kg/m^3).
    sigma : float
        Surface tension coefficient (N/m).  Default: 0.072 (air-water).
    """

    def __init__(
        self,
        d: float,
        rho_c: float,
        rho_d: float,
        sigma: float = 0.072,
    ) -> None:
        self.d = d
        self.rho_c = rho_c
        self.rho_d = rho_d
        self.sigma = sigma

    @property
    def eotvos_number(self) -> float:
        """Eotvos number: Eo = g * |rho_c - rho_d| * d^2 / sigma."""
        g = 9.81
        return g * abs(self.rho_c - self.rho_d) * self.d ** 2 / self.sigma

    def _lift_coefficient(self) -> float:
        """Compute Tomiyama lift coefficient C_L."""
        Eo = self.eotvos_number
        if Eo < 4.0:
            f_Eo = 1.0
        elif Eo < 10.0:
            f_Eo = 1.0 - 0.001 * (Eo - 4.0) ** 2
        else:
            f_Eo = 0.3
        return 0.12 * f_Eo

    def compute(
        self,
        alpha: torch.Tensor,
        U_rel: torch.Tensor,
        vorticity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Tomiyama lift force per unit volume.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_rel : torch.Tensor
            Relative velocity ``(n_cells, 3)``.
        vorticity : torch.Tensor
            Continuous-phase vorticity ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Lift force per unit volume ``(n_cells, 3)``.
        """
        C_L = self._lift_coefficient()

        # F_L = C_L * rho_c * alpha * (U_rel x omega)
        cross = torch.cross(U_rel, vorticity, dim=1)
        return C_L * self.rho_c * alpha.unsqueeze(-1) * cross


@LiftModel.register("saffman")
class SaffmanLift(LiftModel):
    """Saffman lift force for particles in shear flows.

    The Saffman (1965, 1968) lift force acts on a particle in a
    velocity gradient (shear flow).  It is directed toward the
    high-velocity side of the shear layer and is important for
    particle migration in wall-bounded flows.

    The lift force per unit volume is::

        F_L = C_S * rho_c * alpha * sqrt(nu_c * |grad(U)|)
              * (U_rel x omega) / |omega|

    where C_S is a model constant (default 1.615 for low Re).

    Parameters
    ----------
    d : float
        Particle diameter (m).
    rho_c : float
        Continuous phase density (kg/m^3).
    mu_c : float
        Continuous phase dynamic viscosity (Pa*s).
    C_S : float
        Saffman lift coefficient.  Default: 1.615.
    """

    def __init__(
        self,
        d: float,
        rho_c: float,
        mu_c: float,
        C_S: float = 1.615,
    ) -> None:
        self.d = d
        self.rho_c = rho_c
        self.mu_c = mu_c
        self.C_S = C_S

    @property
    def nu_c(self) -> float:
        """Kinematic viscosity of the continuous phase (m^2/s)."""
        return self.mu_c / self.rho_c

    def compute(
        self,
        alpha: torch.Tensor,
        U_rel: torch.Tensor,
        vorticity: torch.Tensor,
        strain_magnitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Saffman lift force per unit volume.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_rel : torch.Tensor
            Relative velocity ``(n_cells, 3)``.
        vorticity : torch.Tensor
            Continuous-phase vorticity magnitude ``(n_cells,)`` or
            vorticity vector ``(n_cells, 3)``.  If vector, the norm
            is used.
        strain_magnitude : torch.Tensor, optional
            |S| = sqrt(2 * S_ij * S_ij) ``(n_cells,)``.  When not
            provided, the vorticity magnitude is used as a proxy for
            the shear rate.

        Returns
        -------
        torch.Tensor
            Lift force per unit volume ``(n_cells, 3)``.
        """
        device = U_rel.device
        dtype = U_rel.dtype

        # Compute vorticity magnitude
        if vorticity.dim() == 2 and vorticity.shape[1] == 3:
            omega_mag = vorticity.norm(dim=1).clamp(min=1e-30)
        else:
            omega_mag = vorticity.clamp(min=1e-30)

        # Shear rate proxy: use strain_magnitude if given, else vorticity
        if strain_magnitude is not None:
            shear = strain_magnitude.clamp(min=1e-30)
        else:
            shear = omega_mag

        # Saffman prefactor: C_S * rho_c * alpha * sqrt(nu_c * shear)
        nu_c = torch.tensor(self.nu_c, dtype=dtype, device=device)
        prefactor = (
            self.C_S * self.rho_c * alpha * torch.sqrt(nu_c * shear)
        )

        # Cross product direction (U_rel x omega)
        if vorticity.dim() == 2 and vorticity.shape[1] == 3:
            cross = torch.cross(U_rel, vorticity, dim=1)
            # Normalize by |omega| to get unit direction
            direction = cross / omega_mag.unsqueeze(-1)
        else:
            # Scalar vorticity: U_rel cross z-hat (assuming 2D shear)
            omega_vec = torch.zeros_like(U_rel)
            omega_vec[:, 2] = vorticity
            cross = torch.cross(U_rel, omega_vec, dim=1)
            direction = cross / omega_mag.unsqueeze(-1)

        return prefactor.unsqueeze(-1) * direction
