"""
Drift-flux models for multiphase drift-flux simulations.

Provides relative velocity (slip) models for the drift-flux approach,
where the dispersed phase velocity is expressed as:

    U_d = U_m + U_slip

where U_m is the mixture velocity and U_slip is the relative (slip)
velocity of the dispersed phase.

Models:
    - :class:`DriftFluxModel` -- abstract base with RTS registry
    - :class:`SimpleDriftFlux` -- algebraic slip model (Stokes settling)
    - :class:`GeneralDriftFlux` -- general drift-flux with relative velocity

In OpenFOAM, drift-flux models are used in ``incompressibleDriftFluxFoam``
and selected via the ``driftFluxModel`` entry in ``phaseProperties``::

    driftFluxModel  simple;

    simpleCoeffs
    {
        V0      (0 0 -0.1);    // Slip velocity coefficient
        alphaMax 0.6;
    }
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "DriftFluxModel",
    "SimpleDriftFlux",
    "GeneralDriftFlux",
]


class DriftFluxModel(ABC):
    """Abstract base class for drift-flux (slip velocity) models.

    Subclasses implement :meth:`compute_slip_velocity` to return the
    local relative velocity between the dispersed and continuous phases.

    Provides an RTS (Run-Time Selection) registry consistent with
    :class:`~pyfoam.boundary.boundary_condition.BoundaryCondition` and
    :class:`~pyfoam.multiphase.bubble_models.BubbleModel`.
    """

    _registry: ClassVar[dict[str, Type["DriftFluxModel"]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a drift-flux model under *name*."""

        def decorator(model_cls: Type[DriftFluxModel]) -> Type[DriftFluxModel]:
            if name in cls._registry:
                raise ValueError(
                    f"DriftFluxModel '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "DriftFluxModel":
        """Factory: create a drift-flux model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown DriftFluxModel '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered drift-flux model names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute_slip_velocity(
        self,
        alpha: torch.Tensor,
        rho_d: float,
        rho_c: float,
        mu_c: float,
        gravity: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the slip (relative) velocity for each cell.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.
        rho_d : float
            Density of the dispersed (particle) phase.
        rho_c : float
            Density of the continuous (carrier) phase.
        mu_c : float
            Dynamic viscosity of the continuous phase.
        gravity : torch.Tensor
            Gravity vector ``(3,)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` slip velocity vector.
        """
        ...

    @abstractmethod
    def compute_drift_flux(
        self,
        alpha: torch.Tensor,
        U_slip: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the drift flux: alpha * (1 - alpha) * U_slip.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.
        U_slip : torch.Tensor
            Slip velocity ``(n_cells, 3)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` drift flux vector.
        """
        ...


@DriftFluxModel.register("simple")
class SimpleDriftFlux(DriftFluxModel):
    """Algebraic slip model (Stokes settling).

    Computes the slip velocity using the Stokes drag law for
    spherical particles:

        U_slip = (rho_d - rho_c) * g * d^2 / (18 * mu_c)

    where:
        rho_d = dispersed-phase density
        rho_c = continuous-phase density
        g = gravity vector
        d = particle diameter
        mu_c = continuous-phase dynamic viscosity

    The drift flux is then:

        J = alpha * (1 - alpha) * U_slip

    The alpha*(1-alpha) factor ensures zero flux at alpha=0 and alpha=1.

    Parameters
    ----------
    d : float
        Particle/bubble diameter (m). Default: 0.001 (1 mm).
    alpha_max : float
        Maximum packing volume fraction. Default: 0.6.
    """

    def __init__(
        self,
        d: float = 0.001,
        alpha_max: float = 0.6,
    ) -> None:
        self._d = d
        self._alpha_max = alpha_max

    @property
    def d(self) -> float:
        """Particle diameter (m)."""
        return self._d

    @property
    def alpha_max(self) -> float:
        """Maximum packing fraction."""
        return self._alpha_max

    def compute_slip_velocity(
        self,
        alpha: torch.Tensor,
        rho_d: float,
        rho_c: float,
        mu_c: float,
        gravity: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Stokes settling slip velocity.

        U_slip = (rho_d - rho_c) * g * d^2 / (18 * mu_c)

        The slip velocity is uniform across all cells (independent of alpha)
        and directed along the gravity vector.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.
        rho_d : float
            Dispersed-phase density.
        rho_c : float
            Continuous-phase density.
        mu_c : float
            Continuous-phase dynamic viscosity.
        gravity : torch.Tensor
            Gravity vector ``(3,)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` slip velocity vector.
        """
        device = get_device()
        dtype = get_default_dtype()
        gravity = gravity.to(device=device, dtype=dtype)

        # Stokes settling: scalar magnitude
        d_rho = rho_d - rho_c
        slip_mag = d_rho * self._d ** 2 / (18.0 * max(mu_c, 1e-30))

        # Slip velocity: uniform, in the gravity direction
        U_slip = slip_mag * gravity.unsqueeze(0).expand(n_cells, -1)

        return U_slip

    def compute_drift_flux(
        self,
        alpha: torch.Tensor,
        U_slip: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute drift flux: alpha * (1 - alpha) * U_slip.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.
        U_slip : torch.Tensor
            Slip velocity ``(n_cells, 3)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` drift flux vector.
        """
        device = get_device()
        dtype = get_default_dtype()
        alpha = alpha.to(device=device, dtype=dtype)

        # Clamp alpha to [0, alpha_max]
        alpha_clamped = alpha.clamp(min=0.0, max=self._alpha_max)

        # Drift flux: alpha * (1 - alpha) * U_slip
        factor = (alpha_clamped * (1.0 - alpha_clamped)).unsqueeze(1)
        return factor * U_slip


@DriftFluxModel.register("general")
class GeneralDriftFlux(DriftFluxModel):
    """General drift-flux model with relative velocity correlation.

    Implements a more general drift-flux model where the relative
    velocity depends on the local volume fraction and flow conditions:

        U_slip = V_0 * (1 - alpha)^n

    where:
        V_0 = terminal velocity coefficient
        n = Richardson-Zaki exponent (default 2.0 for bubbles)

    The drift flux includes the full Zuber-Findlay formulation:

        J = C_0 * alpha * U_m + alpha * V_0 * (1 - alpha)^n

    where C_0 is the distribution parameter (default 1.0).

    Parameters
    ----------
    V0 : list[float]
        Terminal velocity vector [V0x, V0y, V0z] (m/s).
        Default: [0, 0, -0.1] (downward settling).
    n_exp : float
        Richardson-Zaki exponent. Default: 2.0.
    C0 : float
        Distribution parameter. Default: 1.0.
    alpha_max : float
        Maximum packing volume fraction. Default: 0.6.
    """

    def __init__(
        self,
        V0: list[float] | None = None,
        n_exp: float = 2.0,
        C0: float = 1.0,
        alpha_max: float = 0.6,
    ) -> None:
        self._V0 = V0 if V0 is not None else [0.0, 0.0, -0.1]
        self._n_exp = n_exp
        self._C0 = C0
        self._alpha_max = alpha_max

    @property
    def V0(self) -> list[float]:
        """Terminal velocity vector."""
        return self._V0

    @property
    def n_exp(self) -> float:
        """Richardson-Zaki exponent."""
        return self._n_exp

    @property
    def C0(self) -> float:
        """Distribution parameter."""
        return self._C0

    @property
    def alpha_max(self) -> float:
        """Maximum packing fraction."""
        return self._alpha_max

    def compute_slip_velocity(
        self,
        alpha: torch.Tensor,
        rho_d: float,
        rho_c: float,
        mu_c: float,
        gravity: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute relative velocity using Richardson-Zaki correlation.

        U_slip = V_0 * (1 - alpha)^n

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.
        rho_d : float
            Dispersed-phase density (unused in this model).
        rho_c : float
            Continuous-phase density (unused in this model).
        mu_c : float
            Continuous-phase viscosity (unused in this model).
        gravity : torch.Tensor
            Gravity vector (unused in this model; V0 is specified directly).
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` slip velocity vector.
        """
        device = get_device()
        dtype = get_default_dtype()
        alpha = alpha.to(device=device, dtype=dtype)

        alpha_clamped = alpha.clamp(min=0.0, max=self._alpha_max)

        # Richardson-Zaki correction: (1 - alpha)^n
        rz_factor = (1.0 - alpha_clamped).pow(self._n_exp)

        V0_tensor = torch.tensor(self._V0, device=device, dtype=dtype)

        # U_slip = V_0 * (1 - alpha)^n
        U_slip = rz_factor.unsqueeze(1) * V0_tensor.unsqueeze(0)

        return U_slip

    def compute_drift_flux(
        self,
        alpha: torch.Tensor,
        U_slip: torch.Tensor,
        n_cells: int,
        U_m: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute drift flux using Zuber-Findlay formulation.

        J = C_0 * alpha * U_m + alpha * (1 - alpha) * U_slip

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.
        U_slip : torch.Tensor
            Slip velocity ``(n_cells, 3)``.
        n_cells : int
            Number of cells.
        U_m : torch.Tensor, optional
            Mixture velocity ``(n_cells, 3)``. If provided, the
            distribution parameter term is included.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` drift flux vector.
        """
        device = get_device()
        dtype = get_default_dtype()
        alpha = alpha.to(device=device, dtype=dtype)

        alpha_clamped = alpha.clamp(min=0.0, max=self._alpha_max)

        # Drift flux: alpha * (1 - alpha) * U_slip
        factor = (alpha_clamped * (1.0 - alpha_clamped)).unsqueeze(1)
        J = factor * U_slip

        # Add distribution parameter term if mixture velocity is provided
        if U_m is not None and self._C0 != 0.0:
            U_m = U_m.to(device=device, dtype=dtype)
            J = J + self._C0 * alpha_clamped.unsqueeze(1) * U_m

        return J
