"""
Bubble dynamics models for multiphase Euler-Euler simulations.

Provides bubble diameter tracking with constant, breakup, and coalescence
models.  Used by population balance frameworks and Euler-Euler solvers
to model the evolution of bubble size distributions.

Models:
    - :class:`BubbleModel` — abstract base with RTS registry
    - :class:`ConstantBubble` — fixed (constant) bubble diameter
    - :class:`BubbleBreakup` — bubble breakup and coalescence rate model

In OpenFOAM, bubble models are selected via ``bubbleModel`` in the
``phaseProperties`` dictionary::

    bubbleModel     breakupAndCoalescence;

    breakupAndCoalescenceCoeffs
    {
        breakupModel    LuoSvendsen;
        coalescenceModel PrinceBlanch;
        dMin            1e-4;
        dMax            0.01;
    }
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "BubbleModel",
    "ConstantBubble",
    "BubbleBreakup",
]


class BubbleModel(ABC):
    """Abstract base class for bubble diameter models.

    Subclasses implement :meth:`compute_diameter` to return the local
    bubble diameter given the current state.

    Provides an RTS (Run-Time Selection) registry consistent with
    :class:`~pyfoam.boundary.boundary_condition.BoundaryCondition` and
    :class:`~pyfoam.turbulence.turbulence_model.TurbulenceModel`.
    """

    _registry: ClassVar[dict[str, Type["BubbleModel"]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a bubble model under *name*."""

        def decorator(model_cls: Type[BubbleModel]) -> Type[BubbleModel]:
            if name in cls._registry:
                raise ValueError(
                    f"BubbleModel '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "BubbleModel":
        """Factory: create a bubble model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown BubbleModel '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered bubble model names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute_diameter(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute bubble diameter for each cell.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` bubble diameter.
        """
        ...


@BubbleModel.register("constantBubble")
class ConstantBubble(BubbleModel):
    """Constant (uniform) bubble diameter model.

    The simplest model: all bubbles have the same fixed diameter
    regardless of local flow conditions.

    Parameters
    ----------
    d : float
        Fixed bubble diameter (m).  Default: 0.003 (3 mm).
    """

    def __init__(self, d: float = 0.003) -> None:
        self._d = d

    @property
    def d(self) -> float:
        """Bubble diameter (m)."""
        return self._d

    def compute_diameter(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return constant bubble diameter for all cells.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction (unused).
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` tensor filled with ``d``.
        """
        device = get_device()
        dtype = get_default_dtype()
        return torch.full((n_cells,), self._d, device=device, dtype=dtype)


@BubbleModel.register("bubbleBreakup")
class BubbleBreakup(BubbleModel):
    """Bubble breakup and coalescence rate model.

    Models the evolution of bubble diameter through breakup and
    coalescence mechanisms.  The equilibrium diameter is determined
    by the balance between breakup (fragmentation) and coalescence
    (aggregation) rates.

    Breakup rate (Luo & Svendsen, 1996):
        Rate ~ (epsilon / d^2)^(1/3) * exp(-We_crit / We)

    Coalescence rate (Prince & Blanch, 1990):
        Rate ~ (epsilon * d^2)^(1/3) * exp(-d_coal / d)

    The equilibrium diameter is approximated as a function of the
    local turbulent dissipation rate (epsilon) and volume fraction::

        d_eq = d_max * (1 - C_b * alpha^(1/3)) + d_min * C_b * alpha^(1/3)

    Parameters
    ----------
    d_min : float
        Minimum bubble diameter (m).  Default: 1e-4.
    d_max : float
        Maximum bubble diameter (m).  Default: 0.01.
    C_breakup : float
        Breakup rate coefficient.  Default: 0.2.
    We_crit : float
        Critical Weber number for breakup onset.  Default: 1.0.
    """

    def __init__(
        self,
        d_min: float = 1e-4,
        d_max: float = 0.01,
        C_breakup: float = 0.2,
        We_crit: float = 1.0,
    ) -> None:
        self._d_min = d_min
        self._d_max = d_max
        self._C_breakup = C_breakup
        self._We_crit = We_crit

    @property
    def d_min(self) -> float:
        """Minimum bubble diameter (m)."""
        return self._d_min

    @property
    def d_max(self) -> float:
        """Maximum bubble diameter (m)."""
        return self._d_max

    @property
    def C_breakup(self) -> float:
        """Breakup rate coefficient."""
        return self._C_breakup

    @property
    def We_crit(self) -> float:
        """Critical Weber number."""
        return self._We_crit

    def compute_diameter(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        epsilon: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute equilibrium bubble diameter.

        The equilibrium diameter is a volume-fraction-weighted blend
        between the minimum and maximum diameters::

            d_eq = d_min + (d_max - d_min) * (1 - C_b * alpha^(1/3))

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.
        n_cells : int
            Number of cells.
        epsilon : torch.Tensor, optional
            Turbulent dissipation rate ``(n_cells,)``.  If provided,
            the breakup rate is adjusted for local turbulence level.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` equilibrium bubble diameter.
        """
        device = get_device()
        dtype = get_default_dtype()

        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)

        # Equilibrium diameter: blend between d_min and d_max
        # based on volume fraction
        blend = (1.0 - self._C_breakup * alpha_dev.pow(1.0 / 3.0)).clamp(min=0.0, max=1.0)
        d_eq = self._d_min + (self._d_max - self._d_min) * blend

        # If epsilon is provided, adjust for turbulence intensity
        if epsilon is not None:
            eps_dev = epsilon.to(device=device, dtype=dtype).clamp(min=1e-10)
            # Higher epsilon → more breakup → smaller bubbles
            # We_local = rho * epsilon^(2/3) * d / sigma (simplified)
            # Reduce d_eq when We > We_crit
            we_local = eps_dev.pow(2.0 / 3.0) * d_eq
            we_factor = (1.0 / (1.0 + we_local / self._We_crit)).clamp(min=0.1, max=1.0)
            d_eq = d_eq * we_factor

        return d_eq.clamp(min=self._d_min, max=self._d_max)
