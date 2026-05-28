"""
Wall lubrication force models for multiphase Euler-Euler flows.

Provides an abstract base class and two standard wall lubrication
correlations used in gas-liquid multiphase simulations.  Wall
lubrication forces push the dispersed phase away from walls, creating
a near-wall void region.

Models:

- :class:`WallLubricationModel` — abstract base with RTS registry
- :class:`AntalWallLubrication` — Antal et al. (1991) model
- :class:`TomiyamaWallLubrication` — Tomiyama et al. (1998) model

All models register themselves via ``@WallLubricationModel.register(name)``
and can be instantiated at run-time via ``WallLubricationModel.create(name, ...)``.

Usage::

    from pyfoam.multiphase.wall_lubrication_models import WallLubricationModel

    wl = WallLubricationModel.create("antal", d=1e-3, rho_c=998.0, rho_d=1.225)
    F_wl = wl.compute(alpha, U_rel, wall_distance, wall_normal)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "WallLubricationModel",
    "AntalWallLubrication",
    "TomiyamaWallLubrication",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class WallLubricationModel(ABC):
    """Abstract base class for wall lubrication force models.

    Subclasses must implement :meth:`compute` which returns the wall
    lubrication force per unit volume.

    RTS (Run-Time Selection) registry allows string-based lookup::

        @WallLubricationModel.register("antal")
        class AntalWallLubrication(WallLubricationModel):
            ...

        wl = WallLubricationModel.create("antal", d=1e-3, rho_c=998.0, ...)
    """

    _registry: ClassVar[dict[str, Type[WallLubricationModel]]] = {}

    def __init__(self, d: float, rho_c: float, rho_d: float) -> None:
        """
        Parameters
        ----------
        d : float
            Particle/bubble diameter (m).
        rho_c : float
            Continuous phase (liquid) density (kg/m^3).
        rho_d : float
            Dispersed phase (gas) density (kg/m^3).
        """
        self.d = d
        self.rho_c = rho_c
        self.rho_d = rho_d

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a wall lubrication model under *name*."""

        def decorator(model_cls: Type[WallLubricationModel]) -> Type[WallLubricationModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Wall lubrication model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> WallLubricationModel:
        """Factory: create a wall lubrication model by registered *name*.

        Parameters
        ----------
        name : str
            Registered model type name.
        **kwargs
            Constructor arguments forwarded to the model class.

        Returns
        -------
        WallLubricationModel
            Instantiated model.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown wall lubrication model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(
        self,
        alpha: torch.Tensor,
        U_rel: torch.Tensor,
        wall_distance: torch.Tensor,
        wall_normal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute wall lubrication force per unit volume.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_rel : torch.Tensor
            Relative velocity vector ``(n_cells, 3)``.
        wall_distance : torch.Tensor
            Distance to nearest wall ``(n_cells,)``.
        wall_normal : torch.Tensor
            Wall-normal direction vector ``(n_cells, 3)``, pointing
            away from the wall into the fluid domain.

        Returns
        -------
        torch.Tensor
            Wall lubrication force per unit volume ``(n_cells, 3)``.
            Force direction is away from the wall.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _eotvos_number(self, sigma: float) -> float:
        """Compute Eotvos number: Eo = g * |rho_c - rho_d| * d^2 / sigma."""
        g = 9.81
        return g * abs(self.rho_c - self.rho_d) * self.d ** 2 / sigma


# ---------------------------------------------------------------------------
# Antal model
# ---------------------------------------------------------------------------


@WallLubricationModel.register("antal")
class AntalWallLubrication(WallLubricationModel):
    """Antal et al. (1991) wall lubrication force model.

    The wall lubrication force is inversely proportional to the wall
    distance::

        F_wl = C_w * rho_c * alpha * |U_rel|^2 / y_w * n_w

    where:
        - C_w = C_w0 * d / y_w (distance-dependent coefficient)
        - y_w is the wall distance
        - n_w is the wall-normal direction (away from wall)

    The coefficient is capped at C_w_max to prevent singularities
    near the wall.

    Parameters
    ----------
    d : float
        Bubble diameter (m).
    rho_c : float
        Continuous phase density (kg/m^3).
    rho_d : float
        Dispersed phase density (kg/m^3).
    C_w0 : float
        Base wall lubrication coefficient.  Default: 0.05.
    C_w_max : float
        Maximum wall lubrication coefficient.  Default: 10.0.
    """

    def __init__(
        self,
        d: float,
        rho_c: float,
        rho_d: float,
        C_w0: float = 0.05,
        C_w_max: float = 10.0,
    ) -> None:
        super().__init__(d, rho_c, rho_d)
        self.C_w0 = C_w0
        self.C_w_max = C_w_max

    def compute(
        self,
        alpha: torch.Tensor,
        U_rel: torch.Tensor,
        wall_distance: torch.Tensor,
        wall_normal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Antal wall lubrication force per unit volume.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_rel : torch.Tensor
            Relative velocity vector ``(n_cells, 3)``.
        wall_distance : torch.Tensor
            Distance to nearest wall ``(n_cells,)``.
        wall_normal : torch.Tensor
            Wall-normal direction ``(n_cells, 3)``, pointing away from wall.

        Returns
        -------
        torch.Tensor
            Wall lubrication force ``(n_cells, 3)``.
        """
        device = alpha.device
        dtype = alpha.dtype

        y_w = wall_distance.to(device=device, dtype=dtype).clamp(min=1e-10)
        n_w = wall_normal.to(device=device, dtype=dtype)

        # Relative velocity magnitude squared
        U_rel_mag_sq = (U_rel ** 2).sum(dim=1)  # (n_cells,)
        U_rel_mag = U_rel_mag_sq.sqrt().clamp(min=1e-10)

        # Distance-dependent coefficient: C_w = C_w0 * d / y_w, capped
        C_w_raw = self.C_w0 * self.d / y_w
        C_w = C_w_raw.clamp(max=self.C_w_max)

        # Force magnitude: C_w * rho_c * alpha * |U_rel| (per unit volume)
        force_mag = C_w * self.rho_c * alpha * U_rel_mag  # (n_cells,)

        # Force direction: wall normal (away from wall)
        return force_mag.unsqueeze(-1) * n_w


# ---------------------------------------------------------------------------
# Tomiyama model
# ---------------------------------------------------------------------------


@WallLubricationModel.register("tomiyama")
class TomiyamaWallLubrication(WallLubricationModel):
    """Tomiyama et al. (1998) wall lubrication force model.

    The Tomiyama model includes an Eotvos number dependent coefficient
    that accounts for bubble shape effects::

        F_wl = C_wl * rho_c * alpha * d * |U_rel|^2 / y_w * n_w

    where:
        - C_wl depends on the Eotvos number (Eo):
            - Eo < 1:   C_wl = -0.0063 * Eo^2 + 0.078 * Eo + 0.0035
            - 1 <= Eo:  C_wl = 0.0035

    The negative coefficient in the Eo < 1 regime indicates a reversal
    of the wall force direction (attracted to wall) for very small bubbles.

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
    C_wl : float
        Wall lubrication coefficient override.  If ``None``, computed
        from Eotvos number.  Default: ``None``.
    """

    def __init__(
        self,
        d: float,
        rho_c: float,
        rho_d: float,
        sigma: float = 0.072,
        C_wl: float | None = None,
    ) -> None:
        super().__init__(d, rho_c, rho_d)
        self.sigma = sigma
        self._C_wl_override = C_wl

    @property
    def eotvos_number(self) -> float:
        """Eotvos number: Eo = g * |rho_c - rho_d| * d^2 / sigma."""
        return self._eotvos_number(self.sigma)

    def _wall_lubrication_coefficient(self) -> float:
        """Compute Tomiyama wall lubrication coefficient C_wl."""
        if self._C_wl_override is not None:
            return self._C_wl_override

        Eo = self.eotvos_number
        if Eo < 1.0:
            return -0.0063 * Eo ** 2 + 0.078 * Eo + 0.0035
        return 0.0035

    def compute(
        self,
        alpha: torch.Tensor,
        U_rel: torch.Tensor,
        wall_distance: torch.Tensor,
        wall_normal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Tomiyama wall lubrication force per unit volume.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_rel : torch.Tensor
            Relative velocity vector ``(n_cells, 3)``.
        wall_distance : torch.Tensor
            Distance to nearest wall ``(n_cells,)``.
        wall_normal : torch.Tensor
            Wall-normal direction ``(n_cells, 3)``, pointing away from wall.

        Returns
        -------
        torch.Tensor
            Wall lubrication force ``(n_cells, 3)``.
        """
        device = alpha.device
        dtype = alpha.dtype

        y_w = wall_distance.to(device=device, dtype=dtype).clamp(min=1e-10)
        n_w = wall_normal.to(device=device, dtype=dtype)

        # Relative velocity magnitude squared
        U_rel_mag_sq = (U_rel ** 2).sum(dim=1)  # (n_cells,)

        # Wall lubrication coefficient from Eotvos number
        C_wl = self._wall_lubrication_coefficient()

        # Force: C_wl * rho_c * alpha * d * |U_rel|^2 / y_w
        force_mag = (
            C_wl * self.rho_c * alpha * self.d * U_rel_mag_sq / y_w
        )  # (n_cells,)

        # Force direction: wall normal (positive = away from wall)
        return force_mag.unsqueeze(-1) * n_w
