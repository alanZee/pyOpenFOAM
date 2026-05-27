"""
LES filter-width delta models.

Provides alternative definitions of the LES filter width Δ used by
subgrid-scale models.  The standard definition (cube root of cell
volume) is in :mod:`~pyfoam.turbulence.filter_width`; this module
provides additional geometric definitions:

- :class:`CubeRootVolDelta` — identical to the standard filter width
  (``V^(1/3)``), but as a callable class with RTS registration.
- :class:`MaxDeltaXYZ` — the maximum of the three characteristic
  cell-length directions: ``max(dx, dy, dz)``.
- :class:`VanDriestDelta` — cube-root volume with Van Driest wall
  damping: ``Δ = V^(1/3) * (1 - exp(-y+ / A+))``.

All models follow a uniform interface: ``__call__(mesh) -> Tensor(n_cells,)``.

Usage::

    from pyfoam.turbulence.les_deltas import CubeRootVolDelta

    delta_fn = CubeRootVolDelta()
    delta = delta_fn(mesh)   # (n_cells,)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "LESDelta",
    "CubeRootVolDelta",
    "MaxDeltaXYZ",
    "VanDriestDelta",
]


class LESDelta(ABC):
    """Abstract base class for LES filter-width delta models.

    Subclasses implement ``__call__(mesh) -> (n_cells,)``.

    Provides an RTS (Run-Time Selection) registry identical in spirit
    to :class:`BoundaryCondition` and :class:`TurbulenceModel`.
    """

    _registry: ClassVar[dict[str, Type["LESDelta"]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a delta model under *name*."""

        def decorator(delta_cls: Type[LESDelta]) -> Type[LESDelta]:
            if name in cls._registry:
                raise ValueError(
                    f"LESDelta '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = delta_cls
            return delta_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "LESDelta":
        """Factory: create a delta model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown LES delta model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered delta model names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def __call__(self, mesh: Any) -> torch.Tensor:
        """Compute filter-width delta for every cell.

        Args:
            mesh: An ``FvMesh`` with geometry computed.

        Returns:
            ``(n_cells,)`` tensor of filter widths.
        """


@LESDelta.register("cubeRootVol")
class CubeRootVolDelta(LESDelta):
    """Cube root of cell volume filter width.

    The standard LES filter width definition:

        Delta = V^(1/3)

    where V is the cell volume.  This is the same as
    :func:`~pyfoam.turbulence.filter_width.compute_filter_width` but
    packaged as a callable RTS-selectable class.
    """

    def __call__(self, mesh: Any) -> torch.Tensor:
        """Compute cube-root volume delta.

        Args:
            mesh: FvMesh with ``cell_volumes`` attribute.

        Returns:
            ``(n_cells,)`` tensor.
        """
        device = get_device()
        dtype = get_default_dtype()
        V = mesh.cell_volumes.to(device=device, dtype=dtype).clamp(min=1e-30)
        return V.pow(1.0 / 3.0)


@LESDelta.register("maxDeltaxyz")
class MaxDeltaXYZ(LESDelta):
    """Maximum direction delta.

    Computes the characteristic cell size as the maximum of the three
    Cartesian extents:

        Delta = max(dx, dy, dz)

    where dx, dy, dz are estimated from the cube root of the cell
    volume scaled by the aspect-ratio-like factors.  For a uniform
    Cartesian mesh this reduces to the largest cell dimension.

    In the absence of explicit per-axis cell dimensions the model
    approximates using ``V^(1/3)`` as the isotropic baseline (i.e.
    for a cubic cell the result equals the cube-root-volume delta).
    """

    def __call__(self, mesh: Any) -> torch.Tensor:
        """Compute max-direction delta.

        If the mesh exposes ``cell_deltas`` (shape ``(n_cells, 3)``),
        those per-axis lengths are used directly.  Otherwise, falls
        back to the isotropic cube-root-of-volume estimate.

        Args:
            mesh: FvMesh.

        Returns:
            ``(n_cells,)`` tensor.
        """
        device = get_device()
        dtype = get_default_dtype()

        if hasattr(mesh, "cell_deltas"):
            deltas = mesh.cell_deltas.to(device=device, dtype=dtype)
            return deltas.max(dim=-1).values

        # Fallback: isotropic cube-root-of-volume
        V = mesh.cell_volumes.to(device=device, dtype=dtype).clamp(min=1e-30)
        return V.pow(1.0 / 3.0)


@LESDelta.register("vanDriest")
class VanDriestDelta(LESDelta):
    """Van Driest wall-damped filter width.

    Applies Van Driest damping to the cube-root-volume filter width:

        Delta = V^(1/3) * (1 - exp(-y+ / A+))

    where:
        - y+ is the dimensionless wall distance
        - A+ is the Van Driest damping constant (default 25)

    The damping ensures the SGS viscosity goes to zero at walls,
    which is necessary for correct near-wall behaviour in LES.

    Parameters
    ----------
    A_plus : float
        Van Driest damping constant.  Default 25.0.
    """

    def __init__(self, A_plus: float = 25.0) -> None:
        self.A_plus = A_plus

    def __call__(self, mesh: Any) -> torch.Tensor:
        """Compute wall-damped delta.

        If the mesh exposes ``y_plus`` (shape ``(n_cells,)``), Van
        Driest damping is applied.  Otherwise, the plain cube-root
        volume is returned (no damping).

        Args:
            mesh: FvMesh.

        Returns:
            ``(n_cells,)`` tensor.
        """
        device = get_device()
        dtype = get_default_dtype()

        V = mesh.cell_volumes.to(device=device, dtype=dtype).clamp(min=1e-30)
        delta0 = V.pow(1.0 / 3.0)

        if hasattr(mesh, "y_plus"):
            y_plus = mesh.y_plus.to(device=device, dtype=dtype).clamp(min=0.0)
            damping = 1.0 - (-y_plus / self.A_plus).exp()
            return delta0 * damping

        return delta0
