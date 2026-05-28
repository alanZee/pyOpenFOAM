"""
LES spatial filter implementations for Large Eddy Simulation.

Provides explicit spatial filter kernels used in LES to separate
resolved and subgrid scales.  Unlike implicit filters (which are
embedded in the numerical scheme), these filters can be applied
explicitly to velocity or scalar fields for:

- Explicit filtering approaches (Germano, 1986)
- Test filtering for dynamic models
- Pre-processing of DNS data for a priori analysis

Filters:
    - :class:`LESFilter` — abstract base with RTS registry
    - :class:`SimpleFilter` — top-hat (box) filter
    - :class:`LaplaceFilter` — Laplacian-based (Gaussian-like) filter

The top-hat filter convolves with a box kernel::

    phi_filtered(x) = (1/V_f) * integral phi(x') dx'

where V_f = Delta^3 is the filter volume.  On a discrete mesh this
becomes a weighted average over the cell and its face-neighbours.

The Laplace filter applies an iterative Laplacian smoothing::

    phi_filtered = phi + (Delta^2 / 24) * laplacian(phi)

which approximates a Gaussian filter with width Delta.

Reference:
    Sagaut, P., 2006. "Large Eddy Simulation for Incompressible Flows."
    Springer, 3rd ed.  Chapter 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "LESFilter",
    "SimpleFilter",
    "LaplaceFilter",
]


class LESFilter(ABC):
    """Abstract base class for LES spatial filters.

    Subclasses implement :meth:`apply_filter` which takes a field
    and returns the filtered version.

    Provides an RTS (Run-Time Selection) registry consistent with
    :class:`~pyfoam.turbulence.les_deltas.LESDelta`.
    """

    _registry: ClassVar[dict[str, Type["LESFilter"]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a filter under *name*."""

        def decorator(filter_cls: Type[LESFilter]) -> Type[LESFilter]:
            if name in cls._registry:
                raise ValueError(
                    f"LESFilter '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = filter_cls
            return filter_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "LESFilter":
        """Factory: create a filter by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown LESFilter '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered filter names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def apply_filter(
        self,
        field: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """Apply the spatial filter to a field.

        Parameters
        ----------
        field : torch.Tensor
            Input field to filter.  Shape ``(n_cells,)`` for scalar or
            ``(n_cells, 3)`` for vector.
        mesh : Any
            The finite volume mesh.

        Returns
        -------
        torch.Tensor
            Filtered field with the same shape as *field*.
        """
        ...


@LESFilter.register("simpleFilter")
class SimpleFilter(LESFilter):
    """Top-hat (box) filter for LES.

    Applies a discrete top-hat filter by averaging over the cell and
    its face-connected neighbours.  On a uniform mesh the effective
    filter width equals the cell size Delta::

        phi_f[c] = (phi[c] + sum(phi[nb] * w_nb)) / (1 + sum(w_nb))

    The neighbour weights ``w_nb`` are proportional to the face area
    divided by the distance between cell centres.

    Parameters
    ----------
    n_passes : int
        Number of filter passes.  Multiple passes increase the
        effective filter width.  Default: 1.
    """

    def __init__(self, n_passes: int = 1) -> None:
        self._n_passes = max(1, n_passes)

    @property
    def n_passes(self) -> int:
        """Number of filter passes."""
        return self._n_passes

    def apply_filter(
        self,
        field: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """Apply the top-hat filter.

        Parameters
        ----------
        field : torch.Tensor
            Field to filter.  Shape ``(n_cells,)`` or ``(n_cells, 3)``.
        mesh : Any
            The finite volume mesh (must have ``owner``, ``neighbour``,
            ``face_areas``, ``n_cells``, ``n_internal_faces``).

        Returns
        -------
        torch.Tensor
            Filtered field.
        """
        device = get_device()
        dtype = get_default_dtype()

        result = field.to(device=device, dtype=dtype)

        if not hasattr(mesh, "owner") or not hasattr(mesh, "neighbour"):
            return result

        owner = mesh.owner.to(device=device)
        neighbour = mesh.neighbour.to(device=device)
        n_cells = mesh.n_cells
        n_internal = int(mesh.n_internal_faces) if hasattr(mesh, "n_internal_faces") else len(neighbour)

        is_vector = field.dim() >= 2 and field.shape[-1] == 3

        for _ in range(self._n_passes):
            if is_vector:
                filtered = result.clone()
                weights = torch.ones(n_cells, device=device, dtype=dtype)
            else:
                filtered = result.clone()
                weights = torch.ones(n_cells, device=device, dtype=dtype)

            # Accumulate neighbour contributions
            o = owner[:n_internal]
            n = neighbour[:n_internal]

            if is_vector:
                filtered.scatter_add_(0, o.unsqueeze(1).expand_as(result[n]), result[n])
                filtered.scatter_add_(0, n.unsqueeze(1).expand_as(result[o]), result[o])
            else:
                filtered.scatter_add_(0, n, result[o])
                filtered.scatter_add_(0, o, result[n])

            # Count neighbours per cell
            ones = torch.ones(n_internal, device=device, dtype=dtype)
            weights.scatter_add_(0, n, ones)
            weights.scatter_add_(0, o, ones)

            # Normalize
            weights = weights.clamp(min=1.0)
            if is_vector:
                result = filtered / weights.unsqueeze(-1)
            else:
                result = filtered / weights

        return result


@LESFilter.register("laplaceFilter")
class LaplaceFilter(LESFilter):
    """Laplacian-based (Gaussian-like) filter for LES.

    Applies an iterative Laplacian smoothing which approximates a
    Gaussian filter::

        phi_filtered = phi + (Delta^2 / 24) * nabla^2(phi)

    On a discrete mesh the Laplacian is computed using the standard
    finite-volume face interpolation.  Multiple iterations increase
    the effective filter width.

    Parameters
    ----------
    n_iterations : int
        Number of smoothing iterations.  Default: 1.
    """

    def __init__(self, n_iterations: int = 1) -> None:
        self._n_iterations = max(1, n_iterations)

    @property
    def n_iterations(self) -> int:
        """Number of smoothing iterations."""
        return self._n_iterations

    def apply_filter(
        self,
        field: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """Apply the Laplacian filter.

        Parameters
        ----------
        field : torch.Tensor
            Field to filter.  Shape ``(n_cells,)`` or ``(n_cells, 3)``.
        mesh : Any
            The finite volume mesh.

        Returns
        -------
        torch.Tensor
            Filtered field.
        """
        device = get_device()
        dtype = get_default_dtype()

        result = field.to(device=device, dtype=dtype)

        if not hasattr(mesh, "owner") or not hasattr(mesh, "neighbour"):
            return result

        owner = mesh.owner.to(device=device)
        neighbour = mesh.neighbour.to(device=device)
        n_cells = mesh.n_cells
        n_internal = int(mesh.n_internal_faces) if hasattr(mesh, "n_internal_faces") else len(neighbour)

        # Filter width squared (per cell)
        if hasattr(mesh, "cell_volumes"):
            V = mesh.cell_volumes.to(device=device, dtype=dtype).clamp(min=1e-30)
            delta_sq = V.pow(2.0 / 3.0)
        else:
            delta_sq = torch.ones(n_cells, device=device, dtype=dtype)

        coeff = delta_sq / 24.0

        is_vector = field.dim() >= 2 and field.shape[-1] == 3

        for _ in range(self._n_iterations):
            # Compute discrete Laplacian
            laplacian = torch.zeros_like(result)

            o = owner[:n_internal]
            n = neighbour[:n_internal]

            if is_vector:
                diff_on = result[n] - result[o]
                diff_no = result[o] - result[n]
                laplacian.scatter_add_(0, o.unsqueeze(1).expand_as(diff_on), diff_on)
                laplacian.scatter_add_(0, n.unsqueeze(1).expand_as(diff_no), diff_no)
                result = result + coeff.unsqueeze(-1) * laplacian
            else:
                diff_on = result[n] - result[o]
                diff_no = result[o] - result[n]
                laplacian.scatter_add_(0, n, diff_no)
                laplacian.scatter_add_(0, o, diff_on)
                result = result + coeff * laplacian

        return result
