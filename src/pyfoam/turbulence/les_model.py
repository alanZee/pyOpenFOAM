"""
LES model base class for Large Eddy Simulation.

Provides the abstract interface and shared utilities for subgrid-scale
(SGS) turbulence models.  Concrete implementations include:

- :class:`~pyfoam.turbulence.smagorinsky.SmagorinskyModel`
- :class:`~pyfoam.turbulence.wale.WALEModel`

The base class computes:

- **Filter width** Δ = V^(1/3) for each cell
- **Velocity gradient** tensor g_ij = ∂u_i/∂x_j
- **Strain rate tensor** S_ij = ½(g_ij + g_ji)
- **Strain rate magnitude** |S| = √(2 S_ij S_ij)

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvc

from .filter_width import compute_filter_width

__all__ = ["LESModel"]


class LESModel(ABC):
    """Abstract base class for LES subgrid-scale models.

    Subclasses must implement :meth:`nut` and :meth:`correct`.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh (must have ``cell_volumes`` computed).
    U : torch.Tensor
        Velocity field, shape ``(n_cells, 3)``.
    phi : torch.Tensor
        Face flux field, shape ``(n_faces,)``.

    Attributes
    ----------
    _mesh : Any
        Reference to the mesh.
    _U : torch.Tensor
        Current velocity field.
    _phi : torch.Tensor
        Current face flux.
    _delta : torch.Tensor
        Filter width per cell, ``(n_cells,)``.
    _grad_U : torch.Tensor
        Velocity gradient tensor, ``(n_cells, 3, 3)``.
    _S : torch.Tensor
        Strain rate tensor, ``(n_cells, 3, 3)``.
    _mag_S : torch.Tensor
        Strain rate magnitude, ``(n_cells,)``.
    """

    def __init__(
        self,
        mesh: Any,
        U: torch.Tensor,
        phi: torch.Tensor,
    ) -> None:
        self._device = get_device()
        self._dtype = get_default_dtype()
        self._mesh = mesh

        self._U = U.to(device=self._device, dtype=self._dtype)
        self._phi = phi.to(device=self._device, dtype=self._dtype)

        # Pre-compute filter width (constant for a given mesh)
        self._delta = compute_filter_width(
            mesh, device=self._device, dtype=self._dtype,
        )

        # Cached tensors (computed on first correct() call)
        self._grad_U: torch.Tensor | None = None
        self._S: torch.Tensor | None = None
        self._mag_S: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def nut(self) -> torch.Tensor:
        """Return the subgrid-scale (SGS) turbulent viscosity.

        Returns:
            ``(n_cells,)`` tensor of SGS viscosity values.
        """
        ...

    @abstractmethod
    def correct(self) -> None:
        """Update the model with the current velocity field.

        Recomputes the velocity gradient, strain rate, and any
        model-specific quantities.  Must be called after updating
        :attr:`U` before calling :meth:`nut`.
        """
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @property
    def delta(self) -> torch.Tensor:
        """Filter width per cell ``(n_cells,)``."""
        return self._delta

    def _compute_velocity_gradient(self) -> None:
        """Compute the velocity gradient tensor g_ij = ∂u_i/∂x_j.

        Uses :func:`~pyfoam.discretisation.operators.fvc.grad` on each
        component of the velocity field.  The result is stored in
        :attr:`_grad_U` with shape ``(n_cells, 3, 3)``.

        Convention: ``grad_U[c, i, j]`` = ∂u_i/∂x_j at cell *c*.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        dtype = self._dtype
        device = self._device

        grad_U = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        U_data = self._U.to(device=device, dtype=dtype)

        # Compute gradient of each velocity component
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(U_data[:, i], mesh=mesh)

        self._grad_U = grad_U

    def _compute_strain_rate(self) -> None:
        """Compute the strain rate tensor and its magnitude.

        The strain rate tensor is the symmetric part of the velocity
        gradient:

            S_ij = ½(g_ij + g_ji)

        The magnitude is:

            |S| = √(2 S_ij S_ij)

        Results are stored in :attr:`_S` and :attr:`_mag_S`.
        """
        if self._grad_U is None:
            self._compute_velocity_gradient()

        g = self._grad_U  # (n_cells, 3, 3)
        # S_ij = 0.5 * (g_ij + g_ji)
        self._S = 0.5 * (g + g.transpose(-1, -2))

        # |S| = sqrt(2 * S_ij * S_ij)
        # S_ij * S_ij = sum of squares of all tensor components
        S_sq = (self._S * self._S).sum(dim=(-2, -1))
        self._mag_S = (2.0 * S_sq).clamp(min=0.0).sqrt()

    def _compute_gradients(self) -> None:
        """Compute velocity gradient and strain rate tensors.

        Convenience method that calls :meth:`_compute_velocity_gradient`
        followed by :meth:`_compute_strain_rate`.
        """
        self._compute_velocity_gradient()
        self._compute_strain_rate()

    @property
    def grad_U(self) -> torch.Tensor | None:
        """Velocity gradient tensor ``(n_cells, 3, 3)`` or ``None``."""
        return self._grad_U

    @property
    def strain_rate(self) -> torch.Tensor | None:
        """Strain rate tensor ``(n_cells, 3, 3)`` or ``None``."""
        return self._S

    @property
    def mag_strain_rate(self) -> torch.Tensor | None:
        """Strain rate magnitude ``(n_cells,)`` or ``None``."""
        return self._mag_S

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_cells={self._mesh.n_cells}, "
            f"device={self._device}, "
            f"dtype={self._dtype})"
        )
