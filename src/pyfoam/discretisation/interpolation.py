"""
Face interpolation schemes — convert cell-centre values to face values.

Provides the abstract base class and the linear interpolation scheme.
Additional schemes (upwind, linearUpwind, QUICK) are in the
``schemes`` sub-package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from pyfoam.core.backend import gather
from pyfoam.core.device import get_device, get_default_dtype

from pyfoam.discretisation.weights import compute_centre_weights

__all__ = [
    "InterpolationScheme",
    "LinearInterpolation",
]


class InterpolationScheme(ABC):
    """Abstract base for face interpolation schemes.

    An interpolation scheme computes face values from cell-centre values
    and mesh connectivity.  Subclasses implement different accuracy /
    boundedness trade-offs (linear, upwind, TVD, etc.).

    All schemes operate on the internal faces only; boundary face values
    are handled by boundary conditions.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        self._mesh = mesh

    @property
    def mesh(self):
        """The finite volume mesh."""
        return self._mesh

    @abstractmethod
    def interpolate(self, phi: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Interpolate cell-centre values to face values.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).

        Returns:
            ``(n_faces,)`` face values.
        """

    def __call__(self, phi: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Callable interface — delegates to :meth:`interpolate`."""
        return self.interpolate(phi, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mesh={self._mesh})"


class LinearInterpolation(InterpolationScheme):
    """Second-order linear interpolation.

    For each internal face *f* with owner *P* and neighbour *N*:

    .. math::

        \\phi_f = w_f \\phi_P + (1 - w_f) \\phi_N

    where :math:`w_f` is the distance-based interpolation weight.

    Boundary faces use the owner cell value (weight = 1).

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        self._weights = compute_centre_weights(
            mesh.cell_centres,
            mesh.face_centres,
            mesh.owner,
            mesh.neighbour,
            mesh.n_internal_faces,
            mesh.n_faces,
            device=mesh.device,
            dtype=mesh.dtype,
        )

    def interpolate(self, phi: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Linearly interpolate cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).

        Returns:
            ``(n_faces,)`` face values.

        Raises:
            ValueError: If *phi* is not 1-D.
        """
        if phi.dim() != 1:
            raise ValueError(
                f"Expected 1-D input tensor, got {phi.dim()}-D. "
                f"Interpolation operates on scalar fields only."
            )

        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        phi = phi.to(device=device, dtype=dtype)
        face_values = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            face_values = gather(phi, owner)
            return face_values

        phi_P = gather(phi, owner[:n_internal])
        phi_N = gather(phi, neighbour[:n_internal])

        w = self._weights[:n_internal]
        face_values[:n_internal] = w * phi_P + (1.0 - w) * phi_N

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
