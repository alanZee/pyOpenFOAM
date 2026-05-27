"""
Interface compression scheme for Volume-of-Fluid (VOF) methods.

A compressive interpolation scheme that sharpens the interface between
two phases.  Uses the face flux to detect flow direction and applies
a compression velocity to counter numerical diffusion at the interface:

    φ_f = φ_upwind + β * sign(φ_P - φ_N) * |φ_P - φ_N|

where β ∈ [0, 1] is the compression coefficient.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["InterfaceCompressionInterpolation"]


class InterfaceCompressionInterpolation(InterpolationScheme):
    """Interface compression interpolation for VOF methods.

    Applies a compressive correction to reduce numerical smearing at
    the interface between two phases.  The face value is:

    .. math::

        \\phi_f = \\phi_{up} + \\beta \\, \\mathrm{sgn}(\\phi_P - \\phi_N) \\,
        |\\phi_f^{linear} - \\phi_{up}|

    where :math:`\\beta \\in [0, 1]` is the compression coefficient
    (default 1.0 = full compression, 0.0 = pure upwind).

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    beta : float
        Compression coefficient in ``[0, 1]``.  Default ``1.0``.
    """

    def __init__(self, mesh, beta: float = 1.0) -> None:
        super().__init__(mesh)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(
                f"beta must be in [0, 1], got {beta}"
            )
        self._beta = beta
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

    @property
    def beta(self) -> float:
        """Compression coefficient."""
        return self._beta

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Interface compression interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Required.

        Returns:
            ``(n_faces,)`` face values.

        Raises:
            ValueError: If *phi* is not 1-D or *face_flux* is None.
        """
        if phi.dim() != 1:
            raise ValueError(
                f"Expected 1-D input tensor, got {phi.dim()}-D. "
                f"Interpolation operates on scalar fields only."
            )
        if face_flux is None:
            raise ValueError(
                "InterfaceCompressionInterpolation requires 'face_flux'."
            )

        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        phi = phi.to(device=device, dtype=dtype)
        face_flux = face_flux.to(device=device, dtype=dtype)
        face_values = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            face_values = gather(phi, owner)
            return face_values

        phi_P = gather(phi, owner[:n_internal])
        phi_N = gather(phi, neighbour[:n_internal])

        int_flux = face_flux[:n_internal]
        is_positive = int_flux >= 0.0

        # Upwind values
        phi_up = torch.where(is_positive, phi_P, phi_N)

        # Linear interpolation
        w = self._weights[:n_internal]
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        # Compressive correction: push the face value toward upwind
        # when there is a sharp interface (large |φ_P - φ_N|)
        compression = self._beta * (phi_linear - phi_up)
        face_values[:n_internal] = phi_up + compression

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
