"""
Linear-upwind interpolation scheme — second-order upwind-biased.

Uses the upwind cell value plus a gradient-based correction to achieve
second-order accuracy while maintaining some upwind bias for stability.

For flux >= 0 (owner → neighbour):
    φ_f = φ_P + (∇φ)_P · (r_f - r_P)

For flux < 0 (neighbour → owner):
    φ_f = φ_N + (∇φ)_N · (r_f - r_N)
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather
from pyfoam.core.device import get_device, get_default_dtype

from pyfoam.discretisation.interpolation import InterpolationScheme

__all__ = ["LinearUpwindInterpolation"]


class LinearUpwindInterpolation(InterpolationScheme):
    """Second-order linear-upwind interpolation scheme.

    Extrapolates from the upwind cell using the cell-centre gradient.
    More accurate than pure upwind but can produce overshoots in
    regions of sharp gradients.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        # Pre-compute cell-to-face connectivity for gradient computation
        self._build_connectivity()

    def _build_connectivity(self) -> None:
        """Build cell-to-face mapping for gradient computation."""
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        # Build cell → list of face indices
        self._cell_faces: list[list[int]] = [[] for _ in range(n_cells)]
        for f in range(n_faces):
            self._cell_faces[owner[f].item()].append(f)
        for f in range(n_internal):
            self._cell_faces[neighbour[f].item()].append(f)

    def _compute_cell_gradients(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute cell-centre gradients using least-squares fit.

        For each cell, the gradient is computed from the face values
        using the Gauss theorem:

        .. math::

            (\\nabla \\phi)_c = \\frac{1}{V_c} \\sum_f \\phi_f \\vec{S}_f

        Args:
            phi: ``(n_cells,)`` cell-centre values.

        Returns:
            ``(n_cells, 3)`` gradient vectors.
        """
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        face_areas = mesh.face_areas
        cell_volumes = mesh.cell_volumes

        # Interpolate phi to faces using owner values (simple approach)
        # For internal faces, use arithmetic mean
        phi_face = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        if n_internal > 0:
            phi_P = gather(phi, owner[:n_internal])
            phi_N = gather(phi, neighbour[:n_internal])
            phi_face[:n_internal] = 0.5 * (phi_P + phi_N)
        # Boundary: use owner values
        if mesh.n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, owner[n_internal:])

        # Gradient via Gauss theorem: grad = (1/V) * sum(phi_f * S_f)
        grad = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        # Owner: +phi_f * S_f
        face_contrib = phi_face.unsqueeze(-1) * face_areas  # (n_faces, 3)
        int_own = owner[:n_internal]
        int_nei = neighbour[:n_internal]
        grad.index_add_(0, int_own, face_contrib[:n_internal])
        # Neighbour: -phi_f * S_f
        grad.index_add_(0, int_nei, -face_contrib[:n_internal])
        # Boundary: +phi_f * S_f
        if mesh.n_faces > n_internal:
            grad.index_add_(0, owner[n_internal:], face_contrib[n_internal:])

        # Divide by cell volume
        V = cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad = grad / V

        return grad

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Linear-upwind interpolation of cell values to faces.

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
                "LinearUpwindInterpolation requires 'face_flux'."
            )

        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        face_centres = mesh.face_centres
        cell_centres = mesh.cell_centres

        phi = phi.to(device=device, dtype=dtype)
        face_flux = face_flux.to(device=device, dtype=dtype)
        face_values = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            face_values = gather(phi, owner)
            return face_values

        # Compute cell gradients
        grad_phi = self._compute_cell_gradients(phi)  # (n_cells, 3)

        phi_P = gather(phi, owner[:n_internal])
        phi_N = gather(phi, neighbour[:n_internal])

        grad_P = grad_phi[owner[:n_internal]]      # (n_int, 3)
        grad_N = grad_phi[neighbour[:n_internal]]  # (n_int, 3)

        fc = face_centres[:n_internal]                     # (n_int, 3)
        cc_P = cell_centres[owner[:n_internal]]            # (n_int, 3)
        cc_N = cell_centres[neighbour[:n_internal]]        # (n_int, 3)

        # Correction vectors
        d_P = fc - cc_P  # (n_int, 3)
        d_N = fc - cc_N  # (n_int, 3)

        # Extrapolated values
        phi_P_ext = phi_P + (grad_P * d_P).sum(dim=1)
        phi_N_ext = phi_N + (grad_N * d_N).sum(dim=1)

        int_flux = face_flux[:n_internal]
        is_positive = int_flux >= 0.0

        face_values[:n_internal] = torch.where(
            is_positive, phi_P_ext, phi_N_ext
        )

        # Boundary faces: use owner-based reconstruction
        if n_faces > n_internal:
            phi_bnd = gather(phi, owner[n_internal:])
            grad_bnd = grad_phi[owner[n_internal:]]
            fc_bnd = face_centres[n_internal:]
            cc_bnd = cell_centres[owner[n_internal:]]
            d_bnd = fc_bnd - cc_bnd
            face_values[n_internal:] = phi_bnd + (grad_bnd * d_bnd).sum(dim=1)

        return face_values
