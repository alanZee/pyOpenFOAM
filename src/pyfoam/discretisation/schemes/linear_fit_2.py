"""
LinearFit2 interpolation scheme — second variant of linear fit.

Uses a different weighting strategy for the least-squares reconstruction:
distance-squared weighting instead of distance weighting.
This can provide better conditioning on highly stretched meshes.

    w_i = 1 / d_i^2

where d_i is the distance from the cell centre to the face centre.

For meshes with only 2 cells, falls back to linear interpolation.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss linearFit2;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme

__all__ = ["LinearFit2Interpolation"]


class LinearFit2Interpolation(InterpolationScheme):
    """Second variant of linear fit interpolation (distance-squared weighted).

    Reconstructs cell-centre gradients via the Gauss theorem using
    distance-squared weighted face interpolation, then extrapolates to
    face centres.  The face value is the arithmetic mean of the two
    extrapolated values (or upwind-biased when *face_flux* is provided).

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        self._build_connectivity()

    def _build_connectivity(self) -> None:
        """Build cell-to-face mapping for gradient computation."""
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        self._cell_faces: list[list[int]] = [[] for _ in range(n_cells)]
        for f in range(n_faces):
            self._cell_faces[owner[f].item()].append(f)
        for f in range(n_internal):
            self._cell_faces[neighbour[f].item()].append(f)

    def _compute_cell_gradients(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute cell-centre gradients using distance-squared weighted interpolation.

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

        # Distance-squared weighted face interpolation
        phi_face = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        if n_internal > 0:
            phi_P = gather(phi, owner[:n_internal])
            phi_N = gather(phi, neighbour[:n_internal])

            cc_P = mesh.cell_centres[owner[:n_internal]]
            cc_N = mesh.cell_centres[neighbour[:n_internal]]
            fc = mesh.face_centres[:n_internal]

            # Distance-squared weighting: w = d_N^2 / (d_P^2 + d_N^2)
            d_P_sq = (fc - cc_P).pow(2).sum(dim=1)
            d_N_sq = (fc - cc_N).pow(2).sum(dim=1)
            denom = d_P_sq + d_N_sq
            safe_denom = torch.where(denom > 1e-30, denom, torch.ones_like(denom))
            w = d_N_sq / safe_denom
            phi_face[:n_internal] = w * phi_P + (1.0 - w) * phi_N

        if mesh.n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, owner[n_internal:])

        # Gradient via Gauss theorem
        face_contrib = phi_face.unsqueeze(-1) * mesh.face_areas

        grad = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad.index_add_(0, owner[:n_internal], face_contrib[:n_internal])
        grad.index_add_(0, neighbour[:n_internal], -face_contrib[:n_internal])
        if mesh.n_faces > n_internal:
            grad.index_add_(0, owner[n_internal:], face_contrib[n_internal:])

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """LinearFit2 interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Optional; if
                provided, enables upwind-biased mode.

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

        if mesh.n_cells <= 2:
            face_values[:n_internal] = 0.5 * (phi_P + phi_N)
        else:
            grad_phi = self._compute_cell_gradients(phi)

            grad_P = grad_phi[owner[:n_internal]]
            grad_N = grad_phi[neighbour[:n_internal]]

            fc = mesh.face_centres[:n_internal]
            cc_P = mesh.cell_centres[owner[:n_internal]]
            cc_N = mesh.cell_centres[neighbour[:n_internal]]

            d_P = fc - cc_P
            d_N = fc - cc_N

            phi_P_ext = phi_P + (grad_P * d_P).sum(dim=1)
            phi_N_ext = phi_N + (grad_N * d_N).sum(dim=1)

            if face_flux is not None:
                face_flux = face_flux.to(device=device, dtype=dtype)
                int_flux = face_flux[:n_internal]
                is_positive = int_flux >= 0.0
                face_values[:n_internal] = torch.where(
                    is_positive, phi_P_ext, phi_N_ext
                )
            else:
                face_values[:n_internal] = 0.5 * (phi_P_ext + phi_N_ext)

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
