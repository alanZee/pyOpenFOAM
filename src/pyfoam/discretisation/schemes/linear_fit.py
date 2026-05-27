"""
LinearFit interpolation scheme — general weighted least-squares reconstruction.

Reconstructs the face value by fitting a linear function through
neighbouring cell values using a weighted least-squares approach.
The gradient is computed via the Gauss theorem (distance-weighted),
and the face value is extrapolated from the upwind cell.

For flux-independent (pure reconstruction) use, the face value is the
average of both extrapolated values.

For meshes with only 2 cells, falls back to linear interpolation.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss linearFit;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme

__all__ = ["LinearFitInterpolation"]


class LinearFitInterpolation(InterpolationScheme):
    """Weighted least-squares linear fit interpolation scheme.

    Reconstructs cell-centre gradients via the Gauss theorem using
    distance-weighted face interpolation, then extrapolates to face
    centres from both owner and neighbour cells.  The face value is
    the arithmetic mean of the two extrapolated values.

    When *face_flux* is provided, the upwind-biased mode uses only
    the extrapolation from the upwind cell.

    For a 2-cell mesh, falls back to simple linear interpolation.

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
        """Compute cell-centre gradients using weighted least-squares fit.

        Uses the Gauss theorem with distance-weighted face values:

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

        # Distance-weighted face interpolation
        phi_face = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        if n_internal > 0:
            phi_P = gather(phi, owner[:n_internal])
            phi_N = gather(phi, neighbour[:n_internal])
            # Distance-based weights for face values
            cc_P = mesh.cell_centres[owner[:n_internal]]
            cc_N = mesh.cell_centres[neighbour[:n_internal]]
            fc = mesh.face_centres[:n_internal]
            d_P = (fc - cc_P).norm(dim=1)
            d_N = (fc - cc_N).norm(dim=1)
            denom = d_P + d_N
            safe_denom = torch.where(denom > 1e-30, denom, torch.ones_like(denom))
            w = d_N / safe_denom
            phi_face[:n_internal] = w * phi_P + (1.0 - w) * phi_N
        if mesh.n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, owner[n_internal:])

        # Gradient via Gauss theorem: grad = (1/V) * sum(phi_f * S_f)
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
        """LinearFit interpolation of cell values to faces.

        When *face_flux* is provided, upwind-biased extrapolation is
        used (only the upwind cell's gradient extrapolation is taken).
        When *face_flux* is ``None``, the average of both extrapolated
        values is used.

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
            # Insufficient stencil — fall back to linear
            face_values[:n_internal] = 0.5 * (phi_P + phi_N)
        else:
            # Compute cell gradients
            grad_phi = self._compute_cell_gradients(phi)  # (n_cells, 3)

            grad_P = grad_phi[owner[:n_internal]]      # (n_int, 3)
            grad_N = grad_phi[neighbour[:n_internal]]  # (n_int, 3)

            fc = mesh.face_centres[:n_internal]                  # (n_int, 3)
            cc_P = mesh.cell_centres[owner[:n_internal]]         # (n_int, 3)
            cc_N = mesh.cell_centres[neighbour[:n_internal]]     # (n_int, 3)

            # Correction vectors
            d_P = fc - cc_P
            d_N = fc - cc_N

            # Extrapolated values from each cell
            phi_P_ext = phi_P + (grad_P * d_P).sum(dim=1)
            phi_N_ext = phi_N + (grad_N * d_N).sum(dim=1)

            if face_flux is not None:
                # Upwind-biased mode: use only the upwind cell's extrapolation
                face_flux = face_flux.to(device=device, dtype=dtype)
                int_flux = face_flux[:n_internal]
                is_positive = int_flux >= 0.0
                face_values[:n_internal] = torch.where(
                    is_positive, phi_P_ext, phi_N_ext
                )
            else:
                # Pure reconstruction: average of both extrapolations
                face_values[:n_internal] = 0.5 * (phi_P_ext + phi_N_ext)

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
