"""
UpwindFit2 interpolation scheme — v2 variant with cell-averaged gradients.

Improves on upwindFit by averaging the upwind cell gradient with the
neighbour gradient for smoother extrapolation near discontinuities:

    φ_f = φ_up + 0.5 * ((∇φ)_up + (∇φ)_nb) · (r_f - r_up)

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss upwindFit2;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme

__all__ = ["UpwindFit2Interpolation"]


class UpwindFit2Interpolation(InterpolationScheme):
    """Upwind interpolation with averaged gradient reconstruction.

    Extrapolates from the upwind cell using an averaged gradient that
    blends the upwind and downwind cell gradients.  This provides
    smoother behaviour near discontinuities while retaining upwind stability.

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
        """Compute cell-centre gradients using distance-weighted least-squares fit.

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

        phi_face = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        if n_internal > 0:
            phi_P = gather(phi, owner[:n_internal])
            phi_N = gather(phi, neighbour[:n_internal])
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
        """UpwindFit2 interpolation of cell values to faces.

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
                "UpwindFit2Interpolation requires 'face_flux'."
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

        if mesh.n_cells < 3:
            # 不足网格 — 回退到标准上风
            face_values[:n_internal] = torch.where(
                is_positive, phi_P, phi_N
            )
        else:
            grad_phi = self._compute_cell_gradients(phi)

            grad_P = grad_phi[owner[:n_internal]]
            grad_N = grad_phi[neighbour[:n_internal]]

            # v2 改进：使用上风和下风梯度的平均值进行外推
            grad_up = torch.where(
                is_positive.unsqueeze(-1), grad_P, grad_N
            )
            grad_nb = torch.where(
                is_positive.unsqueeze(-1), grad_N, grad_P
            )
            grad_avg = 0.5 * (grad_up + grad_nb)

            fc = mesh.face_centres[:n_internal]
            cc_up = torch.where(
                is_positive.unsqueeze(-1),
                mesh.cell_centres[owner[:n_internal]],
                mesh.cell_centres[neighbour[:n_internal]],
            )
            phi_up = torch.where(is_positive, phi_P, phi_N)

            d = fc - cc_up
            face_values[:n_internal] = phi_up + (grad_avg * d).sum(dim=1)

        # 边界面：使用所有者值
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
