"""
VanLeerV3 interpolation scheme — v3 vector variant with ospre limiter.

v3 改进：使用 ospre 限制器替代 v2 的 superbee 混合，ospre 限制器
在 TVD 区域内提供更平滑的过渡：

    ψ(r) = 1.5 * (r^2 + r) / (r^2 + r + 1)

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss vanLeerV3;
    }
"""

from __future__ import annotations

import torch

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["VanLeerV3Interpolation"]


class VanLeerV3Interpolation(InterpolationScheme):
    """v3 vector variant of Van Leer TVD interpolation.

    Uses the ospre limiter for smoother transitions within the TVD region,
    applied independently to each vector component.

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

    def _compute_cell_gradients(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute cell-centre gradients for each vector component.

        Args:
            phi: ``(n_cells, 3)`` cell-centre vector values.

        Returns:
            ``(n_cells, 3, 3)`` gradient tensors.
        """
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        grad = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)

        for comp in range(3):
            phi_comp = phi[:, comp]
            phi_face = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
            if n_internal > 0:
                phi_P = phi_comp[owner[:n_internal]]
                phi_N = phi_comp[neighbour[:n_internal]]
                phi_face[:n_internal] = 0.5 * (phi_P + phi_N)
            if mesh.n_faces > n_internal:
                phi_face[n_internal:] = phi_comp[owner[n_internal:]]

            face_contrib = phi_face.unsqueeze(-1) * mesh.face_areas
            g = torch.zeros(n_cells, 3, dtype=dtype, device=device)
            g.index_add_(0, owner[:n_internal], face_contrib[:n_internal])
            g.index_add_(0, neighbour[:n_internal], -face_contrib[:n_internal])
            if mesh.n_faces > n_internal:
                g.index_add_(0, owner[n_internal:], face_contrib[n_internal:])

            V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
            grad[:, comp, :] = g / V

        return grad

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """VanLeerV3 interpolation of cell vector values to faces.

        Args:
            phi: ``(n_cells, 3)`` cell-centre vector values.
            face_flux: ``(n_faces,)`` volumetric flux.  Required.

        Returns:
            ``(n_faces, 3)`` face vector values.

        Raises:
            ValueError: If *phi* is not 2-D with last dim 3, or
                *face_flux* is None.
        """
        if phi.dim() != 2 or phi.shape[1] != 3:
            raise ValueError(
                f"Expected (n_cells, 3) input tensor, got shape {phi.shape}. "
                f"VanLeerV3 operates on vector fields."
            )
        if face_flux is None:
            raise ValueError(
                "VanLeerV3Interpolation requires 'face_flux'."
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
        face_values = torch.zeros(n_faces, 3, dtype=dtype, device=device)

        if n_internal == 0:
            face_values[:] = phi[owner]
            return face_values

        idx_own = owner[:n_internal]
        idx_nei = neighbour[:n_internal]
        phi_P = phi[idx_own]
        phi_N = phi[idx_nei]

        int_flux = face_flux[:n_internal]
        is_positive = int_flux >= 0.0
        mask = is_positive.unsqueeze(-1)

        phi_up = torch.where(mask, phi_P, phi_N)

        w = self._weights[:n_internal].unsqueeze(-1)
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        if mesh.n_cells <= 2:
            face_values[:n_internal] = phi_linear
        else:
            grad_phi = self._compute_cell_gradients(phi)

            cc_P = mesh.cell_centres[idx_own]
            cc_N = mesh.cell_centres[idx_nei]
            fc = mesh.face_centres[:n_internal]

            d_up = torch.where(mask, fc - cc_P, fc - cc_N)
            grad_up = torch.where(
                mask.unsqueeze(-1), grad_phi[idx_own], grad_phi[idx_nei],
            )

            phi_face_grad = phi_up + (grad_up * d_up.unsqueeze(1)).sum(dim=2)

            denom = phi_linear - phi_up
            safe_denom = torch.where(
                denom.abs() > 1e-30, denom, torch.ones_like(denom) * 1e-30
            )
            r = (phi_face_grad - phi_up) / safe_denom

            # v3: ospre 限制器: ψ(r) = 1.5 * (r^2 + r) / (r^2 + r + 1)
            r2 = r * r
            psi = 1.5 * (r2 + r) / (r2 + r + 1.0)

            face_values[:n_internal] = phi_up + psi * (phi_linear - phi_up)

        if n_faces > n_internal:
            face_values[n_internal:] = phi[owner[n_internal:]]

        return face_values
