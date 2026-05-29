"""
MUSCLV4 interpolation scheme — v4 vector variant with UMIST limiter.

v4 改进：使用 UMIST (Upstream Monotonic Interpolation for Scalar Transport)
限制器替代 v3 的 Sweby 限制器。UMIST 限制器是双参数限制器的最优组合，
在 TVD 区域内提供更好的分辨率：

    ψ(r) = max(0, min(2r, (1+3r)/4, (3+r)/4, 2))

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss MUSCLV4;
    }
"""

from __future__ import annotations

import torch

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["MUSCLV4Interpolation"]


class MUSCLV4Interpolation(InterpolationScheme):
    """v4 vector variant of MUSCL TVD interpolation.

    Uses the UMIST limiter which provides the optimal balance between
    second-order accuracy and monotonicity preservation.

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
        """计算每个向量分量的单元中心梯度。

        Args:
            phi: ``(n_cells, 3)`` 单元中心向量值。

        Returns:
            ``(n_cells, 3, 3)`` 梯度张量。
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
        """MUSCLV4 interpolation of cell vector values to faces.

        Args:
            phi: ``(n_cells, 3)`` cell-centre vector values.
            face_flux: ``(n_faces,)`` volumetric flux.  Required.

        Returns:
            ``(n_faces, 3)`` face vector values.
        """
        if phi.dim() != 2 or phi.shape[1] != 3:
            raise ValueError(
                f"Expected (n_cells, 3) input tensor, got shape {phi.shape}. "
                f"MUSCLV4 operates on vector fields."
            )
        if face_flux is None:
            raise ValueError(
                "MUSCLV4Interpolation requires 'face_flux'."
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

            # v4: UMIST 限制器
            psi = torch.max(
                torch.zeros_like(r),
                torch.min(
                    torch.min(2.0 * r, (1.0 + 3.0 * r) / 4.0),
                    torch.min(
                        (3.0 + r) / 4.0,
                        2.0 * torch.ones_like(r),
                    ),
                ),
            )

            face_values[:n_internal] = phi_up + psi * (phi_linear - phi_up)

        if n_faces > n_internal:
            face_values[n_internal:] = phi[owner[n_internal:]]

        return face_values
