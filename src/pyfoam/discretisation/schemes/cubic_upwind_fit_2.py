"""
CubicUpwindFit2 interpolation scheme — v2 variant with improved blending.

Improves on cubicUpwindFit by using distance-squared blending for the
curvature correction, providing better accuracy on non-uniform meshes:

    φ_f = φ_up + (∇φ)_up · d + blend2 * curvature

where blend2 = d^2 / (d^2 + d_down^2).

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss cubicUpwindFit2;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["CubicUpwindFit2Interpolation"]


class CubicUpwindFit2Interpolation(InterpolationScheme):
    """Cubic upwind interpolation with distance-squared blending.

    Uses distance-squared weighting for the curvature correction factor,
    providing improved accuracy on non-uniform mesh spacing.

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
        """CubicUpwindFit2 interpolation of cell values to faces.

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
                "CubicUpwindFit2Interpolation requires 'face_flux'."
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
            # 不足网格 — 回退到线性插值
            w = compute_centre_weights(
                mesh.cell_centres, mesh.face_centres,
                owner, neighbour, n_internal, n_faces,
                device=device, dtype=dtype,
            )[:n_internal]
            face_values[:n_internal] = w * phi_P + (1.0 - w) * phi_N
        else:
            grad_phi = self._compute_cell_gradients(phi)

            # 上风单元梯度和位置
            grad_up = torch.where(
                is_positive.unsqueeze(-1),
                grad_phi[owner[:n_internal]],
                grad_phi[neighbour[:n_internal]],
            )
            cc_up = torch.where(
                is_positive.unsqueeze(-1),
                mesh.cell_centres[owner[:n_internal]],
                mesh.cell_centres[neighbour[:n_internal]],
            )

            # 下风单元梯度和位置
            grad_down = torch.where(
                is_positive.unsqueeze(-1),
                grad_phi[neighbour[:n_internal]],
                grad_phi[owner[:n_internal]],
            )
            cc_down = torch.where(
                is_positive.unsqueeze(-1),
                mesh.cell_centres[neighbour[:n_internal]],
                mesh.cell_centres[owner[:n_internal]],
            )

            fc = mesh.face_centres[:n_internal]
            d = fc - cc_up

            # 一阶修正（上风梯度）
            phi_up = torch.where(is_positive, phi_P, phi_N)
            phi_grad = phi_up + (grad_up * d).sum(dim=1)

            # v2 改进：使用距离平方混合因子
            delta_grad = grad_down - grad_up
            d_down = fc - cc_down
            d_mag_sq = (d * d).sum(dim=1).clamp(min=1e-30)
            d_down_mag_sq = (d_down * d_down).sum(dim=1).clamp(min=1e-30)
            blend = d_mag_sq / (d_mag_sq + d_down_mag_sq)
            curvature = 0.5 * blend * (delta_grad * d).sum(dim=1)

            face_values[:n_internal] = phi_grad + curvature

        # 边界面：使用所有者值
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
