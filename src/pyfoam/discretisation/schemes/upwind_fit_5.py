"""
UpwindFit5 interpolation scheme — v5 variant with gradient-weighted reconstruction.

v5 改进：使用梯度加权重建替代 v4 的简单重建，在梯度较大的区域
提供更精确的上风值：

    w_grad = 1 + |∇φ| / (|∇φ|_avg + ε)

    φ_f = φ_up + w_grad · (∇φ)_up · (r_f - r_up)

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss upwindFit5;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["UpwindFit5Interpolation"]


class UpwindFit5Interpolation(InterpolationScheme):
    """Upwind interpolation with gradient-weighted least-squares fit.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    decay_rate : float
        Exponential decay rate parameter.  Default is 1.0.
    """

    def __init__(self, mesh, decay_rate: float = 1.0) -> None:
        super().__init__(mesh)
        self._decay_rate = decay_rate
        self._build_connectivity()

    def _build_connectivity(self) -> None:
        """构建单元到面的映射关系。"""
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
        """使用指数距离加权计算单元中心梯度。"""
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
            alpha = self._decay_rate
            w_P = torch.exp(-alpha * d_P)
            w_N = torch.exp(-alpha * d_N)
            w_sum = w_P + w_N
            safe_sum = torch.where(w_sum > 1e-30, w_sum, torch.ones_like(w_sum))
            w = w_N / safe_sum
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
        """UpwindFit5 interpolation of cell values to faces."""
        if phi.dim() != 1:
            raise ValueError(
                f"Expected 1-D input tensor, got {phi.dim()}-D. "
                f"Interpolation operates on scalar fields only."
            )
        if face_flux is None:
            raise ValueError(
                "UpwindFit5Interpolation requires 'face_flux'."
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
            w = compute_centre_weights(
                mesh.cell_centres, mesh.face_centres,
                owner, neighbour, n_internal, n_faces,
                device=device, dtype=dtype,
            )[:n_internal]
            face_values[:n_internal] = w * phi_P + (1.0 - w) * phi_N
        else:
            grad_phi = self._compute_cell_gradients(phi)

            # v5: 梯度加权重建
            grad_mag = grad_phi.norm(dim=1)
            avg_grad = grad_mag.mean().clamp(min=1e-30)
            grad_weight = 1.0 + grad_mag / avg_grad

            fc = mesh.face_centres[:n_internal]
            cc_P = mesh.cell_centres[owner[:n_internal]]
            cc_N = mesh.cell_centres[neighbour[:n_internal]]

            d_P = fc - cc_P
            d_N = fc - cc_N

            # 上风单元梯度 + 加权修正
            grad_up = torch.where(
                is_positive.unsqueeze(-1),
                grad_phi[owner[:n_internal]],
                grad_phi[neighbour[:n_internal]],
            )
            d_up = torch.where(is_positive.unsqueeze(-1), d_P, d_N)
            w_grad = torch.where(
                is_positive,
                grad_weight[owner[:n_internal]],
                grad_weight[neighbour[:n_internal]],
            )

            phi_up = torch.where(is_positive, phi_P, phi_N)
            face_values[:n_internal] = phi_up + w_grad * (grad_up * d_up).sum(dim=1)

        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
