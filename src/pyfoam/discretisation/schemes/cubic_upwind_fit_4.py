"""
CubicUpwindFit4 interpolation scheme — v4 variant with curvature-aware blending.

v4 改进：使用曲率感知混合替代 v3 的自适应混合系数。通过二阶梯度
信息估算局部曲率，曲率越大越依赖线性插值以避免振荡：

    κ = |∇²φ| / (|∇φ| + ε)
    α = min(1, κ / (1 + κ))

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss cubicUpwindFit4;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["CubicUpwindFit4Interpolation"]


class CubicUpwindFit4Interpolation(InterpolationScheme):
    """Cubic upwind interpolation with curvature-aware blending.

    Uses curvature estimation from second-order gradient information to
    control the blending between linear interpolation and cubic extrapolation.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        super().__init__(mesh)
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
        """计算单元中心梯度。

        Args:
            phi: ``(n_cells,)`` 单元中心值。

        Returns:
            ``(n_cells, 3)`` 梯度向量。
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
        """CubicUpwindFit4 interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Required.

        Returns:
            ``(n_faces,)`` face values.
        """
        if phi.dim() != 1:
            raise ValueError(
                f"Expected 1-D input tensor, got {phi.dim()}-D. "
                f"Interpolation operates on scalar fields only."
            )
        if face_flux is None:
            raise ValueError(
                "CubicUpwindFit4Interpolation requires 'face_flux'."
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

        # 线性插值基底
        w = compute_centre_weights(
            mesh.cell_centres, mesh.face_centres,
            owner, neighbour, n_internal, n_faces,
            device=device, dtype=dtype,
        )[:n_internal]
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        if mesh.n_cells < 3:
            face_values[:n_internal] = phi_linear
        else:
            grad_phi = self._compute_cell_gradients(phi)

            grad_P = grad_phi[owner[:n_internal]]
            grad_N = grad_phi[neighbour[:n_internal]]

            cc_P = mesh.cell_centres[owner[:n_internal]]
            cc_N = mesh.cell_centres[neighbour[:n_internal]]
            fc = mesh.face_centres[:n_internal]

            d_P = fc - cc_P
            d_N = fc - cc_N

            # 上风外推
            phi_up = torch.where(is_positive, phi_P, phi_N)
            grad_up = torch.where(
                is_positive.unsqueeze(-1), grad_P, grad_N
            )
            phi_extrap = phi_up + (grad_up * torch.where(
                is_positive.unsqueeze(-1), d_P, d_N
            )).sum(dim=1)

            # v4 改进：曲率感知混合
            # 使用面两侧梯度差异近似二阶曲率
            grad_diff = grad_P - grad_N
            d_vec = cc_N - cc_P
            d_mag = d_vec.norm(dim=1, keepdim=True).clamp(min=1e-30)
            # 曲率近似: |∇φ_P - ∇φ_N| / |d|
            curvature = grad_diff.norm(dim=1) / d_mag.squeeze(-1)
            grad_mag = torch.where(
                is_positive,
                grad_P.norm(dim=1),
                grad_N.norm(dim=1),
            ).clamp(min=1e-30)
            # 无量纲曲率
            kappa = curvature / grad_mag
            # 混合系数：曲率越大，越依赖线性插值
            alpha = (kappa / (1.0 + kappa)).unsqueeze(-1)

            face_values[:n_internal] = (
                alpha.squeeze(-1) * phi_linear
                + (1.0 - alpha.squeeze(-1)) * phi_extrap
            )

        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
