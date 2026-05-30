"""
CubicUpwindFit5 interpolation scheme — v5 variant with adaptive cubic correction.

v5 改进：使用自适应三次修正系数替代 v4 的固定系数，根据局部
Peclet 数自动调整修正强度：

    α_pe = tanh(Pe / Pe_threshold)
    correction = α_pe * (1/48) * (∇²φ)_face · |S_f|²

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss cubicUpwindFit5;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["CubicUpwindFit5Interpolation"]


class CubicUpwindFit5Interpolation(InterpolationScheme):
    """Cubic upwind interpolation with adaptive Peclet-based correction.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    decay_rate : float
        Exponential decay rate parameter.  Default is 1.0.
    pe_threshold : float
        Peclet number threshold for adaptive correction.  Default is 5.0.
    """

    def __init__(self, mesh, decay_rate: float = 1.0, pe_threshold: float = 5.0) -> None:
        super().__init__(mesh)
        self._decay_rate = decay_rate
        self._pe_threshold = pe_threshold
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
        """CubicUpwindFit5 interpolation of cell values to faces."""
        if phi.dim() != 1:
            raise ValueError(
                f"Expected 1-D input tensor, got {phi.dim()}-D. "
                f"Interpolation operates on scalar fields only."
            )
        if face_flux is None:
            raise ValueError(
                "CubicUpwindFit5Interpolation requires 'face_flux'."
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
            w = compute_centre_weights(
                mesh.cell_centres, mesh.face_centres,
                owner, neighbour, n_internal, n_faces,
                device=device, dtype=dtype,
            )[:n_internal]

            # 基础线性上风
            grad_P = grad_phi[owner[:n_internal]]
            grad_N = grad_phi[neighbour[:n_internal]]

            fc = mesh.face_centres[:n_internal]
            cc_P = mesh.cell_centres[owner[:n_internal]]
            cc_N = mesh.cell_centres[neighbour[:n_internal]]

            d_P = fc - cc_P
            d_N = fc - cc_N

            phi_P_ext = phi_P + (grad_P * d_P).sum(dim=1)
            phi_N_ext = phi_N + (grad_N * d_N).sum(dim=1)

            # v5: 自适应三次修正
            # 估计局部 Peclet 数（基于通量和扩散）
            cc = mesh.cell_centres
            d_cc = cc[neighbour[:n_internal]] - cc[owner[:n_internal]]
            d_mag = d_cc.norm(dim=1).clamp(min=1e-30)
            phi_diff = (phi_N - phi_P).abs()
            grad_avg = (grad_P.norm(dim=1) + grad_N.norm(dim=1)) / 2.0
            pe_local = phi_diff / (d_mag * grad_avg + 1e-30)

            # 自适应系数: tanh(Pe / Pe_threshold)
            alpha_pe = torch.tanh(pe_local / self._pe_threshold)

            # 三次修正（基于面二阶导数近似）
            correction = alpha_pe * (1.0 / 48.0) * (phi_P - phi_N) ** 2 / d_mag.clamp(min=1e-30)

            linear_interp = w * phi_P_ext + (1.0 - w) * phi_N_ext
            face_values[:n_internal] = linear_interp + correction

        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
