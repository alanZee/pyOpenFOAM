"""
GammaV5 interpolation scheme — v5 vector variant with logistic Peclet blending.

v5 改进：使用 logistic 函数替代 v4 的指数函数进行 Peclet 混合，
提供更平滑的过渡和更精确的混合因子：

    α_pe = 1 / (1 + exp(-k * (Pe - Pe_threshold)))

    φ_f = α_pe * φ_up + (1 - α_pe) * φ_linear

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss gammaV5;
    }
"""

from __future__ import annotations

import torch

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["GammaV5Interpolation"]


class GammaV5Interpolation(InterpolationScheme):
    """v5 vector variant of Gamma Peclet-number-based blending.

    Uses a logistic (sigmoid) Peclet blending function for smoother
    transition between upwind and linear interpolation.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    steepness : float
        Steepness of the logistic Peclet transition.  Default is 5.0.
    pe_threshold : float
        Peclet number threshold for blending.  Default is 2.0.
    """

    def __init__(self, mesh, steepness: float = 5.0, pe_threshold: float = 2.0) -> None:
        super().__init__(mesh)
        self._steepness = steepness
        self._pe_threshold = pe_threshold
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
        """计算每个向量分量的单元中心梯度。"""
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
        """GammaV5 interpolation of cell vector values to faces."""
        if phi.dim() != 2 or phi.shape[1] != 3:
            raise ValueError(
                f"Expected (n_cells, 3) input tensor, got shape {phi.shape}. "
                f"GammaV5 operates on vector fields."
            )
        if face_flux is None:
            raise ValueError(
                "GammaV5Interpolation requires 'face_flux'."
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

        # 线性插值
        w = self._weights[:n_internal].unsqueeze(-1)
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        if mesh.n_cells <= 2:
            face_values[:n_internal] = phi_linear
        else:
            grad_phi = self._compute_cell_gradients(phi)

            # 计算面 Peclet 数
            cc_P = mesh.cell_centres[idx_own]
            cc_N = mesh.cell_centres[idx_nei]
            d_mag = (cc_N - cc_P).norm(dim=1).clamp(min=1e-30)

            grad_P = grad_phi[idx_own]
            grad_N = grad_phi[idx_nei]
            grad_f = w * grad_P + (1.0 - w) * grad_N

            # Peclet 数 = |通量| / (|∇φ_f| * d)
            grad_mag = grad_f.norm(dim=1)
            grad_mag_sq = (grad_mag * grad_mag).sum(dim=1) if grad_mag.dim() > 1 else grad_mag
            if grad_mag.dim() > 1:
                grad_mag_sq = grad_mag.norm(dim=1)
            pe = int_flux.abs() / (grad_mag_sq * d_mag + 1e-30)

            # v5: Logistic Peclet 混合
            k = self._steepness
            pe_thresh = self._pe_threshold
            alpha_up = torch.sigmoid(k * (pe - pe_thresh))  # 0→线性, 1→上风

            # 上风插值
            is_positive = int_flux >= 0.0
            mask = is_positive.unsqueeze(-1)
            phi_up = torch.where(mask, phi_P, phi_N)

            # v5: 混合
            alpha = alpha_up.unsqueeze(-1)
            face_values[:n_internal] = alpha * phi_up + (1.0 - alpha) * phi_linear

        if n_faces > n_internal:
            face_values[n_internal:] = phi[owner[n_internal:]]

        return face_values
