"""
FilteredLinear6 interpolation scheme — sixth variant with wavelet-inspired NVD filtering.

v6 改进：使用小波启发的多尺度 NVD 过滤器替代 v5 的三层过滤，
在不同频率尺度上分别进行过滤：

    scale1: 基本范围限制 (tol = 0.003 * range)
    scale2: 基于梯度频率的自适应限制 (tol = 0.001 * range * freq_factor)
    scale3: 小波软阈值过滤 (soft thresholding with wavelet-like shrinkage)

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss filteredLinear6;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["FilteredLinear6Interpolation"]


class FilteredLinear6Interpolation(InterpolationScheme):
    """Sixth variant of filtered linear with wavelet-inspired NVD filtering.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    shrinkage : float
        Wavelet soft-threshold shrinkage parameter.  Default is 0.5.
    """

    def __init__(self, mesh, shrinkage: float = 0.5) -> None:
        super().__init__(mesh)
        self._shrinkage = shrinkage
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

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FilteredLinear6 interpolation of cell values to faces."""
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

        w = self._weights[:n_internal]
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        phi_max = torch.maximum(phi_P, phi_N)
        phi_min = torch.minimum(phi_P, phi_N)
        range_val = phi_max - phi_min

        # 第一尺度：基本范围限制 (0.3%)
        tol1 = 0.003 * range_val
        phi_filtered = torch.clamp(phi_linear, phi_min + tol1, phi_max - tol1)

        # 第二尺度：频率自适应限制
        if face_flux is not None:
            flux_data = face_flux.to(device=device, dtype=dtype)
            int_flux = flux_data[:n_internal].abs()
            freq_factor = int_flux / range_val.clamp(min=1e-30)
            tol2 = 0.001 * range_val * (1.0 + freq_factor)
        else:
            tol2 = 0.001 * range_val

        phi_filtered2 = torch.clamp(phi_filtered, phi_min + tol2, phi_max - tol2)

        # 第三尺度：小波软阈值过滤
        deviation = phi_filtered2 - 0.5 * (phi_P + phi_N)
        half_range = (range_val / 2.0).clamp(min=1e-30)
        normalized_dev = deviation / half_range
        # 软阈值：sign(x) * max(|x| - λ, 0)
        lam = self._shrinkage * 0.1  # 小阈值参数
        soft_thresh = torch.sign(normalized_dev) * torch.clamp(
            normalized_dev.abs() - lam, min=0.0
        )
        phi_filtered3 = 0.5 * (phi_P + phi_N) + soft_thresh * half_range
        face_values[:n_internal] = torch.clamp(
            phi_filtered3, phi_min, phi_max
        )

        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
