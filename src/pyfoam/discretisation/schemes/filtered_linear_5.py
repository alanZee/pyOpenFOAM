"""
FilteredLinear5 interpolation scheme — fifth variant with three-layer NVD filtering.

v5 改进：使用三层 NVD 过滤器替代 v4 的两层过滤，增加一个基于
面通量比率的自适应层：

    layer1: tol1 = 0.005 * range (基本过滤)
    layer2: tol2 = 0.002 * range * (1 + gradient_ratio) (精细过滤)
    layer3: tol3 = tol2 * (1 - flux_ratio^2 / (1 + flux_ratio^2)) (自适应层)

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss filteredLinear5;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["FilteredLinear5Interpolation"]


class FilteredLinear5Interpolation(InterpolationScheme):
    """Fifth variant of filtered linear with three-layer NVD filtering.

    Uses a three-layer NVD filter for enhanced boundedness control.

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

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FilteredLinear5 interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Optional.

        Returns:
            ``(n_faces,)`` face values.
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

        w = self._weights[:n_internal]
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        phi_max = torch.maximum(phi_P, phi_N)
        phi_min = torch.minimum(phi_P, phi_N)
        range_val = phi_max - phi_min

        # 第一层：基本 NVD 过滤 (0.5%)
        tol1 = 0.005 * range_val
        phi_filtered = torch.clamp(phi_linear, phi_min + tol1, phi_max - tol1)

        # 第二层：精细过滤 (0.2%，根据梯度比率缩放)
        if face_flux is not None:
            flux_data = face_flux.to(device=device, dtype=dtype)
            int_flux = flux_data[:n_internal].abs()
            grad_ratio = int_flux / range_val.clamp(min=1e-30)
            tol2 = 0.002 * range_val * (1.0 + grad_ratio)
        else:
            tol2 = 0.002 * range_val

        phi_filtered2 = torch.clamp(phi_filtered, phi_min + tol2, phi_max - tol2)

        # 第三层：自适应层（根据通量比率进一步收紧）
        if face_flux is not None:
            flux_data = face_flux.to(device=device, dtype=dtype)
            int_flux = flux_data[:n_internal].abs()
            # 通量越大，限制越严格
            flux_ratio = int_flux / range_val.clamp(min=1e-30)
            adaptive_factor = 1.0 - flux_ratio ** 2 / (1.0 + flux_ratio ** 2)
            tol3 = tol2 * adaptive_factor
            face_values[:n_internal] = torch.clamp(
                phi_filtered2, phi_min + tol3, phi_max - tol3
            )
        else:
            face_values[:n_internal] = phi_filtered2

        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
