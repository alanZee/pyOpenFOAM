"""
FilteredLinear4 interpolation scheme — fourth variant with wavelet-inspired filtering.

v4 改进：使用小波启发的多尺度过滤，通过两层 NVD 过滤器提供更精确
的有界性控制：

    layer1: tol1 = 0.005 * range (基本过滤)
    layer2: tol2 = 0.002 * range * (1 + gradient_ratio) (精细过滤)

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss filteredLinear4;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["FilteredLinear4Interpolation"]


class FilteredLinear4Interpolation(InterpolationScheme):
    """Fourth variant of filtered linear with multi-scale NVD filtering.

    Uses a two-layer NVD filter: a basic 0.5% filter followed by a finer
    0.2% filter scaled by the local gradient ratio.

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
        """FilteredLinear4 interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Optional.

        Returns:
            ``(n_faces,)`` face values.

        Raises:
            ValueError: If *phi* is not 1-D.
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
            # 梯度比率：通量相对于值范围的大小
            grad_ratio = int_flux / range_val.clamp(min=1e-30)
            tol2 = 0.002 * range_val * (1.0 + grad_ratio)
        else:
            tol2 = 0.002 * range_val

        face_values[:n_internal] = torch.clamp(
            phi_filtered, phi_min + tol2, phi_max - tol2
        )

        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
