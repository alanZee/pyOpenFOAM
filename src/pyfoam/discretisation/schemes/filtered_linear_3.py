"""
FilteredLinear3 interpolation scheme — third variant with adaptive NVD filtering.

Uses adaptive NVD filtering with Peclet-dependent tightness, providing
smoother transition between bounded and unbounded regions:

    tol = 0.005 * (1 + |Pe|) * range
    φ_f = clip(φ_linear, φ_min + tol, φ_max - tol)

where Pe is the local face Peclet number estimate.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss filteredLinear3;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["FilteredLinear3Interpolation"]


class FilteredLinear3Interpolation(InterpolationScheme):
    """Third variant of filtered linear interpolation with adaptive NVD filtering.

    Uses Peclet-number-dependent tightness for the NVD bounding, providing
    tighter filtering in convective regions and looser filtering in diffusive
    regions.

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
        """FilteredLinear3 interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Optional, used for
                adaptive NVD filtering.  When not provided, a default
                tightness of 0.5% is used.

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

        # 线性插值
        w = self._weights[:n_internal]
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        # NVD 过滤
        phi_max = torch.maximum(phi_P, phi_N)
        phi_min = torch.minimum(phi_P, phi_N)

        range_val = phi_max - phi_min

        if face_flux is not None:
            # 自适应 NVD 过滤：根据局部 Peclet 数调整紧度
            flux_data = face_flux.to(device=device, dtype=dtype)
            int_flux = flux_data[:n_internal].abs()
            # 基于通量量级估计局部 Peclet 数
            pe_local = int_flux / range_val.clamp(min=1e-30)
            # 紧度因子随 Peclet 数增加
            tol = 0.005 * (1.0 + pe_local) * range_val
        else:
            # 默认 0.5% 紧度
            tol = 0.005 * range_val

        face_values[:n_internal] = torch.clamp(
            phi_linear, phi_min + tol, phi_max - tol
        )

        # 边界面：使用所有者值
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
