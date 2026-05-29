"""
GammaV2 interpolation scheme — v2 vector variant with modified blending.

Improves on gammaV by using a modified Peclet function with smoother
transition and adjustable blending exponent:

    Γ = clamp(|Pe|^exponent, max=1)

where exponent defaults to 0.5 (square root), providing a more gradual
transition from linear to upwind.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss gammaV2;
    }
"""

from __future__ import annotations

import torch

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["GammaV2Interpolation"]


class GammaV2Interpolation(InterpolationScheme):
    """v2 vector variant of Gamma interpolation.

    Uses a modified Peclet function with an adjustable exponent for
    smoother blending between upwind and linear interpolation.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    exponent : float
        Exponent for the Peclet blending function.  Default is 0.5
        (square root), providing a more gradual transition.
    """

    def __init__(self, mesh, exponent: float = 0.5) -> None:
        super().__init__(mesh)
        self._exponent = exponent
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
        diffusivity: float = 1.0,
    ) -> torch.Tensor:
        """GammaV2 interpolation of cell vector values to faces.

        Args:
            phi: ``(n_cells, 3)`` cell-centre vector values.
            face_flux: ``(n_faces,)`` volumetric flux.  Required.
            diffusivity: Scalar diffusivity for Peclet estimation.
                Default is ``1.0``.

        Returns:
            ``(n_faces, 3)`` face vector values.

        Raises:
            ValueError: If *phi* is not 2-D with last dim 3, or
                *face_flux* is None.
        """
        if phi.dim() != 2 or phi.shape[1] != 3:
            raise ValueError(
                f"Expected (n_cells, 3) input tensor, got shape {phi.shape}. "
                f"GammaV2 operates on vector fields."
            )
        if face_flux is None:
            raise ValueError(
                "GammaV2Interpolation requires 'face_flux'."
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

        # 上风值
        mask = is_positive.unsqueeze(-1)
        phi_upwind = torch.where(mask, phi_P, phi_N)

        # 线性插值
        w = self._weights[:n_internal].unsqueeze(-1)
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        # v2 改进：修改的 Peclet 混合函数
        pe = int_flux.abs() / max(abs(diffusivity), 1e-30)
        pe_pow = torch.pow(pe, self._exponent)
        gamma = torch.clamp(pe_pow, max=1.0).unsqueeze(-1)

        face_values[:n_internal] = gamma * phi_upwind + (1.0 - gamma) * phi_linear

        # 边界面：使用所有者值
        if n_faces > n_internal:
            face_values[n_internal:] = phi[owner[n_internal:]]

        return face_values
