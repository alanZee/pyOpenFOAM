"""
GammaV3 interpolation scheme — v3 vector variant with sigmoid blending.

v3 改进：使用 sigmoid 函数替代 v2 的幂函数进行 Peclet 混合，
提供数学上更平滑的过渡：

    Γ = 1 / (1 + exp(-k * (|Pe| - Pe_threshold)))

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss gammaV3;
    }
"""

from __future__ import annotations

import torch

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["GammaV3Interpolation"]


class GammaV3Interpolation(InterpolationScheme):
    """v3 vector variant of Gamma interpolation with sigmoid blending.

    Uses a sigmoid function for Peclet-number-based blending between
    upwind and linear interpolation, providing an infinitely smooth
    transition.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    steepness : float
        Steepness of the sigmoid transition.  Default is 2.0.
    pe_threshold : float
        Peclet number threshold for the sigmoid midpoint.  Default is 1.0.
    """

    def __init__(
        self, mesh, steepness: float = 2.0, pe_threshold: float = 1.0,
    ) -> None:
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

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
        diffusivity: float = 1.0,
    ) -> torch.Tensor:
        """GammaV3 interpolation of cell vector values to faces.

        Args:
            phi: ``(n_cells, 3)`` cell-centre vector values.
            face_flux: ``(n_faces,)`` volumetric flux.  Required.
            diffusivity: Scalar diffusivity for Peclet estimation.

        Returns:
            ``(n_faces, 3)`` face vector values.

        Raises:
            ValueError: If *phi* is not 2-D with last dim 3, or
                *face_flux* is None.
        """
        if phi.dim() != 2 or phi.shape[1] != 3:
            raise ValueError(
                f"Expected (n_cells, 3) input tensor, got shape {phi.shape}. "
                f"GammaV3 operates on vector fields."
            )
        if face_flux is None:
            raise ValueError(
                "GammaV3Interpolation requires 'face_flux'."
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

        mask = is_positive.unsqueeze(-1)
        phi_upwind = torch.where(mask, phi_P, phi_N)

        w = self._weights[:n_internal].unsqueeze(-1)
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        # v3: sigmoid 混合函数
        pe = int_flux.abs() / max(abs(diffusivity), 1e-30)
        gamma = torch.sigmoid(
            self._steepness * (pe - self._pe_threshold)
        ).unsqueeze(-1)

        face_values[:n_internal] = gamma * phi_upwind + (1.0 - gamma) * phi_linear

        if n_faces > n_internal:
            face_values[n_internal:] = phi[owner[n_internal:]]

        return face_values
