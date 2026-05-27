"""
LUST interpolation scheme — Linear Upstream Stabilised Transport.

Blends linear interpolation with linear-upwind to balance accuracy
and stability:

    φ_f = 0.75 * φ_f_linear + 0.25 * φ_f_linearUpwind

Requires face flux for the linear-upwind component.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme, LinearInterpolation
from pyfoam.discretisation.schemes.linear_upwind import LinearUpwindInterpolation

__all__ = ["LUSTInterpolation"]


class LUSTInterpolation(InterpolationScheme):
    """LUST (Linear Upstream Stabilised Transport) interpolation scheme.

    Blends linear and linear-upwind interpolation:

    .. math::

        \\phi_f = 0.75 \\, \\phi_f^{linear} + 0.25 \\, \\phi_f^{linearUpwind}

    This provides second-order accuracy with reduced overshoots compared
    to pure linear-upwind.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        self._linear = LinearInterpolation(mesh)
        self._linear_upwind = LinearUpwindInterpolation(mesh)

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """LUST interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Required.

        Returns:
            ``(n_faces,)`` face values.

        Raises:
            ValueError: If *phi* is not 1-D or *face_flux* is None.
        """
        if phi.dim() != 1:
            raise ValueError(
                f"Expected 1-D input tensor, got {phi.dim()}-D. "
                f"Interpolation operates on scalar fields only."
            )
        if face_flux is None:
            raise ValueError(
                "LUSTInterpolation requires 'face_flux'."
            )

        phi_linear = self._linear.interpolate(phi)
        phi_lu = self._linear_upwind.interpolate(phi, face_flux)

        return 0.75 * phi_linear + 0.25 * phi_lu
