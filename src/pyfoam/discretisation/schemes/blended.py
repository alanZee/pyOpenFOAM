"""
Blended interpolation scheme — convex combination of two schemes.

Computes a weighted blend of two interpolation schemes:

    phi_f = alpha * phi_scheme1 + (1 - alpha) * phi_scheme2

Commonly used to blend upwind and linear for stability/accuracy trade-off.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss blended(scheme1 scheme2 alpha);
    }
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.discretisation.interpolation import InterpolationScheme

__all__ = ["BlendedInterpolation"]


class BlendedInterpolation(InterpolationScheme):
    """Blended interpolation scheme.

    Computes a convex combination of two interpolation schemes:

    .. math::

        \\phi_f = \\alpha \\, \\phi_{\\text{scheme1}} + (1 - \\alpha) \\, \\phi_{\\text{scheme2}}

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    scheme1 : InterpolationScheme
        The first interpolation scheme.
    scheme2 : InterpolationScheme
        The second interpolation scheme.
    alpha : float
        Blending coefficient in [0, 1].  1.0 gives pure *scheme1*,
        0.0 gives pure *scheme2*.
    """

    def __init__(
        self,
        mesh,
        scheme1: InterpolationScheme | None = None,
        scheme2: InterpolationScheme | None = None,
        alpha: float = 0.5,
    ) -> None:
        super().__init__(mesh)
        if scheme1 is None or scheme2 is None:
            raise ValueError(
                "BlendedInterpolation requires both 'scheme1' and 'scheme2'."
            )
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"alpha must be in [0, 1], got {alpha}."
            )
        self._scheme1 = scheme1
        self._scheme2 = scheme2
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        """Blending coefficient."""
        return self._alpha

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Blended interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Passed through
                to the underlying schemes.

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

        phi1 = self._scheme1.interpolate(phi, face_flux)
        phi2 = self._scheme2.interpolate(phi, face_flux)

        return self._alpha * phi1 + (1.0 - self._alpha) * phi2
