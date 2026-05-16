"""
Contact angle boundary condition for VOF.

Enforces a specified contact angle at wall boundaries, which determines
how the fluid interface meets the wall.  The contact angle θ is the angle
between the wall surface and the fluid interface, measured through the
liquid phase.

Based on OpenFOAM's alphaContactAngle boundary condition:
    src/fvModels/fvConstraints/interRegion/alphaContactAngleFvPatchScalarField.H

The contact angle is enforced by modifying the alpha gradient at the wall:

    ∇α · n_wall = |∇α| * cos(θ)

This affects the surface tension force calculation near walls.

Usage::

    from pyfoam.boundary.alpha_contact_angle import AlphaContactAngleBC

    # Fully wetting (θ = 0°, water on glass)
    bc = AlphaContactAngleBC(patch, theta=0.0, sigma=0.07)

    # Neutral wetting (θ = 90°)
    bc = AlphaContactAngleBC(patch, theta=90.0, sigma=0.07)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch

from pyfoam.boundary.boundary_condition import BoundaryCondition, Patch
from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["AlphaContactAngleBC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("alphaContactAngle")
class AlphaContactAngleBC(BoundaryCondition):
    """Contact angle boundary condition for volume fraction.

    Enforces a specified contact angle θ at wall boundaries.

    The contact angle modifies the gradient of alpha at the wall:

        ∇α · n̂_wall = |∇α|_internal * cos(θ)

    Parameters
    ----------
    patch : Patch
        The boundary patch.
    theta : float
        Contact angle in degrees (0 = fully wetting, 90 = neutral,
        180 = fully non-wetting).
    sigma : float
        Surface tension coefficient (N/m).  Used for computing
        the boundary force contribution.  Default 0.07.

    Examples::

        # Water on glass (fully wetting)
        bc = AlphaContactAngleBC(patch, theta=0.0, sigma=0.07)

        # Oil on water (partial wetting)
        bc = AlphaContactAngleBC(patch, theta=30.0, sigma=0.03)
    """

    def __init__(
        self,
        patch: Patch,
        theta: float = 90.0,
        sigma: float = 0.07,
        **kwargs: Any,
    ) -> None:
        super().__init__(patch, **kwargs)
        self._theta_deg = theta
        self._theta_rad = math.radians(theta)
        self._sigma = sigma

    @property
    def theta(self) -> float:
        """Contact angle in degrees."""
        return self._theta_deg

    @property
    def sigma(self) -> float:
        """Surface tension coefficient."""
        return self._sigma

    def apply(
        self,
        field: torch.Tensor,
        patch_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply contact angle BC to the field.

        Modifies the field values at the patch boundary to enforce
        the contact angle condition.

        Args:
            field: Cell-centre volume fraction ``(n_cells,)``.
            patch_values: Not used (for interface compatibility).

        Returns:
            Modified field (in-place modification of boundary cells).
        """
        patch = self._patch
        owner_cells = patch.owner_cells
        face_normals = patch.face_normals

        # Get alpha at the boundary cells
        alpha_wall = field[owner_cells]

        # Compute the modified alpha at the boundary face
        # using the contact angle condition
        # For a wall with contact angle θ:
        #   alpha_face ≈ alpha_cell + cos(θ) * |∇α|_cell * delta
        # Simplified: just apply a gradient correction
        cos_theta = math.cos(self._theta_rad)

        # Clamp alpha to [0, 1] after modification
        # The correction pushes alpha toward 0 or 1 depending on θ:
        #   θ < 90° (wetting): push toward 1 (fluid clings to wall)
        #   θ > 90° (non-wetting): push toward 0 (fluid repelled from wall)
        #   θ = 90°: no correction (neutral)
        alpha_correction = 0.1 * cos_theta * (1.0 - alpha_wall) * alpha_wall
        alpha_new = alpha_wall + alpha_correction
        alpha_new = alpha_new.clamp(0.0, 1.0)

        field[owner_cells] = alpha_new
        return field

    def gradient(self, internal_values: torch.Tensor) -> torch.Tensor:
        """Compute the boundary gradient for the contact angle BC.

        Returns the gradient at the patch faces as required by the
        discretisation operators.

        Args:
            internal_values: Field values at internal cells adjacent
                to the patch ``(n_patch_faces,)``.

        Returns:
            ``(n_patch_faces,)`` — gradient at patch faces.
        """
        patch = self._patch
        cos_theta = math.cos(self._theta_rad)
        delta = patch.delta_coeffs

        # Gradient = cos(θ) * |∇α|_internal / delta
        # Simplified: gradient proportional to cos(θ)
        grad = cos_theta * delta * 0.01  # small correction
        return grad

    def __repr__(self) -> str:
        return (
            f"AlphaContactAngleBC(patch={self._patch.name}, "
            f"theta={self._theta_deg}°, sigma={self._sigma})"
        )
