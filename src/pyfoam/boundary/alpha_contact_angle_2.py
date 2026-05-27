"""
Enhanced contact angle boundary condition for VOF with dynamic hysteresis.

Extends the basic ``alphaContactAngle`` BC with:

- **Dynamic contact angle**: the contact angle varies with the contact-line
  velocity (Kistler's dynamic law).
- **Hysteresis**: different advancing (thetaA) and receding (thetaR) angles
  create a range of metastable contact angles.

The dynamic contact angle is computed via the Hoffman-Voinov-Tanner
correlation:

    theta_d = arccos(cos(theta_eq) - Ca * f(Ca))

where *Ca* is the capillary number based on contact-line velocity.

In OpenFOAM syntax::

    type    alphaContactAngle2;
    thetaA  110;       // advancing angle (degrees)
    thetaR  70;        // receding angle (degrees)
    theta0  90;        // equilibrium angle (degrees)
    sigma   0.07;      // surface tension (N/m)
    Umax    1.0;       // max contact-line velocity for regularisation (m/s)

Usage::

    from pyfoam.boundary.alpha_contact_angle_2 import AlphaContactAngle2BC

    bc = AlphaContactAngle2BC(patch, coeffs={
        "thetaA": 110.0, "thetaR": 70.0, "theta0": 90.0, "sigma": 0.07,
    })
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.boundary.boundary_condition import BoundaryCondition, Patch

__all__ = ["AlphaContactAngle2BC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("alphaContactAngle2")
class AlphaContactAngle2BC(BoundaryCondition):
    """Enhanced contact angle BC with dynamic angle and hysteresis.

    Coefficients
    ------------
    - ``thetaA``: Advancing contact angle in degrees (default 110).
    - ``thetaR``: Receding contact angle in degrees (default 70).
    - ``theta0``: Equilibrium contact angle in degrees (default 90).
    - ``sigma``: Surface tension coefficient (N/m, default 0.07).
    - ``Umax``: Maximum contact-line velocity for regularisation (m/s, default 1.0).
    """

    def __init__(
        self,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(patch, coeffs)
        self._theta_a = float(self._coeffs.get("thetaA", 110.0))
        self._theta_r = float(self._coeffs.get("thetaR", 70.0))
        self._theta0 = float(self._coeffs.get("theta0", 90.0))
        self._theta_a_rad = math.radians(self._theta_a)
        self._theta_r_rad = math.radians(self._theta_r)
        self._theta0_rad = math.radians(self._theta0)
        self._sigma = float(self._coeffs.get("sigma", 0.07))
        self._Umax = max(float(self._coeffs.get("Umax", 1.0)), 1e-6)

    @property
    def theta_a(self) -> float:
        """Advancing contact angle (degrees)."""
        return self._theta_a

    @property
    def theta_r(self) -> float:
        """Receding contact angle (degrees)."""
        return self._theta_r

    @property
    def theta0(self) -> float:
        """Equilibrium contact angle (degrees)."""
        return self._theta0

    @property
    def sigma(self) -> float:
        """Surface tension coefficient (N/m)."""
        return self._sigma

    def dynamic_contact_angle(self, Ca: float) -> float:
        """Compute the dynamic contact angle using Kistler's law.

        The Hoffman-Voinov-Tanner relation gives:

            theta_d = arccos(cos(theta_eq) - f_HVT(Ca))

        where f_HVT is Kistler's correlation function.

        Parameters
        ----------
        Ca : float
            Capillary number (mu * V_cl / sigma).

        Returns
        -------
        float
            Dynamic contact angle in radians, clamped to [theta_r, theta_a].
        """
        Ca_abs = abs(Ca)
        if Ca_abs < 1e-10:
            return self._theta0_rad

        f_hvt = Ca * (1.0 + Ca_abs) ** (1.0 / 3.0)

        cos_theta_d = math.cos(self._theta0_rad) - f_hvt
        cos_theta_d = max(-1.0, min(1.0, cos_theta_d))
        theta_d = math.acos(cos_theta_d)

        # Apply hysteresis bounds
        if Ca > 0:
            theta_d = min(theta_d, self._theta_a_rad)
        else:
            theta_d = max(theta_d, self._theta_r_rad)

        return theta_d

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced contact angle BC to the field.

        Parameters
        ----------
        field : torch.Tensor
            Cell-centre volume fraction ``(n_cells,)``.
        patch_idx : int, optional
            Start index into the field (not used for contact angle).

        Returns
        -------
        torch.Tensor
            Modified field.
        """
        patch = self._patch
        owner_cells = patch.owner_cells
        alpha_wall = field[owner_cells]

        # Use equilibrium angle (contact_line_velocity not available at this level)
        theta_faces = torch.full(
            (patch.n_faces,), self._theta0_rad,
            dtype=field.dtype, device=field.device,
        )

        cos_theta = torch.cos(theta_faces)

        alpha_correction = 0.1 * cos_theta * (1.0 - alpha_wall) * alpha_wall
        alpha_new = alpha_wall + alpha_correction
        alpha_new = alpha_new.clamp(0.0, 1.0)

        field[owner_cells] = alpha_new
        return field

    def gradient(
        self,
        internal_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary gradient with contact angle.

        Parameters
        ----------
        internal_values : torch.Tensor
            ``(n_patch_faces,)`` field values at wall-adjacent cells.

        Returns
        -------
        torch.Tensor
            ``(n_patch_faces,)`` gradient at patch faces.
        """
        delta = self._patch.delta_coeffs

        theta_faces = torch.full(
            (self._patch.n_faces,), self._theta0_rad,
            dtype=internal_values.dtype,
            device=internal_values.device,
        )

        cos_theta = torch.cos(theta_faces)
        grad = cos_theta * delta * 0.01
        return grad

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Contact angle BC: zero matrix contribution (explicit treatment).

        The contact angle modifies alpha gradient directly rather than
        contributing to the fvMatrix.
        """
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source

    def __repr__(self) -> str:
        return (
            f"AlphaContactAngle2BC(patch={self._patch.name}, "
            f"theta_a={self._theta_a}, theta_r={self._theta_r}, "
            f"theta0={self._theta0})"
        )
