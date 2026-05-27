"""
porousMedium boundary condition.

Applies Darcy-Forchheimer resistance at a boundary patch, modelling a
porous medium interface.

The pressure drop across the porous medium is::

    dP/d = (mu / alpha) * U + 0.5 * C_F * rho * |U| * U

where:
    - mu is the dynamic viscosity
    - alpha is the permeability (Darcy coefficient)
    - C_F is the Forchheimer coefficient (inertial resistance)
    - rho is the density
    - U is the velocity normal to the face
    - d is the porous medium thickness

In OpenFOAM syntax::

    type            porousMedium;
    alpha           1e-8;     // permeability (m^2)
    C_F             0.1;      // Forchheimer coefficient
    thickness       0.1;      // porous medium thickness (m)
    value           uniform 0;

Usage::

    from pyfoam.boundary.porous_medium import PorousMediumBC

    bc = PorousMediumBC(patch, {"alpha": 1e-8, "C_F": 0.1, "thickness": 0.1})
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PorousMediumBC"]


@BoundaryCondition.register("porousMedium")
class PorousMediumBC(BoundaryCondition):
    """Porous medium boundary condition.

    Applies Darcy-Forchheimer resistance at a boundary, producing a
    pressure drop that depends on the local velocity.

    The resistance model:

        dP = d * [ (mu/alpha)*U_n + 0.5*C_F*rho*|U_n|*U_n ]

    where d is the porous medium thickness and U_n is the face-normal
    velocity component.

    Coefficients:
        - ``alpha``: Permeability (m^2), default 1e-7.
        - ``C_F``: Forchheimer coefficient, default 0.1.
        - ``thickness``: Porous medium thickness (m), default 1.0.
        - ``value``: Initial field value (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._alpha = float(self._coeffs.get("alpha", 1e-7))
        self._C_F = float(self._coeffs.get("C_F", 0.1))
        self._thickness = float(self._coeffs.get("thickness", 1.0))

    @property
    def alpha(self) -> float:
        """Return permeability (m^2)."""
        return self._alpha

    @property
    def C_F(self) -> float:
        """Return Forchheimer coefficient."""
        return self._C_F

    @property
    def thickness(self) -> float:
        """Return porous medium thickness (m)."""
        return self._thickness

    def compute_resistance(
        self,
        velocity: torch.Tensor,
        mu: float = 1e-3,
        rho: float = 1.0,
    ) -> torch.Tensor:
        """Compute the Darcy-Forchheimer resistance per unit area.

        Args:
            velocity: ``(n_faces, 3)`` velocity at boundary faces.
            mu: Dynamic viscosity (Pa.s).
            rho: Density (kg/m^3).

        Returns:
            ``(n_faces,)`` resistance force per unit area (Pa/m).
        """
        device = get_device()
        dtype = get_default_dtype()

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        vel = velocity.to(device=device, dtype=dtype)

        # Normal velocity component (positive = outflow)
        U_n = (vel * normals).sum(dim=-1)

        # Darcy term: (mu / alpha) * U_n
        darcy = (mu / self._alpha) * U_n

        # Forchheimer term: 0.5 * C_F * rho * |U_n| * U_n
        forchheimer = 0.5 * self._C_F * rho * U_n.abs() * U_n

        # Total resistance per unit area, scaled by thickness
        R = self._thickness * (darcy + forchheimer)

        return R

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        mu: float | None = None,
        rho: float | None = None,
    ) -> torch.Tensor:
        """Apply porous medium resistance as a pressure correction.

        Without velocity information, applies zero-gradient behaviour.
        With velocity, applies the Darcy-Forchheimer pressure drop.
        """
        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)
        owner_values = field[owners]

        if velocity is not None:
            mu_val = mu if mu is not None else 1e-3
            rho_val = rho if rho is not None else 1.0
            R = self.compute_resistance(velocity, mu=mu_val, rho=rho_val)
            R = R.to(device=device, dtype=dtype)
            boundary_values = owner_values + R
        else:
            boundary_values = owner_values

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = boundary_values
        else:
            field[self._patch.face_indices] = boundary_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Porous medium matrix contribution.

        Adds a linearised Darcy resistance to the diagonal:
            diag[c] += (mu/alpha) * thickness * area

        The Forchheimer (nonlinear) part goes into the source.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)

        # Linearised Darcy coefficient on diagonal
        darcy_coeff = self._thickness * (1.0 / self._alpha) * areas

        diag.scatter_add_(0, owners, darcy_coeff)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
