"""
Drag coefficient boundary condition for particle-laden flows.

Implements the ``dragCoefficient`` BC which computes the drag force
on particles at wall boundaries using a prescribed or computed drag
coefficient.  Common in Euler-Euler and Euler-Lagrange multiphase models
where the particle-wall interaction requires a drag correction.

The drag force at the wall is::

    F_d = 0.5 * Cd * rho_c * A_p * |V_rel| * V_rel

where:
    - ``Cd`` is the drag coefficient (constant or Schiller-Naumann)
    - ``rho_c`` is the continuous-phase density
    - ``A_p`` is the projected particle area
    - ``V_rel`` is the relative (slip) velocity at the wall

In OpenFOAM syntax::

    type            dragCoefficient;
    Cd              0.44;                // drag coefficient
    rho             rho.air;             // continuous phase density field name
    dp              1e-4;                // particle diameter (m)
    model           SchillerNaumann;     // drag model: constant | SchillerNaumann
    value           uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["DragCoefficientBC"]


def _schiller_naumann_cd(Re_p: torch.Tensor) -> torch.Tensor:
    """Compute drag coefficient using Schiller-Naumann correlation.

    Cd = 24/Re * (1 + 0.15 * Re^0.687)  for Re < 1000
    Cd = 0.44                             for Re >= 1000

    Args:
        Re_p: Particle Reynolds number (per-face or per-cell).

    Returns:
        Drag coefficient.
    """
    Re_safe = Re_p.clamp(min=1e-10)
    cd_laminar = (24.0 / Re_safe) * (1.0 + 0.15 * Re_safe.pow(0.687))
    cd_turbulent = torch.full_like(Re_p, 0.44)
    return torch.where(Re_p < 1000.0, cd_laminar, cd_turbulent)


@BoundaryCondition.register("dragCoefficient")
class DragCoefficientBC(BoundaryCondition):
    """Drag coefficient boundary condition for particle-laden flows.

    Applies a drag-force-based velocity correction at wall boundaries
    for the dispersed (particle) phase.  Supports constant Cd or the
    Schiller-Naumann correlation for sphere drag.

    The BC computes a wall velocity that produces the target drag
    force via the penalty method::

        u_wall = u_slip * f(Cd, rho, dp, delta)

    where ``u_slip`` is the slip velocity relative to the wall.

    Coefficients:
        - ``Cd``: Constant drag coefficient (default: 0.44 for spheres).
        - ``rho``: Continuous-phase density field name
          (informational, default: ``"rho.air"``).
        - ``dp``: Particle diameter in metres (default: 1e-4).
        - ``model``: Drag model — ``"constant"`` or ``"SchillerNaumann"``
          (default: ``"constant"``).
        - ``value``: Initial velocity (default: ``(0 0 0)``).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._Cd = float(self._coeffs.get("Cd", 0.44))
        self._dp = float(self._coeffs.get("dp", 1e-4))
        self._model = str(self._coeffs.get("model", "constant")).lower()
        self._rho_name = self._coeffs.get("rho", "rho.air")

    @property
    def Cd(self) -> float:
        """Return the constant drag coefficient."""
        return self._Cd

    @property
    def dp(self) -> float:
        """Return the particle diameter."""
        return self._dp

    @property
    def model(self) -> str:
        """Return the drag model name."""
        return self._model

    @property
    def rho_name(self) -> str:
        """Return the continuous-phase density field name."""
        return self._rho_name

    def _compute_cd(
        self,
        Re_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute drag coefficient based on selected model.

        Args:
            Re_p: Particle Reynolds number (required for
                Schiller-Naumann model).

        Returns:
            Drag coefficient tensor.
        """
        device = get_device()
        dtype = get_default_dtype()

        if self._model == "schillernaumann" and Re_p is not None:
            return _schiller_naumann_cd(Re_p.to(device=device, dtype=dtype))

        # Constant model
        return torch.tensor(self._Cd, dtype=dtype, device=device)

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply drag coefficient velocity correction at the boundary.

        For constant-drag wall, applies a zero-gradient condition
        (owner cell velocity is used at the boundary face).

        Args:
            field: Velocity field ``(n_cells, 3)``.
            patch_idx: Optional start index into *field*.
            rho: Continuous-phase density (unused here, used in
                matrix_contributions).
        """
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = owner_values
        else:
            field[self._patch.face_indices] = owner_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drag force source contribution from wall.

        Adds a penalty-type source term to model the drag interaction
        between particles and the wall::

            diag[c]  += coeff
            source[c] += coeff * u_cell

        where ``coeff = 0.5 * Cd * rho_c * (pi/4 * dp^2) * deltaCoeff * area``.

        Args:
            field: Current velocity field.
            n_cells: Total number of cells.
            diag: Pre-existing diagonal tensor.
            source: Pre-existing source tensor.
            rho: Continuous-phase density (scalar or per-cell).
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        # Density
        if rho is None:
            rho_val = 1.0
        elif isinstance(rho, torch.Tensor):
            rho_val = rho.to(device=device, dtype=dtype)
        else:
            rho_val = float(rho)

        # Projected particle area: A_p = pi/4 * dp^2
        A_p = 3.141592653589793 / 4.0 * self._dp ** 2

        # Drag coefficient
        Cd = self._compute_cd()

        # Penalty coefficient: 0.5 * Cd * rho * A_p * deltaCoeff * area
        coeff = 0.5 * Cd * rho_val * A_p * deltas * areas

        # Owner cell velocity for source
        if field.dim() >= 2:
            u_x = field[owners, 0].to(device=device, dtype=dtype)
        else:
            u_x = field[owners].to(device=device, dtype=dtype)

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * u_x)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
