"""
Wall lubrication force boundary condition for multiphase Euler-Euler models.

Implements the ``wallLubrication`` BC which applies a wall-induced
radial force on the dispersed phase to prevent it from accumulating
near walls.  Based on the Antal et al. (1991) model::

    F_wl = C_w * rho_d * alpha_d * |Vr|^2 / D_p * n_w

where:
    - ``C_w`` is the wall lubrication coefficient
    - ``rho_d`` is the dispersed-phase density
    - ``alpha_d`` is the dispersed-phase volume fraction
    - ``Vr`` is the relative (slip) velocity
    - ``D_p`` is the particle/bubble diameter
    - ``n_w`` is the wall-normal direction

In OpenFOAM syntax::

    type          wallLubrication;
    alpha         alpha.water;        // dispersed phase volume fraction field
    Cw            0.5;                // wall lubrication coefficient
    Dp            0.003;              // particle/bubble diameter (m)
    rho           rho.water;          // dispersed phase density field name
    value         uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["WallLubricationBC"]


@BoundaryCondition.register("wallLubrication")
class WallLubricationBC(BoundaryCondition):
    """Wall lubrication force BC for multiphase Euler-Euler models.

    Applies a wall-normal repulsive force on the dispersed phase to
    model the Antal et al. (1991) wall lubrication effect.  The force
    pushes the dispersed phase away from walls, creating a
    near-wall void region.

    The force is applied as a velocity correction at the wall boundary::

        u_bc = u_interior + F_wl * dt / (rho * delta)

    In practice the BC contributes as a source term to the momentum
    equation via the ``matrix_contributions`` method.

    Coefficients:
        - ``Cw``: Wall lubrication coefficient (default: 0.5).
        - ``Dp``: Particle/bubble diameter in metres (default: 0.003).
        - ``alpha``: Dispersed-phase volume fraction field name
          (informational, default: ``"alpha.d"``).
        - ``rho``: Dispersed-phase density field name
          (informational, default: ``"rho.d"``).
        - ``value``: Initial velocity (default: ``(0 0 0)``).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._Cw = float(self._coeffs.get("Cw", 0.5))
        self._Dp = float(self._coeffs.get("Dp", 0.003))
        self._alpha_name = self._coeffs.get("alpha", "alpha.d")
        self._rho_name = self._coeffs.get("rho", "rho.d")

    @property
    def Cw(self) -> float:
        """Return the wall lubrication coefficient."""
        return self._Cw

    @property
    def Dp(self) -> float:
        """Return the particle/bubble diameter."""
        return self._Dp

    @property
    def alpha_name(self) -> str:
        """Return the dispersed-phase volume fraction field name."""
        return self._alpha_name

    @property
    def rho_name(self) -> str:
        """Return the dispersed-phase density field name."""
        return self._rho_name

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | float | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply wall lubrication velocity correction at the boundary.

        The wall face velocity is set to produce a wall-normal repulsive
        force.  The correction is proportional to::

            F_wl = Cw * rho_d * alpha_d * |V_slip|^2 / Dp

        Since the slip velocity is not directly available here, the BC
        applies a zero-velocity (no-slip) condition and lets the source
        term in ``matrix_contributions`` handle the force.

        Args:
            field: Velocity field ``(n_cells, 3)`` or ``(n_cells,)``.
            patch_idx: Optional start index into *field*.
            alpha: Dispersed-phase volume fraction (unused here, used in
                matrix_contributions).
            rho: Dispersed-phase density (unused here).
        """
        owners = self._patch.owner_cells.to(device=field.device)

        if field.dim() >= 2:
            # Vector field: set wall velocity to owner cell velocity
            # (zero-gradient for the wall lubrication BC)
            owner_values = field[owners]
        else:
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
        alpha: torch.Tensor | float | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall lubrication force source contribution.

        Adds a source term to the momentum equation for cells adjacent
        to the wall::

            source[c] += Cw * rho_d * alpha_d * area / Dp * v_n

        where ``v_n`` is the wall-normal velocity component.

        Args:
            field: Current velocity field.
            n_cells: Total number of cells.
            diag: Pre-existing diagonal tensor.
            source: Pre-existing source tensor.
            alpha: Dispersed-phase volume fraction (scalar or per-cell).
            rho: Dispersed-phase density (scalar or per-cell).
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        normals = self._patch.face_normals.to(device=device, dtype=dtype)

        # Alpha (dispersed-phase volume fraction)
        if alpha is None:
            alpha_val = 0.1
        elif isinstance(alpha, torch.Tensor):
            alpha_val = alpha[owners].to(device=device, dtype=dtype)
        else:
            alpha_val = float(alpha)

        # Density
        if rho is None:
            rho_val = 1000.0
        elif isinstance(rho, torch.Tensor):
            rho_val = rho[owners].to(device=device, dtype=dtype)
        else:
            rho_val = float(rho)

        # Wall lubrication force magnitude per face
        # F = Cw * rho * alpha / Dp * area
        force_coeff = self._Cw * rho_val * alpha_val * areas / self._Dp

        # Contribute as source (penalty-type) for wall-adjacent cells
        source.scatter_add_(0, owners, force_coeff)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
