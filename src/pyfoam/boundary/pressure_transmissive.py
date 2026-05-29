"""
Transmissive pressure boundary condition for non-reflecting outflow.

Implements a transmissive (non-reflecting / characteristic-based) pressure
boundary condition that allows acoustic waves and flow structures to pass
through the boundary without spurious reflections.

The transmissive BC uses a characteristic decomposition to determine
the outgoing wave speed and applies a relaxation towards a reference
pressure:

    p_boundary = p_interior - rho * c * (U · n) * relaxation_factor

where:
    c    — speed of sound
    n    — outward face normal
    rho  — density at boundary
    U    — velocity at boundary

The relaxation factor controls how strongly the BC drives the pressure
towards the reference value, with 1.0 being fully transmissive.

In OpenFOAM syntax::

    type        pressureTransmissive;
    phi         phi;
    rho         rho;
    psi         psi;
    gamma       1.4;
    fieldInf    101325;          // far-field / reference pressure
    lInf        1;               // relaxation length scale
    value       uniform 101325;

Usage::

    bc = BoundaryCondition.create("pressureTransmissive", patch, coeffs={
        "fieldInf": 101325.0,
        "lInf": 1.0,
        "gamma": 1.4,
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureTransmissiveBC"]


@BoundaryCondition.register("pressureTransmissive")
class PressureTransmissiveBC(BoundaryCondition):
    """Transmissive (non-reflecting) pressure boundary condition.

    Allows acoustic waves and flow structures to exit the domain without
    spurious reflections.  Uses a characteristic-based approach with a
    relaxation towards the far-field pressure.

    The boundary pressure is computed as:

        p_b = p_interior - rho * c * U_n * (p - p_inf) / (rho * c + l_inf)

    where:
        - p_interior is the pressure in the owner cell
        - U_n is the normal velocity component at the boundary
        - p_inf is the far-field reference pressure
        - l_inf is a characteristic relaxation length scale

    Coefficients:
        - ``fieldInf``: Far-field reference pressure (Pa).  Default 101325.
        - ``lInf``: Characteristic relaxation length scale (m).  Default 1.0.
        - ``gamma``: Ratio of specific heats (for compressible).  Default 1.4.
        - ``phi``: Flux field name (informational).
        - ``rho``: Density field name (informational).
        - ``psi``: Compressibility field name (informational).
        - ``value``: Initial pressure (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._field_inf = float(self._coeffs.get("fieldInf", 101325.0))
        self._l_inf = float(self._coeffs.get("lInf", 1.0))
        self._gamma = float(self._coeffs.get("gamma", 1.4))

    @property
    def field_inf(self) -> float:
        """Far-field reference pressure."""
        return self._field_inf

    @property
    def l_inf(self) -> float:
        """Characteristic relaxation length scale."""
        return self._l_inf

    @property
    def gamma(self) -> float:
        """Ratio of specific heats."""
        return self._gamma

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        rho: torch.Tensor | float | None = None,
        c: float | None = None,
    ) -> torch.Tensor:
        """Apply non-reflecting pressure BC.

        Uses the characteristic-based formulation to compute the boundary
        pressure from interior values and far-field reference.

        Args:
            field: Pressure field.
            patch_idx: Optional start index into *field*.
            velocity: ``(n_faces, 3)`` velocity at boundary faces.
            rho: Density at boundary (scalar or per-face tensor).
            c: Speed of sound (m/s).  Default: 343.0.
        """
        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)
        owner_values = field[owners]

        if velocity is not None:
            # Compute normal velocity component
            normals = self._patch.face_normals.to(device=device, dtype=dtype)
            u_n = (velocity * normals).sum(dim=-1)  # (n_faces,)

            # Get density
            if isinstance(rho, torch.Tensor):
                rho_val = rho.to(device=device, dtype=dtype)
            elif rho is not None:
                rho_val = torch.full(
                    (self._patch.n_faces,), float(rho),
                    dtype=dtype, device=device,
                )
            else:
                rho_val = torch.full(
                    (self._patch.n_faces,), 1.225,
                    dtype=dtype, device=device,
                )

            # Speed of sound
            c_val = c if c is not None else 343.0

            # Non-reflecting BC correction:
            # p_b = p_int - rho * c * u_n * (p_int - p_inf) / (rho * c * |u_n| + l_inf)
            dp = owner_values - self._field_inf
            wave_impedance = rho_val * c_val
            denom = wave_impedance * u_n.abs() + self._l_inf + 1e-30
            correction = wave_impedance * u_n * dp / denom

            p_boundary = owner_values - correction
        else:
            # No velocity info: fall back to zero-gradient
            p_boundary = owner_values

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = p_boundary
        else:
            field[self._patch.face_indices] = p_boundary
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Minimal matrix contribution.

        The transmissive BC modifies the boundary value directly rather
        than through the matrix, but adds a small relaxation term:
            - Diagonal += rho*c*A / l_inf (relaxation towards reference)
            - Source   += rho*c*A*p_inf / l_inf
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)

        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        # Relaxation coefficient: rho*c*A / l_inf
        # Use default rho*c = 1.225 * 343 ≈ 420 (air at standard conditions)
        rho_c = 1.225 * 343.0
        relax_coeff = rho_c * area_mag / self._l_inf

        diag.scatter_add_(0, owners, relax_coeff)
        source.scatter_add_(0, owners, relax_coeff * self._field_inf)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
