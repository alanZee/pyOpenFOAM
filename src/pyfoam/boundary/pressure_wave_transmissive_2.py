"""
Enhanced wave transmissive pressure boundary condition (v2).

Extends ``pressureWaveTransmissive`` with improved non-reflecting treatment
using NSCBC-style relaxation that blends the outgoing wave with the
far-field pressure through a spatially-varying relaxation coefficient::

    p_face = p_owner - rho * c * (U_n - c) * (p_owner - p_inf)
             / (rho * c * lInf * (1 + Ma)) + blending * (p_far - p_owner)

where ``Ma`` is the local Mach number and ``blending`` controls the
strength of the far-field relaxation.

In OpenFOAM syntax::

    type        pressureWaveTransmissive2;
    phi         phi;
    rho         rho;
    psi         psi;
    gamma       1.4;
    fieldInf    101325;
    lInf        1;
    blending    0.1;
    value       uniform 101325;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureWaveTransmissive2BC"]


@BoundaryCondition.register("pressureWaveTransmissive2")
class PressureWaveTransmissive2BC(BoundaryCondition):
    """Enhanced pressure wave transmissive BC with NSCBC blending.

    Coefficients:
        - ``fieldInf`` (float): Far-field reference pressure (Pa).  Default 101325.
        - ``lInf`` (float): Relaxation length scale (m).  Default 1.0.
        - ``gamma`` (float): Ratio of specific heats.  Default 1.4.
        - ``blending`` (float): NSCBC blending factor (default 0.1).
        - ``rho`` (str): Density field name (informational).
        - ``phi`` (str): Flux field name (informational).
        - ``psi`` (str): Compressibility field name (informational).
        - ``value`` (float): Initial pressure (default 101325).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._field_inf = float(self._coeffs.get("fieldInf", 101325.0))
        self._l_inf = float(self._coeffs.get("lInf", 1.0))
        self._gamma = float(self._coeffs.get("gamma", 1.4))
        self._blending = float(self._coeffs.get("blending", 0.1))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def field_inf(self) -> float:
        """Far-field reference pressure (Pa)."""
        return self._field_inf

    @property
    def l_inf(self) -> float:
        """Relaxation length scale (m)."""
        return self._l_inf

    @property
    def gamma(self) -> float:
        """Ratio of specific heats."""
        return self._gamma

    @property
    def blending(self) -> float:
        """NSCBC blending factor."""
        return self._blending

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        rho: torch.Tensor | float | None = None,
        c: float | None = None,
    ) -> torch.Tensor:
        """Apply enhanced wave transmissive pressure BC with NSCBC blending.

        Args:
            field: Pressure field.
            patch_idx: Optional start index into field.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            rho: Density (scalar or per-face tensor).
            c: Speed of sound (m/s).  Default: 343.0.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        owner_vals = field[owners]

        normals = self._patch.face_normals.to(device=device, dtype=dtype)

        if velocity is not None:
            u_n = (velocity * normals).sum(dim=-1)
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
        else:
            u_n = torch.zeros(n, dtype=dtype, device=device)
            u_mag = torch.zeros(n, dtype=dtype, device=device)

        if isinstance(rho, torch.Tensor):
            rho_val = rho.to(device=device, dtype=dtype)
        elif rho is not None:
            rho_val = torch.full((n,), float(rho), dtype=dtype, device=device)
        else:
            rho_val = torch.full((n,), 1.225, dtype=dtype, device=device)

        c_val = c if c is not None else 343.0

        # Local Mach number
        ma = u_mag / (c_val + 1e-30)

        dp = owner_vals - self._field_inf
        denom = rho_val * c_val * self._l_inf * (1.0 + ma) + 1e-30

        # Wave transmissive base
        p_wave = owner_vals - rho_val * c_val * (u_n - c_val) * dp / denom

        # NSCBC blending with far-field
        p_face = p_wave + self._blending * (self._field_inf - owner_vals)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = p_face
        else:
            field[self._patch.face_indices] = p_face
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Relaxation-based matrix contribution with blending."""
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

        rho_c = 1.225 * 343.0
        relax_coeff = rho_c * area_mag / (self._l_inf + 1e-30)

        # Add blending contribution
        blend_coeff = self._blending * area_mag
        total_coeff = relax_coeff + blend_coeff

        diag.scatter_add_(0, owners, total_coeff)
        source.scatter_add_(
            0, owners, relax_coeff * self._field_inf + blend_coeff * self._field_inf
        )

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
