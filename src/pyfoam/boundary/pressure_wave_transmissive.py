"""
Pressure wave transmissive boundary condition.

Non-reflecting outflow for compressible pressure fields using a
characteristic-based wave transmissive formulation.

In OpenFOAM syntax::

    type        pressureWaveTransmissive;
    phi         phi;
    rho         rho;
    psi         psi;
    gamma       1.4;
    fieldInf    101325;
    lInf        1;
    value       uniform 101325;

The boundary pressure is computed as::

    p_face = p_owner + 0.5 * rho * (U_n - c) * (U_n + c)
             * (p_owner - p_inf) / (rho * c * (U_n + c / lInf))

where ``c`` is the speed of sound and ``U_n`` the face-normal velocity.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureWaveTransmissiveBC"]


@BoundaryCondition.register("pressureWaveTransmissive")
class PressureWaveTransmissiveBC(BoundaryCondition):
    """Pressure wave transmissive boundary condition.

    Non-reflecting outflow for compressible pressure using a
    characteristic-based formulation.  Suitable for unsteady
    compressible simulations where reflections from the outlet
    must be minimised.

    Coefficients:
        - ``fieldInf`` (float): Far-field reference pressure (Pa).  Default 101325.
        - ``lInf`` (float): Relaxation length scale (m).  Default 1.0.
        - ``gamma`` (float): Ratio of specific heats.  Default 1.4.
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
        """Apply wave transmissive pressure BC.

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
        else:
            u_n = torch.zeros(n, dtype=dtype, device=device)

        if isinstance(rho, torch.Tensor):
            rho_val = rho.to(device=device, dtype=dtype)
        elif rho is not None:
            rho_val = torch.full((n,), float(rho), dtype=dtype, device=device)
        else:
            rho_val = torch.full((n,), 1.225, dtype=dtype, device=device)

        c_val = c if c is not None else 343.0

        dp = owner_vals - self._field_inf
        denom = rho_val * c_val * (u_n + c_val / self._l_inf) + 1e-30

        p_face = owner_vals + 0.5 * rho_val * (u_n - c_val) * (u_n + c_val) * dp / denom

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
        """Relaxation-based matrix contribution."""
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

        diag.scatter_add_(0, owners, relax_coeff)
        source.scatter_add_(0, owners, relax_coeff * self._field_inf)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
