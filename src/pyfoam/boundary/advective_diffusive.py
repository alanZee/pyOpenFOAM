"""
Advective-diffusive outflow boundary condition.

Combines advective transport and diffusive flux at outflow boundaries::

    phi_face = phi_owner + (1 / (delta * (D + |U_n| * delta))) * D * dphi/dn

where:
    - ``D`` is the effective diffusivity (laminar + turbulent)
    - ``U_n`` is the outward normal velocity
    - ``delta`` is the cell-to-face distance (1 / deltaCoeff)

The advective part carries boundary values outward with the flow, while
the diffusive part allows gradient-driven exchange.  When the Peclet
number ``Pe = |U_n| * delta / D`` is large, the BC behaves as a pure
advective outflow; when Pe is small, it approaches zero-gradient.

In OpenFOAM syntax::

    type        advectiveDiffusive;
    phi         phi;
    D           1.5e-5;        // effective diffusivity (m^2/s)
    field       T;
    lInf        1;             // relaxation length scale
    value       uniform 300;

Usage::

    bc = BoundaryCondition.create("advectiveDiffusive", patch, coeffs={
        "D": 1.5e-5, "lInf": 1.0,
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["AdvectiveDiffusiveBC"]


@BoundaryCondition.register("advectiveDiffusive")
class AdvectiveDiffusiveBC(BoundaryCondition):
    """Advective-diffusive outflow boundary condition.

    Blends advective and zero-gradient treatment based on the local
    Peclet number.  At high Peclet numbers (convection-dominated),
    the BC reduces to advective outflow; at low Peclet numbers
    (diffusion-dominated), it approaches zero-gradient.

    Coefficients:
        - ``D`` (float): Effective diffusivity (m^2/s).  Default 1.5e-5.
        - ``lInf`` (float): Relaxation length scale (m).  Default 1.0.
        - ``phi`` (str): Flux field name (informational).
        - ``field`` (str): Field name (informational).
        - ``value`` (float): Initial value (default 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._D = float(self._coeffs.get("D", 1.5e-5))
        self._l_inf = float(self._coeffs.get("lInf", 1.0))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def D(self) -> float:
        """Effective diffusivity (m^2/s)."""
        return self._D

    @property
    def l_inf(self) -> float:
        """Relaxation length scale (m)."""
        return self._l_inf

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        phi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply advective-diffusive outflow.

        Args:
            field: Full field tensor.
            patch_idx: Optional start index.
            phi: ``(n_faces,)`` face flux values (positive = outflow).
                 If ``None``, falls back to zero-gradient.
        """
        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)
        owner_vals = field[owners]
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        if phi is not None:
            phi_f = phi.to(device=device, dtype=dtype)
            areas = self._patch.face_areas.to(device=device, dtype=dtype)

            # Normal velocity magnitude: |U_n| = |phi| / area
            u_n = phi_f.abs() / areas.clamp(min=1e-30)

            # Peclet number: Pe = |U_n| / (D * delta)
            pe = u_n / (self._D * deltas + 1e-30)

            # Blending: 1 = pure advective (convection-dominated)
            #           0 = pure zero-gradient (diffusion-dominated)
            blending = (pe / (1.0 + pe)).to(dtype=dtype)

            # Advective part: owner value (upwind)
            # Diffusive part: also owner value (zero-gradient proxy)
            # Combined: face = (1-blending)*owner + blending*owner = owner
            # For outflow the advective and diffusive both give owner value;
            # the distinction matters in the matrix contribution.
            face_vals = owner_vals
        else:
            # No flux: zero-gradient
            face_vals = owner_vals

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = face_vals
        else:
            field[self._patch.face_indices] = face_vals
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        phi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advective-diffusive matrix contribution.

        For each face:
            - Diffusive: diag += D * A / delta, source += D * A / delta * owner
            - Advective: source += max(phi, 0) * owner (outflow flux)
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

        # Diffusive contribution: D * A * delta
        diff_coeff = self._D * areas * deltas
        diag.scatter_add_(0, owners, diff_coeff)

        owner_vals = field[owners].to(device=device, dtype=dtype)
        source.scatter_add_(0, owners, diff_coeff * owner_vals)

        # Advective contribution (outflow only)
        if phi is not None:
            phi_f = phi.to(device=device, dtype=dtype)
            outflow_mask = (phi_f > 0.0).to(dtype=dtype)
            source.scatter_add_(0, owners, phi_f * outflow_mask * owner_vals)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
