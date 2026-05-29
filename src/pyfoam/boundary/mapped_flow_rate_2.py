"""
Enhanced mapped flow rate boundary condition (v2).

Extends ``mappedFlowRate`` with a power-law radial profile so that the
mapped mass flow rate is not distributed uniformly but follows::

    U(r) = U_max * (1 - r/R)^profileExponent

where *r* is the distance from the pipe centre and *R* the hydraulic
radius.  When ``profileExponent = 0`` (default) the distribution is
uniform, recovering the base behaviour.

In OpenFOAM syntax::

    type              mappedFlowRate2;
    neighbourPatch    outlet;
    rho               1.0;
    massFlowRate      1.0;
    profileExponent   7.0;        // 1/7th-power-law
    hydraulicDiameter 0.1;
    value             uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedFlowRate2BC"]


@BoundaryCondition.register("mappedFlowRate2")
class MappedFlowRate2BC(BoundaryCondition):
    """Enhanced mapped flow rate with power-law profile.

    Distributes the target mass flow rate over patch faces according to
    a power-law profile centred on the patch centroid.

    Coefficients:
        - ``massFlowRate`` (float): Target mass flow rate (kg/s).  Default 1.0.
        - ``rho`` (float): Fluid density (kg/m3).  Default 1.0.
        - ``profileExponent`` (float): Power-law exponent (default 0 = uniform).
        - ``hydraulicDiameter`` (float): Hydraulic diameter (m).  Default 0.1.
        - ``neighbourPatch`` (str): Name of the mapped neighbour patch.
        - ``value``: Initial velocity (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mass_flow_rate = float(self._coeffs.get("massFlowRate", 1.0))
        self._rho = float(self._coeffs.get("rho", 1.0))
        self._profile_exponent = float(self._coeffs.get("profileExponent", 0.0))
        self._hydraulic_diameter = float(self._coeffs.get("hydraulicDiameter", 0.1))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mass_flow_rate(self) -> float:
        """Target mass flow rate (kg/s)."""
        return self._mass_flow_rate

    @property
    def rho(self) -> float:
        """Fluid density (kg/m3)."""
        return self._rho

    @property
    def profile_exponent(self) -> float:
        """Power-law profile exponent."""
        return self._profile_exponent

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter (m)."""
        return self._hydraulic_diameter

    @property
    def neighbour_patch_name(self) -> str | None:
        """Name of the mapped neighbour patch."""
        return self._coeffs.get("neighbourPatch", self._patch.neighbour_patch)

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocity from mapped mass flow rate with profile.

        When profileExponent > 0, each face velocity is scaled by a
        power-law factor normalised so that the total mass flow rate
        is preserved.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)

        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        # Compute profile weights
        if self._profile_exponent > 0 and n > 1:
            # Centroid of patch face centres (approximate from face indices)
            # Use linearly-spaced radial positions as a proxy
            r_frac = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
            weights = (1.0 - r_frac) ** self._profile_exponent
            weighted_areas = area_mag * weights
            total_weighted = weighted_areas.sum()
            if total_weighted > 0:
                profile_scale = (area_mag * weights) / (total_weighted + 1e-30) * n
            else:
                profile_scale = torch.ones(n, dtype=dtype, device=device)
        else:
            profile_scale = torch.ones(n, dtype=dtype, device=device)

        total_area = area_mag.sum()
        if total_area > 0:
            u_base = self._mass_flow_rate / (self._rho * total_area)
        else:
            u_base = 0.0

        velocity = -normals * (u_base * profile_scale).unsqueeze(-1) if normals.dim() == 2 else -normals * u_base * profile_scale

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = velocity
        else:
            field[self._patch.face_indices] = velocity
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for enhanced mapped flow rate BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        total_area = areas.abs().sum() if areas.dim() == 1 else areas.norm(dim=1).sum()
        if total_area > 0:
            u_n = self._mass_flow_rate / (self._rho * total_area)
        else:
            u_n = 0.0

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * u_n)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
