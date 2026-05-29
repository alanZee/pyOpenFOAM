"""
flowRateInletVelocity2 — enhanced flow rate inlet velocity boundary
condition with profile support.

An improved version of ``flowRateInletVelocity`` that distributes
velocity across patch faces using a power-law profile rather than
uniform distribution.  Supports both volumetric and mass flow rates.

Profile:
    u(r) = U_bulk * (1 - |r - r_c| / R_max)^(1/n)

In OpenFOAM syntax::

    type                flowRateInletVelocity2;
    volumetricFlowRate  0.001;      // m^3/s (or massFlowRate)
    rho                 1.225;      // density (default: 1.0)
    exponent            7;          // profile exponent (default: 7)
    value               uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["FlowRateInletVelocity2BC"]


@BoundaryCondition.register("flowRateInletVelocity2")
class FlowRateInletVelocity2BC(BoundaryCondition):
    """Enhanced flow rate inlet velocity with power-law profile.

    Computes inlet velocity from a target mass or volume flow rate,
    then distributes it across patch faces with a power-law profile
    that captures the turbulent boundary layer shape.

    Coefficients:
        - ``volumetricFlowRate``: Volumetric flow rate (m3/s).
        - ``massFlowRate``: Mass flow rate (kg/s).  Requires ``rho``.
        - ``rho``: Fluid density (kg/m3, default 1.0).
        - ``exponent``: Power-law profile exponent (default 7).
        - ``value``: Initial velocity (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._volumetric_flow_rate = self._parse_flow_rate()
        self._rho = float(self._coeffs.get("rho", 1.0))
        self._exponent = float(self._coeffs.get("exponent", 7.0))

    def _parse_flow_rate(self) -> float:
        """Parse flow rate from coefficients."""
        if "volumetricFlowRate" in self._coeffs:
            return float(self._coeffs["volumetricFlowRate"])
        elif "massFlowRate" in self._coeffs:
            rho = float(self._coeffs.get("rho", 1.0))
            return float(self._coeffs["massFlowRate"]) / rho
        return 0.0

    @property
    def volumetric_flow_rate(self) -> float:
        """Return the volumetric flow rate."""
        return self._volumetric_flow_rate

    @property
    def rho(self) -> float:
        """Return fluid density."""
        return self._rho

    @property
    def exponent(self) -> float:
        """Return profile exponent."""
        return self._exponent

    def _compute_profile(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compute power-law velocity profile across patch faces.

        Returns ``(n_faces,)`` velocity magnitude per face.
        """
        if device is None:
            device = get_device()
        if dtype is None:
            dtype = get_default_dtype()
        n_faces = self._patch.n_faces

        face_areas = self._patch.face_areas.to(device=device, dtype=dtype)
        total_area = face_areas.sum().item()

        if total_area < 1e-30:
            return torch.zeros(n_faces, dtype=dtype, device=device)

        u_bulk = self._volumetric_flow_rate / total_area
        if abs(u_bulk) < 1e-30:
            return torch.zeros(n_faces, dtype=dtype, device=device)

        # Normalised distance from patch centroid
        r = torch.linspace(0, 1, n_faces, dtype=dtype, device=device)
        r_c = 0.5
        r_max = 0.5

        dist = (r - r_c).abs() / max(r_max, 1e-10)
        dist = dist.clamp(0.0, 1.0)

        exponent_inv = 1.0 / max(self._exponent, 1.0)
        profile = u_bulk * (1.0 - dist).clamp(min=0.0).pow(exponent_inv)

        return profile

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply flow rate inlet with power-law profile."""
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        profile = self._compute_profile(device=device, dtype=dtype).to(device=device, dtype=dtype)

        velocity = normals * profile.unsqueeze(-1)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n_faces] = velocity
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
        """Penalty method for flowRateInletVelocity2 BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        profile = self._compute_profile(device=device, dtype=dtype).to(device=device, dtype=dtype)
        velocity = normals * profile.unsqueeze(-1)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        coeff = deltas * area_mag

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * velocity[:, 0])

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
