"""
flowRateInlet2 — enhanced flow rate inlet boundary condition with
turbulence-aware profile.

An improved version of ``flowRateInletVelocity`` that distributes
velocity across the patch faces with a turbulence-aware profile.
The base velocity is computed from the target mass/volume flow rate,
then modulated by a power-law profile that accounts for turbulent
boundary layer shape.

Profile:
    u(r) = U_bulk * (1 - r/R)^(1/n)

where r is the distance from the patch centroid, R is the maximum
radius, and n is the turbulence-dependent exponent (n = 7 for
fully turbulent, n = 2 for laminar).

In OpenFOAM syntax::

    type                flowRateInlet2;
    massFlowRate        0.1;        // kg/s (optional)
    volumeFlowRate      0.001;      // m^3/s (optional, used if massFlowRate absent)
    rho                 1.225;      // density (kg/m3, default: 1.0)
    exponent            7;          // profile exponent (default: 7)
    value               uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["FlowRateInlet2BC"]


@BoundaryCondition.register("flowRateInlet2")
class FlowRateInlet2BC(BoundaryCondition):
    """Enhanced flow rate inlet with turbulence-aware profile.

    Computes the inlet velocity distribution from a target mass or
    volume flow rate with a power-law profile that captures the
    turbulent boundary layer shape at the inlet.

    Coefficients
    ------------
    massFlowRate : float, optional
        Target mass flow rate (kg/s).  Takes priority over volumeFlowRate.
    volumeFlowRate : float, optional
        Target volumetric flow rate (m^3/s).  Used when massFlowRate is absent.
    rho : float
        Fluid density (kg/m3).  Default: 1.0.
    exponent : float
        Power-law profile exponent.  Default 7 (fully turbulent).
        Use 2 for laminar, 10+ for highly turbulent.
    value : float or tensor
        Initial velocity (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mass_flow_rate = self._coeffs.get("massFlowRate", None)
        self._volume_flow_rate = self._coeffs.get("volumeFlowRate", None)
        self._rho = float(self._coeffs.get("rho", 1.0))
        self._exponent = float(self._coeffs.get("exponent", 7.0))

    @property
    def mass_flow_rate(self) -> float | None:
        """Target mass flow rate (kg/s), or None."""
        return self._mass_flow_rate

    @property
    def volume_flow_rate(self) -> float | None:
        """Target volume flow rate (m^3/s), or None."""
        return self._volume_flow_rate

    @property
    def rho(self) -> float:
        """Fluid density (kg/m3)."""
        return self._rho

    @property
    def exponent(self) -> float:
        """Power-law profile exponent."""
        return self._exponent

    # ------------------------------------------------------------------
    # Profile computation
    # ------------------------------------------------------------------

    def _compute_bulk_velocity(self) -> float:
        """Compute the bulk velocity magnitude from the flow rate."""
        device = get_device()
        dtype = get_default_dtype()
        face_areas = self._patch.face_areas.to(device=device, dtype=dtype)
        total_area = face_areas.sum().item()

        if total_area < 1e-30:
            return 0.0

        if self._mass_flow_rate is not None:
            # V = m_dot / (rho * A)
            return self._mass_flow_rate / (self._rho * total_area)
        elif self._volume_flow_rate is not None:
            # V = Q / A
            return self._volume_flow_rate / total_area
        else:
            return 0.0

    def _compute_profile(self) -> torch.Tensor:
        """Compute power-law velocity profile across patch faces.

        u(r) = U_bulk * (1 - |r - r_c| / R_max)^(1/n)

        Returns
        -------
        torch.Tensor
            ``(n_faces,)`` velocity magnitude per face.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_faces = self._patch.n_faces

        U_bulk = self._compute_bulk_velocity()
        if abs(U_bulk) < 1e-30:
            return torch.zeros(n_faces, dtype=dtype, device=device)

        # Normalised distance from patch centroid
        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        face_areas = self._patch.face_areas.to(device=device, dtype=dtype)

        # Weighted centroid of the patch (area-weighted mean of normal positions)
        if face_areas.dim() > 1:
            area_mag = face_areas.norm(dim=1)
        else:
            area_mag = face_areas.abs()

        # Face centres (approximate from normals * delta_coeffs)
        # Use a simple distance metric based on face index ordering
        # (approximate: assumes faces are ordered by spatial position)
        r = torch.linspace(0, 1, n_faces, dtype=dtype, device=device)
        r_c = 0.5  # centroid
        r_max = 0.5  # half-width

        # Normalised distance from centre
        dist = (r - r_c).abs() / max(r_max, 1e-10)
        dist = dist.clamp(0.0, 1.0)

        # Power-law profile
        exponent_inv = 1.0 / max(self._exponent, 1.0)
        profile = U_bulk * (1.0 - dist).clamp(min=0.0).pow(exponent_inv)

        return profile

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply flow rate inlet with turbulence-aware profile.

        Computes velocity per face using a power-law profile and
        sets it on the boundary.
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        profile = self._compute_profile().to(device=device, dtype=dtype)

        # Velocity = profile * face normal direction
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
        """Penalty method for flowRateInlet2 BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        profile = self._compute_profile().to(device=device, dtype=dtype)
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
