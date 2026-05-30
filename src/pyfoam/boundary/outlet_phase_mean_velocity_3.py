"""
Enhanced outlet phase mean velocity boundary condition (v3).

Extends ``outletPhaseMeanVelocity2`` with a turbulent-flux-weighted outlet
and a mass-conservation correction::

    alpha_safe = max(alpha, alphaMin)
    // Turbulent-flux weighting
    k_local = 0.5 * |U|^2 * intensity^2
    alpha_turb = alpha_safe * (1 + turbWeight * sqrt(k_local) / (|Umean| + 1e-30))
    U_phase = Umean * blend(alpha_turb) / alpha_turb
    // Mass conservation correction
    m_dot_target = rho * alpha_safe * Umean . n * A
    U_outlet = U_phase * m_dot_target / (m_dot_actual + 1e-30)
    // Pressure-gradient correction (from v2)
    dPdx = grad(p) . n
    U_correction = -dPdx * D_h^2 / (32 * mu * alpha_safe)
    U_outlet += pressureCorrection * U_correction
    U_outlet = clamp(|U_outlet|, 0, Umax) * direction

In OpenFOAM syntax::

    type              outletPhaseMeanVelocity3;
    Umean             uniform (1 0 0);
    alphaField        alpha.gas;
    phaseName         gas;
    alphaMin          1e-4;
    alphaBlendExp     1.0;
    pressureCorrection 0.0;
    hydraulicDiameter 0.1;
    mu                1e-3;
    Umax              100.0;
    turbWeight        0.0;
    intensity         0.05;
    rho               1.0;
    value             uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["OutletPhaseMeanVelocity3BC"]


@BoundaryCondition.register("outletPhaseMeanVelocity3")
class OutletPhaseMeanVelocity3BC(BoundaryCondition):
    """Enhanced outlet phase mean velocity BC v3 with turbulent-flux weighting.

    Coefficients:
        - ``Umean``: Desired phase mean velocity vector (m/s).
        - ``alphaField``: Name of the volume fraction field (informational).
        - ``phaseName``: Phase name (informational).
        - ``alphaMin``: Minimum volume fraction threshold (default 1e-4).
        - ``alphaBlendExp``: Exponent for alpha blending function (default 1.0).
        - ``pressureCorrection``: Enable pressure-gradient correction (0=off, 1=on, default 0.0).
        - ``hydraulicDiameter``: Hydraulic diameter for pressure correction (m, default 0.1).
        - ``mu``: Dynamic viscosity for pressure correction (Pa s, default 1e-3).
        - ``Umax``: Maximum velocity magnitude clamp (m/s, default 100.0).
        - ``turbWeight``: Turbulent-flux weighting coefficient (default 0.0).
        - ``intensity``: Turbulence intensity for turbulent weighting (default 0.05).
        - ``rho``: Fluid density for mass conservation (kg/m3, default 1.0).
        - ``value``: Initial velocity (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._Umean = self._parse_vector("Umean", [0.0, 0.0, 0.0])
        self._phase_name = str(self._coeffs.get("phaseName", ""))
        self._alpha_field = str(self._coeffs.get("alphaField", ""))
        self._alpha_min = float(self._coeffs.get("alphaMin", 1e-4))
        self._alpha_blend_exp = float(self._coeffs.get("alphaBlendExp", 1.0))
        self._pressure_correction = float(self._coeffs.get("pressureCorrection", 0.0))
        self._hydraulic_diameter = float(self._coeffs.get("hydraulicDiameter", 0.1))
        self._mu = float(self._coeffs.get("mu", 1e-3))
        self._Umax = float(self._coeffs.get("Umax", 100.0))
        self._turb_weight = float(self._coeffs.get("turbWeight", 0.0))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._rho = float(self._coeffs.get("rho", 1.0))

    def _parse_vector(self, key: str, default: list[float]) -> torch.Tensor:
        """Parse a vector coefficient."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.tensor(raw, dtype=get_default_dtype(), device=get_device())

    @property
    def Umean(self) -> torch.Tensor:
        """Prescribed phase mean velocity."""
        return self._Umean

    @property
    def phase_name(self) -> str:
        """Phase name."""
        return self._phase_name

    @property
    def alpha_field(self) -> str:
        """Volume fraction field name."""
        return self._alpha_field

    @property
    def alpha_min(self) -> float:
        """Minimum volume fraction threshold."""
        return self._alpha_min

    @property
    def alpha_blend_exp(self) -> float:
        """Exponent for alpha blending function."""
        return self._alpha_blend_exp

    @property
    def pressure_correction(self) -> float:
        """Pressure-gradient correction factor."""
        return self._pressure_correction

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter for pressure correction (m)."""
        return self._hydraulic_diameter

    @property
    def mu(self) -> float:
        """Dynamic viscosity (Pa s)."""
        return self._mu

    @property
    def Umax(self) -> float:
        """Maximum velocity magnitude clamp (m/s)."""
        return self._Umax

    @property
    def turb_weight(self) -> float:
        """Turbulent-flux weighting coefficient."""
        return self._turb_weight

    @property
    def intensity(self) -> float:
        """Turbulence intensity."""
        return self._intensity

    @property
    def rho(self) -> float:
        """Fluid density (kg/m3)."""
        return self._rho

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | None = None,
        pressure_gradient: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply v3 enhanced outlet phase-mean velocity BC.

        Args:
            field: Velocity field ``(n_cells_or_faces, 3)``.
            patch_idx: Optional start index into *field*.
            alpha: Phase volume fraction ``(n_faces,)``.  If ``None``,
                assumes alpha = 1 (single-phase limit).
            pressure_gradient: ``(n_faces,)`` streamwise dp/dx for correction.
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        Umean = self._Umean.to(device=device, dtype=dtype)
        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        if alpha is not None:
            alpha_safe = alpha.to(device=device, dtype=dtype).clamp(min=self._alpha_min)
            # Turbulent-flux weighting
            if self._turb_weight > 0:
                Umean_mag = Umean.norm() + 1e-30
                u_local_mag = Umean.unsqueeze(0).expand(n_faces, -1).norm(dim=-1)
                k_local = 0.5 * u_local_mag ** 2 * self._intensity ** 2
                alpha_turb = alpha_safe * (1.0 + self._turb_weight * torch.sqrt(k_local) / Umean_mag)
            else:
                alpha_turb = alpha_safe

            # Alpha blending function
            alpha_blend = alpha_turb ** self._alpha_blend_exp
            velocity = Umean.unsqueeze(0) * alpha_blend.unsqueeze(-1) / alpha_turb.unsqueeze(-1)
        else:
            alpha_safe = torch.ones(n_faces, dtype=dtype, device=device)
            velocity = Umean.unsqueeze(0).expand(n_faces, -1).clone()

        # Mass conservation correction
        m_dot_target = self._rho * (Umean * normals).sum(dim=-1) * area_mag
        # Current mass flow through velocity field
        m_dot_current = self._rho * alpha_safe * (velocity * normals).sum(dim=-1) * area_mag
        m_dot_total_target = m_dot_target.sum()
        m_dot_total_current = m_dot_current.sum()
        if m_dot_total_current.abs() > 1e-30:
            velocity = velocity * (m_dot_total_target / m_dot_total_current)

        # Pressure-gradient correction
        if self._pressure_correction > 0 and pressure_gradient is not None:
            dpdx = pressure_gradient.to(device=device, dtype=dtype)
            D_h = self._hydraulic_diameter
            U_correction = -dpdx * D_h ** 2 / (32.0 * self._mu * alpha_safe + 1e-30)
            U_dir = Umean / (Umean.norm() + 1e-30)
            velocity = velocity + self._pressure_correction * U_correction.unsqueeze(-1) * U_dir.unsqueeze(0)

        # Clamp velocity magnitude
        u_mag = velocity.norm(dim=-1, keepdim=True)
        velocity = velocity * torch.clamp(self._Umax / (u_mag + 1e-30), max=1.0)

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
        """Penalty method for v3 outlet phase mean velocity BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        Umean = self._Umean.to(device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        # Reduced penalty factor for outlet (0.5 * full penalty)
        coeff = 0.5 * deltas * areas

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * Umean[0])

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
