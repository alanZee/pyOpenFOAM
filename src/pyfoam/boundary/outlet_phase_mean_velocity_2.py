"""
Enhanced outlet phase mean velocity boundary condition (v2).

Extends ``outletPhaseMeanVelocity`` with a pressure-gradient-corrected
outlet velocity and a volume-fraction-weighted blending::

    alpha_safe = max(alpha, alphaMin)
    U_phase = Umean * blend(alpha_safe) / alpha_safe
    dPdx = grad(p) . n  (streamwise pressure gradient)
    U_correction = -dPdx * hydraulicDiameter^2 / (32 * mu * alpha_safe)
    U_outlet = U_phase + pressureCorrection * U_correction
    U_outlet = clamp(|U_outlet|, 0, Umax) * direction

In OpenFOAM syntax::

    type              outletPhaseMeanVelocity2;
    Umean             uniform (1 0 0);
    alphaField        alpha.gas;
    phaseName         gas;
    alphaMin          1e-4;
    alphaBlendExp     1.0;
    pressureCorrection 0.0;
    hydraulicDiameter 0.1;
    mu                1e-3;
    Umax              100.0;
    value             uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["OutletPhaseMeanVelocity2BC"]


@BoundaryCondition.register("outletPhaseMeanVelocity2")
class OutletPhaseMeanVelocity2BC(BoundaryCondition):
    """Enhanced outlet phase mean velocity BC v2 with pressure-gradient correction.

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

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | None = None,
        pressure_gradient: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply enhanced outlet phase-mean velocity BC.

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

        if alpha is not None:
            alpha_safe = alpha.to(device=device, dtype=dtype).clamp(min=self._alpha_min)
            # Alpha blending function: smooth transition near alpha_min
            alpha_blend = alpha_safe ** self._alpha_blend_exp
            velocity = Umean.unsqueeze(0) * alpha_blend.unsqueeze(-1) / alpha_safe.unsqueeze(-1)
        else:
            velocity = Umean.unsqueeze(0).expand(n_faces, -1).clone()

        # Pressure-gradient correction
        if self._pressure_correction > 0 and pressure_gradient is not None:
            dpdx = pressure_gradient.to(device=device, dtype=dtype)
            D_h = self._hydraulic_diameter
            alpha_for_corr = alpha_safe if alpha is not None else torch.ones(n_faces, dtype=dtype, device=device)
            U_correction = -dpdx * D_h ** 2 / (32.0 * self._mu * alpha_for_corr + 1e-30)
            # Add correction along the velocity direction
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
        """Penalty method for enhanced outlet phase mean velocity v2.

        Uses reduced penalty for outlet treatment.
        """
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
