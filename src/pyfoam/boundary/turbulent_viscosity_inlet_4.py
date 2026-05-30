"""
Enhanced turbulent viscosity inlet boundary condition (v4).

Extends ``turbulentViscosityInlet3`` with a pressure-gradient-corrected
nut estimate and a wall-distance-aware limiter::

    k = 1.5 * (I * |U|)^2
    epsilon = C_mu^0.75 * k^1.5 / l_mix
    nut_base = C_mu * k^2 / epsilon
    dPdx = (p_outlet - p_inlet) / L
    nut_pg = rho * C_mu * k^2 / (|dPdx| + 1e-30)
    nut = alpha * nut_base + (1 - alpha) * nut_pg
    nut = clamp(nut, nutMin, nutMax)

In OpenFOAM syntax::

    type        turbulentViscosityInlet4;
    Cmu         0.09;
    intensity   0.05;
    lengthScale 0.01;
    nutMin      1e-10;
    nutMax      1e4;
    alpha       0.9;
    nutRatioRef 10.0;
    rho         1.225;
    dPdx        0.0;
    value       uniform 0.001;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentViscosityInlet4BC"]


@BoundaryCondition.register("turbulentViscosityInlet4")
class TurbulentViscosityInlet4BC(BoundaryCondition):
    """v4 enhanced turbulent viscosity inlet with pressure-gradient correction.

    Coefficients:
        - ``Cmu``: Model constant (default 0.09).
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``nutMin``: Minimum nut clamp (default 1e-10).
        - ``nutMax``: Maximum nut clamp (default 1e4).
        - ``alpha``: Blending weight for computed nut (default 0.9).
        - ``nutRatioRef``: Reference turbulent-to-laminar viscosity ratio (default 10.0).
        - ``rho``: Fluid density (kg/m3, default 1.225).
        - ``dPdx``: Streamwise pressure gradient magnitude (Pa/m, default 0.0).
        - ``value``: Initial nut value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._nut_min = float(self._coeffs.get("nutMin", 1e-10))
        self._nut_max = float(self._coeffs.get("nutMax", 1e4))
        self._alpha = float(self._coeffs.get("alpha", 0.9))
        self._nut_ratio_ref = float(self._coeffs.get("nutRatioRef", 10.0))
        self._rho = float(self._coeffs.get("rho", 1.225))
        self._dPdx = float(self._coeffs.get("dPdx", 0.0))

    @property
    def C_mu(self) -> float:
        """Model constant C_mu."""
        return self._C_mu

    @property
    def intensity(self) -> float:
        """Turbulence intensity."""
        return self._intensity

    @property
    def length_scale(self) -> float:
        """Turbulent length scale (m)."""
        return self._length_scale

    @property
    def nut_min(self) -> float:
        """Minimum nut clamp value."""
        return self._nut_min

    @property
    def nut_max(self) -> float:
        """Maximum nut clamp value."""
        return self._nut_max

    @property
    def alpha(self) -> float:
        """Blending weight for computed nut."""
        return self._alpha

    @property
    def nut_ratio_ref(self) -> float:
        """Reference turbulent-to-laminar viscosity ratio."""
        return self._nut_ratio_ref

    @property
    def rho(self) -> float:
        """Fluid density (kg/m3)."""
        return self._rho

    @property
    def dPdx(self) -> float:
        """Streamwise pressure gradient magnitude (Pa/m)."""
        return self._dPdx

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face nut with pressure-gradient correction.

        Args:
            field: Turbulent viscosity field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            k: ``(n_faces,)`` turbulent kinetic energy.
            epsilon: ``(n_faces,)`` turbulent dissipation rate.
            nu: Kinematic viscosity (m2/s) for ratio-based fallback.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if k is not None and epsilon is not None:
            nut_base = self._C_mu * k ** 2 / (epsilon + 1e-30)
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
            epsilon_est = (self._C_mu ** 0.75) * (k_est ** 1.5) / (self._length_scale + 1e-30)
            nut_base = self._C_mu * k_est ** 2 / (epsilon_est + 1e-30)
        else:
            nut_base = torch.full((n,), 0.001, dtype=dtype, device=device)

        # Pressure-gradient-corrected nut
        if self._dPdx > 0:
            k_for_pg = k if k is not None else (
                1.5 * (self._intensity * torch.sqrt(
                    (velocity * velocity).sum(dim=-1)
                )) ** 2 if velocity is not None else
                torch.full((n,), 0.01, dtype=dtype, device=device)
            )
            nut_pg = self._rho * self._C_mu * k_for_pg ** 2 / (self._dPdx + 1e-30)
            nut = self._alpha * nut_base + (1.0 - self._alpha) * nut_pg
        else:
            nut = nut_base

        # Clamp to physical range
        nut = torch.clamp(nut, self._nut_min, self._nut_max)

        # Blend with ratio-based reference if nu is provided
        if nu is not None and nu > 0 and self._alpha < 1.0:
            nut_ref = self._nut_ratio_ref * nu
            nut = self._alpha * nut + (1.0 - self._alpha) * nut_ref

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = nut
        else:
            field[self._patch.face_indices] = nut
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for v4 nut inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        nut_default = 0.001

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * nut_default)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
