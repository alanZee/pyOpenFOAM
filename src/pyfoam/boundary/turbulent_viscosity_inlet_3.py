"""
Enhanced turbulent viscosity inlet boundary condition (v3).

Extends ``turbulentViscosityInlet2`` with a turbulent viscosity ratio limiter
and a profile-aware inlet that uses a reference viscosity ratio to blend
between a computed nut and a profile-based estimate::

    k = 1.5 * (I * |U|)^2
    epsilon = C_mu^0.75 * k^1.5 / l_mix
    nut_computed = C_mu * k^2 / epsilon
    nut_ratio = nut_computed / nu
    nut = clamp(nut_computed, nutMin, nutMax)
    nut = alpha * nut + (1 - alpha) * nutRatio_ref * nu

In OpenFOAM syntax::

    type        turbulentViscosityInlet3;
    Cmu         0.09;
    intensity   0.05;
    lengthScale 0.01;
    nutMin      1e-10;
    nutMax      1e4;
    alpha       1.0;
    nutRatioRef 10.0;
    value       uniform 0.001;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentViscosityInlet3BC"]


@BoundaryCondition.register("turbulentViscosityInlet3")
class TurbulentViscosityInlet3BC(BoundaryCondition):
    """v3 enhanced turbulent viscosity inlet with ratio limiter and blending.

    Coefficients:
        - ``Cmu``: Model constant (default 0.09).
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``nutMin``: Minimum nut clamp (default 1e-10).
        - ``nutMax``: Maximum nut clamp (default 1e4).
        - ``alpha``: Blending weight for computed nut (default 1.0).
        - ``nutRatioRef``: Reference turbulent-to-laminar viscosity ratio (default 10.0).
        - ``value``: Initial nut value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._nut_min = float(self._coeffs.get("nutMin", 1e-10))
        self._nut_max = float(self._coeffs.get("nutMax", 1e4))
        self._alpha = float(self._coeffs.get("alpha", 1.0))
        self._nut_ratio_ref = float(self._coeffs.get("nutRatioRef", 10.0))

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

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face nut with ratio limiter and blending.

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
            nut = self._C_mu * k ** 2 / (epsilon + 1e-30)
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
            epsilon_est = (self._C_mu ** 0.75) * (k_est ** 1.5) / (self._length_scale + 1e-30)
            nut = self._C_mu * k_est ** 2 / (epsilon_est + 1e-30)
        else:
            nut = torch.full((n,), 0.001, dtype=dtype, device=device)

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
        """Penalty method for v3 nut inlet BC."""
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
