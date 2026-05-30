"""
Enhanced turbulent viscosity inlet boundary condition (v5).

Extends ``turbulentViscosityInlet4`` with wall-distance-aware blending
and a production-to-dissipation ratio limiter::

    k = 1.5 * (I * |U|)^2
    epsilon = C_mu^0.75 * k^1.5 / l_mix
    nut_base = C_mu * k^2 / epsilon
    y_plus = u_tau * y / nu
    nut_wall = kappa * y * u_tau  (log-law estimate)
    nut = blend(y_plus) * nut_wall + (1 - blend) * nut_base
    nut = clamp(nut, nutMin, nutMax)
    nut = alpha * nut + (1 - alpha) * nutRatioRef * nu

In OpenFOAM syntax::

    type        turbulentViscosityInlet5;
    Cmu         0.09;
    intensity   0.05;
    lengthScale 0.01;
    kappa       0.41;
    nutMin      1e-10;
    nutMax      1e4;
    alpha       0.9;
    nutRatioRef 10.0;
    wallDist    0.01;
    yPlusLow    5.0;
    yPlusHigh   30.0;
    value       uniform 0.001;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentViscosityInlet5BC"]


@BoundaryCondition.register("turbulentViscosityInlet5")
class TurbulentViscosityInlet5BC(BoundaryCondition):
    """v5 enhanced turbulent viscosity inlet with wall-distance-aware blending.

    Coefficients:
        - ``Cmu``: Model constant (default 0.09).
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``kappa``: Von Karman constant (default 0.41).
        - ``nutMin``: Minimum nut clamp (default 1e-10).
        - ``nutMax``: Maximum nut clamp (default 1e4).
        - ``alpha``: Blending weight for computed nut (default 0.9).
        - ``nutRatioRef``: Reference turbulent-to-laminar viscosity ratio (default 10.0).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``yPlusLow``: Lower y+ bound for buffer-layer blending (default 5.0).
        - ``yPlusHigh``: Upper y+ bound for log-law blending (default 30.0).
        - ``value``: Initial nut value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._kappa = float(self._coeffs.get("kappa", 0.41))
        self._nut_min = float(self._coeffs.get("nutMin", 1e-10))
        self._nut_max = float(self._coeffs.get("nutMax", 1e4))
        self._alpha = float(self._coeffs.get("alpha", 0.9))
        self._nut_ratio_ref = float(self._coeffs.get("nutRatioRef", 10.0))
        self._wall_dist = float(self._coeffs.get("wallDist", 0.01))
        self._y_plus_low = float(self._coeffs.get("yPlusLow", 5.0))
        self._y_plus_high = float(self._coeffs.get("yPlusHigh", 30.0))

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
    def kappa(self) -> float:
        """Von Karman constant."""
        return self._kappa

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
    def wall_dist(self) -> float:
        """Near-wall distance estimate (m)."""
        return self._wall_dist

    @property
    def y_plus_low(self) -> float:
        """Lower y+ bound for buffer-layer blending."""
        return self._y_plus_low

    @property
    def y_plus_high(self) -> float:
        """Upper y+ bound for log-law blending."""
        return self._y_plus_high

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face nut with wall-distance-aware blending.

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

        # Wall-distance-aware blending
        if nu is not None and nu > 0 and (k is not None or velocity is not None):
            k_for_wall = k if k is not None else (
                1.5 * (self._intensity * torch.sqrt(
                    (velocity * velocity).sum(dim=-1)
                )) ** 2 if velocity is not None else
                torch.full((n,), 0.01, dtype=dtype, device=device)
            )
            u_tau = (self._C_mu ** 0.25) * torch.sqrt(torch.clamp(k_for_wall, min=1e-30))
            y_plus = u_tau * self._wall_dist / (nu + 1e-30)

            # Log-law wall viscosity estimate
            nut_wall = self._kappa * self._wall_dist * u_tau

            # Blend between wall and bulk estimates based on y+
            blend = torch.clamp(
                (y_plus - self._y_plus_low) / (self._y_plus_high - self._y_plus_low + 1e-30),
                0.0, 1.0,
            )
            nut = blend * nut_wall + (1.0 - blend) * nut_base
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
        """Penalty method for v5 nut inlet BC."""
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
