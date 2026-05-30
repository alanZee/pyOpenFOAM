"""
Enhanced turbulent length scale inlet boundary condition (v7).

Extends ``turbulentLengthScaleInlet6`` with a two-regime wall model
and a turbulent Prandtl number correction::

    l_computed = C_mu^0.75 * k^1.5 / epsilon
    l_ref = lengthScaleFraction * D_h
    // Two-regime wall model
    y_plus = u_tau * y / nu
    l_visc = kappa * y * (1 - exp(-y_plus / A_visc))
    l_log  = kappa * y / (1 + kappa * y / l_ref)
    l_wall = min(l_visc, l_log)
    // Prandtl number correction
    Pr_t = prandtlCoeff + (1 - prandtlCoeff) * exp(-y_plus / yPlusPrandtl)
    l_pr = l_computed * Pr_t
    // Blending based on y+
    alpha_eff = clamp(y_plus / yPlusBlend, 0, 1)
    l_mix = alpha_eff * l_pr + (1 - alpha_eff) * l_wall
    l_mix = clamp(l_mix, lengthScaleMin, D_h * lengthScaleFraction)

In OpenFOAM syntax::

    type        turbulentLengthScaleInlet7;
    Cmu         0.09;
    intensity   0.05;
    lengthScale 0.01;
    kappa       0.41;
    lengthScaleMin  1e-6;
    lengthScaleFraction 0.07;
    hydraulicDiameter 0.1;
    wallDist    0.01;
    Avisc       26.0;
    yPlusBlend  30.0;
    yPlusPrandtl 50.0;
    prandtlCoeff 0.85;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentLengthScaleInlet7BC"]


@BoundaryCondition.register("turbulentLengthScaleInlet7")
class TurbulentLengthScaleInlet7BC(BoundaryCondition):
    """v7 enhanced turbulent length scale inlet with two-regime wall model.

    Coefficients:
        - ``Cmu``: Model constant (default 0.09).
        - ``intensity``: Fallback turbulence intensity (default 0.05).
        - ``lengthScale``: Fallback length scale (m, default 0.01).
        - ``kappa``: Von Karman constant (default 0.41).
        - ``lengthScaleMin``: Minimum length scale (m, default 1e-6).
        - ``lengthScaleFraction``: Fraction of hydraulic diameter for max (default 0.07).
        - ``hydraulicDiameter``: Hydraulic diameter (m, default 0.1).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``Avisc``: Van Driest damping constant (default 26.0).
        - ``yPlusBlend``: y+ threshold for blending wall and log-law (default 30.0).
        - ``yPlusPrandtl``: y+ threshold for Prandtl correction (default 50.0).
        - ``prandtlCoeff``: Turbulent Prandtl blending coefficient (default 0.85).
        - ``value``: Initial l_mix value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._kappa = float(self._coeffs.get("kappa", 0.41))
        self._length_scale_min = float(self._coeffs.get("lengthScaleMin", 1e-6))
        self._length_scale_fraction = float(self._coeffs.get("lengthScaleFraction", 0.07))
        self._hydraulic_diameter = float(self._coeffs.get("hydraulicDiameter", 0.1))
        self._wall_dist = float(self._coeffs.get("wallDist", 0.01))
        self._A_visc = float(self._coeffs.get("Avisc", 26.0))
        self._y_plus_blend = float(self._coeffs.get("yPlusBlend", 30.0))
        self._y_plus_prandtl = float(self._coeffs.get("yPlusPrandtl", 50.0))
        self._prandtl_coeff = float(self._coeffs.get("prandtlCoeff", 0.85))

    @property
    def C_mu(self) -> float:
        """Model constant C_mu."""
        return self._C_mu

    @property
    def intensity(self) -> float:
        """Fallback turbulence intensity."""
        return self._intensity

    @property
    def length_scale(self) -> float:
        """Fallback length scale (m)."""
        return self._length_scale

    @property
    def kappa(self) -> float:
        """Von Karman constant."""
        return self._kappa

    @property
    def length_scale_min(self) -> float:
        """Minimum length scale (m)."""
        return self._length_scale_min

    @property
    def length_scale_fraction(self) -> float:
        """Fraction of hydraulic diameter for maximum length scale."""
        return self._length_scale_fraction

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter (m)."""
        return self._hydraulic_diameter

    @property
    def wall_dist(self) -> float:
        """Near-wall distance estimate (m)."""
        return self._wall_dist

    @property
    def A_visc(self) -> float:
        """Van Driest damping constant."""
        return self._A_visc

    @property
    def y_plus_blend(self) -> float:
        """y+ threshold for blending wall and log-law."""
        return self._y_plus_blend

    @property
    def y_plus_prandtl(self) -> float:
        """y+ threshold for Prandtl correction."""
        return self._y_plus_prandtl

    @property
    def prandtl_coeff(self) -> float:
        """Turbulent Prandtl blending coefficient."""
        return self._prandtl_coeff

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face mixing length with two-regime wall model.

        Args:
            field: Turbulent length scale field.
            patch_idx: Optional start index.
            k: ``(n_faces,)`` turbulent kinetic energy.
            epsilon: ``(n_faces,)`` turbulent dissipation rate.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for wall model.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        l_max = self._length_scale_fraction * self._hydraulic_diameter

        if k is not None and epsilon is not None:
            l_computed = (self._C_mu ** 0.75) * (k ** 1.5) / (epsilon + 1e-30)
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
            epsilon_est = (self._C_mu ** 0.75) * (k_est ** 1.5) / (self._length_scale + 1e-30)
            l_computed = (self._C_mu ** 0.75) * (k_est ** 1.5) / (epsilon_est + 1e-30)
        else:
            l_computed = torch.full((n,), self._length_scale, dtype=dtype, device=device)

        # Two-regime wall model with Prandtl correction
        if nu is not None and nu > 0 and k is not None:
            u_tau = (self._C_mu ** 0.25) * torch.sqrt(torch.clamp(k, min=1e-30))
            y_plus = u_tau * self._wall_dist / nu

            # Viscous sublayer model
            l_visc = self._kappa * self._wall_dist * (1.0 - torch.exp(-y_plus / self._A_visc))
            # Log-law model (tensor form for torch.min)
            l_log = torch.full_like(l_computed, self._kappa * self._wall_dist / (
                1.0 + self._kappa * self._wall_dist / (l_max + 1e-30)
            ))
            l_wall = torch.min(l_visc, l_log)

            # Prandtl number correction
            Pr_t = self._prandtl_coeff + (1.0 - self._prandtl_coeff) * torch.exp(
                -y_plus / (self._y_plus_prandtl + 1e-30)
            )
            l_pr = l_computed * Pr_t

            # Blending based on y+
            alpha_eff = torch.clamp(y_plus / (self._y_plus_blend + 1e-30), 0.0, 1.0)
            l_mix = alpha_eff * l_pr + (1.0 - alpha_eff) * l_wall
        else:
            l_mix = l_computed

        # Clamp to physical range
        l_mix = torch.clamp(l_mix, self._length_scale_min, l_max)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = l_mix
        else:
            field[self._patch.face_indices] = l_mix
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for v7 mixing length inlet BC."""
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
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * self._length_scale)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
