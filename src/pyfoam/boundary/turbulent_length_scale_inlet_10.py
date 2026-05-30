"""
Enhanced turbulent length scale inlet boundary condition (v10).

Extends ``turbulentLengthScaleInlet9`` with a turbulent Prandtl number
correction and an enhanced wall-function with dynamic transition::

    l_computed = C_mu^0.75 * k^1.5 / epsilon
    l_ref = lengthScaleFraction * D_h
    // Pressure-gradient (from v9)
    // Two-regime wall model (from v9)
    // Prandtl correction (from v9)
    // Wake function (from v9)
    // Dynamic wall-function transition
    y_plus = u_tau * y / nu
    blend_dyn = tanh((y_plus - yPlusTrans) / transWidth)
    l_dyn = blend_dyn * l_wake + (1 - blend_dyn) * l_visc
    // Production-ratio correction
    P_k = 2 * nut * S^2
    P_eps_ratio = P_k / (eps + 1e-30)
    l_prod = l_dyn * (1 + prodCoeff * (P_eps_ratio - 1))
    l_mix = clamp(l_prod, lengthScaleMin, D_h * lengthScaleFraction)

In OpenFOAM syntax::

    type        turbulentLengthScaleInlet10;
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
    wakeCoeff   0.2;
    yPlusWake   15.0;
    ReTauRef    100.0;
    ReTauBlend  50.0;
    pgCoeff     0.05;
    pgNormRef   1.0;
    rho         1.225;
    prodCoeff   0.05;
    yPlusTrans  11.0;
    transWidth  5.0;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentLengthScaleInlet10BC"]


@BoundaryCondition.register("turbulentLengthScaleInlet10")
class TurbulentLengthScaleInlet10BC(BoundaryCondition):
    """v10 enhanced turbulent length scale inlet with dynamic wall transition and production correction.

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
        - ``wakeCoeff``: Wake-function amplitude coefficient (default 0.2).
        - ``yPlusWake``: y+ threshold for wake function (default 15.0).
        - ``ReTauRef``: Reference friction Reynolds number for wake log-correction (default 100.0).
        - ``ReTauBlend``: Friction Reynolds number for asymptotic blending (default 50.0).
        - ``pgCoeff``: Pressure-gradient correction coefficient (default 0.05).
        - ``pgNormRef``: Normalized pressure gradient reference (default 1.0).
        - ``rho``: Fluid density for pressure-gradient normalization (kg/m3, default 1.225).
        - ``prodCoeff``: Production-ratio correction coefficient (default 0.05).
        - ``yPlusTrans``: y+ for dynamic wall-function transition (default 11.0).
        - ``transWidth``: Transition width in y+ units (default 5.0).
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
        self._wake_coeff = float(self._coeffs.get("wakeCoeff", 0.2))
        self._y_plus_wake = float(self._coeffs.get("yPlusWake", 15.0))
        self._Re_tau_ref = float(self._coeffs.get("ReTauRef", 100.0))
        self._Re_tau_blend = float(self._coeffs.get("ReTauBlend", 50.0))
        self._pg_coeff = float(self._coeffs.get("pgCoeff", 0.05))
        self._pg_norm_ref = float(self._coeffs.get("pgNormRef", 1.0))
        self._rho = float(self._coeffs.get("rho", 1.225))
        self._prod_coeff = float(self._coeffs.get("prodCoeff", 0.05))
        self._y_plus_trans = float(self._coeffs.get("yPlusTrans", 11.0))
        self._trans_width = float(self._coeffs.get("transWidth", 5.0))

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

    @property
    def wake_coeff(self) -> float:
        """Wake-function amplitude coefficient."""
        return self._wake_coeff

    @property
    def y_plus_wake(self) -> float:
        """y+ threshold for wake function."""
        return self._y_plus_wake

    @property
    def Re_tau_ref(self) -> float:
        """Reference friction Reynolds number for wake log-correction."""
        return self._Re_tau_ref

    @property
    def Re_tau_blend(self) -> float:
        """Friction Reynolds number for asymptotic blending."""
        return self._Re_tau_blend

    @property
    def pg_coeff(self) -> float:
        """Pressure-gradient correction coefficient."""
        return self._pg_coeff

    @property
    def pg_norm_ref(self) -> float:
        """Normalized pressure gradient reference."""
        return self._pg_norm_ref

    @property
    def rho(self) -> float:
        """Fluid density for pressure-gradient normalization (kg/m3)."""
        return self._rho

    @property
    def prod_coeff(self) -> float:
        """Production-ratio correction coefficient."""
        return self._prod_coeff

    @property
    def y_plus_trans(self) -> float:
        """y+ for dynamic wall-function transition."""
        return self._y_plus_trans

    @property
    def trans_width(self) -> float:
        """Transition width in y+ units."""
        return self._trans_width

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
        pressure_gradient: torch.Tensor | None = None,
        strain_rate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face mixing length with dynamic wall transition and production correction.

        Args:
            field: Turbulent length scale field.
            patch_idx: Optional start index.
            k: ``(n_faces,)`` turbulent kinetic energy.
            epsilon: ``(n_faces,)`` turbulent dissipation rate.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for wall model.
            pressure_gradient: ``(n_faces,)`` streamwise dp/dx for correction.
            strain_rate: ``(n_faces,)`` mean strain rate magnitude |S|.
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

        # Pressure-gradient correction (from v9)
        l_pg = l_computed
        if pressure_gradient is not None and self._pg_coeff > 0:
            dp_dx = pressure_gradient.to(device=device, dtype=dtype)
            if k is not None:
                u_tau_sq = (self._C_mu ** 0.5) * k
            else:
                u_tau_sq = torch.full((n,), 1.0, dtype=dtype, device=device)
            dp_dx_norm = dp_dx.abs() / (self._rho * u_tau_sq + 1e-30)
            l_pg = l_computed * (1.0 - self._pg_coeff * torch.tanh(dp_dx_norm / (self._pg_norm_ref + 1e-30)))

        # Two-regime wall model with dynamic transition
        if nu is not None and nu > 0 and k is not None:
            u_tau = (self._C_mu ** 0.25) * torch.sqrt(torch.clamp(k, min=1e-30))
            y_plus = u_tau * self._wall_dist / nu

            # Viscous sublayer model
            l_visc = self._kappa * self._wall_dist * (1.0 - torch.exp(-y_plus / self._A_visc))
            l_log = torch.full_like(l_pg, self._kappa * self._wall_dist / (
                1.0 + self._kappa * self._wall_dist / (l_max + 1e-30)
            ))
            l_wall = torch.min(l_visc, l_log)

            # Enhanced Prandtl number correction
            Pr_t = self._prandtl_coeff + (1.0 - self._prandtl_coeff) * torch.exp(
                -y_plus / (self._y_plus_prandtl + 1e-30)
            )
            l_pr = l_pg * Pr_t

            # Wake-function correction
            wake = self._wake_coeff * (1.0 - torch.exp(-y_plus / (self._y_plus_wake + 1e-30))) ** 2
            Re_tau = u_tau * self._wall_dist / (nu + 1e-30)
            l_wake = l_pr * (1.0 + wake * torch.log(1.0 + Re_tau / (self._Re_tau_ref + 1e-30)))

            # Dynamic wall-function transition (tanh blending)
            blend_dyn = torch.clamp(
                torch.tanh((y_plus - self._y_plus_trans) / (self._trans_width + 1e-30)),
                0.0, 1.0,
            )
            l_dyn = blend_dyn * l_wake + (1.0 - blend_dyn) * l_visc

            # Reynolds-number asymptotic blending
            Re_blend = torch.clamp(Re_tau / (self._Re_tau_blend + 1e-30), 0.0, 1.0)
            l_mix = Re_blend * l_dyn + (1.0 - Re_blend) * l_wall
        else:
            l_mix = l_pg

        # Production-ratio correction
        if self._prod_coeff > 0 and k is not None and epsilon is not None:
            nut_est = self._C_mu * k ** 2 / (epsilon + 1e-30)
            if strain_rate is not None:
                P_k = 2.0 * nut_est * strain_rate ** 2
            else:
                P_k = epsilon  # equilibrium assumption
            P_eps_ratio = P_k / (epsilon + 1e-30)
            l_mix = l_mix * (1.0 + self._prod_coeff * (P_eps_ratio - 1.0))

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
        """Penalty method for v10 mixing length inlet BC."""
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
