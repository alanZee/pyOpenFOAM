"""
Enhanced turbulent viscosity inlet boundary condition (v8).

Extends ``turbulentViscosityInlet7`` with Reynolds-stress anisotropy
correction and an enhanced production limiter::

    k = 1.5 * (I * |U|)^2
    epsilon = C_mu^0.75 * k^1.5 / l_mix
    nut_base = C_mu * k^2 / epsilon
    // Reynolds-stress anisotropy correction
    a_ij = (P_k / epsilon - 1) * nut / k
    nut_aniso = nut_base * (1 + anisoCoeff * |a_ij|)
    // Wall-distance blending (from v7)
    // Schmidt correction (from v7)
    // Enhanced production limiter with strain-rate ratio
    S_ratio = |S| / (sqrt(epsilon / nu) + 1e-30)
    nut = min(nut, P_max * k / (2 * S_ij^2 + 1e-30) * (1 + strainCoeff * S_ratio))
    nut = clamp(nut, nutMin, nutMax)

In OpenFOAM syntax::

    type        turbulentViscosityInlet8;
    Cmu         0.09;
    intensity   0.05;
    lengthScale 0.01;
    kappa       0.41;
    nutMin      1e-10;
    nutMax      1e4;
    wallDist    0.01;
    yPlusCrit   11.0;
    schmidtCoeff 0.7;
    CprodMax    2.0;
    anisoCoeff  0.1;
    strainCoeff 0.05;
    value       uniform 0.001;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentViscosityInlet8BC"]


@BoundaryCondition.register("turbulentViscosityInlet8")
class TurbulentViscosityInlet8BC(BoundaryCondition):
    """v8 enhanced turbulent viscosity inlet with anisotropy correction.

    Coefficients:
        - ``Cmu``: Model constant (default 0.09).
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``kappa``: Von Karman constant (default 0.41).
        - ``nutMin``: Minimum nut clamp (default 1e-10).
        - ``nutMax``: Maximum nut clamp (default 1e4).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``yPlusCrit``: Critical y+ for wall blending (default 11.0).
        - ``schmidtCoeff``: Turbulent Schmidt blending coefficient (default 0.7).
        - ``CprodMax``: Production-to-dissipation limit ratio (default 2.0).
        - ``anisoCoeff``: Reynolds-stress anisotropy correction coefficient (default 0.1).
        - ``strainCoeff``: Strain-rate ratio limiter coefficient (default 0.05).
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
        self._wall_dist = float(self._coeffs.get("wallDist", 0.01))
        self._y_plus_crit = float(self._coeffs.get("yPlusCrit", 11.0))
        self._schmidt_coeff = float(self._coeffs.get("schmidtCoeff", 0.7))
        self._C_prod_max = float(self._coeffs.get("CprodMax", 2.0))
        self._aniso_coeff = float(self._coeffs.get("anisoCoeff", 0.1))
        self._strain_coeff = float(self._coeffs.get("strainCoeff", 0.05))

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
    def wall_dist(self) -> float:
        """Near-wall distance estimate (m)."""
        return self._wall_dist

    @property
    def y_plus_crit(self) -> float:
        """Critical y+ for wall blending."""
        return self._y_plus_crit

    @property
    def schmidt_coeff(self) -> float:
        """Turbulent Schmidt blending coefficient."""
        return self._schmidt_coeff

    @property
    def C_prod_max(self) -> float:
        """Production-to-dissipation limit ratio."""
        return self._C_prod_max

    @property
    def aniso_coeff(self) -> float:
        """Reynolds-stress anisotropy correction coefficient."""
        return self._aniso_coeff

    @property
    def strain_coeff(self) -> float:
        """Strain-rate ratio limiter coefficient."""
        return self._strain_coeff

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        nu: float | None = None,
        strain_rate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face nut with anisotropy correction and wall-distance blending.

        Args:
            field: Turbulent viscosity field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            k: ``(n_faces,)`` turbulent kinetic energy.
            epsilon: ``(n_faces,)`` turbulent dissipation rate.
            nu: Kinematic viscosity (m2/s).
            strain_rate: ``(n_faces,)`` mean strain rate magnitude |S|.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if k is not None and epsilon is not None:
            nut_base = self._C_mu * k ** 2 / (epsilon + 1e-30)
            # Reynolds-stress anisotropy correction
            nut_est_for_prod = nut_base
            P_k = 2.0 * nut_est_for_prod * (strain_rate ** 2 if strain_rate is not None else
                                              epsilon / (2.0 * nut_est_for_prod + 1e-30))
            a_ij = torch.clamp((P_k / (epsilon + 1e-30) - 1.0) * nut_base / (k + 1e-30), -1.0, 1.0)
            nut_base = nut_base * (1.0 + self._aniso_coeff * a_ij.abs())
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
            epsilon_est = (self._C_mu ** 0.75) * (k_est ** 1.5) / (self._length_scale + 1e-30)
            nut_base = self._C_mu * k_est ** 2 / (epsilon_est + 1e-30)
        else:
            nut_base = torch.full((n,), 0.001, dtype=dtype, device=device)

        # Wall-distance blending
        if nu is not None and nu > 0:
            k_for_wall = k if k is not None else nut_base / (self._C_mu + 1e-30) * 0.1
            u_tau = (self._C_mu ** 0.25) * torch.sqrt(torch.clamp(k_for_wall, min=1e-30))
            y_plus = u_tau * self._wall_dist / nu

            # Wall-viscosity model
            nut_wall = self._kappa * u_tau * self._wall_dist * torch.exp(-y_plus / (self._y_plus_crit + 1e-30))

            # Schmidt correction
            Sc_t = 1.0 / (self._schmidt_coeff + (1.0 - self._schmidt_coeff) *
                          torch.exp(-y_plus / (self._y_plus_crit + 1e-30)))
            nut_sc = nut_base * Sc_t

            # Blending function
            blend = torch.clamp(y_plus / (self._y_plus_crit + 1e-30), 0.0, 1.0)
            nut = blend * nut_sc + (1.0 - blend) * nut_wall
        else:
            nut = nut_base

        # Enhanced production limiter with strain-rate ratio
        if strain_rate is not None and self._C_prod_max > 0:
            k_for_lim = k if k is not None else nut / (self._C_mu + 1e-30) * 0.1
            eps_for_lim = (self._C_mu ** 0.75) * (k_for_lim ** 1.5) / (self._length_scale + 1e-30)
            P_max = self._C_prod_max * eps_for_lim / (k_for_lim + 1e-30)
            # Strain-rate ratio correction
            if nu is not None and nu > 0:
                S_ratio = strain_rate / (torch.sqrt(eps_for_lim / (nu + 1e-30)) + 1e-30)
                nut_limit = P_max * k_for_lim / (2.0 * strain_rate ** 2 + 1e-30) * (1.0 + self._strain_coeff * S_ratio)
            else:
                nut_limit = P_max * k_for_lim / (2.0 * strain_rate ** 2 + 1e-30)
            nut = torch.min(nut, nut_limit)

        # Clamp to physical range
        nut = torch.clamp(nut, self._nut_min, self._nut_max)

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
        """Penalty method for v8 nut inlet BC."""
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
