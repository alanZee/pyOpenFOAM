"""
Enhanced turbulent length scale inlet boundary condition (v6).

Extends ``turbulentLengthScaleInlet5`` with anisotropy-aware length scale
and a strain-rate-dependent correction::

    l_computed = C_mu^0.75 * k^1.5 / epsilon
    l_ref = lengthScaleFraction * D_h
    // Anisotropy correction
    a_ij = (u_i * u_j) / k - 2/3 * delta_ij
    II_a = a_ij : a_ij  (second invariant)
    l_aniso = l_computed * (1 + anisoCoeff * sqrt(II_a / 6))
    // Strain-rate correction
    S_mag = |S_ij|
    l_strain = l_computed * exp(-strainCoeff * S_mag * l_computed / (sqrt(k) + 1e-30))
    alpha_eff = alpha * (1 + beta * log10(1 + Re_t / ReTRef))
    l_mix = alpha_eff * l_aniso + (1 - alpha_eff) * l_strain
    l_mix = clamp(l_mix, lengthScaleMin, D_h * lengthScaleFraction)

In OpenFOAM syntax::

    type        turbulentLengthScaleInlet6;
    Cmu         0.09;
    intensity   0.05;
    lengthScale 0.01;
    kappa       0.41;
    lengthScaleMin  1e-6;
    lengthScaleFraction 0.07;
    hydraulicDiameter 0.1;
    alpha       0.8;
    beta        0.05;
    ReTRef      100.0;
    anisoCoeff  0.1;
    strainCoeff 0.5;
    value       uniform 0.01;
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentLengthScaleInlet6BC"]


@BoundaryCondition.register("turbulentLengthScaleInlet6")
class TurbulentLengthScaleInlet6BC(BoundaryCondition):
    """v6 enhanced turbulent length scale inlet with anisotropy and strain-rate correction.

    Coefficients:
        - ``Cmu``: Model constant (default 0.09).
        - ``intensity``: Fallback turbulence intensity (default 0.05).
        - ``lengthScale``: Fallback length scale (m, default 0.01).
        - ``kappa``: Von Karman constant (default 0.41).
        - ``lengthScaleMin``: Minimum length scale (m, default 1e-6).
        - ``lengthScaleFraction``: Fraction of hydraulic diameter for max (default 0.07).
        - ``hydraulicDiameter``: Hydraulic diameter (m, default 0.1).
        - ``alpha``: Base blending weight (default 0.8).
        - ``beta``: Re_t sensitivity coefficient (default 0.05).
        - ``ReTRef``: Reference turbulent Reynolds number (default 100.0).
        - ``anisoCoeff``: Anisotropy correction coefficient (default 0.1).
        - ``strainCoeff``: Strain-rate correction coefficient (default 0.5).
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
        self._alpha = float(self._coeffs.get("alpha", 0.8))
        self._beta = float(self._coeffs.get("beta", 0.05))
        self._Re_t_ref = float(self._coeffs.get("ReTRef", 100.0))
        self._aniso_coeff = float(self._coeffs.get("anisoCoeff", 0.1))
        self._strain_coeff = float(self._coeffs.get("strainCoeff", 0.5))

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
    def alpha(self) -> float:
        """Base blending weight."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Re_t sensitivity coefficient."""
        return self._beta

    @property
    def Re_t_ref(self) -> float:
        """Reference turbulent Reynolds number."""
        return self._Re_t_ref

    @property
    def aniso_coeff(self) -> float:
        """Anisotropy correction coefficient."""
        return self._aniso_coeff

    @property
    def strain_coeff(self) -> float:
        """Strain-rate correction coefficient."""
        return self._strain_coeff

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
        strain_rate: torch.Tensor | None = None,
        anisotropy_invariant: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face mixing length with anisotropy and strain-rate correction.

        Args:
            field: Turbulent length scale field.
            patch_idx: Optional start index.
            k: ``(n_faces,)`` turbulent kinetic energy.
            epsilon: ``(n_faces,)`` turbulent dissipation rate.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for Re_t estimation.
            strain_rate: ``(n_faces,)`` mean strain rate magnitude |S|.
            anisotropy_invariant: ``(n_faces,)`` second invariant II_a of Reynolds stress anisotropy.
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

        # Anisotropy correction
        if anisotropy_invariant is not None and self._aniso_coeff != 0:
            II_a = torch.clamp(anisotropy_invariant, min=0.0)
            l_aniso = l_computed * (1.0 + self._aniso_coeff * torch.sqrt(II_a / 6.0 + 1e-30))
        else:
            l_aniso = l_computed

        # Strain-rate correction
        if strain_rate is not None and self._strain_coeff != 0:
            k_for_strain = k if k is not None else torch.full(
                (n,), 1.5 * (self._intensity * 10.0) ** 2, dtype=dtype, device=device
            )
            l_strain = l_computed * torch.exp(
                -self._strain_coeff * strain_rate * l_computed / (torch.sqrt(k_for_strain) + 1e-30)
            )
        else:
            l_strain = l_computed

        # Adaptive blending coefficient based on Re_t
        alpha_eff = self._alpha
        if k is not None and epsilon is not None and nu is not None and nu > 0 and self._beta != 0:
            Re_t = k ** 2 / (nu * epsilon + 1e-30)
            Re_t_mean = Re_t.mean().item()
            alpha_eff = float(torch.clamp(
                torch.tensor(self._alpha * (1.0 + self._beta * math.log10(
                    1.0 + Re_t_mean / self._Re_t_ref
                ))),
                0.0, 1.0,
            ))

        # Hybrid blending
        l_mix = alpha_eff * l_aniso + (1.0 - alpha_eff) * l_strain

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
        """Penalty method for v6 mixing length inlet BC."""
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
