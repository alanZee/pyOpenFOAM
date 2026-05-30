"""
Enhanced turbulent kinetic energy inlet boundary condition (v10).

Extends ``turbulentKineticEnergyInlet9`` with a compressibility-aware
correction and an enhanced production limiter based on local Mach number::

    k_intensity = 1.5 * (I * |U|)^2
    k_length = (epsilon * l_mix / C_mu^0.75)^(2/3)
    // Adaptive blending (from v9)
    // Dynamic production/dissipation balance (from v9)
    // Spectral energy correction (from v9)
    // Production anisotropy correction (from v9)
    // Compressibility correction
    Ma_t = sqrt(2 * k) / c
    Ma_eff = clamp(Ma_t, 0, MaLimit)
    k_comp = k * (1 + compCoeff * Ma_eff^2)
    // Enhanced production limiter
    P_k = 2 * nut * S_ij^2
    P_limit = CprodMax * eps
    k_out = min(k_comp, k_intensity + P_limit * dt / (1 + dt * eps / k))
    k_out = clamp(k_out, kMin, kMax)

In OpenFOAM syntax::

    type        turbulentKineticEnergyInlet10;
    intensity   0.05;
    lengthScale 0.01;
    Cmu         0.09;
    kappa       0.41;
    alpha       0.8;
    beta        0.05;
    ReTRef      100.0;
    kMin        1e-10;
    kMax        100.0;
    dynamicCoeff 0.1;
    wallDist    0.01;
    spectralCoeff 0.05;
    spectralRatioRef 5.0;
    anisoProdCoeff 0.05;
    compCoeff   0.1;
    MaLimit     0.5;
    CprodMax    2.0;
    dt          1e-3;
    nu          1e-5;
    value       uniform 0.01;
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentKineticEnergyInlet10BC"]


@BoundaryCondition.register("turbulentKineticEnergyInlet10")
class TurbulentKineticEnergyInlet10BC(BoundaryCondition):
    """v10 enhanced turbulent kinetic energy inlet with compressibility correction.

    Coefficients:
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``kappa``: Von Karman constant (default 0.41).
        - ``alpha``: Base blending weight (default 0.8).
        - ``beta``: Re_t sensitivity coefficient (default 0.05).
        - ``ReTRef``: Reference turbulent Reynolds number (default 100.0).
        - ``kMin``: Minimum k clamp (default 1e-10).
        - ``kMax``: Maximum k clamp (default 100.0).
        - ``dynamicCoeff``: Dynamic production/dissipation balance coefficient (default 0.1).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``spectralCoeff``: Spectral energy correction coefficient (default 0.05).
        - ``spectralRatioRef``: Reference frequency ratio for spectral correction (default 5.0).
        - ``anisoProdCoeff``: Production anisotropy correction coefficient (default 0.05).
        - ``compCoeff``: Compressibility correction coefficient (default 0.1).
        - ``MaLimit``: Turbulent Mach number limit (default 0.5).
        - ``CprodMax``: Production-to-dissipation limit ratio (default 2.0).
        - ``dt``: Time step for production limiting (s, default 1e-3).
        - ``nu``: Kinematic viscosity for Kolmogorov scale (m2/s, default 1e-5).
        - ``value``: Initial k value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._kappa = float(self._coeffs.get("kappa", 0.41))
        self._alpha = float(self._coeffs.get("alpha", 0.8))
        self._beta = float(self._coeffs.get("beta", 0.05))
        self._Re_t_ref = float(self._coeffs.get("ReTRef", 100.0))
        self._k_min = float(self._coeffs.get("kMin", 1e-10))
        self._k_max = float(self._coeffs.get("kMax", 100.0))
        self._dynamic_coeff = float(self._coeffs.get("dynamicCoeff", 0.1))
        self._wall_dist = float(self._coeffs.get("wallDist", 0.01))
        self._spectral_coeff = float(self._coeffs.get("spectralCoeff", 0.05))
        self._spectral_ratio_ref = float(self._coeffs.get("spectralRatioRef", 5.0))
        self._aniso_prod_coeff = float(self._coeffs.get("anisoProdCoeff", 0.05))
        self._comp_coeff = float(self._coeffs.get("compCoeff", 0.1))
        self._Ma_limit = float(self._coeffs.get("MaLimit", 0.5))
        self._C_prod_max = float(self._coeffs.get("CprodMax", 2.0))
        self._dt = float(self._coeffs.get("dt", 1e-3))
        self._nu = float(self._coeffs.get("nu", 1e-5))

    @property
    def intensity(self) -> float:
        """Turbulence intensity."""
        return self._intensity

    @property
    def length_scale(self) -> float:
        """Turbulent length scale (m)."""
        return self._length_scale

    @property
    def C_mu(self) -> float:
        """Model constant C_mu."""
        return self._C_mu

    @property
    def kappa(self) -> float:
        """Von Karman constant."""
        return self._kappa

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
    def k_min(self) -> float:
        """Minimum k clamp."""
        return self._k_min

    @property
    def k_max(self) -> float:
        """Maximum k clamp."""
        return self._k_max

    @property
    def dynamic_coeff(self) -> float:
        """Dynamic production/dissipation balance coefficient."""
        return self._dynamic_coeff

    @property
    def wall_dist(self) -> float:
        """Near-wall distance estimate (m)."""
        return self._wall_dist

    @property
    def spectral_coeff(self) -> float:
        """Spectral energy correction coefficient."""
        return self._spectral_coeff

    @property
    def spectral_ratio_ref(self) -> float:
        """Reference frequency ratio for spectral correction."""
        return self._spectral_ratio_ref

    @property
    def aniso_prod_coeff(self) -> float:
        """Production anisotropy correction coefficient."""
        return self._aniso_prod_coeff

    @property
    def comp_coeff(self) -> float:
        """Compressibility correction coefficient."""
        return self._comp_coeff

    @property
    def Ma_limit(self) -> float:
        """Turbulent Mach number limit."""
        return self._Ma_limit

    @property
    def C_prod_max(self) -> float:
        """Production-to-dissipation limit ratio."""
        return self._C_prod_max

    @property
    def dt(self) -> float:
        """Time step for production limiting (s)."""
        return self._dt

    @property
    def nu(self) -> float:
        """Kinematic viscosity for Kolmogorov scale (m2/s)."""
        return self._nu

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        nu: float | None = None,
        strain_rate: torch.Tensor | None = None,
        c: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k with compressibility and enhanced production limiter.

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            epsilon: ``(n_faces,)`` dissipation rate.
            nu: Kinematic viscosity (m2/s) for Re_t estimation.
            strain_rate: ``(n_faces,)`` mean strain rate magnitude |S|.
            c: Speed of sound (m/s) for compressibility correction.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_intensity = 1.5 * (self._intensity * u_mag) ** 2

            if epsilon is not None:
                k_length = (epsilon * self._length_scale / (self._C_mu ** 0.75 + 1e-30)) ** (2.0 / 3.0)

                # Adaptive blending coefficient
                alpha_eff = self._alpha
                if nu is not None and nu > 0 and self._beta != 0:
                    eps_est = (self._C_mu ** 0.75) * (k_intensity ** 1.5) / (self._length_scale + 1e-30)
                    Re_t = k_intensity ** 2 / (nu * eps_est + 1e-30)
                    Re_t_mean = Re_t.mean().item()
                    alpha_eff = float(torch.clamp(
                        torch.tensor(self._alpha * (1.0 + self._beta * math.log10(
                            1.0 + Re_t_mean / self._Re_t_ref
                        ))),
                        0.0, 1.0,
                    ))

                k_base = alpha_eff * k_intensity + (1.0 - alpha_eff) * k_length

                # Dynamic production/dissipation balance (from v9)
                nut_est = self._C_mu * k_base ** 2 / (epsilon + 1e-30)
                S_ij_est = torch.sqrt(epsilon / (2.0 * nut_est + 1e-30))
                P_k = 2.0 * nut_est * S_ij_est ** 2
                eps_k = (self._C_mu ** 0.75) * (k_base ** 1.5) / (self._length_scale + 1e-30)
                tau_ratio = P_k / (eps_k + 1e-30)
                k_dyn = k_base * (1.0 + self._dynamic_coeff * torch.tanh(tau_ratio - 1.0))

                # von Karman length-scale correction
                kappa_y = self._kappa * self._wall_dist
                l_vk = kappa_y / (1.0 + kappa_y / (self._length_scale + 1e-30))
                k_vk = (epsilon * l_vk / (self._C_mu ** 0.75 + 1e-30)) ** (2.0 / 3.0)

                k = alpha_eff * k_dyn + (1.0 - alpha_eff) * k_vk

                # Spectral energy correction (from v9)
                nu_loc = nu if nu is not None and nu > 0 else self._nu
                f_cutoff = u_mag / (2.0 * math.pi * self._length_scale + 1e-30)
                f_kolm = (epsilon / (nu_loc ** 3 + 1e-30)) ** 0.25
                E_ratio = f_cutoff / (f_kolm + 1e-30)
                k = k * (1.0 + self._spectral_coeff * torch.tanh(E_ratio / (self._spectral_ratio_ref + 1e-30)))

                # Production anisotropy correction (from v9)
                if strain_rate is not None:
                    P_k_actual = 2.0 * nut_est * strain_rate ** 2
                    P_ratio = P_k_actual / (eps_k + 1e-30)
                    k = k * (1.0 + self._aniso_prod_coeff * (P_ratio - 1.0))

                # Compressibility correction
                c_eff = c if c is not None else 343.0
                Ma_t = torch.sqrt(2.0 * torch.clamp(k, min=1e-30)) / (c_eff + 1e-30)
                Ma_eff = torch.clamp(Ma_t, max=self._Ma_limit)
                k = k * (1.0 + self._comp_coeff * Ma_eff ** 2)

                # Enhanced production limiter
                P_limit = self._C_prod_max * eps_k
                dt_eff = self._dt
                k_limited = k_intensity + P_limit * dt_eff / (1.0 + dt_eff * eps_k / (k + 1e-30))
                k = torch.min(k, k_limited)
            else:
                k = k_intensity

            # Clamp to physical range
            k = torch.clamp(k, self._k_min, self._k_max)
        else:
            k = torch.full((n,), 0.01, dtype=dtype, device=device)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = k
        else:
            field[self._patch.face_indices] = k
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for v10 enhanced k inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        k_default = max(self._k_min, 0.01)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * k_default)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
