"""
Enhanced turbulent dissipation inlet boundary condition (v11).

Extends ``turbulentDissipationInlet10`` with anisotropic dissipation
correction and a spectral energy cascade model::

    k = 1.5 * (I * |U|)^2
    // Two-layer model (from v10)
    // Kolmogorov-scale limiter (from v10)
    // Dynamic time-scale limiter (from v10)
    // Vortex-stretching correction (from v10)
    // Anisotropic dissipation correction
    P_k = 2 * nut * S_ij^2
    ratio_P = P_k / (eps + 1e-30)
    eps_aniso = eps * (1 + anisoCoeff * (ratio_P - 1)^2)
    // Spectral cascade model
    tau_eta = sqrt(nu / (eps_aniso + 1e-30))
    tau_integral = k / (eps_aniso + 1e-30)
    cascade_ratio = tau_integral / (tau_eta + 1e-30)
    eps_out = eps_aniso * (1 + cascadeCoeff * log(1 + cascade_ratio / cascadeRatioRef))
    eps_out = clamp(eps_out, epsilonMin, epsilonMax)

In OpenFOAM syntax::

    type        turbulentDissipationInlet11;
    mixingLength 0.01;
    Cmu         0.09;
    kappa       0.41;
    C1          1.44;
    intensity   0.05;
    wallDist    0.01;
    yPlusLow    5.0;
    yPlusHigh   30.0;
    epsilonMin  1e-10;
    epsilonMax  1e6;
    gridScale   0.001;
    dynCoeff    0.1;
    dynRatioRef 10.0;
    vsCoeff     0.05;
    anisoCoeff  0.05;
    cascadeCoeff 0.02;
    cascadeRatioRef 10.0;
    value       uniform 0.01;
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentDissipationInlet11BC"]


@BoundaryCondition.register("turbulentDissipationInlet11")
class TurbulentDissipationInlet11BC(BoundaryCondition):
    """v11 enhanced turbulent dissipation inlet with anisotropic correction.

    Coefficients:
        - ``mixingLength``: Mixing length (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``kappa``: Von Karman constant (default 0.41).
        - ``C1``: k-epsilon model constant C1_epsilon (default 1.44).
        - ``intensity``: Fallback turbulence intensity (default 0.05).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``yPlusLow``: Lower y+ bound for buffer-layer blending (default 5.0).
        - ``yPlusHigh``: Upper y+ bound for log-law blending (default 30.0).
        - ``epsilonMin``: Minimum epsilon clamp (default 1e-10).
        - ``epsilonMax``: Maximum epsilon clamp (default 1e6).
        - ``gridScale``: Local grid scale for Kolmogorov limiter (m, default 0.001).
        - ``dynCoeff``: Dynamic time-scale correction coefficient (default 0.1).
        - ``dynRatioRef``: Reference time-scale ratio for dynamic correction (default 10.0).
        - ``vsCoeff``: Vortex-stretching correction coefficient (default 0.05).
        - ``anisoCoeff``: Anisotropic dissipation correction coefficient (default 0.05).
        - ``cascadeCoeff``: Spectral cascade model coefficient (default 0.02).
        - ``cascadeRatioRef``: Reference cascade time-scale ratio (default 10.0).
        - ``value``: Initial epsilon value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._kappa = float(self._coeffs.get("kappa", 0.41))
        self._C1 = float(self._coeffs.get("C1", 1.44))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._wall_dist = float(self._coeffs.get("wallDist", 0.01))
        self._y_plus_low = float(self._coeffs.get("yPlusLow", 5.0))
        self._y_plus_high = float(self._coeffs.get("yPlusHigh", 30.0))
        self._epsilon_min = float(self._coeffs.get("epsilonMin", 1e-10))
        self._epsilon_max = float(self._coeffs.get("epsilonMax", 1e6))
        self._grid_scale = float(self._coeffs.get("gridScale", 0.001))
        self._dyn_coeff = float(self._coeffs.get("dynCoeff", 0.1))
        self._dyn_ratio_ref = float(self._coeffs.get("dynRatioRef", 10.0))
        self._vs_coeff = float(self._coeffs.get("vsCoeff", 0.05))
        self._aniso_coeff = float(self._coeffs.get("anisoCoeff", 0.05))
        self._cascade_coeff = float(self._coeffs.get("cascadeCoeff", 0.02))
        self._cascade_ratio_ref = float(self._coeffs.get("cascadeRatioRef", 10.0))

    @property
    def mixing_length(self) -> float:
        """Mixing length (m)."""
        return self._mixing_length

    @property
    def C_mu(self) -> float:
        """Model constant C_mu."""
        return self._C_mu

    @property
    def kappa(self) -> float:
        """Von Karman constant."""
        return self._kappa

    @property
    def C1(self) -> float:
        """k-epsilon model constant C1_epsilon."""
        return self._C1

    @property
    def intensity(self) -> float:
        """Fallback turbulence intensity."""
        return self._intensity

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

    @property
    def epsilon_min(self) -> float:
        """Minimum epsilon clamp."""
        return self._epsilon_min

    @property
    def epsilon_max(self) -> float:
        """Maximum epsilon clamp."""
        return self._epsilon_max

    @property
    def grid_scale(self) -> float:
        """Local grid scale for Kolmogorov limiter (m)."""
        return self._grid_scale

    @property
    def dyn_coeff(self) -> float:
        """Dynamic time-scale correction coefficient."""
        return self._dyn_coeff

    @property
    def dyn_ratio_ref(self) -> float:
        """Reference time-scale ratio for dynamic correction."""
        return self._dyn_ratio_ref

    @property
    def vs_coeff(self) -> float:
        """Vortex-stretching correction coefficient."""
        return self._vs_coeff

    @property
    def aniso_coeff(self) -> float:
        """Anisotropic dissipation correction coefficient."""
        return self._aniso_coeff

    @property
    def cascade_coeff(self) -> float:
        """Spectral cascade model coefficient."""
        return self._cascade_coeff

    @property
    def cascade_ratio_ref(self) -> float:
        """Reference cascade time-scale ratio."""
        return self._cascade_ratio_ref

    def _two_layer_epsilon(
        self, k: torch.Tensor, y: float, nu: float, n: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute two-layer epsilon with buffer/log-law blending."""
        eps_log = (self._C_mu ** 0.75) * (k ** 1.5) / (self._kappa * y + 1e-30)
        eps_buf = 2.0 * nu * k / (y ** 2 + 1e-30)

        u_tau = (self._C_mu ** 0.25) * torch.sqrt(k)
        y_plus = u_tau * y / (nu + 1e-30)

        blend = torch.clamp(
            (y_plus - self._y_plus_low) / (self._y_plus_high - self._y_plus_low + 1e-30),
            0.0, 1.0,
        )

        return blend * eps_log + (1.0 - blend) * eps_buf

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
        strain_rate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face epsilon with anisotropic and cascade corrections.

        Args:
            field: Turbulent dissipation rate field.
            patch_idx: Optional start index.
            k: ``(n_faces,)`` turbulent kinetic energy.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for two-layer model.
            strain_rate: ``(n_faces,)`` mean strain rate magnitude |S|.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if k is not None and nu is not None and nu > 0:
            eps_base = self._two_layer_epsilon(k, self._wall_dist, nu, n, device, dtype)

            # Kolmogorov-scale limiter
            eps_kolm = nu ** 3 / (self._grid_scale ** 4 + 1e-30)
            eps = torch.max(eps_base, torch.tensor(eps_kolm, dtype=dtype, device=device))

            # Dynamic time-scale limiter
            tau_t = k / (eps + 1e-30)
            tau_K = torch.sqrt(nu / (eps + 1e-30))
            tau_ratio = tau_t / (tau_K + 1e-30)
            eps_dyn = eps * (1.0 + self._dyn_coeff * torch.tanh(tau_ratio / self._dyn_ratio_ref))

            # Vortex-stretching correction
            omega_mag = torch.sqrt(eps_dyn / (self._C_mu * k + 1e-30))
            if strain_rate is not None:
                eps_vs = eps_dyn * (1.0 + self._vs_coeff * strain_rate / (omega_mag + 1e-30))
            else:
                eps_vs = eps_dyn

            # Anisotropic dissipation correction
            if strain_rate is not None:
                nut_est = self._C_mu * k ** 2 / (eps_vs + 1e-30)
                P_k = 2.0 * nut_est * strain_rate ** 2
                ratio_P = P_k / (eps_vs + 1e-30)
                eps_aniso = eps_vs * (1.0 + self._aniso_coeff * (ratio_P - 1.0) ** 2)
            else:
                eps_aniso = eps_vs

            # Spectral cascade model
            tau_eta = torch.sqrt(nu / (eps_aniso + 1e-30))
            tau_integral = k / (eps_aniso + 1e-30)
            cascade_ratio = tau_integral / (tau_eta + 1e-30)
            epsilon = eps_aniso * (1.0 + self._cascade_coeff * torch.log(1.0 + cascade_ratio / (self._cascade_ratio_ref + 1e-30)))

        elif k is not None:
            l = torch.full((n,), self._mixing_length, dtype=dtype, device=device)
            epsilon = (self._C_mu ** 0.75) * (k ** 1.5) / (l + 1e-30)
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
            l = torch.full((n,), self._mixing_length, dtype=dtype, device=device)
            epsilon = (self._C_mu ** 0.75) * (k_est ** 1.5) / (l + 1e-30)
        else:
            epsilon = torch.full((n,), 0.01, dtype=dtype, device=device)

        # Clamp to physical range
        epsilon = torch.clamp(epsilon, self._epsilon_min, self._epsilon_max)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = epsilon
        else:
            field[self._patch.face_indices] = epsilon
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for v11 enhanced epsilon inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        epsilon_default = 0.01

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * epsilon_default)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
