"""
Enhanced turbulent frequency inlet boundary condition (v9).

Extends ``turbulentFrequencyInlet8`` with a realizability constraint
and an SST-style limiter for robust near-wall treatment::

    k = 1.5 * (I * |U|)^2
    // Two-layer model (from v8)
    omega_log = k^0.5 / (C_mu^0.25 * kappa * y)
    omega_buf = 6 * nu / (beta1 * y^2)
    omega_base = blend(y_plus) * omega_log + (1 - blend) * omega_buf
    // Kolmogorov-scale limiter (from v8)
    omega_kolm = sqrt(eps_kolm / (C_mu * k))
    omega = max(omega_base, omega_kolm)
    // Realizability: omega >= sqrt(2 * epsilon_real / (C_mu * k))
    omega_real = sqrt(2 * k / (C_mu * k * l_max^2))
    omega = max(omega, omega_real)
    // SST-style limiter
    omega_sst = max(omega, sqrt(k) / (C_mu^0.25 * kappa * y))
    omega = omega_sst
    omega = clamp(omega, omegaMin, omegaMax)

In OpenFOAM syntax::

    type        turbulentFrequencyInlet9;
    mixingLength 0.01;
    Cmu         0.09;
    kappa       0.41;
    beta1       0.075;
    betaStar    0.09;
    intensity   0.05;
    wallDist    0.01;
    yPlusLow    5.0;
    yPlusHigh   30.0;
    omegaMin    1e-4;
    omegaMax    1e6;
    productionRatio 1.5;
    gridScale   0.001;
    lMax        1.0;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentFrequencyInlet9BC"]


@BoundaryCondition.register("turbulentFrequencyInlet9")
class TurbulentFrequencyInlet9BC(BoundaryCondition):
    """v9 enhanced turbulent frequency inlet with realizability and SST limiter.

    Coefficients:
        - ``mixingLength``: Mixing length (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``kappa``: Von Karman constant (default 0.41).
        - ``beta1``: k-omega model constant (default 0.075).
        - ``betaStar``: k-omega model constant beta* (default 0.09).
        - ``intensity``: Fallback turbulence intensity (default 0.05).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``yPlusLow``: Lower y+ bound for buffer-layer blending (default 5.0).
        - ``yPlusHigh``: Upper y+ bound for log-law blending (default 30.0).
        - ``omegaMin``: Minimum omega clamp (default 1e-4).
        - ``omegaMax``: Maximum omega clamp (default 1e6).
        - ``productionRatio``: Ratio of production-to-frequency limiter (default 1.5).
        - ``gridScale``: Local grid scale for Kolmogorov limiter (m, default 0.001).
        - ``lMax``: Maximum length scale for realizability (m, default 1.0).
        - ``value``: Initial omega value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._kappa = float(self._coeffs.get("kappa", 0.41))
        self._beta1 = float(self._coeffs.get("beta1", 0.075))
        self._beta_star = float(self._coeffs.get("betaStar", 0.09))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._wall_dist = float(self._coeffs.get("wallDist", 0.01))
        self._y_plus_low = float(self._coeffs.get("yPlusLow", 5.0))
        self._y_plus_high = float(self._coeffs.get("yPlusHigh", 30.0))
        self._omega_min = float(self._coeffs.get("omegaMin", 1e-4))
        self._omega_max = float(self._coeffs.get("omegaMax", 1e6))
        self._production_ratio = float(self._coeffs.get("productionRatio", 1.5))
        self._grid_scale = float(self._coeffs.get("gridScale", 0.001))
        self._l_max = float(self._coeffs.get("lMax", 1.0))

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
    def beta1(self) -> float:
        """k-omega model constant beta_1."""
        return self._beta1

    @property
    def beta_star(self) -> float:
        """k-omega model constant beta*."""
        return self._beta_star

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
    def omega_min(self) -> float:
        """Minimum omega clamp."""
        return self._omega_min

    @property
    def omega_max(self) -> float:
        """Maximum omega clamp."""
        return self._omega_max

    @property
    def production_ratio(self) -> float:
        """Ratio of production-to-frequency limiter."""
        return self._production_ratio

    @property
    def grid_scale(self) -> float:
        """Local grid scale for Kolmogorov limiter (m)."""
        return self._grid_scale

    @property
    def l_max(self) -> float:
        """Maximum length scale for realizability (m)."""
        return self._l_max

    def _two_layer_omega(
        self, k: torch.Tensor, y: float, nu: float, n: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute two-layer omega with buffer/log-law blending."""
        omega_log = torch.sqrt(k) / (self._C_mu ** 0.25 * self._kappa * y + 1e-30)
        omega_buf = 6.0 * nu / (self._beta1 * y ** 2 + 1e-30)

        u_tau = (self._C_mu ** 0.25) * torch.sqrt(k)
        y_plus = u_tau * y / (nu + 1e-30)

        blend = torch.clamp(
            (y_plus - self._y_plus_low) / (self._y_plus_high - self._y_plus_low + 1e-30),
            0.0, 1.0,
        )

        return blend * omega_log + (1.0 - blend) * omega_buf

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
        strain_rate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face omega with realizability and SST limiter.

        Args:
            field: Specific dissipation rate field.
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
            omega_base = self._two_layer_omega(k, self._wall_dist, nu, n, device, dtype)

            # Kolmogorov-scale limiter
            eps_kolm = nu ** 3 / (self._grid_scale ** 4 + 1e-30)
            omega_kolm = torch.sqrt(torch.tensor(eps_kolm, dtype=dtype, device=device) / (self._C_mu * k + 1e-30))
            omega = torch.max(omega_base, omega_kolm)

            # Realizability constraint
            omega_real = torch.sqrt(2.0 * k / (self._C_mu * k * self._l_max ** 2 + 1e-30))
            omega = torch.max(omega, omega_real)

            # SST-style limiter: omega >= k^0.5 / (C_mu^0.25 * kappa * y)
            omega_sst = torch.sqrt(k) / (self._C_mu ** 0.25 * self._kappa * self._wall_dist + 1e-30)
            omega = torch.max(omega, omega_sst)

            # Production limiter
            if strain_rate is not None:
                nut_est = k / (self._beta_star * omega + 1e-30)
                P_k = 2.0 * nut_est * strain_rate ** 2
                omega_prod = P_k / (self._beta1 * k + 1e-30)
                omega = torch.max(omega, omega_prod * self._production_ratio)
        elif k is not None:
            l = torch.full((n,), self._mixing_length, dtype=dtype, device=device)
            omega = torch.sqrt(k) / (self._C_mu ** 0.25 * l + 1e-30)
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
            l = torch.full((n,), self._mixing_length, dtype=dtype, device=device)
            omega = torch.sqrt(k_est) / (self._C_mu ** 0.25 * l + 1e-30)
        else:
            omega = torch.full((n,), 0.01, dtype=dtype, device=device)

        # Clamp to physical range
        omega = torch.clamp(omega, self._omega_min, self._omega_max)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = omega
        else:
            field[self._patch.face_indices] = omega
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for v9 enhanced omega inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        omega_default = 0.01

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * omega_default)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
