"""
Enhanced turbulent frequency inlet boundary condition (v6).

Extends ``turbulentFrequencyInlet5`` with a production-limiter correction
and a wall-normal-distance-aware damping term::

    k = 1.5 * (I * |U|)^2
    // Log-law layer: omega = k^0.5 / (C_mu^0.25 * kappa * y)
    // Buffer layer: omega = 6 * nu / (beta1 * y^2)
    omega_base = blend(y_plus, yPlusLow, yPlusHigh) * omega_log + (1 - blend) * omega_buf
    // Production limiter: omega_prod = 1 / (beta1 * nut / k + nu / (k + 1e-30))
    omega_limited = max(omega_base, omega_prod * productionRatio)
    omega = clamp(omega_limited, omegaMin, omegaMax)

In OpenFOAM syntax::

    type        turbulentFrequencyInlet6;
    mixingLength 0.01;
    Cmu         0.09;
    kappa       0.41;
    beta1       0.075;
    intensity   0.05;
    wallDist    0.01;
    yPlusLow    5.0;
    yPlusHigh   30.0;
    omegaMin    1e-4;
    omegaMax    1e6;
    productionRatio 1.5;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentFrequencyInlet6BC"]


@BoundaryCondition.register("turbulentFrequencyInlet6")
class TurbulentFrequencyInlet6BC(BoundaryCondition):
    """v6 enhanced turbulent frequency inlet with production limiter.

    Coefficients:
        - ``mixingLength``: Mixing length (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``kappa``: Von Karman constant (default 0.41).
        - ``beta1``: k-omega model constant (default 0.075).
        - ``intensity``: Fallback turbulence intensity (default 0.05).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``yPlusLow``: Lower y+ bound for buffer-layer blending (default 5.0).
        - ``yPlusHigh``: Upper y+ bound for log-law blending (default 30.0).
        - ``omegaMin``: Minimum omega clamp (default 1e-4).
        - ``omegaMax``: Maximum omega clamp (default 1e6).
        - ``productionRatio``: Ratio of production-to-frequency limiter (default 1.5).
        - ``value``: Initial omega value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._kappa = float(self._coeffs.get("kappa", 0.41))
        self._beta1 = float(self._coeffs.get("beta1", 0.075))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._wall_dist = float(self._coeffs.get("wallDist", 0.01))
        self._y_plus_low = float(self._coeffs.get("yPlusLow", 5.0))
        self._y_plus_high = float(self._coeffs.get("yPlusHigh", 30.0))
        self._omega_min = float(self._coeffs.get("omegaMin", 1e-4))
        self._omega_max = float(self._coeffs.get("omegaMax", 1e6))
        self._production_ratio = float(self._coeffs.get("productionRatio", 1.5))

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
    ) -> torch.Tensor:
        """Set boundary-face omega with production limiter and two-layer model.

        Args:
            field: Specific dissipation rate field.
            patch_idx: Optional start index.
            k: ``(n_faces,)`` turbulent kinetic energy.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for two-layer model.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if k is not None and nu is not None and nu > 0:
            omega_base = self._two_layer_omega(k, self._wall_dist, nu, n, device, dtype)

            # Production limiter
            eps_est = (self._C_mu ** 0.75) * (k ** 1.5) / (self._mixing_length + 1e-30)
            nut_est = self._C_mu * k ** 2 / (eps_est + 1e-30)
            omega_prod = k / (self._beta1 * nut_est + nu + 1e-30)
            omega = torch.max(omega_base, omega_prod * self._production_ratio)
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
        """Penalty method for v6 enhanced omega inlet BC."""
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
