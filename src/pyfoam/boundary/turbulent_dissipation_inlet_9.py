"""
Enhanced turbulent dissipation inlet boundary condition (v9).

Extends ``turbulentDissipationInlet8`` with a realizability constraint
and an anisotropy-aware production limiter::

    k = 1.5 * (I * |U|)^2
    // Two-layer model (from v8)
    eps_log = C_mu^0.75 * k^1.5 / (kappa * y)
    eps_buf = 2 * nu * k / y^2
    eps_base = blend(y_plus) * eps_log + (1 - blend) * eps_buf
    // Kolmogorov-scale limiter (from v8)
    eps_kolm = nu^3 / gridScale^4
    eps = max(eps_base, eps_kolm)
    // Realizability constraint: eps <= k^1.5 / (C_mu^0.75 * l_min)
    eps_real = k^1.5 / (C_mu^0.75 * l_min)
    eps = min(eps, eps_real)
    // Anisotropy-aware production limiter
    P_k = 2 * nut * (S_ij + 0.5 * a_ij * S_mag)
    eps_prod = C1 * P_k
    eps = max(eps, eps_prod * productionRatio)
    eps = clamp(eps, epsilonMin, epsilonMax)

In OpenFOAM syntax::

    type        turbulentDissipationInlet9;
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
    productionRatio 1.5;
    gridScale   0.001;
    lMin        1e-6;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentDissipationInlet9BC"]


@BoundaryCondition.register("turbulentDissipationInlet9")
class TurbulentDissipationInlet9BC(BoundaryCondition):
    """v9 enhanced turbulent dissipation inlet with realizability constraint.

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
        - ``productionRatio``: Ratio of production-to-dissipation limiter (default 1.5).
        - ``gridScale``: Local grid scale for Kolmogorov limiter (m, default 0.001).
        - ``lMin``: Minimum length scale for realizability (m, default 1e-6).
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
        self._production_ratio = float(self._coeffs.get("productionRatio", 1.5))
        self._grid_scale = float(self._coeffs.get("gridScale", 0.001))
        self._l_min = float(self._coeffs.get("lMin", 1e-6))

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
    def production_ratio(self) -> float:
        """Ratio of production-to-dissipation limiter."""
        return self._production_ratio

    @property
    def grid_scale(self) -> float:
        """Local grid scale for Kolmogorov limiter (m)."""
        return self._grid_scale

    @property
    def l_min(self) -> float:
        """Minimum length scale for realizability (m)."""
        return self._l_min

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
        """Set boundary-face epsilon with realizability constraint.

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

            # Realizability constraint: eps <= k^1.5 / (C_mu^0.75 * l_min)
            eps_real = k ** 1.5 / (self._C_mu ** 0.75 * self._l_min + 1e-30)
            eps = torch.min(eps, eps_real)

            # Anisotropy-aware production limiter
            if strain_rate is not None:
                nut_est = self._C_mu * k ** 2 / (eps + 1e-30)
                P_k = 2.0 * nut_est * strain_rate ** 2
                eps_prod = self._C1 * P_k
                epsilon = torch.max(eps, eps_prod * self._production_ratio)
            else:
                nut_est = self._C_mu * k ** 2 / (eps + 1e-30)
                eps_prod = self._C_mu * k ** 2 / (nut_est + nu + 1e-30)
                epsilon = torch.max(eps, eps_prod * self._production_ratio)
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
        """Penalty method for v9 enhanced epsilon inlet BC."""
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
