"""
Enhanced turbulent dissipation inlet boundary condition (v5).

Extends v4 with a two-layer model that estimates epsilon differently in the
log-law and buffer layers, plus a turbulent production limiter::

    k = 1.5 * (I * |U|)^2
    y_plus = u_tau * y / nu
    // Log-law layer: eps = C_mu^0.75 * k^1.5 / (kappa * y)
    // Buffer layer: eps = 2 * nu * k / y^2
    eps_log = C_mu^0.75 * k^1.5 / (kappa * y + 1e-30)
    eps_buf = 2 * nu * k / (y^2 + 1e-30)
    epsilon = blend(y_plus, yPlusLow, yPlusHigh) * eps_buf
              + (1 - blend) * eps_log
    epsilon = clamp(epsilon, epsilonMin, epsilonMax)

In OpenFOAM syntax::

    type        turbulentDissipationInlet5;
    mixingLength 0.01;
    Cmu         0.09;
    kappa       0.41;
    intensity   0.05;
    alpha       1.0;
    wallDist    0.01;
    yPlusLow    5.0;
    yPlusHigh   30.0;
    epsilonMin  1e-10;
    epsilonMax  1e6;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentDissipationInlet5BC"]


@BoundaryCondition.register("turbulentDissipationInlet5")
class TurbulentDissipationInlet5BC(BoundaryCondition):
    """v5 enhanced turbulent dissipation inlet with two-layer blending.

    Coefficients:
        - ``mixingLength``: Mixing length (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``kappa``: Von Karman constant (default 0.41).
        - ``intensity``: Fallback turbulence intensity (default 0.05).
        - ``alpha``: Base blending weight for intensity-based epsilon (default 1.0).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``yPlusLow``: Lower y+ bound for buffer-layer blending (default 5.0).
        - ``yPlusHigh``: Upper y+ bound for log-law blending (default 30.0).
        - ``epsilonMin``: Minimum epsilon clamp (default 1e-10).
        - ``epsilonMax``: Maximum epsilon clamp (default 1e6).
        - ``value``: Initial epsilon value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._kappa = float(self._coeffs.get("kappa", 0.41))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._alpha = float(self._coeffs.get("alpha", 1.0))
        self._wall_dist = float(self._coeffs.get("wallDist", 0.01))
        self._y_plus_low = float(self._coeffs.get("yPlusLow", 5.0))
        self._y_plus_high = float(self._coeffs.get("yPlusHigh", 30.0))
        self._epsilon_min = float(self._coeffs.get("epsilonMin", 1e-10))
        self._epsilon_max = float(self._coeffs.get("epsilonMax", 1e6))

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
    def intensity(self) -> float:
        """Fallback turbulence intensity."""
        return self._intensity

    @property
    def alpha(self) -> float:
        """Base blending weight for intensity-based epsilon."""
        return self._alpha

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

    def _two_layer_epsilon(
        self, k: torch.Tensor, y: float, nu: float, n: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute two-layer epsilon with buffer/log-law blending."""
        # Log-law layer epsilon: eps = C_mu^0.75 * k^1.5 / (kappa * y)
        eps_log = (self._C_mu ** 0.75) * (k ** 1.5) / (self._kappa * y + 1e-30)

        # Buffer layer epsilon: eps = 2 * nu * k / y^2
        eps_buf = 2.0 * nu * k / (y ** 2 + 1e-30)

        # Estimate y+ for blending (simplified: use k-based friction velocity)
        u_tau = (self._C_mu ** 0.25) * torch.sqrt(k)
        y_plus = u_tau * y / (nu + 1e-30)

        # Smooth blending function
        blend = torch.clamp(
            (y_plus - self._y_plus_low) / (self._y_plus_high - self._y_plus_low + 1e-30),
            0.0, 1.0,
        )

        # In buffer layer (blend -> 0): use eps_buf
        # In log-law layer (blend -> 1): use eps_log
        return blend * eps_log + (1.0 - blend) * eps_buf

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face epsilon with two-layer model and clamping.

        Args:
            field: Turbulent dissipation rate field.
            patch_idx: Optional start index.
            k: ``(n_faces,)`` turbulent kinetic energy.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for two-layer model.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if k is not None and nu is not None and nu > 0:
            epsilon = self._two_layer_epsilon(k, self._wall_dist, nu, n, device, dtype)
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
        """Penalty method for v5 enhanced epsilon inlet BC."""
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
