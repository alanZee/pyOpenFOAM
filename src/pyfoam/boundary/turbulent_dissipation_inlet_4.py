"""
Enhanced turbulent dissipation inlet boundary condition (v4).

Extends v3 with a Reynolds-number-dependent blending coefficient, a wall-function
aware epsilon estimate, and a turbulent-to-laminar viscosity ratio limiter::

    k = 1.5 * (I * |U|)^2
    Re_t = k^2 / (nu * epsilon)
    alpha_eff = alpha * (1 + beta * log10(1 + Re_t / Re_t_ref))
    epsilon_intensity = C_mu^0.75 * k^1.5 / l_mix
    epsilon = alpha_eff * epsilon_intensity + (1 - alpha_eff) * epsilon_k
    epsilon = clamp(epsilon, epsilonMin, epsilonMax)

In OpenFOAM syntax::

    type        turbulentDissipationInlet4;
    mixingLength 0.01;
    Cmu         0.09;
    intensity   0.05;
    alpha       1.0;
    beta        0.05;
    ReTRef      100.0;
    epsilonMin  1e-10;
    epsilonMax  1e6;
    value       uniform 0.01;
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentDissipationInlet4BC"]


@BoundaryCondition.register("turbulentDissipationInlet4")
class TurbulentDissipationInlet4BC(BoundaryCondition):
    """v4 enhanced turbulent dissipation inlet with adaptive blending and clamping.

    Coefficients:
        - ``mixingLength``: Mixing length (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``intensity``: Fallback turbulence intensity (default 0.05).
        - ``alpha``: Base blending weight for intensity-based epsilon (default 1.0).
        - ``beta``: Re_t sensitivity coefficient (default 0.05).
        - ``ReTRef``: Reference turbulent Reynolds number (default 100.0).
        - ``epsilonMin``: Minimum epsilon clamp (default 1e-10).
        - ``epsilonMax``: Maximum epsilon clamp (default 1e6).
        - ``value``: Initial epsilon value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._alpha = float(self._coeffs.get("alpha", 1.0))
        self._beta = float(self._coeffs.get("beta", 0.05))
        self._Re_t_ref = float(self._coeffs.get("ReTRef", 100.0))
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
    def intensity(self) -> float:
        """Fallback turbulence intensity."""
        return self._intensity

    @property
    def alpha(self) -> float:
        """Base blending weight for intensity-based epsilon."""
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
    def epsilon_min(self) -> float:
        """Minimum epsilon clamp."""
        return self._epsilon_min

    @property
    def epsilon_max(self) -> float:
        """Maximum epsilon clamp."""
        return self._epsilon_max

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        l_mix: torch.Tensor | float | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face epsilon with adaptive blending and clamping.

        Args:
            field: Turbulent dissipation rate field.
            patch_idx: Optional start index.
            k: ``(n_faces,)`` turbulent kinetic energy.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            l_mix: Per-face or scalar mixing length override.
            nu: Kinematic viscosity for Re_t estimation.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if l_mix is not None:
            l = l_mix if isinstance(l_mix, torch.Tensor) else torch.full(
                (n,), float(l_mix), dtype=dtype, device=device
            )
        else:
            l = torch.full((n,), self._mixing_length, dtype=dtype, device=device)

        if k is not None:
            eps_k = (self._C_mu ** 0.75) * (k ** 1.5) / (l + 1e-30)

            if velocity is not None:
                u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
                k_est = 1.5 * (self._intensity * u_mag) ** 2
                eps_intensity = (self._C_mu ** 0.75) * (k_est ** 1.5) / (l + 1e-30)

                # Adaptive blending coefficient
                alpha_eff = self._alpha
                if nu is not None and nu > 0 and self._beta != 0:
                    Re_t = k_est ** 2 / (nu * eps_intensity + 1e-30)
                    Re_t_mean = Re_t.mean().item()
                    alpha_eff = min(
                        1.0,
                        self._alpha * (1.0 + self._beta * math.log10(
                            1.0 + Re_t_mean / self._Re_t_ref
                        ))
                    )

                epsilon = alpha_eff * eps_intensity + (1.0 - alpha_eff) * eps_k
            else:
                epsilon = eps_k
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
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
        """Penalty method for v4 enhanced epsilon inlet BC."""
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
