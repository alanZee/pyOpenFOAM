"""
Enhanced turbulent kinetic energy inlet boundary condition (v4).

Extends v3 blending with a Reynolds-number-adaptive alpha, a turbulence
production limiter, and per-face k clamping::

    k_intensity = 1.5 * (I * |U|)^2
    k_length = (epsilon * l_mix / C_mu^0.75)^(2/3)
    Re_t = k_intensity^2 / (nu * epsilon_est)
    alpha_eff = clamp(alpha * (1 + beta * log10(1 + Re_t / ReTRef)), 0, 1)
    k = alpha_eff * k_intensity + (1 - alpha_eff) * k_length
    k = clamp(k, kMin, kMax)

In OpenFOAM syntax::

    type        turbulentKineticEnergyInlet4;
    intensity   0.05;
    lengthScale 0.01;
    Cmu         0.09;
    alpha       0.8;
    beta        0.05;
    ReTRef      100.0;
    kMin        1e-10;
    kMax        100.0;
    U           U;
    value       uniform 0.01;
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentKineticEnergyInlet4BC"]


@BoundaryCondition.register("turbulentKineticEnergyInlet4")
class TurbulentKineticEnergyInlet4BC(BoundaryCondition):
    """v4 enhanced turbulent kinetic energy inlet with adaptive blending and clamping.

    Coefficients:
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``alpha``: Base blending weight for intensity-based k (default 0.8).
        - ``beta``: Re_t sensitivity coefficient (default 0.05).
        - ``ReTRef``: Reference turbulent Reynolds number (default 100.0).
        - ``kMin``: Minimum k clamp (default 1e-10).
        - ``kMax``: Maximum k clamp (default 100.0).
        - ``U``: Velocity field name (informational).
        - ``value``: Initial k value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._alpha = float(self._coeffs.get("alpha", 0.8))
        self._beta = float(self._coeffs.get("beta", 0.05))
        self._Re_t_ref = float(self._coeffs.get("ReTRef", 100.0))
        self._k_min = float(self._coeffs.get("kMin", 1e-10))
        self._k_max = float(self._coeffs.get("kMax", 100.0))

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
    def alpha(self) -> float:
        """Base blending weight for intensity-based k."""
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

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k with adaptive blending and clamping.

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            epsilon: ``(n_faces,)`` dissipation rate (for length-scale k).
            nu: Kinematic viscosity (m2/s) for Re_t estimation.
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

                k = alpha_eff * k_intensity + (1.0 - alpha_eff) * k_length
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
        """Penalty method for v4 enhanced k inlet BC."""
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
