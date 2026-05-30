"""
Enhanced turbulent intensity inlet boundary condition (v6).

Extends ``turbulentIntensityInlet5`` with a production-to-dissipation ratio
limiter and a local Reynolds number correction::

    k_raw = 1.5 * (I * |U|)^2
    Re_t = k_raw^2 / (nu * epsilon_est)
    I_eff = I * (1 + alpha * log10(1 + Re_t / ReTRef))
    // Production-to-dissipation limiter
    P_ratio = C_prodRatio * epsilon / k
    I_limited = min(I_eff, sqrt(P_ratio / (1.5 * |U|^2)))
    k = 1.5 * (I_limited * |U|)^2
    k *= anisotropyFactor
    k = clamp(k, kMin, kMax)

In OpenFOAM syntax::

    type        turbulentIntensityInlet6;
    intensity   0.05;
    kMin        1e-10;
    kMax        100.0;
    alpha       0.1;
    ReTRef      100.0;
    anisotropyFactor 1.0;
    CprodRatio  2.0;
    U           U;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentIntensityInlet6BC"]


@BoundaryCondition.register("turbulentIntensityInlet6")
class TurbulentIntensityInlet6BC(BoundaryCondition):
    """v6 enhanced turbulent intensity inlet with production limiter.

    Coefficients:
        - ``intensity``: Base turbulence intensity (default 0.05).
        - ``kMin``: Minimum turbulent kinetic energy (default 1e-10).
        - ``kMax``: Maximum turbulent kinetic energy (default 100.0).
        - ``alpha``: Reynolds-number sensitivity (default 0.1).
        - ``ReTRef``: Reference turbulent Reynolds number (default 100.0).
        - ``anisotropyFactor``: Streamwise anisotropy correction (default 1.0).
        - ``CprodRatio``: Production-to-dissipation limit ratio (default 2.0).
        - ``U``: Velocity field name (informational).
        - ``value``: Initial k value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._k_min = float(self._coeffs.get("kMin", 1e-10))
        self._k_max = float(self._coeffs.get("kMax", 100.0))
        self._alpha = float(self._coeffs.get("alpha", 0.1))
        self._Re_t_ref = float(self._coeffs.get("ReTRef", 100.0))
        self._anisotropy_factor = float(self._coeffs.get("anisotropyFactor", 1.0))
        self._C_prod_ratio = float(self._coeffs.get("CprodRatio", 2.0))

    @property
    def intensity(self) -> float:
        """Base turbulence intensity."""
        return self._intensity

    @property
    def k_min(self) -> float:
        """Minimum turbulent kinetic energy."""
        return self._k_min

    @property
    def k_max(self) -> float:
        """Maximum turbulent kinetic energy."""
        return self._k_max

    @property
    def alpha(self) -> float:
        """Reynolds-number sensitivity coefficient."""
        return self._alpha

    @property
    def Re_t_ref(self) -> float:
        """Reference turbulent Reynolds number."""
        return self._Re_t_ref

    @property
    def anisotropy_factor(self) -> float:
        """Streamwise anisotropy correction factor."""
        return self._anisotropy_factor

    @property
    def C_prod_ratio(self) -> float:
        """Production-to-dissipation limit ratio."""
        return self._C_prod_ratio

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k with production-to-dissipation limiter.

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for Re estimation.
            epsilon: ``(n_faces,)`` existing epsilon for production ratio limiter.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_raw = 1.5 * (self._intensity * u_mag) ** 2

            if nu is not None and nu > 0 and self._alpha != 0:
                # Turbulent Reynolds number estimation
                epsilon_est = k_raw ** 1.5 / (0.01 + 1e-30)
                Re_t = k_raw ** 2 / (nu * epsilon_est + 1e-30)
                I_eff = self._intensity * (
                    1.0 + self._alpha * torch.log10(1.0 + Re_t / self._Re_t_ref)
                )
            else:
                I_eff = torch.full((n,), self._intensity, dtype=dtype, device=device)

            # Production-to-dissipation ratio limiter
            if epsilon is not None and self._C_prod_ratio > 0:
                P_ratio = self._C_prod_ratio * epsilon / (k_raw + 1e-30)
                I_limited = torch.min(
                    I_eff,
                    torch.sqrt(P_ratio / (1.5 * u_mag ** 2 + 1e-30)),
                )
                k = 1.5 * (I_limited * u_mag) ** 2
            else:
                k = 1.5 * (I_eff * u_mag) ** 2

            # Anisotropy correction
            k = k * self._anisotropy_factor
            k = torch.clamp(k, self._k_min, self._k_max)
        else:
            k = torch.full((n,), max(self._k_min, 0.01), dtype=dtype, device=device)

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
        """Penalty method for v6 intensity inlet BC."""
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
