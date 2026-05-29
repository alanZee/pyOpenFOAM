"""
Enhanced turbulent intensity inlet boundary condition (v3).

Extends ``turbulentIntensityInlet2`` with adaptive intensity scaling based on
the local turbulence Reynolds number and an optional k blending with a
reference value::

    k_raw = 1.5 * (I * |U|)^2
    Re_t = k_raw^2 / (nu * epsilon_est)
    I_eff = I * (1 + alpha * log10(1 + Re_t / Re_t_ref))
    k = 1.5 * (I_eff * |U|)^2
    k = clamp(k, kMin, kMax)

In OpenFOAM syntax::

    type        turbulentIntensityInlet3;
    intensity   0.05;
    kMin        1e-10;
    kMax        100.0;
    alpha       0.1;
    ReTRef      100.0;
    U           U;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentIntensityInlet3BC"]


@BoundaryCondition.register("turbulentIntensityInlet3")
class TurbulentIntensityInlet3BC(BoundaryCondition):
    """v3 enhanced turbulent intensity inlet with adaptive scaling.

    Coefficients:
        - ``intensity``: Base turbulence intensity (default 0.05).
        - ``kMin``: Minimum turbulent kinetic energy (default 1e-10).
        - ``kMax``: Maximum turbulent kinetic energy (default 100.0).
        - ``alpha``: Reynolds-number sensitivity (default 0.1).
        - ``ReTRef``: Reference turbulent Reynolds number (default 100.0).
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

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k with adaptive intensity scaling.

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for Re_t estimation.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_raw = 1.5 * (self._intensity * u_mag) ** 2

            # Adaptive intensity scaling based on turbulence Reynolds number
            if nu is not None and nu > 0 and self._alpha != 0:
                epsilon_est = k_raw ** 1.5 / (0.01 + 1e-30)  # rough l_mix ~ 0.01
                Re_t = k_raw ** 2 / (nu * epsilon_est + 1e-30)
                I_eff = self._intensity * (
                    1.0 + self._alpha * torch.log10(1.0 + Re_t / self._Re_t_ref)
                )
                k = 1.5 * (I_eff * u_mag) ** 2
            else:
                k = k_raw

            k = torch.clamp(k, self._k_min, self._k_max)
        else:
            k = torch.full((n,), max(self._k_min, 0.01), dtype=dtype, device=device)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = k
        else:
            field[self._patch.face_indices] = k
        return field

    def compute_kinetic_energy(
        self,
        velocity: torch.Tensor,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Return the computed and clamped k values without modifying a field.

        Args:
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s).

        Returns:
            ``(n_faces,)`` turbulent kinetic energy.
        """
        u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
        k_raw = 1.5 * (self._intensity * u_mag) ** 2

        if nu is not None and nu > 0 and self._alpha != 0:
            epsilon_est = k_raw ** 1.5 / (0.01 + 1e-30)
            Re_t = k_raw ** 2 / (nu * epsilon_est + 1e-30)
            I_eff = self._intensity * (
                1.0 + self._alpha * torch.log10(1.0 + Re_t / self._Re_t_ref)
            )
            k = 1.5 * (I_eff * u_mag) ** 2
        else:
            k = k_raw

        return torch.clamp(k, self._k_min, self._k_max)

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for v3 adaptive intensity inlet BC."""
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
