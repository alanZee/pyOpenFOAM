"""
Enhanced turbulent intensity inlet boundary condition (v2).

Extends ``turbulentIntensityInlet`` with intensity clamping and
optional k_min / k_max bounds::

    k = 1.5 * (I * |U|)^2
    k = clamp(k, kMin, kMax)

In OpenFOAM syntax::

    type        turbulentIntensityInlet2;
    intensity   0.05;
    kMin        1e-10;
    kMax        100.0;
    U           U;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentIntensityInlet2BC"]


@BoundaryCondition.register("turbulentIntensityInlet2")
class TurbulentIntensityInlet2BC(BoundaryCondition):
    """Enhanced turbulent intensity inlet with clamping.

    k = 1.5 * (I * |U|)^2, clamped to [kMin, kMax].

    Coefficients:
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``kMin``: Minimum turbulent kinetic energy (default 1e-10).
        - ``kMax``: Maximum turbulent kinetic energy (default 100.0).
        - ``U``: Velocity field name (informational).
        - ``value``: Initial k value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._k_min = float(self._coeffs.get("kMin", 1e-10))
        self._k_max = float(self._coeffs.get("kMax", 100.0))

    @property
    def intensity(self) -> float:
        """Turbulence intensity."""
        return self._intensity

    @property
    def k_min(self) -> float:
        """Minimum turbulent kinetic energy."""
        return self._k_min

    @property
    def k_max(self) -> float:
        """Maximum turbulent kinetic energy."""
        return self._k_max

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k from turbulence intensity with clamping.

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k = 1.5 * (self._intensity * u_mag) ** 2
            k = torch.clamp(k, self._k_min, self._k_max)
        else:
            k = torch.full((n,), max(self._k_min, 0.01), dtype=dtype, device=device)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = k
        else:
            field[self._patch.face_indices] = k
        return field

    def compute_kinetic_energy(self, velocity: torch.Tensor) -> torch.Tensor:
        """Return the computed and clamped k values without modifying a field.

        Args:
            velocity: ``(n_faces, 3)`` velocity at boundary.

        Returns:
            ``(n_faces,)`` turbulent kinetic energy.
        """
        u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
        k = 1.5 * (self._intensity * u_mag) ** 2
        return torch.clamp(k, self._k_min, self._k_max)

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for enhanced intensity inlet BC."""
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
