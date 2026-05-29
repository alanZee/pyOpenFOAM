"""
Enhanced turbulent kinetic energy inlet boundary condition (v3).

v3 extends v2 with a blending between intensity-based and
length-scale-based k estimates, plus a reference turbulence field
for coupled simulations::

    k_intensity = 1.5 * (I * |U|)^2
    k_length = (epsilon * l_mix / C_mu^0.75)^(2/3)
    k = alpha * k_intensity + (1 - alpha) * k_length

In OpenFOAM syntax::

    type        turbulentKineticEnergyInlet3;
    intensity   0.05;
    lengthScale 0.01;
    Cmu         0.09;
    alpha       0.8;
    U           U;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentKineticEnergyInlet3BC"]


@BoundaryCondition.register("turbulentKineticEnergyInlet3")
class TurbulentKineticEnergyInlet3BC(BoundaryCondition):
    """v3 enhanced turbulent kinetic energy inlet with blending.

    Blends intensity-based and length-scale-based k estimates.

    Coefficients:
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``alpha``: Blending weight for intensity-based k (default 0.8).
        - ``U``: Velocity field name (informational).
        - ``value``: Initial k value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._alpha = float(self._coeffs.get("alpha", 0.8))

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
        """Blending weight for intensity-based k."""
        return self._alpha

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k with blended estimate.

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            epsilon: ``(n_faces,)`` dissipation rate (for length-scale k).
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_intensity = 1.5 * (self._intensity * u_mag) ** 2

            if epsilon is not None:
                k_length = (epsilon * self._length_scale / (self._C_mu ** 0.75 + 1e-30)) ** (2.0 / 3.0)
                k = self._alpha * k_intensity + (1.0 - self._alpha) * k_length
            else:
                k = k_intensity
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
        """Penalty method for v3 enhanced k inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        k_default = 0.01

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * k_default)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
