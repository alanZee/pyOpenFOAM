"""
Enhanced turbulent viscosity inlet boundary condition (v2).

Extends ``turbulentViscosityInlet`` with explicit turbulence intensity
and length scale parameters, plus per-face clamping::

    k = 1.5 * (I * |U|)^2
    epsilon = C_mu^0.75 * k^1.5 / l_mix
    nut = C_mu * k^2 / epsilon
    nut = clamp(nut, nutMin, nutMax)

In OpenFOAM syntax::

    type        turbulentViscosityInlet2;
    Cmu         0.09;
    intensity   0.05;
    lengthScale 0.01;
    nutMin      1e-10;
    nutMax      1e4;
    value       uniform 0.001;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentViscosityInlet2BC"]


@BoundaryCondition.register("turbulentViscosityInlet2")
class TurbulentViscosityInlet2BC(BoundaryCondition):
    """Enhanced turbulent viscosity inlet with intensity and length scale.

    Coefficients:
        - ``Cmu``: Model constant (default 0.09).
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``nutMin``: Minimum nut clamp (default 1e-10).
        - ``nutMax``: Maximum nut clamp (default 1e4).
        - ``value``: Initial nut value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._nut_min = float(self._coeffs.get("nutMin", 1e-10))
        self._nut_max = float(self._coeffs.get("nutMax", 1e4))

    @property
    def C_mu(self) -> float:
        """Model constant C_mu."""
        return self._C_mu

    @property
    def intensity(self) -> float:
        """Turbulence intensity."""
        return self._intensity

    @property
    def length_scale(self) -> float:
        """Turbulent length scale (m)."""
        return self._length_scale

    @property
    def nut_min(self) -> float:
        """Minimum nut clamp value."""
        return self._nut_min

    @property
    def nut_max(self) -> float:
        """Maximum nut clamp value."""
        return self._nut_max

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face nut with clamping.

        Args:
            field: Turbulent viscosity field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            k: ``(n_faces,)`` turbulent kinetic energy.
            epsilon: ``(n_faces,)`` turbulent dissipation rate.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if k is not None and epsilon is not None:
            nut = self._C_mu * k ** 2 / (epsilon + 1e-30)
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
            epsilon_est = (self._C_mu ** 0.75) * (k_est ** 1.5) / (self._length_scale + 1e-30)
            nut = self._C_mu * k_est ** 2 / (epsilon_est + 1e-30)
        else:
            nut = torch.full((n,), 0.001, dtype=dtype, device=device)

        # Clamp to physical range
        nut = torch.clamp(nut, self._nut_min, self._nut_max)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = nut
        else:
            field[self._patch.face_indices] = nut
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for enhanced nut inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        nut_default = 0.001

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * nut_default)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
