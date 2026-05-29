"""
Enhanced turbulent dissipation inlet boundary condition (v2).

Computes epsilon from turbulence intensity and length scale, combining
the intensity-based k estimation with the mixing-length dissipation formula::

    k = 1.5 * (I * |U|)^2
    epsilon = C_mu^0.75 * k^1.5 / l_mix

In OpenFOAM syntax::

    type        turbulentDissipationInlet2;
    intensity   0.05;
    lengthScale 0.01;
    Cmu         0.09;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentDissipationInlet2BC"]


@BoundaryCondition.register("turbulentDissipationInlet2")
class TurbulentDissipationInlet2BC(BoundaryCondition):
    """Enhanced turbulent dissipation inlet with intensity and length scale.

    k = 1.5 * (I * |U|)^2
    epsilon = C_mu^0.75 * k^1.5 / l_mix

    Coefficients:
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``value``: Initial epsilon value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))

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

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face epsilon from intensity, velocity, and length scale.

        Args:
            field: Turbulent dissipation rate field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            k: ``(n_faces,)`` pre-computed turbulent kinetic energy.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if k is None and velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k = 1.5 * (self._intensity * u_mag) ** 2

        if k is not None:
            epsilon = (self._C_mu ** 0.75) * (k ** 1.5) / (self._length_scale + 1e-30)
        else:
            epsilon = torch.full((n,), 0.01, dtype=dtype, device=device)

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
        """Penalty method for enhanced epsilon inlet BC."""
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
