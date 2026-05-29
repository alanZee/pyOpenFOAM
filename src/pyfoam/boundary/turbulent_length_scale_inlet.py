"""
Turbulent length scale inlet boundary condition.

Computes the turbulent mixing length scale from the turbulent kinetic
energy k and the turbulent dissipation rate epsilon::

    l_mix = C_mu^0.75 * k^1.5 / epsilon

This boundary condition is typically used at inlets for the
k-epsilon turbulence model where a mixing length specification
is needed.

In OpenFOAM syntax::

    type        turbulentLengthScaleInlet;
    Cmu         0.09;
    intensity   0.05;
    mixingLength 0.01;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentLengthScaleInletBC"]


@BoundaryCondition.register("turbulentLengthScaleInlet")
class TurbulentLengthScaleInletBC(BoundaryCondition):
    """Turbulent length scale inlet boundary condition.

    l_mix = C_mu^0.75 * k^1.5 / epsilon

    Coefficients:
        - ``Cmu``: Model constant (default 0.09).
        - ``intensity``: Fallback turbulence intensity (default 0.05).
        - ``mixingLength``: Fallback mixing length (m, default 0.01).
        - ``value``: Initial l_mix value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))

    @property
    def C_mu(self) -> float:
        """Model constant C_mu."""
        return self._C_mu

    @property
    def intensity(self) -> float:
        """Fallback turbulence intensity."""
        return self._intensity

    @property
    def mixing_length(self) -> float:
        """Fallback mixing length (m)."""
        return self._mixing_length

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face mixing length.

        l_mix = C_mu^0.75 * k^1.5 / epsilon

        Args:
            field: Turbulent length scale field.
            patch_idx: Optional start index.
            k: ``(n_faces,)`` turbulent kinetic energy.
            epsilon: ``(n_faces,)`` turbulent dissipation rate.
            velocity: ``(n_faces, 3)`` velocity at boundary.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if k is not None and epsilon is not None:
            l_mix = (self._C_mu ** 0.75) * (k ** 1.5) / (epsilon + 1e-30)
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
            epsilon_est = (self._C_mu ** 0.75) * (k_est ** 1.5) / (self._mixing_length + 1e-30)
            l_mix = (self._C_mu ** 0.75) * (k_est ** 1.5) / (epsilon_est + 1e-30)
        else:
            l_mix = torch.full((n,), self._mixing_length, dtype=dtype, device=device)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = l_mix
        else:
            field[self._patch.face_indices] = l_mix
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for mixing length inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * self._mixing_length)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
