"""
Turbulent dissipation inlet boundary condition.

Computes the turbulent dissipation rate epsilon from the turbulent
kinetic energy k and a mixing length l_mix::

    epsilon = C_mu^0.75 * k^1.5 / l_mix

If k is not provided, it is estimated from the velocity and turbulence
intensity.  This differs from ``turbulentDissipationRateInlet`` by
supporting explicit k and l_mix override parameters.

In OpenFOAM syntax::

    type        turbulentDissipationInlet;
    mixingLength 0.01;
    Cmu         0.09;
    intensity   0.05;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentDissipationInletBC"]


@BoundaryCondition.register("turbulentDissipationInlet")
class TurbulentDissipationInletBC(BoundaryCondition):
    """Turbulent dissipation rate inlet boundary condition.

    epsilon = C_mu^0.75 * k^1.5 / l_mix

    Coefficients:
        - ``mixingLength``: Mixing length (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``intensity``: Fallback turbulence intensity (default 0.05).
        - ``value``: Initial epsilon value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._intensity = float(self._coeffs.get("intensity", 0.05))

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

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        l_mix: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face epsilon.

        epsilon = C_mu^0.75 * k^1.5 / l_mix

        Args:
            field: Turbulent dissipation rate field.
            patch_idx: Optional start index.
            k: ``(n_faces,)`` turbulent kinetic energy.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            l_mix: Per-face or scalar mixing length override.
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
            epsilon = (self._C_mu ** 0.75) * (k ** 1.5) / (l + 1e-30)
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
            epsilon = (self._C_mu ** 0.75) * (k_est ** 1.5) / (l + 1e-30)
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
        """Penalty method for epsilon inlet BC."""
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
