"""
Mixed temperature boundary condition.

Blends a fixed-value (Dirichlet) and gradient (Neumann) condition to
produce a Robin / mixed (type III) thermal boundary condition::

    q = alpha * (T - T_ref)          (convective part)
    T_face = (grad_coeff * T_interior + value_coeff * T_ref) / (grad_coeff + value_coeff)

where:
    - ``alpha`` is the heat transfer coefficient
    - ``T_ref`` is the reference (ambient) temperature
    - grad_coeff and value_coeff are blending weights derived from
      face geometry and alpha.

In OpenFOAM syntax::

    type        mixedTemperature;
    T_ref       300;         // reference / ambient temperature
    alpha       10;          // heat transfer coefficient (W/m²K)
    value       uniform 300;

Usage::

    bc = BoundaryCondition.create("mixedTemperature", patch, coeffs={
        "T_ref": 300.0,
        "alpha": 10.0,
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MixedTemperatureBC"]


@BoundaryCondition.register("mixedTemperature")
class MixedTemperatureBC(BoundaryCondition):
    """Mixed (Robin) temperature boundary condition.

    Blends fixed-value and gradient contributions to produce a Robin BC::

        T_face = (h * T_ref + k/delta * T_interior) / (h + k/delta)

    where h = alpha (heat transfer coefficient), k/delta is the
    diffusion coefficient over the near-wall distance, T_ref is the
    reference temperature, and T_interior is the owner-cell value.

    Coefficients:
        - ``T_ref``: Reference / ambient temperature (K).  Default 300.
        - ``alpha``: Heat transfer coefficient (W/m²K).  Default 10.
        - ``value``: Initial temperature (default 300).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._t_ref = float(self._coeffs.get("T_ref", 300.0))
        self._alpha = float(self._coeffs.get("alpha", 10.0))

    @property
    def t_ref(self) -> float:
        """Reference / ambient temperature (K)."""
        return self._t_ref

    @property
    def alpha(self) -> float:
        """Heat transfer coefficient (W/m²K)."""
        return self._alpha

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply mixed temperature BC.

        Computes a Robin (type III) blend of fixed-value and gradient::

            T_face = (h * T_ref + (k/delta) * T_interior) / (h + k/delta)

        For simplicity, the diffusion coefficient k/delta is taken as
        the patch's ``delta_coeffs`` (1/distance), which serves as the
        geometric weight.

        Args:
            field: Temperature field.
            patch_idx: Optional start index into *field*.
        """
        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)
        owner_vals = field[owners]

        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        h = self._alpha

        # Robin blend: T = (h * T_ref + delta * T_int) / (h + delta)
        denom = h + deltas + 1e-30
        t_face = (h * self._t_ref + deltas * owner_vals) / denom

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = t_face
        else:
            field[self._patch.face_indices] = t_face
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Robin-type matrix contribution.

        Adds implicit diagonal and source from the Robin BC formulation::

            diag   += h * A / (h + k/delta)
            source += h * A * T_ref / (h + k/delta)
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        h = self._alpha

        denom = h + deltas + 1e-30
        coeff = h * areas / denom

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * self._t_ref)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
