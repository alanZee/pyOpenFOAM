"""
Energy boundary conditions for temperature/enthalpy fields.

Implements OpenFOAM energy boundary conditions:
- fixedEnergy: Fixed temperature/enthalpy BC
- mixedEnergy: Mixed (Robin) temperature BC

``gradientEnergy`` has been moved to :mod:`pyfoam.boundary.gradient_energy`
and is re-exported here for backward compatibility.

In OpenFOAM syntax::

    // fixedEnergy
    type    fixedEnergy;
    value   uniform 300;      // temperature (K)

    // mixedEnergy
    type    mixedEnergy;
    refValue    uniform 300;  // reference temperature (K)
    refGradient uniform 0;    // reference gradient (K/m)
    valueFraction uniform 0.5;// blending weight [0,1]
    value       uniform 300;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

# 向后兼容：从独立模块重新导出 GradientEnergyBC
from .gradient_energy import GradientEnergyBC  # noqa: F401

__all__ = [
    "FixedEnergyBC",
    "GradientEnergyBC",
    "MixedEnergyBC",
]


@BoundaryCondition.register("fixedEnergy")
class FixedEnergyBC(BoundaryCondition):
    """Fixed temperature/enthalpy boundary condition.

    Prescribes a fixed value (temperature or enthalpy) at each
    boundary face.  Identical in form to ``fixedValue`` but
    semantically scoped for energy fields.

    Coefficients:
        - ``value``: Temperature or enthalpy (default: 300 K).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._value = self._resolve_value()

    def _resolve_value(self) -> torch.Tensor:
        """Parse the ``value`` coefficient into a tensor."""
        raw = self._coeffs.get("value", 300.0)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @property
    def value(self) -> torch.Tensor:
        """Return the prescribed temperature/enthalpy."""
        return self._value

    @value.setter
    def value(self, new_value: float | torch.Tensor) -> None:
        """Update the prescribed boundary values."""
        if isinstance(new_value, torch.Tensor):
            self._value = new_value.to(
                dtype=get_default_dtype(), device=get_device()
            )
        else:
            self._value = torch.full_like(
                self._patch.face_areas, float(new_value)
            )

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face energy values to the prescribed value."""
        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = self._value
        else:
            field[self._patch.face_indices] = self._value
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method: large diagonal + matching source.

        diag[c]   += deltaCoeff * faceArea
        source[c] += deltaCoeff * faceArea * prescribedValue
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
        values = self._value.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * values)

        return diag, source


@BoundaryCondition.register("mixedEnergy")
class MixedEnergyBC(BoundaryCondition):
    """Mixed (Robin) temperature boundary condition.

    Blends between a fixed value and a zero-gradient condition using a
    per-face weight (``valueFraction``).  This is the standard OpenFOAM
    ``mixed`` BC applied to energy fields::

        phi_face = f * refValue + (1 - f) * (phi_owner + refGradient / deltaCoeff)

    where ``f = valueFraction`` is in [0, 1]:
        - f = 1 → fixed value
        - f = 0 → zero gradient
        - 0 < f < 1 → Robin / convective condition

    Coefficients:
        - ``refValue``: Reference temperature (default: 300).
        - ``refGradient``: Reference gradient (default: 0).
        - ``valueFraction``: Blending weight [0,1] (default: 1 → fixed value).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._ref_value = self._parse_coeff("refValue", 300.0)
        self._ref_gradient = self._parse_coeff("refGradient", 0.0)
        self._value_fraction = self._parse_coeff("valueFraction", 1.0).clamp(0.0, 1.0)

    def _parse_coeff(self, key: str, default: float) -> torch.Tensor:
        """Parse a coefficient into a per-face tensor."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @property
    def ref_value(self) -> torch.Tensor:
        """Return the reference temperature."""
        return self._ref_value

    @property
    def ref_gradient(self) -> torch.Tensor:
        """Return the reference gradient."""
        return self._ref_gradient

    @property
    def value_fraction(self) -> torch.Tensor:
        """Return the blending weight."""
        return self._value_fraction

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply mixed (Robin) blending.

        phi_face = f * refValue + (1 - f) * (phi_owner + refGradient / deltaCoeff)
        """
        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)
        owner_values = field[owners]
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        f = self._value_fraction.to(device=device, dtype=dtype)
        ref_val = self._ref_value.to(device=device, dtype=dtype)
        ref_grad = self._ref_gradient.to(device=device, dtype=dtype)

        dist = 1.0 / deltas
        zero_grad_face = owner_values + ref_grad * dist
        face_values = f * ref_val + (1.0 - f) * zero_grad_face

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = face_values
        else:
            field[self._patch.face_indices] = face_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mixed (Robin) matrix contribution.

        diag[c]   += f * deltaCoeff * faceArea
        source[c] += f * deltaCoeff * faceArea * refValue + (1-f) * gradient * faceArea
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

        f = self._value_fraction.to(device=device, dtype=dtype)
        ref_val = self._ref_value.to(device=device, dtype=dtype)
        ref_grad = self._ref_gradient.to(device=device, dtype=dtype)

        coeff = deltas * areas

        # Fixed-value part (weighted by f)
        diag.scatter_add_(0, owners, f * coeff)
        source.scatter_add_(0, owners, f * coeff * ref_val)

        # Gradient part (weighted by 1-f)
        source.scatter_add_(0, owners, (1.0 - f) * ref_grad * areas)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
