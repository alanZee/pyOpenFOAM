"""
outletInlet boundary condition.

Behaves as fixedValue when flow enters the domain and as zeroGradient
when flow exits.  This is the opposite behaviour of ``inletOutlet``.

In OpenFOAM syntax::

    type       outletInlet;
    inletValue uniform 0;

The direction is determined by the sign of ``v · n`` (velocity dot
face normal): negative means inflow, positive means outflow.

- **Inflow** (``v · n < 0``): prescribed value applied.
- **Outflow** (``v · n >= 0``): zero-gradient (copy from owner cell).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["OutletInletBC"]


@BoundaryCondition.register("outletInlet")
class OutletInletBC(BoundaryCondition):
    """Outlet/inlet boundary condition.

    - **Inflow** (``v · n < 0``): applies prescribed inlet value.
    - **Outflow** (``v · n >= 0``): applies zero gradient.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._inlet_value = self._resolve_value()

    def _resolve_value(self) -> torch.Tensor:
        """Parse the ``inletValue`` coefficient into a tensor."""
        raw = self._coeffs.get("inletValue", self._coeffs.get("value", 0.0))
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @property
    def inlet_value(self) -> torch.Tensor:
        """Return the prescribed inlet value."""
        return self._inlet_value

    def _flow_direction(
        self,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Determine flow direction at each face.

        Returns:
            Boolean tensor: ``True`` for inflow (``v · n < 0``),
            ``False`` for outflow.
        """
        normals = self._patch.face_normals.to(
            device=velocity.device, dtype=velocity.dtype
        )
        vn = (velocity * normals).sum(dim=-1)
        return vn < 0.0  # True = inflow

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply outlet/inlet behaviour.

        Args:
            field: Field to modify.
            patch_idx: Optional start index into field.
            velocity: ``(n_faces, 3)`` velocity at boundary faces.
                If ``None``, defaults to zero-gradient behaviour.
        """
        if velocity is None:
            # 无速度信息时回退到零梯度（拷贝 owner 值）
            owners = self._patch.owner_cells.to(device=field.device)
            owner_values = field[owners]
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = owner_values
            else:
                field[self._patch.face_indices] = owner_values
            return field

        is_inflow = self._flow_direction(velocity)
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]
        inlet_vals = self._inlet_value.to(device=field.device, dtype=field.dtype)

        # 入流 → 固定值，出流 → owner 值（零梯度）
        face_values = torch.where(is_inflow, inlet_vals, owner_values)

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
        velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matrix contributions depend on flow direction.

        - **Inflow**: penalty method (like fixedValue).
        - **Outflow**: zero contribution (like zeroGradient).
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
        coeff = deltas * areas

        if velocity is not None:
            is_inflow = self._flow_direction(velocity.to(device=device))
        else:
            is_inflow = torch.zeros(
                self._patch.n_faces, dtype=torch.bool, device=device
            )

        inlet_vals = self._inlet_value.to(device=device, dtype=dtype)

        # 仅入流面有贡献
        masked_coeff = coeff * is_inflow.to(dtype=dtype)

        diag.scatter_add_(0, owners, masked_coeff)
        source.scatter_add_(0, owners, masked_coeff * inlet_vals)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
