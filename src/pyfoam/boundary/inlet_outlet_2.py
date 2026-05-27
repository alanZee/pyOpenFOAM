"""
pressureInletOutlet boundary condition.

Pressure-based inlet/outlet condition.  In OpenFOAM syntax::

    type       pressureInletOutlet;
    phi        phi;               (flux field name, informational)
    p0         uniform 101325;    (total pressure at inlet)
    value      uniform 0;

Behaviour:
- **Inflow** (``phi < 0``): applies total pressure (p0) as fixed value.
- **Outflow** (``phi >= 0``): applies zero gradient (∂p/∂n = 0).

This is the pressure analogue of inletOutlet, commonly used for
atmospheric/open boundaries where the total pressure is known at
inlets and zero-gradient is applied at outlets.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureInletOutletBC"]


@BoundaryCondition.register("pressureInletOutlet")
class PressureInletOutletBC(BoundaryCondition):
    """Pressure-based inlet/outlet boundary condition.

    - **Inflow** (flux < 0): applies total pressure p0 as fixed value.
    - **Outflow** (flux >= 0): applies zero gradient (copies owner values).

    The flow direction is determined by the sign of the face flux
    (or the velocity dot product with the face normal if no flux provided).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._p0 = float(self._coeffs.get("p0", 0.0))
        self._inlet_value = self._resolve_value()

    def _resolve_value(self) -> torch.Tensor:
        """Parse the ``value`` coefficient into a per-face tensor."""
        raw = self._coeffs.get("value", 0.0)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @property
    def p0(self) -> float:
        """Return the total pressure for inflow."""
        return self._p0

    @property
    def inlet_value(self) -> torch.Tensor:
        """Return the inlet prescribed value."""
        return self._inlet_value

    def _flow_direction(
        self,
        flux: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Determine flow direction at each face.

        Args:
            flux: ``(n_faces,)`` face flux.  Negative = inflow.
            velocity: ``(n_faces, 3)`` velocity at faces (fallback).
                Uses v·n to determine direction.

        Returns:
            Boolean tensor: ``True`` for inflow, ``False`` for outflow.
        """
        device = get_device()
        if flux is not None:
            flux_dev = flux.to(device=device, dtype=get_default_dtype())
            return flux_dev < 0.0  # negative flux = inflow

        if velocity is not None:
            normals = self._patch.face_normals.to(
                device=velocity.device, dtype=velocity.dtype
            )
            vn = (velocity * normals).sum(dim=-1)
            return vn < 0.0  # v·n < 0 = inflow

        # No flux/velocity info → assume all outflow
        return torch.zeros(self._patch.n_faces, dtype=torch.bool, device=device)

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        flux: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply pressure inlet/outlet behaviour.

        Args:
            field: Scalar pressure field to modify.
            patch_idx: Optional start index into field.
            flux: ``(n_faces,)`` face flux for direction detection.
            velocity: ``(n_faces, 3)`` velocity (fallback for direction).
        """
        is_inflow = self._flow_direction(flux, velocity)
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]

        # Inflow: use p0 as fixed value; outflow: copy owner values (zeroGrad)
        p0_tensor = torch.full(
            (self._patch.n_faces,),
            self._p0,
            dtype=field.dtype,
            device=field.device,
        )
        face_values = torch.where(is_inflow, p0_tensor, owner_values)

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
        flux: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matrix contributions depend on flow direction.

        - **Inflow**: penalty method (like fixedValue with p0).
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

        is_inflow = self._flow_direction(flux, velocity)
        if flux is not None:
            is_inflow = is_inflow.to(device=device)
        elif velocity is not None:
            is_inflow = is_inflow.to(device=device)

        # Mask: only inflow faces contribute
        masked_coeff = coeff * is_inflow.to(dtype=dtype)

        p0_tensor = torch.full(
            (self._patch.n_faces,),
            self._p0,
            dtype=dtype,
            device=device,
        )

        diag.scatter_add_(0, owners, masked_coeff)
        source.scatter_add_(0, owners, masked_coeff * p0_tensor)

        return diag, source
