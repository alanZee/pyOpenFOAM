"""
Processor cyclic boundary condition for parallel decompositions.

Handles data exchange between **cyclic** (coupled) patches that span
processor boundaries in parallel runs.  Unlike the plain ``processor``
BC which exchanges arbitrary patch data, ``processorCyclic`` is aware
of the cyclic (periodic) coupling and applies the correct coordinate
transformation between the two halves::

    type   processorCyclic;
    myProcNo     0;
    neighbProcNo 1;
    transform    rotational;   // rotational | translational | noOrdering

The BC:
1. Exchanges face values between coupled processor patches.
2. Applies the cyclic coordinate transform (rotation / translation).
3. Falls back to zero-gradient when no coupled data is available.

Usage::

    bc = BoundaryCondition.create("processorCyclic", patch, coeffs={
        "myProcNo": 0, "neighbProcNo": 1, "transform": "rotational",
    })
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ProcessorCyclicBC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("processorCyclic")
class ProcessorCyclicBC(BoundaryCondition):
    """Processor cyclic boundary condition for parallel decompositions.

    Combines the ``processor`` BC's inter-process data exchange with the
    ``cyclic`` BC's coordinate transformation awareness.

    Coefficients:
        - ``myProcNo`` (int): This processor's rank.  Default 0.
        - ``neighbProcNo`` (int): Neighbouring processor's rank.  Default 1.
        - ``transform`` (str): Coordinate transform type.
          ``"rotational"``, ``"translational"``, or ``"noOrdering"``.
          Default ``"noOrdering"``.
        - ``rotationAxis`` (list[float]): Rotation axis for rotational
          transform.  Default ``[0, 0, 1]``.
        - ``rotationAngle`` (float): Rotation angle in degrees.
          Default 180.
    """

    def __init__(
        self,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(patch, coeffs)

        self._neighbour_field: torch.Tensor | None = None
        self._my_proc: int = int(self._coeffs.get("myProcNo", 0))
        self._neighbour_proc: int = int(self._coeffs.get("neighbProcNo", 1))
        self._transform: str = self._coeffs.get("transform", "noOrdering")
        self._rotation_axis = self._coeffs.get("rotationAxis", [0.0, 0.0, 1.0])
        self._rotation_angle = float(self._coeffs.get("rotationAngle", 180.0))

        logger.debug(
            "ProcessorCyclicBC: proc %d <-> proc %d, transform=%s, n_faces=%d",
            self._my_proc, self._neighbour_proc, self._transform, self._patch.n_faces,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def my_proc(self) -> int:
        """This processor's rank."""
        return self._my_proc

    @property
    def neighbour_proc(self) -> int:
        """Neighbouring processor's rank."""
        return self._neighbour_proc

    @property
    def transform(self) -> str:
        """Coordinate transform type."""
        return self._transform

    # ------------------------------------------------------------------
    # Communication interface
    # ------------------------------------------------------------------

    def set_neighbour_field(self, neighbour_field: torch.Tensor) -> None:
        """Set the field values received from the coupled processor.

        Values are automatically transformed according to the cyclic
        transform specification.

        Args:
            neighbour_field: Face values from the coupled processor patch.
        """
        raw = neighbour_field.to(dtype=get_default_dtype(), device=get_device())
        self._neighbour_field = self._apply_transform(raw)

    def prepare_send_buffer(self, field: torch.Tensor) -> torch.Tensor:
        """Prepare a send buffer with this patch's face values.

        Args:
            field: Full field tensor.

        Returns:
            Tensor of face values to send.
        """
        if self._patch.face_indices.numel() == 0:
            return torch.zeros(0, dtype=field.dtype, device=field.device)
        return field[self._patch.face_indices].clone()

    def receive_buffer(self, buffer: torch.Tensor) -> None:
        """Process a received buffer from the coupled processor.

        Args:
            buffer: Face values received from the neighbour.
        """
        self.set_neighbour_field(buffer)

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _apply_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the cyclic coordinate transform to neighbour values.

        For vector fields (dim >= 2):
        - ``rotational``: Apply rotation matrix.
        - ``translational``: No-op (pass-through).
        - ``noOrdering``: No-op (pass-through).

        For scalar fields: always pass-through.
        """
        if values.dim() < 2 or self._transform == "noOrdering":
            return values

        if self._transform == "translational":
            return values

        if self._transform == "rotational":
            return self._rotate_vectors(values)

        return values

    def _rotate_vectors(self, values: torch.Tensor) -> torch.Tensor:
        """Apply rotation to vector values.

        Uses Rodrigues' rotation formula around the specified axis.
        """
        device = values.device
        dtype = values.dtype

        axis = torch.tensor(self._rotation_axis, dtype=dtype, device=device)
        axis = axis / (axis.norm() + 1e-30)

        angle_rad = self._rotation_angle * 3.141592653589793 / 180.0
        cos_a = torch.cos(torch.tensor(angle_rad, dtype=dtype, device=device))
        sin_a = torch.sin(torch.tensor(angle_rad, dtype=dtype, device=device))

        # Rodrigues: v' = v*cos + (k x v)*sin + k*(k.v)*(1-cos)
        k = axis
        v = values
        k_dot_v = (k.unsqueeze(0) * v).sum(dim=-1, keepdim=True)
        k_cross_v = torch.cross(k.expand_as(v), v, dim=-1)

        rotated = v * cos_a + k_cross_v * sin_a + k.unsqueeze(0) * k_dot_v * (1.0 - cos_a)
        return rotated

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply processor cyclic BC.

        If neighbour data is available, uses the (transformed) coupled
        patch values.  Otherwise falls back to owner cell values.
        """
        if self._neighbour_field is not None:
            values = self._neighbour_field
        else:
            owners = self._patch.owner_cells.to(device=field.device)
            values = field[owners]

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = values
        else:
            field[self._patch.face_indices] = values

        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matrix contributions for processor cyclic coupling.

        Uses penalty coupling with transformed neighbour values:
            diag[c]   += deltaCoeff * area
            source[c] += deltaCoeff * area * neighbourValue
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
        diag.scatter_add_(0, owners, coeff)

        if self._neighbour_field is not None:
            nvalues = self._neighbour_field.to(device=device, dtype=dtype)
            source.scatter_add_(0, owners, coeff * nvalues)
        else:
            owner_vals = field[owners].to(dtype=dtype)
            source.scatter_add_(0, owners, coeff * owner_vals)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
