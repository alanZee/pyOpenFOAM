"""
processor boundary condition for parallel decompositions.

Handles data exchange between processor patches in parallel runs.
In OpenFOAM syntax::

    type   processor;

In a decomposed domain, each processor boundary face is coupled with
a face on a neighbouring processor.  The processor BC:

1. Exchanges boundary face values between coupled processors
2. Supports both blocking and non-blocking communication patterns
3. Maintains data consistency across processor boundaries

In the serial (single-process) case, processor BCs behave as identity
operations — values from the coupled patch are directly available.

Usage::

    @BoundaryCondition.register("processor")
    class ProcessorBC(BoundaryCondition):
        ...
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ProcessorBC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("processor")
class ProcessorBC(BoundaryCondition):
    """Processor boundary condition for parallel decompositions.

    Handles data exchange at processor-to-processor boundaries.
    In serial mode, acts as a pass-through that copies coupled-patch
    values (or zeros if no coupled data is available).

    Attributes
    ----------
    _neighbour_field : torch.Tensor or None
        Cached field values from the coupled processor patch.
    _my_proc : int
        This processor's rank (default 0 for serial).
    _neighbour_proc : int
        Neighbouring processor's rank (default 1 for serial).
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

        logger.debug(
            "ProcessorBC: proc %d <-> proc %d, n_faces=%d",
            self._my_proc, self._neighbour_proc, self._patch.n_faces,
        )

    # ------------------------------------------------------------------
    # Communication interface
    # ------------------------------------------------------------------

    def set_neighbour_field(self, neighbour_field: torch.Tensor) -> None:
        """Set the field values received from the coupled processor.

        In a real parallel run, this would be populated by MPI
        communication.  For serial testing, values can be set directly.

        Args:
            neighbour_field: Tensor of face values from the coupled
                processor patch.
        """
        self._neighbour_field = neighbour_field.to(
            dtype=get_default_dtype(), device=get_device()
        )

    def prepare_send_buffer(self, field: torch.Tensor) -> torch.Tensor:
        """Prepare a send buffer with this patch's face values.

        Extracts the boundary face values that need to be sent to
        the neighbouring processor.

        Args:
            field: Full field tensor.

        Returns:
            Tensor of face values to send.
        """
        if self._patch.face_indices.numel() == 0:
            device = field.device
            dtype = field.dtype
            if field.dim() == 1:
                return torch.zeros(0, dtype=dtype, device=device)
            return torch.zeros(0, 3, dtype=dtype, device=device)

        if field.dim() == 1:
            return field[self._patch.face_indices].clone()
        else:
            return field[self._patch.face_indices].clone()

    def receive_buffer(self, buffer: torch.Tensor) -> None:
        """Process a received buffer from the coupled processor.

        Args:
            buffer: Tensor of face values received from the neighbour.
        """
        self.set_neighbour_field(buffer)

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply processor BC to the field.

        If neighbour data is available, copies the coupled patch
        values to this patch's boundary faces.  Otherwise, falls
        back to copying from owner cells (zero-gradient-like).

        In a serial run with properly coupled patches, this ensures
        continuity across processor boundaries.

        Args:
            field: Full field tensor.
            patch_idx: Optional explicit start index.

        Returns:
            Modified field with processor boundary values set.
        """
        if self._neighbour_field is not None:
            values = self._neighbour_field
        else:
            # Serial fallback: copy from owner cells
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
        """Processor BC matrix contributions.

        Uses the coupled-patch values for implicit coupling:
            diag[c]   += deltaCoeff * area
            source[c] += deltaCoeff * area * neighbourValue

        This is analogous to a fixedValue penalty approach but with
        values from the coupled processor.

        Args:
            field: Current field values.
            n_cells: Total number of cells.
            diag: Pre-existing diagonal tensor.
            source: Pre-existing source tensor.

        Returns:
            ``(diag, source)`` tuple.
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

        # Add diagonal contribution
        diag.scatter_add_(0, owners, coeff)

        # Add source contribution from neighbour values
        if self._neighbour_field is not None:
            nvalues = self._neighbour_field.to(device=device, dtype=dtype)
            source.scatter_add_(0, owners, coeff * nvalues)
        else:
            # No neighbour data: use owner values (zero-flux)
            owner_vals = field[owners].to(dtype=dtype)
            source.scatter_add_(0, owners, coeff * owner_vals)

        return diag, source
