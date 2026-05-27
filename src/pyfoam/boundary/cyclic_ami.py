"""
cyclicAMI boundary condition.

Arbitrary Mesh Interface — maps values between non-conformal patches::

    type            cyclicAMI;
    neighbourPatch  cyclicAMI_half2;
    transform       noOrdering;

Unlike the standard cyclic BC which requires matching face meshes,
cyclicAMI performs interpolation between patches that may have
different face counts or layouts.

apply(): maps values between non-conformal patches via interpolation.

The interpolation is driven by an explicit mapping matrix: each face
on this patch receives a weighted sum of neighbour-patch face values.
The mapping is set externally via :meth:`set_ami_weights`.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["CyclicAMI"]


@BoundaryCondition.register("cyclicAMI")
class CyclicAMI(BoundaryCondition):
    """Arbitrary Mesh Interface boundary condition.

    Maps values between two patches whose faces are not geometrically
    conformal.  An interpolation weight matrix (AMI weights) encodes
    the geometric mapping: for each face on this patch, the weights
    specify a sparse linear combination of neighbour-patch face values.

    Coefficients:
        - ``neighbourPatch`` (str): Name of the coupled AMI patch.
          Default: ``None`` (falls back to ``patch.neighbour_patch``).
        - ``transform`` (str): Coordinate transform type.
          Default: ``"noOrdering"``.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._neighbour_field: torch.Tensor | None = None
        # AMI weights: (n_this_faces, n_neighbour_faces)
        # _weights[i, j] = fraction of neighbour face j contributing to face i
        self._weights: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def neighbour_patch_name(self) -> str | None:
        """Return the name of the coupled AMI patch."""
        return self._coeffs.get("neighbourPatch", self._patch.neighbour_patch)

    @property
    def transform(self) -> str:
        """Return the coordinate transform type."""
        return self._coeffs.get("transform", "noOrdering")

    # ------------------------------------------------------------------
    # Coupling data
    # ------------------------------------------------------------------

    def set_neighbour_field(self, neighbour_field: torch.Tensor) -> None:
        """Set the neighbour-patch face values.

        Args:
            neighbour_field: Tensor of face values from the coupled patch.
        """
        self._neighbour_field = neighbour_field.to(
            dtype=get_default_dtype(), device=get_device()
        )

    def set_ami_weights(self, weights: torch.Tensor) -> None:
        """Set the AMI interpolation weight matrix.

        Args:
            weights: Shape ``(n_this_faces, n_neighbour_faces)``.
                Each row sums to 1 and gives the linear combination
                of neighbour face values interpolated onto one
                face of this patch.
        """
        self._weights = weights.to(
            dtype=get_default_dtype(), device=get_device()
        )

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Map neighbour values onto this patch via AMI interpolation.

        If weights and neighbour data are both available, the face
        values are computed as ``weights @ neighbour_field``.  Otherwise
        falls back to zero-gradient (owner cell values).
        """
        if self._neighbour_field is not None and self._weights is not None:
            nfield = self._neighbour_field.to(
                device=field.device, dtype=field.dtype
            )
            w = self._weights.to(device=field.device, dtype=field.dtype)
            values = torch.mv(w, nfield)
        else:
            # Fallback: zero-gradient (copy from owner cells)
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
        """Implicit coupling via AMI weights.

        For each face *i* on this patch:

        - interpolated value:  v_i = sum_j (w_ij * neighbour_j)
        - diag[c(i)]   += deltaCoeff_i * area_i
        - source[c(i)] += deltaCoeff_i * area_i * v_i

        If weights / neighbour data are not available, the source
        contribution is zero (treated as zero-flux).
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

        if self._neighbour_field is not None and self._weights is not None:
            nfield = self._neighbour_field.to(device=device, dtype=dtype)
            w = self._weights.to(device=device, dtype=dtype)
            interpolated = torch.mv(w, nfield)
            source.scatter_add_(0, owners, coeff * interpolated)
        # else: no AMI data -> treat as zero-flux

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
