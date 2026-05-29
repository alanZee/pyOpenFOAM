"""
Enhanced cyclic AMI boundary condition.

Improves on the base ``cyclicAMI`` BC with:

1. **Weighted interpolation** with optional distance-based weighting.
2. **Conservation enforcement** ensuring flux balance between coupled
   patches.
3. **Non-orthogonal correction** for improved accuracy on skewed meshes.

The interpolation applies a corrected AMI mapping::

    phi_face = sum_j (w_ij * alpha_ij * phi_neighbour_j)

where ``w_ij`` are the AMI geometric weights and ``alpha_ij`` are
optional distance-based correction factors.

In OpenFOAM syntax::

    type            cyclicAMI2;
    neighbourPatch  cyclicAMI2_half2;
    transform       noOrdering;
    conserve        true;          // enforce flux conservation
    nonOrthoCorrect true;          // apply non-orthogonal correction
    value           uniform 0;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["CyclicAMI2BC"]


@BoundaryCondition.register("cyclicAMI2")
class CyclicAMI2BC(BoundaryCondition):
    """Enhanced Arbitrary Mesh Interface boundary condition.

    Extends the base cyclicAMI with conservation enforcement and
    non-orthogonal correction.

    Coefficients:
        - ``neighbourPatch`` (str): Coupled AMI patch name.  Default:
          ``patch.neighbour_patch``.
        - ``transform`` (str): Coordinate transform type.
          Default ``"noOrdering"``.
        - ``conserve`` (bool): Enforce flux conservation.  Default True.
        - ``nonOrthoCorrect`` (bool): Apply non-orthogonal correction.
          Default True.
        - ``tolerance`` (float): Conservation tolerance.  Default 1e-6.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._neighbour_field: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None
        self._correction_factors: torch.Tensor | None = None

        self._conserve = bool(self._coeffs.get("conserve", True))
        self._non_ortho_correct = bool(self._coeffs.get("nonOrthoCorrect", True))
        self._tolerance = float(self._coeffs.get("tolerance", 1e-6))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def neighbour_patch_name(self) -> str | None:
        """Name of the coupled AMI patch."""
        return self._coeffs.get("neighbourPatch", self._patch.neighbour_patch)

    @property
    def transform(self) -> str:
        """Coordinate transform type."""
        return self._coeffs.get("transform", "noOrdering")

    @property
    def conserve(self) -> bool:
        """Whether flux conservation is enforced."""
        return self._conserve

    @property
    def non_ortho_correct(self) -> bool:
        """Whether non-orthogonal correction is applied."""
        return self._non_ortho_correct

    @property
    def tolerance(self) -> float:
        """Conservation tolerance."""
        return self._tolerance

    # ------------------------------------------------------------------
    # Coupling data
    # ------------------------------------------------------------------

    def set_neighbour_field(self, neighbour_field: torch.Tensor) -> None:
        """Set the neighbour-patch face values.

        Args:
            neighbour_field: Tensor from the coupled patch.
        """
        self._neighbour_field = neighbour_field.to(
            dtype=get_default_dtype(), device=get_device()
        )

    def set_ami_weights(self, weights: torch.Tensor) -> None:
        """Set the AMI interpolation weight matrix.

        Args:
            weights: Shape ``(n_this_faces, n_neighbour_faces)``.
        """
        self._weights = weights.to(
            dtype=get_default_dtype(), device=get_device()
        )

    def set_correction_factors(self, factors: torch.Tensor) -> None:
        """Set non-orthogonal correction factors.

        Args:
            factors: Shape ``(n_this_faces,)`` with correction multipliers.
        """
        self._correction_factors = factors.to(
            dtype=get_default_dtype(), device=get_device()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interpolated_values(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute interpolated neighbour values with corrections."""
        n = self._patch.n_faces
        if self._neighbour_field is not None and self._weights is not None:
            nfield = self._neighbour_field.to(device=device, dtype=dtype)
            w = self._weights.to(device=device, dtype=dtype)
            values = torch.mv(w, nfield)

            # Apply non-orthogonal correction
            if self._non_ortho_correct and self._correction_factors is not None:
                corr = self._correction_factors[:n].to(device=device, dtype=dtype)
                values = values * corr

            # Enforce conservation: scale to match total flux balance
            if self._conserve:
                total_neighbour = nfield.sum()
                total_interpolated = values.sum()
                if total_interpolated.abs() > self._tolerance:
                    scale = total_neighbour / (total_interpolated + 1e-30)
                    values = values * scale

            return values
        else:
            # Fallback: zero-gradient
            owners = self._patch.owner_cells.to(device=device)
            # Need to index from a full field; use zeros as proxy
            return torch.zeros(n, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply enhanced AMI interpolation.

        Args:
            field: Full field tensor.
            patch_idx: Optional start index.
        """
        device = field.device
        dtype = field.dtype

        if self._neighbour_field is not None and self._weights is not None:
            values = self._interpolated_values(device, dtype)
        else:
            # Fallback: zero-gradient
            owners = self._patch.owner_cells.to(device=device)
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
        """Implicit coupling via enhanced AMI weights.

        For each face i:
            diag[c(i)]   += deltaCoeff_i * area_i
            source[c(i)] += deltaCoeff_i * area_i * interpolated_i
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
            interpolated = self._interpolated_values(device, dtype)
            source.scatter_add_(0, owners, coeff * interpolated)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
