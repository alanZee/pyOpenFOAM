"""
zeroGradient2 boundary condition.

Zero gradient with optional non-orthogonal correction::

    type                    zeroGradient2;
    nonOrthogonalCorrection true;
    correctionFactor        1.0;

apply(): zero gradient + optional non-orthogonal correction

The base behaviour copies owner-cell values to boundary faces (zero normal
gradient).  When ``nonOrthogonalCorrection`` is enabled, an externally
computed correction field can be applied, weighted by ``correctionFactor``.
For uniform fields or with no correction data, behaviour reduces to plain
zero gradient.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ZeroGradient2BC"]


@BoundaryCondition.register("zeroGradient2")
class ZeroGradient2BC(BoundaryCondition):
    """Zero-gradient BC with optional non-orthogonal correction.

    The base behaviour is identical to ``zeroGradient``: copy owner-cell
    values to boundary faces.  When a correction field is provided via
    :meth:`set_correction`, the face values are adjusted::

        face_value = owner_value + correction_factor * correction

    Coefficients:
        - ``nonOrthogonalCorrection`` (bool): Enable correction.
          Default: ``False``.
        - ``correctionFactor`` (float): Relaxation factor for the
          correction (0 = no correction, 1 = full).  Default: 1.0.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._correction_enabled: bool = bool(
            self._coeffs.get("nonOrthogonalCorrection", False)
        )
        self._correction_factor: float = float(
            self._coeffs.get("correctionFactor", 1.0)
        )
        self._correction: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def correction_enabled(self) -> bool:
        """Whether non-orthogonal correction is active."""
        return self._correction_enabled

    @property
    def correction_factor(self) -> float:
        """Relaxation factor for the non-orthogonal correction."""
        return self._correction_factor

    # ------------------------------------------------------------------
    # Correction data
    # ------------------------------------------------------------------

    def set_correction(self, correction: torch.Tensor) -> None:
        """Set the non-orthogonal correction tensor (one value per face).

        Args:
            correction: Tensor of per-face correction values.
        """
        self._correction = correction.to(
            dtype=get_default_dtype(), device=get_device()
        )

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Copy owner-cell values to boundary faces with optional correction.

        Without correction, this is a standard zero-gradient BC.
        With correction enabled and data set, face values are adjusted::

            face = owner + factor * correction
        """
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]

        if (
            self._correction_enabled
            and self._correction_factor != 0.0
            and self._correction is not None
        ):
            corr = self._correction.to(device=field.device, dtype=field.dtype)
            owner_values = owner_values + self._correction_factor * corr

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = owner_values
        else:
            field[self._patch.face_indices] = owner_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero gradient: no matrix contribution (even with correction).

        The non-orthogonal correction is an explicit update applied in
        :meth:`apply` and does not affect the implicit matrix.
        """
        device = get_device()
        dtype = field.dtype
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
