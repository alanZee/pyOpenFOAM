"""
Coded fixed-value boundary condition.

Evaluates a user-defined Python expression at runtime to produce a
**fixed** boundary value.  Unlike the generic ``coded`` BC which can
return arbitrary computations, this BC enforces a fixed-value constraint
once the expression is evaluated::

    type        codedFixedValue;
    code        "lambda patch, field: 300 + 10 * torch.sin(field[patch.owner_cells])";
    value       uniform 300;

The expression is evaluated each ``apply()`` call and the result is
directly imposed as the boundary face value (Dirichlet).

In contrast to ``coded`` (which can be any BC logic), ``codedFixedValue``
always treats the computed value as a strict Dirichlet condition.

Usage::

    bc = BoundaryCondition.create("codedFixedValue", patch, coeffs={
        "code": lambda p, f: torch.full((p.n_faces,), 350.0),
    })
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["CodedFixedValueBC"]


@BoundaryCondition.register("codedFixedValue")
class CodedFixedValueBC(BoundaryCondition):
    """Coded fixed-value boundary condition.

    Evaluates a user-supplied expression and applies the result as a
    Dirichlet (fixed-value) boundary condition.

    Coefficients:
        - ``code`` (callable | str): Python callable
          ``f(patch, field) -> Tensor`` or eval-able string.  Required.
        - ``value`` (float): Initial / fallback value (default: 0).
        - ``scale`` (float): Scaling factor applied to the computed
          values.  Default 1.0.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._user_fn = self._resolve_code()
        self._scale = float(self._coeffs.get("scale", 1.0))

    def _resolve_code(self) -> Callable[..., torch.Tensor]:
        """Parse the ``code`` coefficient into a callable."""
        code = self._coeffs.get("code")
        if code is None:
            raise KeyError("'codedFixedValue' BC requires a 'code' coefficient")
        if callable(code):
            return code
        if isinstance(code, str):
            return eval(code)  # noqa: S307
        raise TypeError(
            f"'code' must be a callable or eval-able string, got {type(code)}"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def user_fn(self) -> Callable[..., torch.Tensor]:
        """Return the user-supplied function."""
        return self._user_fn

    @property
    def scale(self) -> float:
        """Scaling factor for computed values."""
        return self._scale

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply coded fixed-value BC.

        Evaluates the user expression and sets boundary faces to the
        (possibly scaled) result.

        Args:
            field: Full field tensor.
            patch_idx: Optional start index.
        """
        face_values = self._user_fn(self._patch, field)

        if face_values.dim() == 0:
            face_values = face_values.expand(self._patch.n_faces)

        face_values = face_values * self._scale

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
        """Fixed-value penalty matrix contribution.

        diag[c]   += deltaCoeff * faceArea
        source[c] += deltaCoeff * faceArea * codedValue
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

        face_values = self._user_fn(self._patch, field).to(device=device, dtype=dtype)
        if face_values.dim() == 0:
            face_values = face_values.expand(self._patch.n_faces)
        face_values = face_values * self._scale

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * face_values)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
