"""
Coded (user-defined function) boundary condition.

Allows the user to supply a Python callable that is evaluated at each
``apply()`` call.  This is analogous to OpenFOAM's ``coded`` BC::

    type    coded;
    name    myCustomBC;
    code    "lambda patch, field: torch.full((patch.n_faces,), 300.0)";
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["CodedBC"]


@BoundaryCondition.register("coded")
class CodedBC(BoundaryCondition):
    """User-defined coded boundary condition.

    Accepts a Python callable (or code string) that computes boundary
    face values.  The callable receives the patch info and current
    field, and must return a 1-D tensor of face values.

    Coefficients:
        - ``code``: A Python callable ``f(patch, field) -> Tensor``
          **or** a string containing a lambda expression.
          Required.
        - ``value``: Initial/fallback value (default: 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._user_fn = self._resolve_code()

    def _resolve_code(self) -> Callable[..., torch.Tensor]:
        """Parse the ``code`` coefficient into a callable."""
        code = self._coeffs.get("code")
        if code is None:
            raise KeyError("'coded' BC requires a 'code' coefficient")
        if callable(code):
            return code
        if isinstance(code, str):
            # 通过 eval 解析字符串形式的函数
            return eval(code)  # noqa: S307 — 受信输入
        raise TypeError(
            f"'code' must be a callable or eval-able string, got {type(code)}"
        )

    @property
    def user_fn(self) -> Callable[..., torch.Tensor]:
        """Return the user-supplied function."""
        return self._user_fn

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply the user function to set boundary-face values.

        Args:
            field: Full field tensor.
            patch_idx: Optional explicit start index into *field*.
        """
        face_values = self._user_fn(self._patch, field)
        # 确保形状正确
        if face_values.dim() == 0:
            face_values = face_values.expand(self._patch.n_faces)

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
        """Penalty method with user-defined values.

        diag[c]   += deltaCoeff * faceArea
        source[c] += deltaCoeff * faceArea * userValue
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

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * face_values)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
