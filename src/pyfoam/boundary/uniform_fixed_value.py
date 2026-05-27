"""
uniformFixedValue boundary condition.

Time-varying boundary value specified by a function object.  In
OpenFOAM syntax::

    type            uniformFixedValue;
    uniformValue    table ((0 0) (1 1) (2 0));

or using a coded function::

    uniformValue    sine;

The value is evaluated at each time-step via a callable stored in
``coeffs["uniformValue"]``.  The callable signature is::

    value = f(t: float) -> float | torch.Tensor

If ``uniformValue`` is a plain number the BC behaves like
``fixedValue`` with that constant.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition

__all__ = ["UniformFixedValueBC"]


# ---------------------------------------------------------------------------
# Built-in function objects
# ---------------------------------------------------------------------------


def _make_ramp(coeffs: dict[str, Any]) -> Callable[[float], float]:
    """Create a linear ramp ``start + (end - start) * t / duration``."""
    start = float(coeffs.get("start", 0.0))
    end = float(coeffs.get("end", 1.0))
    duration = float(coeffs.get("duration", 1.0))

    def ramp(t: float) -> float:
        frac = min(max(t / duration, 0.0), 1.0)
        return start + (end - start) * frac

    return ramp


def _make_sine(coeffs: dict[str, Any]) -> Callable[[float], float]:
    """Create ``amplitude * sin(2*pi*frequency*t + phase) + offset``."""
    amplitude = float(coeffs.get("amplitude", 1.0))
    frequency = float(coeffs.get("frequency", 1.0))
    phase = float(coeffs.get("phase", 0.0))
    offset = float(coeffs.get("offset", 0.0))
    import math

    def sine(t: float) -> float:
        return amplitude * math.sin(2.0 * math.pi * frequency * t + phase) + offset

    return sine


_BUILTIN_FUNCTIONS: dict[str, Callable[[dict[str, Any]], Callable]] = {
    "ramp": _make_ramp,
    "sine": _make_sine,
}


# ---------------------------------------------------------------------------
# Main BC
# ---------------------------------------------------------------------------


@BoundaryCondition.register("uniformFixedValue")
class UniformFixedValueBC(BoundaryCondition):
    """Time-varying uniform boundary value.

    The value at each time-step is obtained by calling a function
    object stored in ``coeffs["uniformValue"]``.
    """

    def __init__(self, patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._func = self._resolve_function()

    def _resolve_function(self) -> Callable[[float], float | torch.Tensor]:
        """Parse the ``uniformValue`` coefficient.

        Priority:
        1. Callable (function / lambda) — used directly.
        2. String naming a built-in (``"ramp"``, ``"sine"``).
        3. Plain number — constant value.
        """
        raw = self._coeffs.get("uniformValue", 0.0)
        if callable(raw):
            return raw
        if isinstance(raw, str):
            key = raw.lower()
            if key in _BUILTIN_FUNCTIONS:
                return _BUILTIN_FUNCTIONS[key](self._coeffs)
            raise ValueError(
                f"Unknown uniformValue function '{raw}'. "
                f"Built-ins: {sorted(_BUILTIN_FUNCTIONS)}"
            )
        # Plain number → constant
        val = float(raw)
        return lambda _t: val

    @property
    def func(self) -> Callable[[float], float | torch.Tensor]:
        """Return the value function."""
        return self._func

    def evaluate(self, t: float) -> torch.Tensor:
        """Evaluate the boundary value at time *t*.

        Returns:
            ``(n_faces,)`` tensor with the evaluated value broadcast
            to every face.
        """
        raw = self._func(t)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        t: float = 0.0,
    ) -> torch.Tensor:
        """Set boundary-face values from the time function.

        Args:
            field: Full field tensor.
            patch_idx: Optional start index into field.
            t: Current simulation time.
        """
        values = self.evaluate(t).to(device=field.device, dtype=field.dtype)
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
        t: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method, same as fixedValue but using the time-
        evaluated value.
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
        values = self.evaluate(t).to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * values)

        return diag, source
