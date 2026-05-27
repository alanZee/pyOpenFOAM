"""
fixedValue2 boundary condition.

Fixed value with time-dependent coefficient::

    type        fixedValue2;
    baseValue   uniform 1;
    timeFunction sine;
    amplitude   0.5;
    frequency   1.0;

apply(): value = base_value * time_function(t)

Supported time functions:
    - ``constant``: factor = 1.0 (default, same as fixedValue)
    - ``linear``:   factor = slope * t
    - ``sine``:     factor = amplitude * sin(2 * pi * frequency * t)
    - ``cosine``:   factor = amplitude * cos(2 * pi * frequency * t)
    - ``step``:     factor = amplitude if t >= onset, else 0
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["FixedValue2BC"]

# Time-function dispatch table
_TIME_FUNCTIONS = {
    "constant": lambda t, amp, freq, phase: 1.0,
    "linear": lambda t, amp, freq, phase: amp * t,
    "sine": lambda t, amp, freq, phase: amp * math.sin(2.0 * math.pi * freq * t + phase),
    "cosine": lambda t, amp, freq, phase: amp * math.cos(2.0 * math.pi * freq * t + phase),
    "step": lambda t, amp, freq, phase: amp if t >= phase else 0.0,
}


@BoundaryCondition.register("fixedValue2")
class FixedValue2BC(BoundaryCondition):
    """Fixed-value BC with time-dependent coefficient.

    The boundary face value at time *t* is::

        value(t) = base_value * time_function(t)

    Coefficients:
        - ``baseValue`` (float | list): Base value, uniform or per-face.
          Default: 0.
        - ``timeFunction`` (str): One of ``constant``, ``linear``,
          ``sine``, ``cosine``, ``step``.  Default: ``constant``.
        - ``amplitude`` (float): Amplitude for the time function.  Default: 1.
        - ``frequency`` (float): Frequency [Hz] for sine/cosine.  Default: 1.
        - ``phase`` (float): Phase offset [rad] for sine/cosine, or onset
          time for step.  Default: 0.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._base_value = self._resolve_base_value()
        self._func_name: str = self._coeffs.get("timeFunction", "constant")
        self._amplitude: float = float(self._coeffs.get("amplitude", 1.0))
        self._frequency: float = float(self._coeffs.get("frequency", 1.0))
        self._phase: float = float(self._coeffs.get("phase", 0.0))

        if self._func_name not in _TIME_FUNCTIONS:
            raise ValueError(
                f"Unknown time function '{self._func_name}'. "
                f"Available: {sorted(_TIME_FUNCTIONS)}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_base_value(self) -> torch.Tensor:
        """Parse the ``baseValue`` coefficient into a tensor."""
        raw = self._coeffs.get("baseValue", self._coeffs.get("value", 0.0))
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    def _time_factor(self, t: float) -> float:
        """Evaluate the time function at time *t*."""
        func = _TIME_FUNCTIONS[self._func_name]
        return func(t, self._amplitude, self._frequency, self._phase)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_value(self) -> torch.Tensor:
        """Return the base (time-independent) values."""
        return self._base_value

    @property
    def time_function(self) -> str:
        """Return the name of the time function."""
        return self._func_name

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        time: float = 0.0,
    ) -> torch.Tensor:
        """Set boundary-face values to ``base_value * time_function(t)``.

        Args:
            field: Full field tensor.
            patch_idx: Optional explicit start index.
            time: Current simulation time.
        """
        factor = self._time_factor(time)
        values = self._base_value.to(device=field.device, dtype=field.dtype) * factor

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
        """Penalty method with base value (implicit-safe).

        diag[c]   += deltaCoeff * area
        source[c] += deltaCoeff * area * base_value

        The time-dependent part is applied only through :meth:`apply`,
        keeping the implicit matrix well-conditioned.
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
        values = self._base_value.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * values)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
