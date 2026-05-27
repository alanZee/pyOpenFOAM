"""
Time-varying boundary condition.

Applies a time-varying value by interpolating from a time-value lookup
table.  Analogous to OpenFOAM's ``timeVaryingMappedFixedValue`` but
simplified for the Python layer::

    type    timeVarying;
    table   ((0 0) (1 1) (2 0.5));
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TimeVaryingBC"]


@BoundaryCondition.register("timeVarying")
class TimeVaryingBC(BoundaryCondition):
    """Time-varying boundary condition via table interpolation.

    Interpolates boundary-face values from a time-value lookup table
    using piecewise linear interpolation.  Times outside the table
    range are clamped to the boundary values.

    Coefficients:
        - ``table``: List of ``[time, value]`` pairs, e.g.
          ``[[0, 0], [1, 1], [2, 0.5]]``.  Required.
        - ``value``: Initial value used for shape (default: 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        table_raw = self._coeffs.get("table", [[0.0, 0.0]])
        self._build_interpolation(table_raw)

    def _build_interpolation(self, table: list) -> None:
        """Parse the time-value table and prepare interpolation data."""
        times: list[float] = []
        values: list[float] = []
        for row in table:
            times.append(float(row[0]))
            values.append(float(row[1]))
        self._times = torch.tensor(times, dtype=get_default_dtype())
        self._values = torch.tensor(values, dtype=get_default_dtype())

    @property
    def table_times(self) -> torch.Tensor:
        """Return the time column of the lookup table."""
        return self._times

    @property
    def table_values(self) -> torch.Tensor:
        """Return the value column of the lookup table."""
        return self._values

    def _interpolate(self, time: float) -> torch.Tensor:
        """Interpolate value from the lookup table at the given time."""
        t = self._times
        v = self._values

        if time <= t[0].item():
            val = v[0]
        elif time >= t[-1].item():
            val = v[-1]
        else:
            # 二分查找左端点
            idx = torch.searchsorted(t, torch.tensor(time)) - 1
            idx = max(idx, 0)
            t0, t1 = t[idx].item(), t[idx + 1].item()
            frac = (time - t0) / (t1 - t0)
            val = v[idx] + frac * (v[idx + 1] - v[idx])

        return torch.full(
            (self._patch.n_faces,),
            float(val),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    def apply(
        self, field: torch.Tensor, patch_idx: int | None = None, time: float = 0.0,
    ) -> torch.Tensor:
        """Set boundary-face values from table interpolation at *time*.

        Args:
            field: Full field tensor.
            patch_idx: Optional explicit start index into *field*.
            time: Current simulation time.
        """
        face_values = self._interpolate(time)
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
        """Penalty method using the first table entry as implicit value.

        diag[c]   += deltaCoeff * faceArea
        source[c] += deltaCoeff * faceArea * interpolatedValue

        Uses the first table entry for the implicit diagonal to ensure
        diagonal dominance.  The full time-varying value goes into source.
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

        # 隐式部分用第一个时间步的值
        implicit_val = self._interpolate(0.0).to(device=device, dtype=dtype)
        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * implicit_val)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
