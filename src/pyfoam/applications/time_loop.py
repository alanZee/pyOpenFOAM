"""
Time-stepping loop for OpenFOAM-style solvers.

Provides a :class:`TimeLoop` generator that yields ``(time, step)``
pairs and handles ``writeControl`` / ``writeInterval`` semantics.

Supports two ``writeControl`` modes (matching OpenFOAM):

- ``"timeStep"`` — write every *N* steps.
- ``"runTime"``  — write every *T* seconds of simulation time.

Usage::

    loop = TimeLoop(start_time=0, end_time=100, delta_t=0.001,
                    write_interval=10, write_control="timeStep")
    for t, step in loop:
        solve(...)
        if loop.should_write():
            write_fields(t)
"""

from __future__ import annotations

import logging
import math
from typing import Iterator

__all__ = ["TimeLoop"]

logger = logging.getLogger(__name__)


class TimeLoop:
    """Iterates through simulation time steps.

    Parameters
    ----------
    start_time : float
        Initial simulation time.
    end_time : float
        Final simulation time.
    delta_t : float
        Time-step size.
    write_interval : float
        Write interval (steps or seconds depending on *write_control*).
    write_control : str
        ``"timeStep"`` or ``"runTime"``.
    """

    def __init__(
        self,
        start_time: float = 0.0,
        end_time: float = 100.0,
        delta_t: float = 1.0,
        write_interval: float = 1.0,
        write_control: str = "timeStep",
    ) -> None:
        self._start = start_time
        self._end = end_time
        self._dt = delta_t
        self._write_interval = write_interval
        self._write_control = write_control

        # Mutable state
        self._time = start_time
        self._step = 0
        self._last_write_step = -1  # step at which we last wrote
        self._last_write_time = start_time - write_interval  # ensure first step writes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time

    @property
    def step(self) -> int:
        """Current step number (0-based)."""
        return self._step

    @property
    def delta_t(self) -> float:
        """Time-step size."""
        return self._dt

    @property
    def end_time(self) -> float:
        """End time."""
        return self._end

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[float, int]]:
        """Yield ``(time, step)`` for each time step.

        The loop runs from ``start_time`` (inclusive) to ``end_time``
        (exclusive) in increments of ``delta_t``.
        """
        self._time = self._start
        self._step = 0

        while self._time < self._end - 1e-12 * self._dt:
            yield self._time, self._step
            self._advance()

    def _advance(self) -> None:
        """Advance one time step."""
        self._step += 1
        self._time = self._start + self._step * self._dt

        # Clamp to end_time
        if self._time > self._end:
            self._time = self._end

    # ------------------------------------------------------------------
    # Write control
    # ------------------------------------------------------------------

    def should_write(self) -> bool:
        """Return ``True`` if fields should be written at the current step.

        Honours ``writeControl`` and ``writeInterval``.
        """
        if self._write_control == "timeStep":
            interval = max(1, int(round(self._write_interval)))
            if self._step % interval == 0 and self._step != self._last_write_step:
                return True
        elif self._write_control == "runTime":
            if self._time - self._last_write_time >= self._write_interval - 1e-12:
                return True
        return False

    def mark_written(self) -> None:
        """Mark that fields were written at the current step/time."""
        self._last_write_step = self._step
        self._last_write_time = self._time

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TimeLoop(start={self._start}, end={self._end}, "
            f"dt={self._dt}, step={self._step}, time={self._time:.6g})"
        )
