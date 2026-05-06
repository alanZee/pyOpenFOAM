"""
Convergence monitoring for application-level solvers.

Provides a :class:`ConvergenceMonitor` that tracks field residuals across
time steps (or outer iterations) and determines when the simulation has
converged.

Two convergence modes (matching OpenFOAM):

- **Per-field**: Each field's residual drops below its tolerance.
- **Global**: A combined metric (e.g. continuity error) drops below tolerance.

Usage::

    monitor = ConvergenceMonitor(tolerance=1e-4)
    for t in time_loop:
        ...
        converged = monitor.update(step, {"U": u_res, "p": p_res, "cont": cont_err})
        if converged:
            break
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ConvergenceMonitor", "ConvergenceRecord"]

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceRecord:
    """A single convergence snapshot.

    Attributes
    ----------
    step : int
        Iteration or time-step number.
    residuals : dict[str, float]
        Named residual values (e.g. ``{"U": 1e-3, "p": 1e-4, "cont": 1e-5}``).
    """

    step: int
    residuals: dict[str, float] = field(default_factory=dict)


class ConvergenceMonitor:
    """Monitors convergence across iterations or time steps.

    Parameters
    ----------
    tolerance : float
        Absolute convergence tolerance.  A step is considered converged when
        **all** tracked residuals drop below this value.
    min_steps : int
        Minimum number of steps before convergence can be declared.
    residual_names : list[str], optional
        Names of residuals to track.  If ``None``, any key passed to
        :meth:`update` is tracked.
    """

    def __init__(
        self,
        tolerance: float = 1e-4,
        min_steps: int = 1,
        residual_names: list[str] | None = None,
    ) -> None:
        self._tolerance = tolerance
        self._min_steps = min_steps
        self._names = residual_names
        self._history: list[ConvergenceRecord] = []
        self._initial: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tolerance(self) -> float:
        """Absolute convergence tolerance."""
        return self._tolerance

    @property
    def history(self) -> list[ConvergenceRecord]:
        """Full convergence history."""
        return list(self._history)

    @property
    def n_steps(self) -> int:
        """Number of recorded steps."""
        return len(self._history)

    @property
    def initial_residuals(self) -> dict[str, float] | None:
        """Residuals from the first step (for relative convergence)."""
        return self._initial

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, step: int, residuals: dict[str, float]) -> bool:
        """Record residuals for *step* and check convergence.

        Convergence is declared when:
        1. ``step >= min_steps``, AND
        2. **every** residual value is below ``tolerance``.

        Args:
            step: Current iteration / time-step number.
            residuals: ``{name: value}`` residual map.

        Returns:
            ``True`` if converged.
        """
        record = ConvergenceRecord(step=step, residuals=dict(residuals))
        self._history.append(record)

        if self._initial is None:
            self._initial = dict(residuals)

        # Log every 10 steps or the first 5
        n = len(self._history)
        if n <= 5 or n % 10 == 0:
            parts = ", ".join(f"{k}={v:.6e}" for k, v in residuals.items())
            logger.info("Step %d: %s", step, parts)

        # Convergence check
        if len(self._history) < self._min_steps:
            return False

        names = self._names or list(residuals.keys())
        for name in names:
            val = residuals.get(name)
            if val is None:
                continue
            if val >= self._tolerance:
                return False

        logger.info(
            "Converged at step %d after %d steps",
            step, len(self._history),
        )
        return True

    def reset(self) -> None:
        """Reset the monitor for a new run."""
        self._history.clear()
        self._initial = None

    def get_residual_series(self, name: str) -> list[float]:
        """Extract a single residual's history as a list.

        Args:
            name: Residual name (e.g. ``"U"``).

        Returns:
            List of residual values (one per recorded step).
        """
        return [
            r.residuals.get(name, 0.0) for r in self._history
        ]

    def __repr__(self) -> str:
        return (
            f"ConvergenceMonitor(tolerance={self._tolerance}, "
            f"min_steps={self._min_steps}, n_steps={len(self._history)})"
        )
