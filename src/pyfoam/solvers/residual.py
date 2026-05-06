"""
Residual monitoring for iterative linear solvers.

Provides a :class:`ResidualMonitor` that tracks convergence history, computes
relative residuals, and determines when the solver has converged.

Supports two convergence criteria (matching OpenFOAM):
- **Absolute**: |r| < tolerance
- **Relative**: |r| / |r₀| < relTol  (or |r| / |b| < relTol)

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

__all__ = ["ResidualMonitor", "ConvergenceInfo"]

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceInfo:
    """Result of a linear solve.

    Attributes
    ----------
    converged : bool
        Whether the solver converged within the iteration limit.
    iterations : int
        Number of iterations performed.
    final_residual : float
        Final absolute residual |r|.
    initial_residual : float
        Initial absolute residual |r₀|.
    residual_ratio : float
        Final / initial residual ratio.
    tolerance : float
        Requested tolerance.
    residual_history : list[float]
        Absolute residual at each iteration.
    """

    converged: bool
    iterations: int
    final_residual: float
    initial_residual: float
    residual_ratio: float
    tolerance: float
    residual_history: list[float] = field(default_factory=list)


class ResidualMonitor:
    """Tracks residual convergence for iterative solvers.

    Parameters
    ----------
    tolerance : float
        Absolute convergence tolerance.
    rel_tol : float
        Relative convergence tolerance (|r|/|r₀| < rel_tol).
    min_iter : int
        Minimum iterations before declaring convergence.
    verbose : bool
        If True, log residuals at each iteration.
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        rel_tol: float = 0.01,
        min_iter: int = 0,
        verbose: bool = False,
    ) -> None:
        self._tolerance = tolerance
        self._rel_tol = rel_tol
        self._min_iter = min_iter
        self._verbose = verbose
        self._history: list[float] = []
        self._initial_residual: float | None = None

    @property
    def tolerance(self) -> float:
        """Absolute convergence tolerance."""
        return self._tolerance

    @property
    def rel_tol(self) -> float:
        """Relative convergence tolerance."""
        return self._rel_tol

    @property
    def history(self) -> list[float]:
        """Residual history."""
        return self._history

    @property
    def initial_residual(self) -> float | None:
        """Initial residual (set on first update)."""
        return self._initial_residual

    def reset(self) -> None:
        """Reset the monitor for a new solve."""
        self._history.clear()
        self._initial_residual = None

    def update(self, residual: torch.Tensor, iteration: int) -> bool:
        """Update with a new residual and check convergence.

        Args:
            residual: ``(n_cells,)`` residual vector r.
            iteration: Current iteration number (0-based).

        Returns:
            True if converged.
        """
        res_norm = float(torch.norm(residual).item())
        self._history.append(res_norm)

        if self._initial_residual is None:
            self._initial_residual = res_norm

        if self._verbose:
            ratio = res_norm / self._initial_residual if self._initial_residual > 0 else 0.0
            logger.info(
                "  Iteration %d: residual = %.6e (ratio = %.6e)",
                iteration, res_norm, ratio,
            )

        # Check convergence (only after minimum iterations)
        if iteration >= self._min_iter:
            # Absolute convergence
            if res_norm < self._tolerance:
                return True
            # Relative convergence
            if self._initial_residual > 0:
                if res_norm / self._initial_residual < self._rel_tol:
                    return True

        return False

    def build_info(self, converged: bool) -> ConvergenceInfo:
        """Build a ConvergenceInfo from the current state.

        Args:
            converged: Whether the solver converged.

        Returns:
            ConvergenceInfo with solve statistics.
        """
        initial = self._initial_residual or 0.0
        final = self._history[-1] if self._history else 0.0
        ratio = final / initial if initial > 0 else 0.0

        return ConvergenceInfo(
            converged=converged,
            iterations=len(self._history),
            final_residual=final,
            initial_residual=initial,
            residual_ratio=ratio,
            tolerance=self._tolerance,
            residual_history=list(self._history),
        )
