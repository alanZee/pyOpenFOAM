"""
Base linear solver infrastructure and solver selection.

Provides:

- :class:`LinearSolverBase` — abstract base for all iterative solvers
- :func:`create_solver` — factory function for solver selection from name
- :func:`solver_from_dict` — create solver from fvSolution dictionary

Solver names (case-insensitive):
- ``"PCG"`` — Preconditioned Conjugate Gradient (symmetric)
- ``"PBiCGStab"`` / ``"PBiCGSTAB"`` — Preconditioned BiCG Stabilised (asymmetric)
- ``"GAMG"`` — Algebraic Multigrid

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import assert_floating
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.residual import ConvergenceInfo, ResidualMonitor

__all__ = [
    "LinearSolverBase",
    "create_solver",
    "solver_from_dict",
]


class LinearSolverBase(ABC):
    """Abstract base class for iterative linear solvers.

    Subclasses implement :meth:`_solve` with the specific algorithm.
    The base class handles:
    - Input validation
    - Residual monitoring
    - Convergence reporting

    Parameters
    ----------
    tolerance : float
        Absolute convergence tolerance.
    rel_tol : float
        Relative convergence tolerance.
    max_iter : int
        Maximum number of iterations.
    min_iter : int
        Minimum iterations before declaring convergence.
    verbose : bool
        If True, log residuals at each iteration.
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        rel_tol: float = 0.01,
        max_iter: int = 1000,
        min_iter: int = 0,
        verbose: bool = False,
    ) -> None:
        self._tolerance = tolerance
        self._rel_tol = rel_tol
        self._max_iter = max_iter
        self._min_iter = min_iter
        self._verbose = verbose

    @property
    def tolerance(self) -> float:
        """Absolute convergence tolerance."""
        return self._tolerance

    @property
    def max_iter(self) -> int:
        """Maximum iterations."""
        return self._max_iter

    def __call__(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        tolerance: float | None = None,
        max_iter: int | None = None,
    ) -> tuple[torch.Tensor, int, float]:
        """Solve A x = b.

        This is the main entry point, matching the LinearSolver protocol
        expected by FvMatrix.solve().

        Args:
            matrix: The LDU matrix A.
            source: The right-hand side vector b.
            x0: Initial guess.
            tolerance: Override convergence tolerance.
            max_iter: Override maximum iterations.

        Returns:
            Tuple of ``(solution, iterations, final_residual)``.
        """
        assert_floating(source, "source")
        assert_floating(x0, "x0")

        tol = tolerance if tolerance is not None else self._tolerance
        max_it = max_iter if max_iter is not None else self._max_iter

        monitor = ResidualMonitor(
            tolerance=tol,
            rel_tol=self._rel_tol,
            min_iter=self._min_iter,
            verbose=self._verbose,
        )

        solution, info = self._solve(matrix, source, x0, monitor, max_it)

        return solution, info.iterations, info.final_residual

    @abstractmethod
    def _solve(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        monitor: ResidualMonitor,
        max_iter: int,
    ) -> tuple[torch.Tensor, ConvergenceInfo]:
        """Implement the specific solver algorithm.

        Args:
            matrix: The LDU matrix A.
            source: The right-hand side vector b.
            x0: Initial guess.
            monitor: Residual monitor for convergence tracking.
            max_iter: Maximum iterations.

        Returns:
            Tuple of ``(solution, convergence_info)``.
        """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tolerance={self._tolerance}, "
            f"rel_tol={self._rel_tol}, "
            f"max_iter={self._max_iter})"
        )


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

_SOLVER_REGISTRY: dict[str, type[LinearSolverBase]] = {}


def _register_solver(name: str, cls: type[LinearSolverBase]) -> None:
    """Register a solver class by name."""
    _SOLVER_REGISTRY[name.upper()] = cls
    # Also register common aliases
    if name.upper() == "PBICGSTAB":
        _SOLVER_REGISTRY["PBICGSTAB"] = cls
        _SOLVER_REGISTRY["PBICGSTAB"] = cls


def create_solver(
    name: str,
    *,
    tolerance: float = 1e-6,
    rel_tol: float = 0.01,
    max_iter: int = 1000,
    min_iter: int = 0,
    verbose: bool = False,
    **kwargs: Any,
) -> LinearSolverBase:
    """Create a linear solver by name.

    Args:
        name: Solver name (case-insensitive). One of:
            ``"PCG"``, ``"PBiCGStab"``, ``"GAMG"``.
        tolerance: Absolute convergence tolerance.
        rel_tol: Relative convergence tolerance.
        max_iter: Maximum iterations.
        min_iter: Minimum iterations before convergence check.
        verbose: If True, log residuals.
        **kwargs: Additional solver-specific parameters.

    Returns:
        Solver instance.

    Raises:
        ValueError: If solver name is not recognised.

    Examples::

        solver = create_solver("PCG", tolerance=1e-6, max_iter=1000)
        solution, iters, residual = solver(matrix, source, x0)
    """
    # Lazy import to avoid circular imports
    if not _SOLVER_REGISTRY:
        from pyfoam.solvers.pcg import PCGSolver
        from pyfoam.solvers.pbicgstab import PBiCGSTABSolver
        from pyfoam.solvers.gamg import GAMGSolver

        _register_solver("PCG", PCGSolver)
        _register_solver("PBICGSTAB", PBiCGSTABSolver)
        _register_solver("GAMG", GAMGSolver)

    key = name.upper().replace("-", "").replace("_", "")
    # Normalise PBiCGStab variants
    if key in ("PBICGSTAB", "PBICGStab".upper()):
        key = "PBICGSTAB"

    if key not in _SOLVER_REGISTRY:
        available = ", ".join(sorted(_SOLVER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown solver '{name}'. Available: {available}"
        )

    cls = _SOLVER_REGISTRY[key]
    return cls(
        tolerance=tolerance,
        rel_tol=rel_tol,
        max_iter=max_iter,
        min_iter=min_iter,
        verbose=verbose,
        **kwargs,
    )


def solver_from_dict(
    solver_name: str,
    solver_dict: dict[str, Any],
) -> LinearSolverBase:
    """Create a solver from an fvSolution sub-dictionary.

    Parses the standard OpenFOAM solver settings::

        solver          PCG;
        tolerance       1e-06;
        relTol          0.01;
        maxIter         1000;

    Args:
        solver_name: Solver algorithm name (e.g., ``"PCG"``).
        solver_dict: Dictionary of solver parameters.

    Returns:
        Configured solver instance.
    """
    return create_solver(
        solver_name,
        tolerance=float(solver_dict.get("tolerance", 1e-6)),
        rel_tol=float(solver_dict.get("relTol", 0.01)),
        max_iter=int(solver_dict.get("maxIter", 1000)),
    )
