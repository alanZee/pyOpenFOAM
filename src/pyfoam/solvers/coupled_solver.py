"""
Base class for pressure-velocity coupled solvers (SIMPLE, PISO, PIMPLE).

Provides common infrastructure for incompressible flow solvers that couple
the momentum and continuity equations via a pressure-correction step.

The coupled solver manages:
- Field references (velocity U, pressure p, face flux phi)
- Linear solver selection for momentum and pressure equations
- Under-relaxation parameters
- Convergence monitoring (residuals for each field)
- Time step control

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import assert_floating
from pyfoam.solvers.linear_solver import LinearSolverBase, create_solver

__all__ = [
    "CoupledSolverConfig",
    "CoupledSolverBase",
    "ConvergenceData",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CoupledSolverConfig:
    """Configuration for pressure-velocity coupled solvers.

    Attributes
    ----------
    p_solver : str
        Linear solver name for the pressure equation.
    U_solver : str
        Linear solver name for the momentum equation.
    p_tolerance : float
        Convergence tolerance for the pressure equation.
    U_tolerance : float
        Convergence tolerance for the momentum equation.
    p_max_iter : int
        Maximum iterations for the pressure linear solver.
    U_max_iter : int
        Maximum iterations for the momentum linear solver.
    n_non_orthogonal_correctors : int
        Number of non-orthogonal correction loops (0 for orthogonal meshes).
    relaxation_factor_p : float
        Under-relaxation factor for pressure (0 < α ≤ 1).
    relaxation_factor_U : float
        Under-relaxation factor for velocity (0 < α ≤ 1).
    relaxation_factor_phi : float
        Under-relaxation factor for face flux (0 < α ≤ 1).
    """

    p_solver: str = "PCG"
    U_solver: str = "PBiCGStab"
    p_tolerance: float = 1e-6
    U_tolerance: float = 1e-6
    p_max_iter: int = 1000
    U_max_iter: int = 1000
    n_non_orthogonal_correctors: int = 0
    relaxation_factor_p: float = 1.0
    relaxation_factor_U: float = 0.7
    relaxation_factor_phi: float = 1.0


# ---------------------------------------------------------------------------
# Convergence tracking
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceData:
    """Tracks convergence data for a coupled solve.

    Attributes
    ----------
    p_residual : float
        Final pressure equation residual.
    U_residual : float
        Final momentum equation residual (max component).
    continuity_error : float
        Global continuity error (sum of flux imbalance).
    outer_iterations : int
        Number of outer (SIMPLE/PIMPLE) iterations.
    converged : bool
        Whether the solution converged within tolerance.
    residual_history : list[dict]
        Per-iteration residual records.
    """

    p_residual: float = 0.0
    U_residual: float = 0.0
    continuity_error: float = 0.0
    outer_iterations: int = 0
    converged: bool = False
    residual_history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Coupled solver base
# ---------------------------------------------------------------------------


class CoupledSolverBase(ABC):
    """Abstract base class for pressure-velocity coupled solvers.

    Provides the common framework for SIMPLE, PISO, and PIMPLE algorithms.
    Subclasses implement :meth:`solve` with the specific algorithm.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh (must have ``n_cells``, ``owner``,
        ``neighbour``, ``n_internal_faces``, ``cell_volumes``,
        ``face_areas``, ``delta_coefficients`` attributes).
    config : CoupledSolverConfig
        Solver configuration.
    """

    def __init__(
        self,
        mesh: Any,
        config: CoupledSolverConfig | None = None,
    ) -> None:
        self._mesh = mesh
        self._config = config or CoupledSolverConfig()

        # Create linear solvers
        self._p_solver = create_solver(
            self._config.p_solver,
            tolerance=self._config.p_tolerance,
            max_iter=self._config.p_max_iter,
        )
        self._U_solver = create_solver(
            self._config.U_solver,
            tolerance=self._config.U_tolerance,
            max_iter=self._config.U_max_iter,
        )

        self._device = mesh.device
        self._dtype = mesh.dtype

    @property
    def mesh(self) -> Any:
        """The finite volume mesh."""
        return self._mesh

    @property
    def config(self) -> CoupledSolverConfig:
        """Solver configuration."""
        return self._config

    @property
    def p_solver(self) -> LinearSolverBase:
        """Pressure linear solver."""
        return self._p_solver

    @property
    def U_solver(self) -> LinearSolverBase:
        """Momentum linear solver."""
        return self._U_solver

    @abstractmethod
    def solve(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        *,
        U_old: torch.Tensor | None = None,
        p_old: torch.Tensor | None = None,
        max_outer_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run the coupled pressure-velocity algorithm.

        Args:
            U: ``(n_cells, 3)`` velocity field.
            p: ``(n_cells,)`` pressure field.
            phi: ``(n_faces,)`` face flux field.
            U_old: Previous time-step velocity (for time derivative).
            p_old: Previous time-step pressure (for time derivative).
            max_outer_iterations: Maximum outer-loop iterations.
            tolerance: Convergence tolerance on residuals.

        Returns:
            Tuple of ``(U, p, phi, convergence_data)``.
        """

    def _compute_residual(
        self,
        field: torch.Tensor,
        field_old: torch.Tensor,
    ) -> float:
        """Compute the L2 norm of the field change.

        Args:
            field: Current field values.
            field_old: Previous iteration field values.

        Returns:
            L2 norm of the difference, normalised by field magnitude.
        """
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"p_solver={self._config.p_solver}, "
            f"U_solver={self._config.U_solver}, "
            f"relax_U={self._config.relaxation_factor_U:.2f}, "
            f"relax_p={self._config.relaxation_factor_p:.2f})"
        )
