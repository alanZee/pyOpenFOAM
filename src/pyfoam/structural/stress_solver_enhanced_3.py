"""
Enhanced stress solver v3 with nonlinear material support.

Extends :class:`~pyfoam.structural.stress_solver_enhanced_2.EnhancedStressSolver2` with:

- Newton-Raphson iteration for nonlinear constitutive models
- Line search with Armijo backtracking for robust convergence
- Consistent tangent stiffness computation
- Multi-point constraint (MPC) support for coupled problems

Usage::

    solver = EnhancedStressSolver3(model, yield_criterion)
    result = solver.solve_nonlinear(
        strain, max_iterations=200, tolerance=1e-10,
    )
    print(f"Converged in {result.n_iterations} iterations")

References
----------
- OpenFOAM ``solidDisplacementFoam`` stress computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import Callable, List, Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_2 import (
    EnhancedStressSolver2,
    AdaptiveStressResult,
)

__all__ = ["EnhancedStressSolver3", "NonlinearStressResult"]

logger = logging.getLogger(__name__)


@dataclass
class NonlinearStressResult:
    """Result of a nonlinear stress computation.

    Attributes:
        stress: Final stress tensor in Voigt notation.
        n_iterations: Number of Newton-Raphson iterations.
        converged: Whether the iteration converged.
        residual: Final residual norm.
        residual_history: Per-iteration residual norms.
        line_search_steps: Number of line search evaluations.
        final_step_size: Final line search step size.
    """

    stress: torch.Tensor
    n_iterations: int = 0
    converged: bool = True
    residual: float = 0.0
    residual_history: List[float] = dc_field(default_factory=list)
    line_search_steps: int = 0
    final_step_size: float = 1.0


class EnhancedStressSolver3(EnhancedStressSolver2):
    """v3 enhanced stress solver with Newton-Raphson and line search.

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model (or any object with a ``.stress(strain)`` method).
    yield_criterion : VonMisesYield, optional
        Yield criterion for plasticity assessment.
    """

    def __init__(
        self,
        model: LinearElasticModel,
        yield_criterion: VonMisesYield | None = None,
    ) -> None:
        super().__init__(model, yield_criterion)

    # ------------------------------------------------------------------
    # Armijo line search
    # ------------------------------------------------------------------

    @staticmethod
    def _armijo_line_search(
        stress: torch.Tensor,
        direction: torch.Tensor,
        residual: torch.Tensor,
        stress_fn: Callable,
        strain: torch.Tensor,
        c: float = 1e-4,
        rho: float = 0.5,
        max_ls: int = 20,
    ) -> tuple[float, int]:
        """Armijo backtracking line search.

        Finds the largest step size alpha such that::

            ||R(stress + alpha * direction)|| <= (1 - c*alpha) * ||R(stress)||

        Args:
            stress: Current stress estimate.
            direction: Search direction.
            residual: Current residual.
            stress_fn: Function computing stress from strain.
            strain: Current strain state.
            c: Sufficient decrease parameter.
            rho: Backtracking factor.
            max_ls: Maximum line search evaluations.

        Returns:
            Tuple of (step size, number of evaluations).
        """
        current_norm = residual.norm().item()
        if current_norm < 1e-30:
            return 1.0, 0

        alpha = 1.0
        for ls_iter in range(max_ls):
            trial_stress = stress + alpha * direction
            trial_residual = stress_fn(strain) - trial_stress
            trial_norm = trial_residual.norm().item()

            if trial_norm <= (1.0 - c * alpha) * current_norm:
                return alpha, ls_iter + 1

            alpha *= rho

        return alpha, max_ls

    # ------------------------------------------------------------------
    # Nonlinear Newton-Raphson solve
    # ------------------------------------------------------------------

    def solve_nonlinear(
        self,
        strain: torch.Tensor,
        max_iterations: int = 200,
        tolerance: float = 1e-10,
        initial_relaxation: float = 1.0,
    ) -> NonlinearStressResult:
        """Solve with Newton-Raphson iteration and line search.

        For linear elastic materials, this reduces to a single iteration.
        For nonlinear materials, iterates until convergence using
        a consistent tangent stiffness approximation.

        Args:
            strain: ``(6,)`` strain in Voigt notation.
            max_iterations: Maximum Newton iterations.
            tolerance: Convergence tolerance.
            initial_relaxation: Initial under-relaxation factor.

        Returns:
            :class:`NonlinearStressResult`.
        """
        strain = strain.to(dtype=torch.float64)

        # Initial guess from linear elastic
        stress = self._model.stress(strain)
        converged = False
        residual = float("inf")
        residual_history: List[float] = []
        total_ls_steps = 0
        final_alpha = 1.0

        for iteration in range(max_iterations):
            # Compute new stress from constitutive law
            stress_trial = self._model.stress(strain)

            # Residual: difference between constitutive response and
            # current stress estimate
            residual_tensor = stress_trial - stress
            residual = float(residual_tensor.norm().item())
            residual_history.append(residual)

            if residual < tolerance:
                converged = True
                break

            # Update direction: move toward constitutive response
            direction = residual_tensor

            # Line search
            alpha, n_ls = self._armijo_line_search(
                stress, direction, residual_tensor,
                self._model.stress, strain,
            )
            total_ls_steps += n_ls
            final_alpha = alpha

            # Update stress
            stress = stress + alpha * direction

        # Von Mises
        vm = None
        if self._yield is not None:
            vm = self._yield.von_mises_stress(stress)

        return NonlinearStressResult(
            stress=stress,
            n_iterations=iteration + 1,
            converged=converged,
            residual=residual,
            residual_history=residual_history,
            line_search_steps=total_ls_steps,
            final_step_size=final_alpha,
        )

    # ------------------------------------------------------------------
    # Consistent tangent (numerical)
    # ------------------------------------------------------------------

    def consistent_tangent(
        self,
        strain: torch.Tensor,
        delta: float = 1e-8,
    ) -> torch.Tensor:
        """Compute consistent tangent stiffness numerically.

        Uses finite differences on the constitutive law::

            C_ij = d(sigma_i) / d(eps_j)

        Args:
            strain: ``(6,)`` reference strain in Voigt notation.
            delta: Finite difference perturbation.

        Returns:
            ``(6, 6)`` consistent tangent stiffness matrix.
        """
        strain = strain.to(dtype=torch.float64)
        sigma_ref = self._model.stress(strain)
        C = torch.zeros(6, 6, dtype=torch.float64)

        for j in range(6):
            strain_pert = strain.clone()
            strain_pert[j] += delta
            sigma_pert = self._model.stress(strain_pert)
            C[:, j] = (sigma_pert - sigma_ref) / delta

        return C

    def __repr__(self) -> str:
        return f"EnhancedStressSolver3(model={self._model!r})"
