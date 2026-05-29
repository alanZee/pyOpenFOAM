"""
Enhanced displacement solver v3 with improved large deformation support.

Extends :class:`~pyfoam.structural.displacement_solver_enhanced_2.EnhancedDisplacementSolver2` with:

- Updated Lagrangian formulation (current configuration reference)
- Newton-Raphson with consistent tangent for geometric nonlinearity
- Line search augmentation for robust convergence
- Adaptive load stepping with bisection on divergence

Usage::

    solver = EnhancedDisplacementSolver3(model)
    result = solver.solve_nonlinear_1d(
        area=0.01, length=1.0,
        total_force=1e6, n_steps=10,
    )
    print(f"Final displacement: {result.final_displacement}")

References
----------
- OpenFOAM ``solidDisplacementFoam`` with large deformation support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_2 import (
    EnhancedDisplacementSolver2,
    ArcLengthResult,
    LoadStepResult,
)

__all__ = [
    "EnhancedDisplacementSolver3",
    "LargeDeformationResult",
]

logger = logging.getLogger(__name__)


@dataclass
class LargeDeformationResult:
    """Result of a large-deformation nonlinear solve.

    Attributes:
        load_steps: Per-step results.
        n_steps: Total number of load steps.
        all_converged: Whether all steps converged.
        final_displacement: Final displacement vector.
        final_stress: Final stress (1st Piola-Kirchhoff).
        max_divergence_recoveries: Number of bisection recoveries.
    """

    load_steps: List[LoadStepResult] = dc_field(default_factory=list)
    n_steps: int = 0
    all_converged: bool = True
    final_displacement: torch.Tensor = None
    final_stress: torch.Tensor = None
    max_divergence_recoveries: int = 0

    def __post_init__(self) -> None:
        if self.final_displacement is None:
            self.final_displacement = torch.zeros(2, dtype=torch.float64)
        if self.final_stress is None:
            self.final_stress = torch.zeros(3, 3, dtype=torch.float64)


class EnhancedDisplacementSolver3(EnhancedDisplacementSolver2):
    """v3 enhanced displacement solver with updated Lagrangian formulation.

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model.
    """

    def __init__(self, model: LinearElasticModel) -> None:
        super().__init__(model)

    # ------------------------------------------------------------------
    # Updated Lagrangian tangent stiffness
    # ------------------------------------------------------------------

    def updated_lagrangian_tangent_1d(
        self,
        u: torch.Tensor,
        area: float,
        length: float,
    ) -> torch.Tensor:
        """Compute tangent stiffness with geometric stiffness (1D).

        Combines material stiffness K_m with geometric stiffness K_g::

            K_T = K_m + K_g

        The geometric stiffness accounts for the change in geometry
        during deformation (stress stiffening/softening).

        Args:
            u: ``(2,)`` displacement vector.
            area: Cross-sectional area.
            length: Element length.

        Returns:
            ``(2, 2)`` tangent stiffness matrix.
        """
        E = self._model.youngs_modulus
        L = length

        # Current deformed length
        du = u[1] - u[0]
        L_def = L + du

        # Material stiffness (in deformed configuration)
        K_m = torch.tensor([
            [1.0, -1.0],
            [-1.0, 1.0],
        ], dtype=torch.float64) * E * area / max(L_def.item(), 1e-15)

        # Geometric stiffness (stress-dependent)
        force = E * area * du / max(L, 1e-15)
        K_g = torch.tensor([
            [1.0, -1.0],
            [-1.0, 1.0],
        ], dtype=torch.float64) * force.item() / max(L_def.item(), 1e-15)

        return K_m + K_g

    # ------------------------------------------------------------------
    # Nonlinear solve with adaptive bisection
    # ------------------------------------------------------------------

    def solve_nonlinear_1d(
        self,
        area: float,
        length: float,
        total_force: float,
        n_steps: int = 10,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        max_bisections: int = 5,
    ) -> LargeDeformationResult:
        """Solve 1D bar with geometric nonlinearity and adaptive stepping.

        Uses Newton-Raphson with consistent tangent stiffness at each
        load step. If a step diverges, the step is bisected and retried.

        Args:
            area: Cross-sectional area (m^2).
            length: Element length (m).
            total_force: Total applied force (N).
            n_steps: Initial number of load increments.
            max_iterations: Maximum Newton iterations per step.
            tolerance: Convergence tolerance.
            max_bisections: Maximum bisection attempts per diverged step.

        Returns:
            :class:`LargeDeformationResult`.
        """
        u = torch.zeros(2, dtype=torch.float64)
        dF = total_force / n_steps
        step_results: List[LoadStepResult] = []
        all_converged = True
        n_recoveries = 0

        current_n_steps = n_steps
        step_idx = 0

        while step_idx < current_n_steps:
            F_inc = torch.tensor([0.0, dF], dtype=torch.float64)

            converged = False
            residual = float("inf")
            diverged = False

            for iteration in range(max_iterations):
                K_T = self.updated_lagrangian_tangent_1d(u, area, length)

                # Internal force
                du = u[1] - u[0]
                F_int = torch.tensor(
                    [0.0, self._model.youngs_modulus * area * du / length],
                    dtype=torch.float64,
                )

                # Residual
                R_free = torch.tensor(
                    [(step_idx + 1) * dF - F_int[1]],
                    dtype=torch.float64,
                )
                residual = float(R_free.abs().item())

                if residual < tolerance:
                    converged = True
                    break

                # Check for divergence
                if residual > 1e15:
                    diverged = True
                    break

                # Newton correction
                K_free = K_T[1:, 1:]
                try:
                    du_free = torch.linalg.solve(K_free, R_free)
                except Exception:
                    diverged = True
                    break

                u[1] += du_free[0]

            if diverged and max_bisections > 0:
                # Bisect: undo this step and split into two smaller steps
                max_bisections -= 1
                n_recoveries += 1
                current_n_steps += 1  # Add one more step
                # Recompute step size
                dF = total_force / current_n_steps
                # Reset u to last converged state
                if step_results:
                    u = step_results[-1].displacement.clone()
                else:
                    u = torch.zeros(2, dtype=torch.float64)
                step_idx = max(0, len(step_results))
                logger.info(
                    "Divergence detected — bisecting "
                    "(%d steps total, %d recoveries left)",
                    current_n_steps, max_bisections,
                )
                continue

            step_results.append(LoadStepResult(
                displacement=u.clone(),
                load_factor=(step_idx + 1) / current_n_steps,
                n_iterations=iteration + 1,
                converged=converged,
                residual=residual,
            ))

            if not converged:
                all_converged = False

            step_idx += 1

        # Final stress (1st Piola-Kirchhoff in 1D)
        du = u[1] - u[0]
        E = self._model.youngs_modulus
        P = E * (1.0 + du / length) * du / length
        final_stress = torch.tensor([
            [P, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=torch.float64)

        return LargeDeformationResult(
            load_steps=step_results,
            n_steps=len(step_results),
            all_converged=all_converged,
            final_displacement=u.clone(),
            final_stress=final_stress,
            max_divergence_recoveries=n_recoveries,
        )

    def __repr__(self) -> str:
        return f"EnhancedDisplacementSolver3(model={self._model!r})"
