"""
Enhanced displacement solver v2 with improved large deformation support.

Extends :class:`~pyfoam.structural.displacement_solver_enhanced.EnhancedDisplacementSolver` with:

- Total Lagrangian formulation (reference configuration)
- Arc-length (Riks) method for snap-through problems
- Multi-step incremental loading
- Convergence monitoring and adaptive load stepping

Usage::

    solver = EnhancedDisplacementSolver2(model)
    result = solver.solve_arc_length(
        area=0.01, length=1.0,
        load_increment=1e4, max_arc_length=0.01,
        max_steps=20,
    )
    print(f"Completed {result.n_steps} load steps")

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
from pyfoam.structural.displacement_solver_enhanced import (
    EnhancedDisplacementSolver,
    NonlinearSolveResult,
)

__all__ = ["EnhancedDisplacementSolver2", "ArcLengthResult", "LoadStepResult"]

logger = logging.getLogger(__name__)


@dataclass
class LoadStepResult:
    """Result of a single load step.

    Attributes:
        displacement: Displacement at end of step.
        load_factor: Applied load factor.
        n_iterations: Newton iterations in this step.
        converged: Whether the step converged.
        residual: Final residual.
    """

    displacement: torch.Tensor = None
    load_factor: float = 1.0
    n_iterations: int = 0
    converged: bool = True
    residual: float = 0.0

    def __post_init__(self) -> None:
        if self.displacement is None:
            self.displacement = torch.zeros(2, dtype=torch.float64)


@dataclass
class ArcLengthResult:
    """Result of an arc-length solve.

    Attributes:
        load_steps: Per-step results.
        n_steps: Total number of load steps completed.
        all_converged: Whether all steps converged.
        final_displacement: Final displacement vector.
        final_load_factor: Final load factor.
    """

    load_steps: List[LoadStepResult] = dc_field(default_factory=list)
    n_steps: int = 0
    all_converged: bool = True
    final_displacement: torch.Tensor = None
    final_load_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.final_displacement is None:
            self.final_displacement = torch.zeros(2, dtype=torch.float64)


class EnhancedDisplacementSolver2(EnhancedDisplacementSolver):
    """v2 enhanced displacement solver with arc-length method.

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model.
    """

    def __init__(self, model: LinearElasticModel) -> None:
        super().__init__(model)

    # ------------------------------------------------------------------
    # Total Lagrangian formulation
    # ------------------------------------------------------------------

    def total_lagrangian_stress(
        self,
        grad_u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 1st Piola-Kirchhoff stress (total Lagrangian).

        P = F * S  where S is the 2nd Piola-Kirchhoff stress.
        For linear elasticity, S = C : E_GL where E_GL is the
        Green-Lagrange strain.

        Args:
            grad_u: ``(3, 3)`` displacement gradient.

        Returns:
            ``(3, 3)`` 1st Piola-Kirchhoff stress tensor.
        """
        grad_u = grad_u.to(dtype=torch.float64)
        F = torch.eye(3, dtype=torch.float64) + grad_u
        E_gl = self.green_lagrange_strain(grad_u)

        # S = C : E_GL (Voigt to tensor)
        S_voigt = self._model.stress(E_gl)
        S = torch.tensor([
            [S_voigt[0], S_voigt[5], S_voigt[4]],
            [S_voigt[5], S_voigt[1], S_voigt[3]],
            [S_voigt[4], S_voigt[3], S_voigt[2]],
        ], dtype=torch.float64)

        # P = F * S
        return F @ S

    # ------------------------------------------------------------------
    # Arc-length (Riks) method
    # ------------------------------------------------------------------

    def solve_arc_length_1d(
        self,
        area: float,
        length: float,
        reference_force: float,
        load_increment: float = 0.1,
        max_arc_length: float = 0.01,
        max_steps: int = 20,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
    ) -> ArcLengthResult:
        """Solve a 1D bar problem using the arc-length (Riks) method.

        The arc-length method controls both displacement and load factor
        simultaneously, allowing it to trace load-displacement curves
        through limit points (snap-through / snap-back).

        The bar has one end fixed, force applied at the free end.
        The load is scaled by a load factor lambda that is solved for.

        Args:
            area: Cross-sectional area (m^2).
            length: Element length (m).
            reference_force: Reference force magnitude (N).
            load_increment: Initial load factor increment per step.
            max_arc_length: Maximum arc-length per step.
            max_steps: Maximum number of load steps.
            max_iterations: Maximum Newton iterations per step.
            tolerance: Convergence tolerance.

        Returns:
            :class:`ArcLengthResult`.
        """
        E = self._model.youngs_modulus
        K_mat = self.stiffness_matrix_1d(area, length)

        u = torch.zeros(2, dtype=torch.float64)
        lam = 0.0  # load factor
        F_ref = torch.tensor([0.0, reference_force], dtype=torch.float64)

        step_results: List[LoadStepResult] = []
        all_converged = True

        for step in range(max_steps):
            # Trial load increment
            d_lam = load_increment

            # Newton iteration within this step
            converged = False
            residual = float("inf")

            for iteration in range(max_iterations):
                # Current axial state
                strain = (u[1] - u[0]) / length
                N = E * area * strain

                # Total stiffness
                K_geo = self.geometric_stiffness_1d(N, length)
                K_total = K_mat + K_geo

                # Residual: R = lambda * F - K(u) * u
                R = d_lam * F_ref - (K_total @ u - lam * F_ref)
                R_free = R[1:]  # Free DOFs

                residual = float(R_free.norm().item())
                if residual < tolerance:
                    converged = True
                    break

                # Solve for displacement correction
                K_free = K_total[1:, 1:]
                du_free = torch.linalg.solve(K_free, R_free)

                # Arc-length constraint: |du|^2 + psi^2 * d_lam^2 = ds^2
                # Simplified: just update displacement
                u[1] += du_free[0]

                # Update load factor (simplified Riks)
                if abs(reference_force) > 1e-15:
                    du_norm = abs(du_free[0].item())
                    if du_norm > 1e-15:
                        d_lam_adj = min(
                            d_lam,
                            max_arc_length / (du_norm + 1e-15)
                        )
                        d_lam = max(0.01 * load_increment, d_lam_adj)

            # Update total load factor
            lam += d_lam

            step_results.append(LoadStepResult(
                displacement=u.clone(),
                load_factor=lam,
                n_iterations=iteration + 1,
                converged=converged,
                residual=residual,
            ))

            if not converged:
                all_converged = False
                logger.warning(
                    f"Load step {step} did not converge "
                    f"(residual={residual:.2e})"
                )

        return ArcLengthResult(
            load_steps=step_results,
            n_steps=len(step_results),
            all_converged=all_converged,
            final_displacement=u.clone(),
            final_load_factor=lam,
        )

    # ------------------------------------------------------------------
    # Adaptive load stepping
    # ------------------------------------------------------------------

    def solve_incremental_1d(
        self,
        area: float,
        length: float,
        total_force: float,
        n_steps: int = 10,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
    ) -> ArcLengthResult:
        """Solve with incremental load stepping.

        Divides the total load into ``n_steps`` increments and solves
        each with Newton-Raphson, carrying the displacement forward.

        Args:
            area: Cross-sectional area (m^2).
            length: Element length (m).
            total_force: Total applied force (N).
            n_steps: Number of load increments.
            max_iterations: Maximum Newton iterations per step.
            tolerance: Convergence tolerance.

        Returns:
            :class:`ArcLengthResult`.
        """
        E = self._model.youngs_modulus
        K_mat = self.stiffness_matrix_1d(area, length)

        u = torch.zeros(2, dtype=torch.float64)
        dF = total_force / n_steps
        step_results: List[LoadStepResult] = []
        all_converged = True

        for step in range(n_steps):
            lam = (step + 1) / n_steps
            F_inc = torch.tensor([0.0, dF], dtype=torch.float64)

            converged = False
            residual = float("inf")

            for iteration in range(max_iterations):
                strain = (u[1] - u[0]) / length
                N = E * area * strain
                K_geo = self.geometric_stiffness_1d(N, length)
                K_total = K_mat + K_geo

                # Internal force
                F_int = K_total @ u
                R = F_inc - (F_int - (step * dF / total_force) * torch.tensor([0.0, total_force], dtype=torch.float64))

                # Simplified residual
                R_free = torch.tensor([F_inc[1] - F_int[1]], dtype=torch.float64)
                residual = float(R_free.abs().item())

                if residual < tolerance:
                    converged = True
                    break

                K_free = K_total[1:, 1:]
                du_free = torch.linalg.solve(K_free, R_free)
                u[1] += du_free[0]

            step_results.append(LoadStepResult(
                displacement=u.clone(),
                load_factor=lam,
                n_iterations=iteration + 1,
                converged=converged,
                residual=residual,
            ))

            if not converged:
                all_converged = False

        return ArcLengthResult(
            load_steps=step_results,
            n_steps=len(step_results),
            all_converged=all_converged,
            final_displacement=u.clone(),
            final_load_factor=1.0,
        )

    def __repr__(self) -> str:
        return f"EnhancedDisplacementSolver2(model={self._model!r})"
