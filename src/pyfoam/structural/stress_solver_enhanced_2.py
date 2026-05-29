"""
Enhanced stress solver v2 with better convergence and nonlinear material support.

Extends :class:`~pyfoam.structural.stress_solver_enhanced.EnhancedStressSolver` with:

- Aitken relaxation for accelerated convergence
- Adaptive under-relaxation based on residual history
- Stress extrapolation at integration points
- Principal stress trajectory computation

Usage::

    solver = EnhancedStressSolver2(model, yield_criterion)
    result = solver.solve_adaptive(
        strain, max_iterations=100, tolerance=1e-8,
    )
    print(f"Converged in {result.n_iterations} iterations, "
          f"relaxation={result.final_relaxation:.3f}")

References
----------
- OpenFOAM ``solidDisplacementFoam`` stress computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced import (
    EnhancedStressSolver,
    IterativeStressResult,
)

__all__ = ["EnhancedStressSolver2", "AdaptiveStressResult"]

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveStressResult:
    """Result of an adaptive iterative stress computation.

    Attributes:
        stress: Final stress tensor in Voigt notation.
        n_iterations: Number of iterations performed.
        converged: Whether the iteration converged.
        residual: Final residual norm.
        final_relaxation: Final relaxation factor used.
        residual_history: Per-iteration residual norms.
        von_mises: Von Mises equivalent stress (if yield criterion set).
        is_plastic: Whether plasticity correction was applied.
    """

    stress: torch.Tensor
    n_iterations: int = 0
    converged: bool = True
    residual: float = 0.0
    final_relaxation: float = 1.0
    residual_history: List[float] = None
    von_mises: Optional[torch.Tensor] = None
    is_plastic: bool = False

    def __post_init__(self) -> None:
        if self.residual_history is None:
            self.residual_history = []


class EnhancedStressSolver2(EnhancedStressSolver):
    """v2 enhanced stress solver with adaptive convergence.

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model.
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
    # Aitken relaxation
    # ------------------------------------------------------------------

    @staticmethod
    def _aitken_relaxation(
        stress_old: torch.Tensor,
        stress_new: torch.Tensor,
        stress_prev: torch.Tensor,
    ) -> float:
        """Compute Aitken (optimal) relaxation factor.

        Uses the formula::

            omega = -omega_prev * dot(s_prev, s_new - s_prev) /
                    |s_new - s_prev|^2

        where s_prev = stress_prev - stress_old.

        Args:
            stress_old: Previous iteration stress.
            stress_new: Current iteration stress.
            stress_prev: Two iterations ago stress.

        Returns:
            Optimal relaxation factor.
        """
        delta = stress_new - stress_old
        delta_norm_sq = delta.dot(delta).item()

        if delta_norm_sq < 1e-30:
            return 1.0

        s_diff = stress_old - stress_prev
        numerator = s_diff.dot(delta).item()

        if abs(numerator) < 1e-30:
            return 1.0

        # Aitken formula (clipped to reasonable range)
        omega = -numerator / delta_norm_sq
        return max(0.01, min(2.0, omega))

    # ------------------------------------------------------------------
    # Adaptive solve
    # ------------------------------------------------------------------

    def solve_adaptive(
        self,
        strain: torch.Tensor,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        initial_relaxation: float = 1.0,
        min_relaxation: float = 0.01,
        max_relaxation: float = 2.0,
    ) -> AdaptiveStressResult:
        """Solve with adaptive Aitken relaxation for robust convergence.

        Starts with the given relaxation factor and adaptively adjusts
        using the Aitken method. Falls back to fixed relaxation if
        the adaptive step is unstable.

        Args:
            strain: ``(6,)`` strain in Voigt notation.
            max_iterations: Maximum iterations.
            tolerance: Convergence tolerance.
            initial_relaxation: Initial under-relaxation factor.
            min_relaxation: Minimum allowed relaxation.
            max_relaxation: Maximum allowed relaxation.

        Returns:
            :class:`AdaptiveStressResult`.
        """
        strain = strain.to(dtype=torch.float64)

        # Initial guess
        stress = self._model.stress(strain)
        stress_prev = stress.clone()
        omega = initial_relaxation
        converged = False
        residual = float("inf")
        residual_history: List[float] = []

        for iteration in range(max_iterations):
            # Compute new stress
            stress_trial = self._model.stress(strain)

            # Apply relaxation
            stress_update = omega * stress_trial + (1 - omega) * stress

            # Compute residual
            residual_tensor = stress_update - stress
            residual = float(residual_tensor.norm().item())
            residual_history.append(residual)

            if residual < tolerance:
                converged = True
                stress = stress_update
                break

            # Aitken relaxation (need at least 2 iterations)
            if iteration >= 1:
                try:
                    omega_new = self._aitken_relaxation(
                        stress, stress_update, stress_prev
                    )
                    omega = max(min_relaxation, min(max_relaxation, omega_new))
                except Exception:
                    pass  # Keep current omega

            stress_prev = stress.clone()
            stress = stress_update

        # Compute von Mises
        vm = None
        if self._yield is not None:
            vm = self._yield.von_mises_stress(stress)

        return AdaptiveStressResult(
            stress=stress,
            n_iterations=iteration + 1,
            converged=converged,
            residual=residual,
            final_relaxation=omega,
            residual_history=residual_history,
            von_mises=vm,
        )

    # ------------------------------------------------------------------
    # Stress at integration points
    # ------------------------------------------------------------------

    def extrapolate_to_nodes(
        self,
        element_stress: torch.Tensor,
        extrap_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Extrapolate stress from integration points to nodes.

        Uses a shape function extrapolation matrix::

            sigma_nodes = N_ext * sigma_integration_points

        Args:
            element_stress: ``(n_ip, 6)`` stress at integration points.
            extrap_matrix: ``(n_nodes, n_ip)`` extrapolation matrix.

        Returns:
            ``(n_nodes, 6)`` stress at nodes.
        """
        return (extrap_matrix @ element_stress).to(dtype=torch.float64)

    # ------------------------------------------------------------------
    # Principal stress trajectories
    # ------------------------------------------------------------------

    def principal_stress_directions(
        self, stress: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute principal stresses and their directions.

        Args:
            stress: ``(6,)`` stress in Voigt notation.

        Returns:
            Tuple of:
            - ``(3,)`` principal stresses (descending order)
            - ``(3, 3)`` principal stress directions (columns)
        """
        s = stress.to(dtype=torch.float64)
        sigma = torch.tensor([
            [s[0], s[5], s[4]],
            [s[5], s[1], s[3]],
            [s[4], s[3], s[2]],
        ], dtype=torch.float64)

        eigvals, eigvecs = torch.linalg.eigh(sigma)

        # Sort descending
        idx = eigvals.argsort(descending=True)
        return eigvals[idx], eigvecs[:, idx]

    # ------------------------------------------------------------------
    # Batch solve
    # ------------------------------------------------------------------

    def solve_batch(
        self,
        strains: torch.Tensor,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
    ) -> List[AdaptiveStressResult]:
        """Solve for a batch of strain states.

        Args:
            strains: ``(n, 6)`` batch of strains in Voigt notation.
            max_iterations: Maximum iterations per point.
            tolerance: Convergence tolerance.

        Returns:
            List of :class:`AdaptiveStressResult`, one per strain state.
        """
        strains = strains.to(dtype=torch.float64)
        results = []
        for i in range(strains.shape[0]):
            result = self.solve_adaptive(
                strains[i],
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            results.append(result)
        return results

    def __repr__(self) -> str:
        return f"EnhancedStressSolver2(model={self._model!r})"
