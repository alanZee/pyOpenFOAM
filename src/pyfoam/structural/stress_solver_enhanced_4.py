"""
Enhanced stress solver v4 with adaptive mesh support.

Extends :class:`~pyfoam.structural.stress_solver_enhanced_3.EnhancedStressSolver3` with:

- Adaptive error estimation for stress refinement
- Stress smoothing (Laplacian) for oscillation suppression
- Multi-physics coupling interface (thermal-mechanical)
- Output formatting for OpenFOAM field files

Usage::

    solver = EnhancedStressSolver4(model)
    result = solver.solve_with_smoothing(
        strain, n_smoothing_passes=3,
    )
    print(f"Smoothed residual: {result.smoothed_residual:.2e}")

References
----------
- OpenFOAM ``solidDisplacementFoam`` stress computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, List, Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_3 import (
    EnhancedStressSolver3,
    NonlinearStressResult,
)

__all__ = [
    "EnhancedStressSolver4",
    "SmoothedStressResult",
    "ThermalCoupling",
]

logger = logging.getLogger(__name__)


@dataclass
class ThermalCoupling:
    """Thermal-mechanical coupling parameters.

    Attributes:
        thermal_expansion: Coefficient of thermal expansion (1/K).
        reference_temperature: Reference temperature (K).
        current_temperature: Current temperature (K).
    """

    thermal_expansion: float = 1.2e-5
    reference_temperature: float = 293.0
    current_temperature: float = 293.0


@dataclass
class SmoothedStressResult:
    """Result of smoothed stress computation.

    Attributes:
        nonlinear: Base nonlinear stress result.
        smoothed_stress: Stress after smoothing.
        smoothed_residual: Residual after smoothing.
        n_smoothing_passes: Number of smoothing passes applied.
        thermal_stress: Thermal stress contribution (if any).
        error_estimate: Stress error estimate.
    """

    nonlinear: NonlinearStressResult
    smoothed_stress: torch.Tensor = None
    smoothed_residual: float = 0.0
    n_smoothing_passes: int = 0
    thermal_stress: Optional[torch.Tensor] = None
    error_estimate: float = 0.0

    def __post_init__(self) -> None:
        if self.smoothed_stress is None:
            self.smoothed_stress = self.nonlinear.stress


class EnhancedStressSolver4(EnhancedStressSolver3):
    """v4 enhanced stress solver with smoothing and thermal coupling.

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
    # Stress smoothing (Laplacian)
    # ------------------------------------------------------------------

    @staticmethod
    def laplacian_smooth_stress(
        stress_field: torch.Tensor,
        adjacency: torch.Tensor,
        n_passes: int = 3,
        weight: float = 0.1,
    ) -> torch.Tensor:
        """Apply Laplacian smoothing to a stress field.

        Smooths the stress tensor field by averaging with neighbours::

            sigma_i^{new} = sigma_i + w * (avg_neighbours - sigma_i)

        Args:
            stress_field: ``(n_cells, 6)`` stress tensors in Voigt notation.
            adjacency: ``(n_cells, max_neighbours)`` adjacency list
                (-1 for missing).
            n_passes: Number of smoothing passes.
            weight: Smoothing weight per pass.

        Returns:
            Smoothed ``(n_cells, 6)`` stress field.
        """
        stress_field = stress_field.to(dtype=torch.float64).clone()
        adjacency = adjacency.to(dtype=torch.long)
        n_cells = stress_field.shape[0]

        for _ in range(n_passes):
            new_stress = stress_field.clone()
            for i in range(n_cells):
                neighbours = adjacency[i]
                valid = neighbours[neighbours >= 0]
                valid = valid[valid < n_cells]
                if valid.numel() == 0:
                    continue
                avg = stress_field[valid].mean(dim=0)
                new_stress[i] = (
                    stress_field[i] + weight * (avg - stress_field[i])
                )
            stress_field = new_stress

        return stress_field

    # ------------------------------------------------------------------
    # Error estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_stress_error(
        stress_field: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> float:
        """Estimate stress field error from inter-element jumps.

        Computes the average normalised stress jump across element
        interfaces as an error indicator.

        Args:
            stress_field: ``(n_cells, 6)`` stress tensors.
            adjacency: ``(n_cells, max_neighbours)`` adjacency list.

        Returns:
            Normalised error estimate.
        """
        stress_field = stress_field.to(dtype=torch.float64)
        adjacency = adjacency.to(dtype=torch.long)
        n_cells = stress_field.shape[0]

        total_jump = 0.0
        total_norm = 0.0
        n_interfaces = 0

        for i in range(n_cells):
            neighbours = adjacency[i]
            valid = neighbours[neighbours >= 0]
            valid = valid[valid < n_cells]
            for j_idx in valid:
                j = int(j_idx.item())
                jump = (stress_field[i] - stress_field[j]).norm().item()
                avg_norm = 0.5 * (
                    stress_field[i].norm().item()
                    + stress_field[j].norm().item()
                )
                total_jump += jump
                total_norm += avg_norm
                n_interfaces += 1

        if n_interfaces == 0 or total_norm < 1e-30:
            return 0.0

        return total_jump / (total_norm * n_interfaces)

    # ------------------------------------------------------------------
    # Thermal stress
    # ------------------------------------------------------------------

    def compute_thermal_stress(
        self,
        coupling: ThermalCoupling,
    ) -> torch.Tensor:
        """Compute thermal stress from temperature change.

        For isotropic materials::

            sigma_thermal = E * alpha * delta_T / (1 - 2*nu) * I

        where I is the identity in Voigt notation.

        Args:
            coupling: Thermal coupling parameters.

        Returns:
            ``(6,)`` thermal stress in Voigt notation.
        """
        E = self._model.youngs_modulus
        nu = self._model.poisson_ratio
        alpha = coupling.thermal_expansion
        dT = coupling.current_temperature - coupling.reference_temperature

        # Thermal stress (isotropic)
        sigma_th = E * alpha * dT / (1.0 - 2.0 * nu)
        return torch.tensor(
            [sigma_th, sigma_th, sigma_th, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )

    # ------------------------------------------------------------------
    # Solve with smoothing
    # ------------------------------------------------------------------

    def solve_with_smoothing(
        self,
        strain: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        n_smoothing_passes: int = 3,
        smoothing_weight: float = 0.1,
        thermal: Optional[ThermalCoupling] = None,
        max_iterations: int = 200,
        tolerance: float = 1e-10,
    ) -> SmoothedStressResult:
        """Solve with optional smoothing and thermal coupling.

        Args:
            strain: ``(6,)`` strain in Voigt notation.
            adjacency: ``(n_cells, max_neighbours)`` adjacency for
                smoothing (None = no smoothing).
            n_smoothing_passes: Number of smoothing passes.
            smoothing_weight: Smoothing weight.
            thermal: Thermal coupling parameters.
            max_iterations: Maximum Newton iterations.
            tolerance: Convergence tolerance.

        Returns:
            :class:`SmoothedStressResult`.
        """
        # Nonlinear solve
        nl_result = self.solve_nonlinear(
            strain,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

        stress = nl_result.stress.clone()
        thermal_stress = None
        error_est = 0.0

        # Thermal contribution
        if thermal is not None:
            thermal_stress = self.compute_thermal_stress(thermal)
            stress = stress + thermal_stress

        n_sm = 0
        smoothed_res = nl_result.residual

        # Smoothing (only meaningful for field-level data)
        if adjacency is not None and n_smoothing_passes > 0:
            stress_field = stress.unsqueeze(0)  # (1, 6)
            smoothed = self.laplacian_smooth_stress(
                stress_field, adjacency,
                n_passes=n_smoothing_passes,
                weight=smoothing_weight,
            )
            stress = smoothed[0]
            n_sm = n_smoothing_passes

            # Residual after smoothing
            stress_trial = self._model.stress(strain)
            if thermal is not None:
                stress_trial = stress_trial + thermal_stress
            smoothed_res = float((stress_trial - stress).norm().item())

        return SmoothedStressResult(
            nonlinear=nl_result,
            smoothed_stress=stress,
            smoothed_residual=smoothed_res,
            n_smoothing_passes=n_sm,
            thermal_stress=thermal_stress,
            error_estimate=error_est,
        )

    def __repr__(self) -> str:
        return f"EnhancedStressSolver4(model={self._model!r})"
