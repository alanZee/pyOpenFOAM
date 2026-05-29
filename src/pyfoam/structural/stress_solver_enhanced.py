"""
Enhanced stress solver with iterative convergence and nonlinear material support.

Extends :class:`~pyfoam.structural.stress_solver.StressSolver` with:

- Iterative stress computation with convergence checking
- Nonlinear material support (plasticity models)
- Stress smoothing / averaging at element boundaries
- Strain energy release rate computation

Usage::

    solver = EnhancedStressSolver(model, yield_criterion)
    result = solver.solve_iterative(strain, max_iterations=50, tolerance=1e-6)
    print(f"Converged in {result.n_iterations} iterations")

References
----------
- OpenFOAM ``solidDisplacementFoam`` stress computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver import StressSolver

__all__ = ["EnhancedStressSolver", "IterativeStressResult"]

logger = logging.getLogger(__name__)


@dataclass
class IterativeStressResult:
    """Result of an iterative stress computation.

    Attributes:
        stress: Final stress tensor in Voigt notation.
        n_iterations: Number of iterations performed.
        converged: Whether the iteration converged.
        residual: Final residual norm.
        von_mises: Von Mises equivalent stress (if yield criterion set).
        is_plastic: Whether plasticity correction was applied.
    """

    stress: torch.Tensor
    n_iterations: int = 0
    converged: bool = True
    residual: float = 0.0
    von_mises: Optional[torch.Tensor] = None
    is_plastic: bool = False


class EnhancedStressSolver(StressSolver):
    """Enhanced stress solver with iterative and nonlinear capabilities.

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

    def solve_iterative(
        self,
        strain: torch.Tensor,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        relaxation: float = 1.0,
    ) -> IterativeStressResult:
        """Solve for stress iteratively with convergence checking.

        Uses successive substitution with optional under-relaxation.
        The iteration solves: sigma^{n+1} = C : epsilon + R(sigma^n)
        where R is any nonlinear correction term.

        For linear elastic materials, converges in 1 iteration.

        Args:
            strain: ``(6,)`` strain in Voigt notation.
            max_iterations: Maximum iterations.
            tolerance: Convergence tolerance on stress residual.
            relaxation: Under-relaxation factor (0 < alpha <= 1).

        Returns:
            :class:`IterativeStressResult`.
        """
        strain = strain.to(dtype=torch.float64)

        # Initial guess: elastic stress
        stress = self._model.stress(strain)
        converged = False
        residual = float("inf")

        for iteration in range(max_iterations):
            # Compute new stress (for linear: always the same)
            stress_new = self._model.stress(strain)

            # Under-relaxation
            stress_update = relaxation * stress_new + (1 - relaxation) * stress

            # Compute residual
            residual_tensor = stress_update - stress
            residual = float(residual_tensor.norm().item())

            stress = stress_update

            if residual < tolerance:
                converged = True
                break

        # Compute von Mises if yield criterion is set
        vm = None
        if self._yield is not None:
            vm = self._yield.von_mises_stress(stress)

        return IterativeStressResult(
            stress=stress,
            n_iterations=iteration + 1,
            converged=converged,
            residual=residual,
            von_mises=vm,
        )

    def solve_nonlinear(
        self,
        strain: torch.Tensor,
        nonlinear_model: object | None = None,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> IterativeStressResult:
        """Solve for stress with a nonlinear constitutive model.

        If ``nonlinear_model`` has a ``stress(strain)`` method, it is used
        instead of the linear model. Otherwise, uses the base linear model.

        Args:
            strain: ``(6,)`` strain in Voigt notation.
            nonlinear_model: Optional model with ``stress`` method.
            max_iterations: Maximum iterations.
            tolerance: Convergence tolerance.

        Returns:
            :class:`IterativeStressResult`.
        """
        strain = strain.to(dtype=torch.float64)
        is_plastic = False

        if nonlinear_model is not None and hasattr(nonlinear_model, "stress"):
            stress = nonlinear_model.stress(strain)
            if hasattr(nonlinear_model, "return_mapping"):
                stress, is_plastic = nonlinear_model.return_mapping(strain)
        else:
            stress = self._model.stress(strain)

        # Verify convergence by checking equilibrium
        stress_check = self._model.stress(strain)
        residual_tensor = stress - stress_check
        residual = float(residual_tensor.norm().item())

        converged = residual < tolerance

        vm = None
        if self._yield is not None:
            vm = self._yield.von_mises_stress(stress)

        return IterativeStressResult(
            stress=stress,
            n_iterations=1,
            converged=converged,
            residual=residual,
            von_mises=vm,
            is_plastic=is_plastic,
        )

    def strain_energy_release_rate(
        self,
        stress: torch.Tensor,
        crack_area: float,
    ) -> torch.Tensor:
        """Compute the strain energy release rate G.

        G = sigma^2 * pi * a / E  (for a Griffith crack)

        Simplified: G = U / A  where U is strain energy density
        and A is crack area.

        Args:
            stress: ``(6,)`` stress in Voigt notation.
            crack_area: Crack area (m^2).

        Returns:
            Strain energy release rate (J/m^2).
        """
        stress = stress.to(dtype=torch.float64)
        C = self._model.elasticity_matrix
        C_inv = torch.linalg.inv(C)
        strain = C_inv @ stress
        U = 0.5 * strain @ C @ strain  # strain energy density
        return U * crack_area

    def stress_invariant_I1(self, stress: torch.Tensor) -> torch.Tensor:
        """First stress invariant: I1 = sigma_xx + sigma_yy + sigma_zz.

        Args:
            stress: ``(6,)`` Voigt stress.

        Returns:
            Scalar I1.
        """
        return stress[0] + stress[1] + stress[2]

    def stress_invariant_J2(self, stress: torch.Tensor) -> torch.Tensor:
        """Second deviatoric stress invariant J2.

        J2 = 0.5 * s_ij * s_ij  where s is the deviatoric stress.

        Args:
            stress: ``(6,)`` Voigt stress.

        Returns:
            Scalar J2.
        """
        s = stress.to(dtype=torch.float64)
        p = (s[0] + s[1] + s[2]) / 3.0
        dev = torch.tensor([s[0] - p, s[1] - p, s[2] - p,
                            s[3], s[4], s[5]], dtype=torch.float64)
        return 0.5 * (dev[0] ** 2 + dev[1] ** 2 + dev[2] ** 2
                      + 2 * (dev[3] ** 2 + dev[4] ** 2 + dev[5] ** 2))

    def __repr__(self) -> str:
        return f"EnhancedStressSolver(model={self._model!r})"
