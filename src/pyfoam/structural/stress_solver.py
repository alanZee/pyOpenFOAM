"""
Stress solver for structural mechanics.

Computes the stress field from a strain field using a constitutive
model (e.g. :class:`LinearElasticModel`).  Supports both single-point
and batch (field) computations.

In OpenFOAM, the stress solver is part of the ``solidDisplacementFoam``
and ``solidEquilibriumDisplacementFoam`` solvers.  This module provides
a standalone Python equivalent.

Usage::

    model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
    solver = StressSolver(model)

    strain = torch.tensor([0.001, -0.0003, -0.0003, 0, 0, 0], dtype=torch.float64)
    stress = solver.solve(strain)
"""

from __future__ import annotations

import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield

__all__ = ["StressSolver"]


class StressSolver:
    """Compute stress fields from strain fields.

    Wraps a constitutive model and optional yield criterion to provide
    a complete stress analysis pipeline:

    1. Compute stress: ``sigma = C : epsilon``
    2. Compute von Mises equivalent stress
    3. Check yielding and safety factor

    Args:
        model: Constitutive model (e.g. :class:`LinearElasticModel`).
        yield_criterion: Optional yield criterion (e.g. :class:`VonMisesYield`).
    """

    def __init__(
        self,
        model: LinearElasticModel,
        yield_criterion: VonMisesYield | None = None,
    ) -> None:
        self._model = model
        self._yield = yield_criterion

    @property
    def model(self) -> LinearElasticModel:
        return self._model

    @property
    def yield_criterion(self) -> VonMisesYield | None:
        return self._yield

    def solve(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress from strain.

        Args:
            strain: ``(6,)`` or ``(n, 6)`` strain in Voigt notation.

        Returns:
            ``(6,)`` or ``(n, 6)`` stress in Voigt notation.
        """
        return self._model.stress(strain)

    def solve_full(self, strain: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute stress with full diagnostics.

        Args:
            strain: ``(6,)`` or ``(n, 6)`` strain in Voigt notation.

        Returns:
            Dictionary with keys:

            - ``"stress"``: stress tensor (Voigt)
            - ``"von_mises"``: von Mises equivalent stress
            - ``"is_yielding"``: boolean yield flag (if yield criterion set)
            - ``"safety_factor"``: safety factor (if yield criterion set)
        """
        stress = self.solve(strain)
        result: dict[str, torch.Tensor] = {"stress": stress}

        if self._yield is not None:
            result["von_mises"] = self._yield.von_mises_stress(stress)
            result["is_yielding"] = self._yield.is_yielding(stress)
            result["safety_factor"] = self._yield.safety_factor(stress)

        return result

    def principal_stresses(self, stress: torch.Tensor) -> torch.Tensor:
        """Compute principal stresses from the stress tensor.

        Converts the 6-component Voigt stress to a 3x3 symmetric
        tensor and computes eigenvalues.

        Args:
            stress: ``(6,)`` stress in Voigt notation.

        Returns:
            ``(3,)`` principal stresses sorted descending.
        """
        s = stress.to(dtype=torch.float64)
        # Voigt -> full 3x3 symmetric tensor
        sigma = torch.tensor([
            [s[0], s[5], s[4]],
            [s[5], s[1], s[3]],
            [s[4], s[3], s[2]],
        ], dtype=torch.float64)
        eigvals = torch.linalg.eigvalsh(sigma)
        return eigvals.flip(0)  # descending order

    def __repr__(self) -> str:
        return f"StressSolver(model={self._model!r})"
