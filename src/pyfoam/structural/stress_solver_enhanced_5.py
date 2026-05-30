"""
Enhanced stress solver v5 with multi-criteria failure assessment.

Extends :class:`~pyfoam.structural.stress_solver_enhanced_4.EnhancedStressSolver4` with:

- Multi-criteria failure assessment (max principal, von Mises, Tresca)
- Stress invariant computation (I1, I2, I3, J2, J3)
- Mohr-Coulomb failure envelope check
- Stress triaxiality and Lode angle computation

Usage::

    solver = EnhancedStressSolver5(model)
    result = solver.assess_failure(strain)
    print(f"Triaxiality: {result.triaxiality:.3f}")

References
----------
- OpenFOAM ``solidDisplacementFoam`` stress computation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field as dc_field
from typing import Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_4 import (
    EnhancedStressSolver4,
    SmoothedStressResult,
    ThermalCoupling,
)

__all__ = [
    "EnhancedStressSolver5",
    "FailureAssessment",
    "StressInvariants",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stress invariants
# ---------------------------------------------------------------------------


@dataclass
class StressInvariants:
    """Stress invariants for a 3D stress state.

    Attributes:
        I1: First invariant (trace of stress tensor).
        I2: Second invariant.
        I3: Third invariant.
        J2: Second deviatoric invariant (related to von Mises stress).
        J3: Third deviatoric invariant.
        von_mises: Von Mises equivalent stress.
        max_principal: Maximum principal stress.
        min_principal: Minimum principal stress.
    """

    I1: float = 0.0
    I2: float = 0.0
    I3: float = 0.0
    J2: float = 0.0
    J3: float = 0.0
    von_mises: float = 0.0
    max_principal: float = 0.0
    min_principal: float = 0.0


# ---------------------------------------------------------------------------
# Failure assessment
# ---------------------------------------------------------------------------


@dataclass
class FailureAssessment:
    """Multi-criteria failure assessment result.

    Attributes:
        invariants: Stress invariants.
        triaxiality: Stress triaxiality (sigma_m / sigma_eq).
        lode_angle: Lode angle (rad).
        von_mises_ratio: Von Mises stress / yield stress.
        tresca_ratio: Tresca stress / (2 * yield stress).
        mohr_coulomb_ratio: Mohr-Coulomb ratio (>1 means failure).
        max_principal_ratio: Max principal stress / tensile strength.
        is_yielding: Whether von Mises criterion indicates yielding.
    """

    invariants: StressInvariants = None
    triaxiality: float = 0.0
    lode_angle: float = 0.0
    von_mises_ratio: float = 0.0
    tresca_ratio: float = 0.0
    mohr_coulomb_ratio: float = 0.0
    max_principal_ratio: float = 0.0
    is_yielding: bool = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EnhancedStressSolver5(EnhancedStressSolver4):
    """v5 enhanced stress solver with multi-criteria failure assessment.

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model.
    yield_criterion : VonMisesYield, optional
        Yield criterion.
    tensile_strength : float
        Ultimate tensile strength (Pa).
    cohesion : float
        Mohr-Coulomb cohesion (Pa).
    friction_angle : float
        Mohr-Coulomb friction angle (rad).
    """

    def __init__(
        self,
        model: LinearElasticModel,
        yield_criterion: VonMisesYield | None = None,
        tensile_strength: float = 400e6,
        cohesion: float = 30e6,
        friction_angle: float = 0.5,
    ) -> None:
        super().__init__(model, yield_criterion)
        self._sigma_UTS = tensile_strength
        self._c = cohesion
        self._phi = friction_angle

    # ------------------------------------------------------------------
    # Stress invariants
    # ------------------------------------------------------------------

    @staticmethod
    def compute_invariants(stress: torch.Tensor) -> StressInvariants:
        """Compute stress invariants from Voigt stress.

        Args:
            stress: ``(6,)`` stress in Voigt notation
                [s11, s22, s33, s23, s13, s12].

        Returns:
            :class:`StressInvariants`.
        """
        s = stress.to(dtype=torch.float64)
        s11, s22, s33, s23, s13, s12 = s[0], s[1], s[2], s[3], s[4], s[5]

        # First invariant
        I1 = s11 + s22 + s33

        # Second invariant
        I2 = (
            s11 * s22 + s22 * s33 + s33 * s11
            - s12 ** 2 - s13 ** 2 - s23 ** 2
        )

        # Third invariant
        I3 = (
            s11 * s22 * s33
            + 2.0 * s12 * s13 * s23
            - s11 * s23 ** 2
            - s22 * s13 ** 2
            - s33 * s12 ** 2
        )

        # Deviatoric stress
        p = I1 / 3.0
        s11_d = s11 - p
        s22_d = s22 - p
        s33_d = s33 - p

        # J2
        J2 = (
            0.5 * (s11_d ** 2 + s22_d ** 2 + s33_d ** 2)
            + s12 ** 2 + s13 ** 2 + s23 ** 2
        )

        # J3
        J3 = (
            s11_d * s22_d * s33_d
            + 2.0 * s12 * s13 * s23
            - s11_d * s23 ** 2
            - s22_d * s13 ** 2
            - s33_d * s12 ** 2
        )

        # Von Mises
        von_mises = math.sqrt(max(3.0 * J2, 0.0))

        # Principal stresses (from cubic equation, simplified)
        # For the general case, use the invariants
        # Simplified: max/min principal from diagonal + off-diagonal
        sigma_m = p
        R = math.sqrt(max(2.0 * J2 / 3.0, 0.0))
        max_principal = sigma_m + R
        min_principal = sigma_m - R

        return StressInvariants(
            I1=I1, I2=I2, I3=I3,
            J2=J2, J3=J3,
            von_mises=von_mises,
            max_principal=max_principal,
            min_principal=min_principal,
        )

    # ------------------------------------------------------------------
    # Triaxiality and Lode angle
    # ------------------------------------------------------------------

    @staticmethod
    def compute_triaxiality(stress: torch.Tensor) -> float:
        """Compute stress triaxiality: eta = sigma_m / sigma_eq.

        Args:
            stress: ``(6,)`` stress in Voigt notation.

        Returns:
            Stress triaxiality (dimensionless).
        """
        invariants = EnhancedStressSolver5.compute_invariants(stress)
        sigma_m = invariants.I1 / 3.0
        sigma_eq = invariants.von_mises

        if sigma_eq < 1e-30:
            return 0.0

        return sigma_m / sigma_eq

    @staticmethod
    def compute_lode_angle(stress: torch.Tensor) -> float:
        """Compute Lode angle from stress state.

        Args:
            stress: ``(6,)`` stress in Voigt notation.

        Returns:
            Lode angle (rad, range [0, pi/3]).
        """
        invariants = EnhancedStressSolver5.compute_invariants(stress)
        J2 = invariants.J2
        J3 = invariants.J3

        if J2 < 1e-30:
            return 0.0

        # Lode angle: sin(3*theta) = -3*sqrt(3)*J3 / (2*J2^(3/2))
        sin_3theta = -3.0 * math.sqrt(3.0) * J3 / (2.0 * J2 ** 1.5)
        sin_3theta = max(-1.0, min(1.0, sin_3theta))
        theta = math.asin(sin_3theta) / 3.0

        # Map to [0, pi/3]
        return max(0.0, min(math.pi / 3.0, theta + math.pi / 6.0))

    # ------------------------------------------------------------------
    # Failure assessment
    # ------------------------------------------------------------------

    def assess_failure(
        self,
        strain: torch.Tensor,
        yield_stress: float | None = None,
    ) -> FailureAssessment:
        """Perform multi-criteria failure assessment.

        Args:
            strain: ``(6,)`` strain in Voigt notation.
            yield_stress: Yield stress (uses model default if None).

        Returns:
            :class:`FailureAssessment`.
        """
        stress = self._model.stress(strain.to(dtype=torch.float64))
        invariants = self.compute_invariants(stress)

        sy = yield_stress if yield_stress is not None else 250e6

        # Von Mises ratio
        vm_ratio = invariants.von_mises / max(sy, 1e-30)

        # Tresca: tau_max = (sigma_max - sigma_min) / 2
        tau_max = (invariants.max_principal - invariants.min_principal) / 2.0
        tresca_ratio = tau_max / max(sy, 1e-30)

        # Mohr-Coulomb
        sigma_n = (invariants.max_principal + invariants.min_principal) / 2.0
        mc_lhs = tau_max
        mc_rhs = self._c * math.cos(self._phi) + sigma_n * math.sin(self._phi)
        mc_ratio = mc_lhs / max(mc_rhs, 1e-30)

        # Max principal ratio (tensile)
        principal_ratio = invariants.max_principal / max(self._sigma_UTS, 1e-30)

        # Triaxiality and Lode angle
        triax = self.compute_triaxiality(stress)
        lode = self.compute_lode_angle(stress)

        return FailureAssessment(
            invariants=invariants,
            triaxiality=triax,
            lode_angle=lode,
            von_mises_ratio=vm_ratio,
            tresca_ratio=tresca_ratio,
            mohr_coulomb_ratio=mc_ratio,
            max_principal_ratio=principal_ratio,
            is_yielding=vm_ratio > 1.0,
        )

    def __repr__(self) -> str:
        return f"EnhancedStressSolver5(model={self._model!r})"
