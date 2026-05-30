"""
Enhanced elastic material models v5 with advanced constitutive behaviour.

Extends :class:`~pyfoam.structural.elastic_model_enhanced_4` with:

- :class:`GradientPlasticityModel` -- strain gradient plasticity with internal length
- :class:`CoupledDamagePlasticityModel` -- Lemaitre damage coupled with plasticity
- :class:`HyperelasticOgdenModel` -- Ogden hyperelastic model for rubber-like materials

Usage::

    model = GradientPlasticityModel(E=210e9, nu=0.3, sigma_y=250e6, l=1e-4)
    stress = model.stress(strain)

References
----------
- OpenFOAM ``mechanicalModel`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.elastic_model_enhanced_4 import (
    GursonDamageModel,
    CrystalPlasticityModel,
    PhaseFieldFractureModel,
)

__all__ = [
    "GradientPlasticityModel",
    "CoupledDamagePlasticityModel",
    "HyperelasticOgdenModel",
]


class GradientPlasticityModel:
    """Strain gradient plasticity model with internal length scale.

    Extends classical plasticity with higher-order strain gradient terms::

        sigma_y_eff = sigma_y + l^2 * |grad(eps_p)|^2

    where l is the internal length scale parameter that regularises
    strain localisation.

    Args:
        E: Young's modulus (Pa).
        nu: Poisson's ratio.
        sigma_y: Initial yield stress (Pa).
        l: Internal length scale (m).
        hardening_modulus: Isotropic hardening modulus H (Pa).
    """

    def __init__(
        self,
        E: float = 210e9,
        nu: float = 0.3,
        sigma_y: float = 250e6,
        l: float = 1e-4,
        hardening_modulus: float = 1e9,
    ) -> None:
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )
        self._sigma_y0 = sigma_y
        self._sigma_y = sigma_y
        self._l = l
        self._H = hardening_modulus
        self._eps_p_accumulated = 0.0
        self._E = E
        self._nu = nu

    @property
    def internal_length(self) -> float:
        """Internal length scale parameter."""
        return self._l

    @property
    def yield_stress(self) -> float:
        """Current yield stress."""
        return self._sigma_y

    @property
    def accumulated_plastic_strain(self) -> float:
        """Accumulated equivalent plastic strain."""
        return self._eps_p_accumulated

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the 6x6 elasticity matrix."""
        return self._model.elasticity_matrix

    def effective_yield_stress(
        self,
        strain_gradient_norm: float = 0.0,
    ) -> float:
        """Compute effective yield stress with gradient enhancement.

        Args:
            strain_gradient_norm: Norm of the plastic strain gradient.

        Returns:
            Effective yield stress (Pa).
        """
        gradient_contribution = self._l ** 2 * strain_gradient_norm ** 2
        return self._sigma_y + gradient_contribution

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress (elastic).

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            ``(6,)`` stress.
        """
        return self._model.stress(strain.to(dtype=torch.float64))

    def check_yield(
        self,
        stress: torch.Tensor,
        strain_gradient_norm: float = 0.0,
    ) -> bool:
        """Check if yielding occurs with gradient enhancement.

        Args:
            stress: ``(6,)`` stress in Voigt notation.
            strain_gradient_norm: Norm of plastic strain gradient.

        Returns:
            True if yielding occurs.
        """
        yield_criterion = VonMisesYield(sigma_y=self.effective_yield_stress(strain_gradient_norm))
        return yield_criterion.yield_function(stress) > 0

    def update_hardening(
        self,
        d_eps_p: float,
        strain_gradient_norm: float = 0.0,
    ) -> float:
        """Update yield stress from plastic strain increment.

        Args:
            d_eps_p: Incremental equivalent plastic strain.
            strain_gradient_norm: Norm of plastic strain gradient.

        Returns:
            Updated effective yield stress.
        """
        self._eps_p_accumulated += d_eps_p
        self._sigma_y = self._sigma_y0 + self._H * self._eps_p_accumulated
        return self.effective_yield_stress(strain_gradient_norm)

    def reset_state(self) -> None:
        """Reset plastic state."""
        self._sigma_y = self._sigma_y0
        self._eps_p_accumulated = 0.0

    def __repr__(self) -> str:
        return (
            f"GradientPlasticityModel(E={self._E:.2e}, "
            f"sigma_y={self._sigma_y:.2e}, l={self._l:.2e})"
        )


class CoupledDamagePlasticityModel:
    """Lemaitre damage model coupled with isotropic plasticity.

    Combines damage evolution with plastic deformation::

        sigma_eff = sigma / (1 - D)
        dD/dp = (sigma_y * R * p^(R-1)) / (S * (1-D)^(beta+1))

    where D is the damage variable, p is accumulated plastic strain,
    R and S are material parameters, and beta controls damage growth.

    Args:
        E: Young's modulus (Pa).
        nu: Poisson's ratio.
        sigma_y: Initial yield stress (Pa).
        S: Damage resistance parameter (Pa).
        R: Damage exponent (dimensionless).
        beta: Damage growth exponent.
    """

    def __init__(
        self,
        E: float = 210e9,
        nu: float = 0.3,
        sigma_y: float = 250e6,
        S: float = 1e6,
        R: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )
        self._sigma_y = sigma_y
        self._S = S
        self._R = R
        self._beta = beta
        self._D = 0.0  # damage variable
        self._p = 0.0  # accumulated plastic strain
        self._E = E
        self._nu = nu

    @property
    def damage(self) -> float:
        """Current damage variable (0 = intact, 1 = failed)."""
        return self._D

    @property
    def accumulated_plastic_strain(self) -> float:
        """Accumulated plastic strain."""
        return self._p

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the degraded 6x6 elasticity matrix."""
        return (1.0 - self._D) * self._model.elasticity_matrix

    def effective_stress(self, stress: torch.Tensor) -> torch.Tensor:
        """Compute effective stress (damage-corrected).

        Args:
            stress: ``(6,)`` nominal stress.

        Returns:
            ``(6,)`` effective stress.
        """
        return stress.to(dtype=torch.float64) / max(1.0 - self._D, 1e-10)

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress (elastic, degraded by damage).

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            ``(6,)`` stress.
        """
        C = self.elasticity_matrix
        return C @ strain.to(dtype=torch.float64)

    def update_damage(self, d_eps_p: float) -> float:
        """Update damage from plastic strain increment.

        Uses the Lemaitre damage evolution law.

        Args:
            d_eps_p: Incremental equivalent plastic strain.

        Returns:
            Updated damage variable.
        """
        if d_eps_p <= 0 or self._D >= 1.0:
            return self._D

        self._p += d_eps_p
        # Lemaitre damage evolution
        R = self._R
        p = max(self._p, 1e-30)
        dD = (self._sigma_y * R * p ** (R - 1)) / (
            self._S * max((1.0 - self._D) ** (self._beta + 1), 1e-30)
        ) * d_eps_p

        self._D = min(1.0, self._D + max(dD, 0.0))
        return self._D

    def is_failed(self) -> bool:
        """Check if material has failed (D >= 1)."""
        return self._D >= 1.0

    def reset_state(self) -> None:
        """Reset damage and plastic state."""
        self._D = 0.0
        self._p = 0.0

    def __repr__(self) -> str:
        return (
            f"CoupledDamagePlasticityModel(E={self._E:.2e}, "
            f"D={self._D:.6f}, p={self._p:.6f})"
        )


class HyperelasticOgdenModel:
    """Ogden hyperelastic model for rubber-like materials.

    Strain energy density::

        W = sum_i (mu_i / alpha_i) * (lambda1^alpha_i + lambda2^alpha_i
            + lambda3^alpha_i - 3) + K/2 * (J - 1)^2

    where lambda_i are principal stretches, mu_i and alpha_i are Ogden
    parameters, K is bulk modulus, and J is volume ratio.

    Args:
        mu: List of Ogden mu parameters (Pa).
        alpha: List of Ogden alpha parameters (dimensionless).
        bulk_modulus: Bulk modulus K (Pa).
    """

    def __init__(
        self,
        mu: list[float] | None = None,
        alpha: list[float] | None = None,
        bulk_modulus: float = 1e9,
    ) -> None:
        self._mu = mu if mu is not None else [1.0, 0.0]
        self._alpha = alpha if alpha is not None else [2.0, 4.0]
        self._K = bulk_modulus

        if len(self._mu) != len(self._alpha):
            raise ValueError("mu and alpha must have the same length.")

    @property
    def n_terms(self) -> int:
        """Number of Ogden terms."""
        return len(self._mu)

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus."""
        return self._K

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Linearised elasticity matrix for small strain (Voigt notation)."""
        # For small strains, Ogden reduces to isotropic with mu_sum = sum(mu_i*alpha_i)
        mu_sum = sum(m * a for m, a in zip(self._mu, self._alpha))
        # Equivalent E and nu from mu_sum and K
        E = 9.0 * self._K * mu_sum / (3.0 * self._K + mu_sum)
        nu = (3.0 * self._K - 2.0 * mu_sum) / (2.0 * (3.0 * self._K + mu_sum))
        model = LinearElasticModel(youngs_modulus=E, poisson_ratio=max(min(nu, 0.4999), -0.9999))
        return model.elasticity_matrix

    def strain_energy(
        self,
        stretches: torch.Tensor,
    ) -> float:
        """Compute strain energy density from principal stretches.

        Args:
            stretches: ``(3,)`` principal stretches (lambda1, lambda2, lambda3).

        Returns:
            Strain energy density (J/m^3).
        """
        stretches = stretches.to(dtype=torch.float64)
        J = stretches.prod()

        W = 0.0
        for mu_i, alpha_i in zip(self._mu, self._alpha):
            a = max(abs(alpha_i), 1e-15)
            W += mu_i / a * (stretches.pow(alpha_i).sum().item() - 3.0)

        # Volumetric penalty
        W += self._K / 2.0 * (J - 1.0).item() ** 2

        return W

    def stress_from_stretches(
        self,
        stretches: torch.Tensor,
    ) -> torch.Tensor:
        """Compute principal stresses from principal stretches.

        Args:
            stretches: ``(3,)`` principal stretches.

        Returns:
            ``(3,)`` principal stresses (Pa).
        """
        stretches = stretches.to(dtype=torch.float64)
        J = stretches.prod()

        stresses = torch.zeros(3, dtype=torch.float64)
        for mu_i, alpha_i in zip(self._mu, self._alpha):
            stresses += mu_i * stretches.pow(alpha_i - 1.0)

        # Volumetric: p = K * (J - 1)
        p = self._K * (J - 1.0)
        stresses += p

        return stresses

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress using linearised model for small strains.

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            ``(6,)`` stress.
        """
        C = self.elasticity_matrix
        return C @ strain.to(dtype=torch.float64)

    def reset_state(self) -> None:
        """Reset state (no state variables for this model)."""
        pass

    def __repr__(self) -> str:
        return (
            f"HyperelasticOgdenModel(n_terms={self.n_terms}, "
            f"K={self._K:.2e})"
        )
