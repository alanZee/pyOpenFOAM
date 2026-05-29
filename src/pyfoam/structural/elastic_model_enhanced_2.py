"""
Enhanced elastic material models v2 with advanced constitutive behaviour.

Extends :class:`~pyfoam.structural.elastic_model_enhanced` with:

- :class:`TransverselyIsotropicModel` — material with one axis of symmetry (5 constants)
- :class:`HyperelasticNeoHookean` — Neo-Hookean hyperelastic model for large deformations
- :class:`CombinedPlasticityModel` — isotropic + kinematic hardening plasticity

Usage::

    # Transversely isotropic (fibre-reinforced)
    model = TransverselyIsotropicModel(
        E_axial=150e9, E_transverse=10e9,
        nu_axial=0.3, nu_transverse=0.4,
        G_axial=6e9,
    )

    # Neo-Hookean hyperelasticity
    model = HyperelasticNeoHookean(mu=1e6, kappa=1e9)

References
----------
- OpenFOAM ``mechanicalModel`` framework
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.elastic_model_enhanced import IsotropicPlasticModel

__all__ = [
    "TransverselyIsotropicModel",
    "HyperelasticNeoHookean",
    "CombinedPlasticityModel",
]


class TransverselyIsotropicModel:
    """Transversely isotropic elastic model with 5 independent constants.

    Materials with one axis of symmetry (e.g., fibre-reinforced composites,
    rolled metals, crystals with hexagonal symmetry).

    Independent constants:
    - E_axial: Young's modulus along the symmetry axis
    - E_transverse: Young's modulus in the transverse plane
    - nu_axial: Poisson's ratio for loading along the symmetry axis
    - nu_transverse: Poisson's ratio within the transverse plane
    - G_axial: Shear modulus for shear involving the symmetry axis

    Args:
        E_axial: Axial Young's modulus (Pa).
        E_transverse: Transverse Young's modulus (Pa).
        nu_axial: Axial Poisson's ratio.
        nu_transverse: Transverse Poisson's ratio.
        G_axial: Axial shear modulus (Pa).
        symmetry_axis: Symmetry axis index (0=x, 1=y, 2=z). Default 2 (z).
    """

    def __init__(
        self,
        E_axial: float = 210e9,
        E_transverse: float = 210e9,
        nu_axial: float = 0.3,
        nu_transverse: float = 0.3,
        G_axial: float = 80.77e9,
        symmetry_axis: int = 2,
    ) -> None:
        if symmetry_axis not in (0, 1, 2):
            raise ValueError("symmetry_axis must be 0, 1, or 2.")
        self._E_a = E_axial
        self._E_t = E_transverse
        self._nu_a = nu_axial
        self._nu_t = nu_transverse
        self._G_a = G_axial
        self._axis = symmetry_axis
        self._C = self._build_elasticity_matrix()

    def _build_elasticity_matrix(self) -> torch.Tensor:
        """Build the 6x6 elasticity matrix in Voigt notation.

        For a transversely isotropic material with symmetry axis z::

            S = [[1/E_t, -nu_t/E_t, -nu_a/E_a, 0,      0,      0     ],
                 [-nu_t/E_t, 1/E_t, -nu_a/E_a, 0,      0,      0     ],
                 [-nu_a/E_a, -nu_a/E_a, 1/E_a, 0,      0,      0     ],
                 [0,         0,        0,        1/G_a,  0,      0     ],
                 [0,         0,        0,        0,      1/G_a,  0     ],
                 [0,         0,        0,        0,      0,      2*(1+nu_t)/E_t]]
        """
        E_a = self._E_a
        E_t = self._E_t
        nu_a = self._nu_a
        nu_t = self._nu_t
        G_a = self._G_a
        G_t = E_t / (2.0 * (1.0 + nu_t))

        # Build compliance matrix assuming z is the symmetry axis
        S = torch.tensor([
            [1.0 / E_t, -nu_t / E_t, -nu_a / E_a, 0.0, 0.0, 0.0],
            [-nu_t / E_t, 1.0 / E_t, -nu_a / E_a, 0.0, 0.0, 0.0],
            [-nu_a / E_a, -nu_a / E_a, 1.0 / E_a, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / G_a, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / G_a, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / G_t],
        ], dtype=torch.float64)

        return torch.linalg.inv(S)

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the 6x6 elasticity matrix."""
        return self._C.clone()

    @property
    def youngs_moduli(self) -> tuple[float, float]:
        """Return (E_axial, E_transverse)."""
        return (self._E_a, self._E_t)

    @property
    def shear_modulus(self) -> float:
        """Return the axial shear modulus."""
        return self._G_a

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """sigma = C : epsilon."""
        strain = strain.to(dtype=torch.float64)
        if strain.dim() == 1:
            return self._C @ strain
        return (self._C @ strain.T).T

    def strain(self, stress: torch.Tensor) -> torch.Tensor:
        """epsilon = C^{-1} : sigma."""
        stress = stress.to(dtype=torch.float64)
        C_inv = torch.linalg.inv(self._C)
        if stress.dim() == 1:
            return C_inv @ stress
        return (C_inv @ stress.T).T

    def __repr__(self) -> str:
        return (
            f"TransverselyIsotropicModel(E_a={self._E_a:.2e}, "
            f"E_t={self._E_t:.2e})"
        )


class HyperelasticNeoHookean:
    """Neo-Hookean hyperelastic model for large deformation analysis.

    The strain energy density is::

        W = (mu/2) * (I1 - 3) - mu * ln(J) + (kappa/2) * (ln(J))^2

    where:
    - I1 = tr(C) is the first invariant of the right Cauchy-Green tensor
    - J = det(F) is the volume ratio
    - mu is the shear modulus
    - kappa is the bulk modulus

    Args:
        mu: Shear modulus (Pa).
        kappa: Bulk modulus (Pa).
    """

    def __init__(
        self,
        mu: float = 1e6,
        kappa: float = 1e9,
    ) -> None:
        self._mu = mu
        self._kappa = kappa

    @property
    def mu(self) -> float:
        """Shear modulus."""
        return self._mu

    @property
    def kappa(self) -> float:
        """Bulk modulus."""
        return self._kappa

    def strain_energy(
        self,
        grad_u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute strain energy density W.

        Args:
            grad_u: ``(3, 3)`` displacement gradient.

        Returns:
            Scalar strain energy density.
        """
        F = torch.eye(3, dtype=torch.float64) + grad_u.to(dtype=torch.float64)
        C = F.T @ F
        I1 = torch.trace(C)
        J = torch.det(F)

        W = (
            0.5 * self._mu * (I1 - 3.0)
            - self._mu * torch.log(J)
            + 0.5 * self._kappa * torch.log(J) ** 2
        )
        return W

    def pk2_stress(
        self,
        grad_u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute second Piola-Kirchhoff stress tensor.

        S = mu * (I - C^{-1}) + kappa * ln(J) * C^{-1}

        Args:
            grad_u: ``(3, 3)`` displacement gradient.

        Returns:
            ``(3, 3)`` second Piola-Kirchhoff stress.
        """
        F = torch.eye(3, dtype=torch.float64) + grad_u.to(dtype=torch.float64)
        C = F.T @ F
        J = torch.det(F)
        C_inv = torch.linalg.inv(C)
        I = torch.eye(3, dtype=torch.float64)

        S = self._mu * (I - C_inv) + self._kappa * torch.log(J) * C_inv
        return S

    def cauchy_stress(
        self,
        grad_u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Cauchy stress tensor.

        sigma = (1/J) * F * S * F^T

        Args:
            grad_u: ``(3, 3)`` displacement gradient.

        Returns:
            ``(3, 3)`` Cauchy stress tensor.
        """
        F = torch.eye(3, dtype=torch.float64) + grad_u.to(dtype=torch.float64)
        J = torch.det(F)
        S = self.pk2_stress(grad_u)
        sigma = (1.0 / J) * F @ S @ F.T
        return sigma

    def stress_voigt(
        self,
        grad_u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Cauchy stress in Voigt notation.

        Args:
            grad_u: ``(3, 3)`` displacement gradient.

        Returns:
            ``(6,)`` stress in Voigt notation.
        """
        sigma = self.cauchy_stress(grad_u)
        return torch.tensor([
            sigma[0, 0],
            sigma[1, 1],
            sigma[2, 2],
            sigma[1, 2],
            sigma[0, 2],
            sigma[0, 1],
        ], dtype=torch.float64)

    def __repr__(self) -> str:
        return (
            f"HyperelasticNeoHookean(mu={self._mu:.2e}, "
            f"kappa={self._kappa:.2e})"
        )


class CombinedPlasticityModel:
    """Isotropic + kinematic hardening plasticity (Chaboche model).

    Combines linear isotropic hardening with nonlinear kinematic
    hardening to model the Bauschinger effect and ratcheting::

        f = |sigma - alpha| - (sigma_y + R)
        d_alpha = C * d_eps_p - gamma * alpha * d_p
        d_R = b * (Q - R) * d_p

    where:
    - alpha: back stress tensor (kinematic hardening)
    - R: isotropic hardening variable
    - C, gamma: kinematic hardening parameters
    - b, Q: isotropic hardening parameters

    Args:
        youngs_modulus: Young's modulus (Pa).
        poisson_ratio: Poisson's ratio.
        yield_stress: Initial yield stress (Pa).
        kinematic_C: Kinematic hardening modulus (Pa).
        kinematic_gamma: Kinematic hardening recall rate.
        isotropic_b: Isotropic hardening rate.
        isotropic_Q: Isotropic hardening saturation value (Pa).
    """

    def __init__(
        self,
        youngs_modulus: float = 210e9,
        poisson_ratio: float = 0.3,
        yield_stress: float = 250e6,
        kinematic_C: float = 1e9,
        kinematic_gamma: float = 10.0,
        isotropic_b: float = 1.0,
        isotropic_Q: float = 50e6,
    ) -> None:
        self._elastic = LinearElasticModel(youngs_modulus, poisson_ratio)
        self._G = self._elastic.shear_modulus
        self._sigma_y = yield_stress
        self._C_kin = kinematic_C
        self._gamma = kinematic_gamma
        self._b_iso = isotropic_b
        self._Q_iso = isotropic_Q

        # Internal state
        self._back_stress = torch.zeros(6, dtype=torch.float64)
        self._R: float = 0.0  # isotropic hardening variable
        self._eq_plastic_strain: float = 0.0

    @property
    def yield_stress(self) -> float:
        """Current yield stress (including isotropic hardening)."""
        return self._sigma_y + self._R

    @property
    def back_stress(self) -> torch.Tensor:
        """Current back stress tensor."""
        return self._back_stress.clone()

    @property
    def equivalent_plastic_strain(self) -> float:
        """Accumulated equivalent plastic strain."""
        return self._eq_plastic_strain

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the 6x6 elastic elasticity matrix."""
        return self._elastic.elasticity_matrix

    def reset_state(self) -> None:
        """Reset all internal state variables."""
        self._back_stress.zero_()
        self._R = 0.0
        self._eq_plastic_strain = 0.0

    def return_mapping(
        self, strain: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """Perform stress return-mapping with combined hardening.

        Args:
            strain: ``(6,)`` total strain in Voigt notation.

        Returns:
            Tuple of (corrected stress, is_plastic).
        """
        strain = strain.to(dtype=torch.float64)
        trial_stress = self._elastic.stress(strain)

        # Effective stress relative to back stress
        effective = trial_stress - self._back_stress

        # Deviatoric part of effective stress
        hydro = (effective[0] + effective[1] + effective[2]) / 3.0
        dev = effective.clone()
        dev[0] -= hydro
        dev[1] -= hydro
        dev[2] -= hydro

        # Von Mises of effective stress
        vm_sq = (
            1.5 * (dev[0] ** 2 + dev[1] ** 2 + dev[2] ** 2)
            + 3.0 * (dev[3] ** 2 + dev[4] ** 2 + dev[5] ** 2)
        )
        sigma_vm = torch.sqrt(torch.clamp(vm_sq, min=0.0))

        current_yield = self._sigma_y + self._R
        if sigma_vm.item() <= current_yield:
            return trial_stress, False

        # Plastic multiplier (simplified radial return)
        delta_lambda = (sigma_vm.item() - current_yield) / (
            3.0 * self._G + self._C_kin + self._b_iso * self._Q_iso
        )

        # Update back stress (kinematic hardening)
        normal = dev / torch.clamp(sigma_vm, min=1e-15)
        d_alpha = self._C_kin * delta_lambda * normal - self._gamma * self._back_stress * delta_lambda
        self._back_stress += d_alpha

        # Update isotropic hardening
        self._R += self._b_iso * (self._Q_iso - self._R) * delta_lambda

        # Update plastic strain
        self._eq_plastic_strain += delta_lambda

        # Correct stress: scale deviatoric part
        scale = 1.0 - 3.0 * self._G * delta_lambda / sigma_vm.item()
        corrected = trial_stress.clone()
        corrected[0] = scale * dev[0] + hydro
        corrected[1] = scale * dev[1] + hydro
        corrected[2] = scale * dev[2] + hydro
        corrected[3] = scale * dev[3]
        corrected[4] = scale * dev[4]
        corrected[5] = scale * dev[5]

        return corrected, True

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress with combined hardening return-mapping.

        Args:
            strain: ``(6,)`` or ``(n, 6)`` strain in Voigt notation.

        Returns:
            ``(6,)`` or ``(n, 6)`` stress.
        """
        strain = strain.to(dtype=torch.float64)
        if strain.dim() == 1:
            result, _ = self.return_mapping(strain)
            return result
        results = []
        for s in strain:
            r, _ = self.return_mapping(s)
            results.append(r)
        return torch.stack(results)

    def __repr__(self) -> str:
        return (
            f"CombinedPlasticityModel(E={self._elastic.youngs_modulus:.2e}, "
            f"sigma_y={self._sigma_y:.2e})"
        )
