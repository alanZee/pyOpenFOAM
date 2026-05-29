"""
Enhanced elastic material models with anisotropy and plasticity.

Extends :class:`~pyfoam.structural.elastic_model.LinearElasticModel` with:

- :class:`AnisotropicElasticModel` — fully anisotropic 6x6 elasticity matrix
- :class:`OrthotropicElasticModel` — orthotropic (3 orthogonal symmetry planes)
- :class:`IsotropicPlasticModel` — isotropic elastic + isotropic hardening plasticity

Usage::

    # Orthotropic (wood-like)
    model = OrthotropicElasticModel(
        E1=12e9, E2=1e9, E3=1e9,
        nu23=0.3, nu13=0.3, nu12=0.05,
        G23=0.5e9, G13=5e9, G12=5e9,
    )

    # Plasticity
    model = IsotropicPlasticModel(
        youngs_modulus=210e9,
        poisson_ratio=0.3,
        yield_stress=250e6,
        hardening_modulus=1e9,
    )

References
----------
- OpenFOAM ``mechanicalModel`` framework
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield

__all__ = [
    "AnisotropicElasticModel",
    "OrthotropicElasticModel",
    "IsotropicPlasticModel",
]


class AnisotropicElasticModel:
    """Fully anisotropic elastic constitutive model.

    The user provides the full 6x6 elasticity matrix ``C`` directly.
    All 21 independent elastic constants can be specified.

    Args:
        C: ``(6, 6)`` symmetric positive-definite elasticity matrix
            in Voigt notation. Units: Pa.
    """

    def __init__(self, C: torch.Tensor) -> None:
        if C.shape != (6, 6):
            raise ValueError(f"C must be 6x6, got {C.shape}")
        if not torch.allclose(C, C.T, atol=1e-6 * C.abs().max()):
            raise ValueError("C must be symmetric.")
        self._C = C.to(dtype=torch.float64)

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the 6x6 elasticity matrix ``C``."""
        return self._C.clone()

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """sigma = C : epsilon.

        Args:
            strain: ``(6,)`` or ``(n, 6)`` strain in Voigt notation.

        Returns:
            ``(6,)`` or ``(n, 6)`` stress in Voigt notation.
        """
        strain = strain.to(dtype=torch.float64)
        if strain.dim() == 1:
            return self._C @ strain
        return (self._C @ strain.T).T

    def strain(self, stress: torch.Tensor) -> torch.Tensor:
        """epsilon = C^{-1} : sigma.

        Args:
            stress: ``(6,)`` or ``(n, 6)`` stress in Voigt notation.

        Returns:
            ``(6,)`` or ``(n, 6)`` strain.
        """
        stress = stress.to(dtype=torch.float64)
        C_inv = torch.linalg.inv(self._C)
        if stress.dim() == 1:
            return C_inv @ stress
        return (C_inv @ stress.T).T

    def compliance_matrix(self) -> torch.Tensor:
        """Return the 6x6 compliance matrix S = C^{-1}."""
        return torch.linalg.inv(self._C)

    def __repr__(self) -> str:
        return f"AnisotropicElasticModel(C shape={self._C.shape})"


class OrthotropicElasticModel:
    """Orthotropic elastic model with 9 independent constants.

    Orthotropic materials have three orthogonal planes of symmetry.
    Common examples: wood, rolled metals, fibre-reinforced composites.

    The 9 independent constants are:

    - E1, E2, E3: Young's moduli along the three axes
    - nu23, nu13, nu12: Poisson's ratios
    - G23, G13, G12: Shear moduli

    The compliance matrix in Voigt notation is::

        S = [[1/E1,   -nu12/E1, -nu13/E1,  0,     0,     0    ],
             [-nu12/E1, 1/E2,   -nu23/E2,  0,     0,     0    ],
             [-nu13/E1, -nu23/E2, 1/E3,    0,     0,     0    ],
             [0,        0,        0,        1/G23, 0,     0    ],
             [0,        0,        0,        0,     1/G13, 0    ],
             [0,        0,        0,        0,     0,     1/G12]]

    Args:
        E1, E2, E3: Young's moduli (Pa).
        nu23, nu13, nu12: Poisson's ratios.
        G23, G13, G12: Shear moduli (Pa).
    """

    def __init__(
        self,
        E1: float = 210e9,
        E2: float = 210e9,
        E3: float = 210e9,
        nu23: float = 0.3,
        nu13: float = 0.3,
        nu12: float = 0.3,
        G23: float = 80.77e9,
        G13: float = 80.77e9,
        G12: float = 80.77e9,
    ) -> None:
        self._E = (E1, E2, E3)
        self._nu = (nu23, nu13, nu12)
        self._G = (G23, G13, G12)

        # Build compliance matrix
        S = torch.tensor([
            [1.0 / E1, -nu12 / E1, -nu13 / E1, 0.0, 0.0, 0.0],
            [-nu12 / E1, 1.0 / E2, -nu23 / E2, 0.0, 0.0, 0.0],
            [-nu13 / E1, -nu23 / E2, 1.0 / E3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / G23, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / G13, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / G12],
        ], dtype=torch.float64)

        self._S = S
        self._C = torch.linalg.inv(S)

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the 6x6 elasticity matrix ``C``."""
        return self._C.clone()

    @property
    def compliance_matrix(self) -> torch.Tensor:
        """Return the 6x6 compliance matrix ``S``."""
        return self._S.clone()

    @property
    def youngs_moduli(self) -> tuple[float, float, float]:
        """Return (E1, E2, E3)."""
        return self._E

    @property
    def poisson_ratios(self) -> tuple[float, float, float]:
        """Return (nu23, nu13, nu12)."""
        return self._nu

    @property
    def shear_moduli(self) -> tuple[float, float, float]:
        """Return (G23, G13, G12)."""
        return self._G

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """sigma = C : epsilon."""
        strain = strain.to(dtype=torch.float64)
        if strain.dim() == 1:
            return self._C @ strain
        return (self._C @ strain.T).T

    def strain(self, stress: torch.Tensor) -> torch.Tensor:
        """epsilon = S : sigma."""
        stress = stress.to(dtype=torch.float64)
        if stress.dim() == 1:
            return self._S @ stress
        return (self._S @ stress.T).T

    def __repr__(self) -> str:
        return (
            f"OrthotropicElasticModel(E=({self._E[0]:.2e}, "
            f"{self._E[1]:.2e}, {self._E[2]:.2e}))"
        )


class IsotropicPlasticModel:
    """Isotropic elastic model with linear isotropic hardening plasticity.

    Implements a simple return-mapping algorithm for J2 (von Mises)
    plasticity with linear isotropic hardening.

    The yield function is::

        f(sigma) = sigma_vm - (sigma_y + H * eq_plastic_strain)

    where:
    - sigma_vm is the von Mises equivalent stress
    - sigma_y is the initial yield stress
    - H is the hardening modulus
    - eq_plastic_strain is the accumulated equivalent plastic strain

    Args:
        youngs_modulus: Young's modulus (Pa).
        poisson_ratio: Poisson's ratio.
        yield_stress: Initial yield stress (Pa).
        hardening_modulus: Isotropic hardening modulus *H* (Pa).
    """

    def __init__(
        self,
        youngs_modulus: float = 210e9,
        poisson_ratio: float = 0.3,
        yield_stress: float = 250e6,
        hardening_modulus: float = 0.0,
    ) -> None:
        self._elastic = LinearElasticModel(youngs_modulus, poisson_ratio)
        self._yield_criterion = VonMisesYield(yield_stress)
        self._sigma_y = yield_stress
        self._H = hardening_modulus
        self._eq_plastic_strain: float = 0.0

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the 6x6 elastic elasticity matrix."""
        return self._elastic.elasticity_matrix

    @property
    def yield_stress(self) -> float:
        """Current yield stress (including hardening)."""
        return self._sigma_y + self._H * self._eq_plastic_strain

    @property
    def hardening_modulus(self) -> float:
        """Hardening modulus H."""
        return self._H

    @property
    def equivalent_plastic_strain(self) -> float:
        """Accumulated equivalent plastic strain."""
        return self._eq_plastic_strain

    def reset_plastic_strain(self) -> None:
        """Reset accumulated plastic strain to zero."""
        self._eq_plastic_strain = 0.0

    def stress_trial(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute trial (elastic) stress from total strain.

        Args:
            strain: ``(6,)`` total strain in Voigt notation.

        Returns:
            ``(6,)`` trial stress.
        """
        return self._elastic.stress(strain)

    def return_mapping(
        self, strain: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """Perform stress return-mapping for J2 plasticity.

        Uses the closed-form radial return to the yield surface.
        For isotropic hardening, the return is::

            sigma_corrected = (sigma_y_new / sigma_vm_trial) * sigma_dev_trial
            + sigma_hydro * I

        where sigma_y_new = sigma_y + H * delta_p and delta_p is the
        plastic multiplier.

        Args:
            strain: ``(6,)`` total strain in Voigt notation.

        Returns:
            Tuple of (corrected stress, is_plastic).
        """
        strain = strain.to(dtype=torch.float64)
        trial_stress = self._elastic.stress(strain)
        sigma_vm = self._yield_criterion.von_mises_stress(trial_stress)

        current_yield = self.yield_stress
        if sigma_vm <= current_yield:
            return trial_stress, False

        # Compute hydrostatic and deviatoric parts
        hydro = (trial_stress[0] + trial_stress[1] + trial_stress[2]) / 3.0
        dev = trial_stress.clone()
        dev[0] -= hydro
        dev[1] -= hydro
        dev[2] -= hydro

        # Shear modulus
        G = self._elastic.shear_modulus

        # Closed-form plastic multiplier:
        # delta_lambda = (sigma_vm - sigma_y) / (3G + H)
        delta_lambda = (sigma_vm - current_yield) / (3.0 * G + self._H)

        # Corrected stress:
        # Scale deviatoric part to new yield surface
        # sigma_new = (1 - 3G*delta_lambda/sigma_vm) * dev + hydro * I
        scale = 1.0 - 3.0 * G * delta_lambda / sigma_vm
        corrected = trial_stress.clone()
        corrected[0] = scale * dev[0] + hydro
        corrected[1] = scale * dev[1] + hydro
        corrected[2] = scale * dev[2] + hydro
        corrected[3] = scale * dev[3]
        corrected[4] = scale * dev[4]
        corrected[5] = scale * dev[5]

        # Update accumulated plastic strain
        self._eq_plastic_strain += delta_lambda

        return corrected, True

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress with plastic return-mapping.

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
            f"IsotropicPlasticModel(E={self._elastic.youngs_modulus:.2e}, "
            f"sigma_y={self._sigma_y:.2e}, H={self._H:.2e})"
        )
