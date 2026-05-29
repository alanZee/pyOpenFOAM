"""
Elastic material models and yield criteria for structural mechanics.

Provides constitutive models for linear elastic materials and yield
criteria for plasticity assessment:

- :class:`LinearElasticModel` — Hooke's law: ``sigma = C : epsilon``
- :class:`VonMisesYield` — von Mises yield criterion

These models are used by :class:`StressSolver` and
:class:`DisplacementSolver` to compute stress and displacement fields.

Usage::

    model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
    stress = model.stress(strain)

    yield_criterion = VonMisesYield(yield_stress=250e6)
    is_yielding = yield_criterion.is_yielding(stress)
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["LinearElasticModel", "VonMisesYield"]


class LinearElasticModel:
    """Isotropic linear elastic constitutive model (Hooke's law).

    Computes the 6x6 elasticity matrix ``C`` for isotropic materials
    using Voigt notation (xx, yy, zz, yz, xz, xy) and provides
    stress computation via ``sigma = C : epsilon``.

    In OpenFOAM, the ``linearElastic`` mechanical model provides
    exactly this constitutive relationship for FSI solvers.

    Args:
        youngs_modulus: Young's modulus *E* (Pa).
        poisson_ratio: Poisson's ratio *nu* (dimensionless, < 0.5).
    """

    def __init__(
        self,
        youngs_modulus: float = 210e9,
        poisson_ratio: float = 0.3,
    ) -> None:
        if poisson_ratio >= 0.5 or poisson_ratio < 0.0:
            raise ValueError(
                f"Poisson's ratio must be in [0, 0.5), got {poisson_ratio}."
            )
        self._E = youngs_modulus
        self._nu = poisson_ratio
        self._C = self._build_elasticity_matrix()

    def _build_elasticity_matrix(self) -> torch.Tensor:
        """Build the 6x6 symmetric elasticity matrix in Voigt notation.

        For isotropic materials::

            C = E / ((1+nu)*(1-2*nu)) *
                [[1-nu,  nu,    nu,    0,       0,       0      ],
                 [nu,    1-nu,  nu,    0,       0,       0      ],
                 [nu,    nu,    1-nu,  0,       0,       0      ],
                 [0,     0,     0,     (1-2nu)/2, 0,     0      ],
                 [0,     0,     0,     0,   (1-2nu)/2,   0      ],
                 [0,     0,     0,     0,       0,   (1-2nu)/2 ]]
        """
        E = self._E
        nu = self._nu
        factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        d = 1.0 - nu
        g = (1.0 - 2.0 * nu) / 2.0

        C = torch.tensor([
            [d,    nu,   nu,   0.0,  0.0,  0.0],
            [nu,   d,    nu,   0.0,  0.0,  0.0],
            [nu,   nu,   d,    0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  g,    0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  g,    0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  g  ],
        ], dtype=torch.float64) * factor
        return C

    @property
    def youngs_modulus(self) -> float:
        return self._E

    @property
    def poisson_ratio(self) -> float:
        return self._nu

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the 6x6 elasticity matrix ``C`` (Voigt notation)."""
        return self._C.clone()

    @property
    def shear_modulus(self) -> float:
        """G = E / (2 * (1 + nu))."""
        return self._E / (2.0 * (1.0 + self._nu))

    @property
    def bulk_modulus(self) -> float:
        """K = E / (3 * (1 - 2*nu))."""
        return self._E / (3.0 * (1.0 - 2.0 * self._nu))

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress from strain via Hooke's law.

        Args:
            strain: ``(6,)`` strain vector in Voigt notation
                ``(eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy)``
                or ``(n, 6)`` batch.

        Returns:
            ``(6,)`` or ``(n, 6)`` stress vector in Voigt notation.
        """
        strain = strain.to(dtype=torch.float64)
        if strain.dim() == 1:
            return self._C @ strain
        return (self._C @ strain.T).T

    def strain(self, stress: torch.Tensor) -> torch.Tensor:
        """Compute strain from stress (inverse Hooke's law).

        Args:
            stress: ``(6,)`` or ``(n, 6)`` stress in Voigt notation.

        Returns:
            ``(6,)`` or ``(n, 6)`` strain in Voigt notation.
        """
        stress = stress.to(dtype=torch.float64)
        C_inv = torch.linalg.inv(self._C)
        if stress.dim() == 1:
            return C_inv @ stress
        return (C_inv @ stress.T).T

    def __repr__(self) -> str:
        return (
            f"LinearElasticModel(E={self._E:.3e}, nu={self._nu})"
        )


class VonMisesYield:
    """von Mises yield criterion.

    Computes the equivalent (von Mises) stress from a stress tensor
    in Voigt notation and checks yielding against a specified yield
    stress.

    In OpenFOAM, the ``vonMises`` yield criterion is used in
    structural solvers for plasticity assessment::

        sigma_vm = sqrt(0.5 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2))

    where ``s1, s2, s3`` are principal stresses.

    For the 6-component Voigt stress vector
    ``(sigma_xx, sigma_yy, sigma_zz, tau_yz, tau_xz, tau_xy)``::

        sigma_vm = sqrt(
            0.5 * ((sigma_xx - sigma_yy)^2 + (sigma_yy - sigma_zz)^2 +
                   (sigma_zz - sigma_xx)^2) +
            3 * (tau_yz^2 + tau_xz^2 + tau_xy^2)
        )

    Args:
        yield_stress: Material yield stress *sigma_y* (Pa).
    """

    def __init__(self, yield_stress: float = 250e6) -> None:
        self._sigma_y = yield_stress

    @property
    def yield_stress(self) -> float:
        return self._sigma_y

    def von_mises_stress(self, stress: torch.Tensor) -> torch.Tensor:
        """Compute von Mises equivalent stress.

        Args:
            stress: ``(6,)`` stress in Voigt notation, or ``(n, 6)`` batch.

        Returns:
            Scalar von Mises stress, or ``(n,)`` batch.
        """
        stress = stress.to(dtype=torch.float64)
        if stress.dim() == 1:
            return self._vm_single(stress)
        return torch.stack([self._vm_single(s) for s in stress])

    @staticmethod
    def _vm_single(s: torch.Tensor) -> torch.Tensor:
        """Von Mises stress for a single 6-component Voigt vector."""
        sx, sy, sz = s[0], s[1], s[2]
        tyz, txz, txy = s[3], s[4], s[5]
        vm_sq = (
            0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
            + 3.0 * (tyz ** 2 + txz ** 2 + txy ** 2)
        )
        return torch.sqrt(torch.clamp(vm_sq, min=0.0))

    def is_yielding(self, stress: torch.Tensor) -> torch.Tensor:
        """Check if the stress state exceeds the yield stress.

        Args:
            stress: ``(6,)`` or ``(n, 6)`` stress in Voigt notation.

        Returns:
            Boolean tensor — ``True`` where von Mises stress > yield stress.
        """
        vm = self.von_mises_stress(stress)
        return vm > self._sigma_y

    def safety_factor(self, stress: torch.Tensor) -> torch.Tensor:
        """Compute the safety factor against yielding.

        safety_factor = sigma_y / sigma_vm

        Args:
            stress: ``(6,)`` or ``(n, 6)`` stress in Voigt notation.

        Returns:
            Safety factor (scalar or ``(n,)``).  Values > 1 mean safe.
        """
        vm = self.von_mises_stress(stress)
        return self._sigma_y / torch.clamp(vm, min=1e-30)

    def __repr__(self) -> str:
        return f"VonMisesYield(sigma_y={self._sigma_y:.3e})"
