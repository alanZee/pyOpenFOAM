"""
Enhanced elastic material models v3 with advanced constitutive behaviour.

Extends :class:`~pyfoam.structural.elastic_model_enhanced_2` with:

- :class:`OrthotropicPlasticModel` — orthotropic elasticity + Hill yield criterion
- :class:`ViscoelasticMaxwellModel` — Maxwell viscoelastic model for time-dependent behaviour
- :class:`DamageModel` — isotropic damage model with stiffness degradation

Usage::

    # Viscoelastic Maxwell
    model = ViscoelasticMaxwellModel(
        E_inf=1e9, E_1=5e8, eta_1=1e6,
    )
    stress = model.stress(strain, dt=0.001)

References
----------
- OpenFOAM ``mechanicalModel`` framework
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.elastic_model_enhanced import OrthotropicElasticModel
from pyfoam.structural.elastic_model_enhanced_2 import (
    TransverselyIsotropicModel,
    CombinedPlasticityModel,
)

__all__ = [
    "OrthotropicPlasticModel",
    "ViscoelasticMaxwellModel",
    "DamageModel",
]


class OrthotropicPlasticModel:
    """Orthotropic elasticity combined with Hill yield criterion.

    Extends orthotropic elasticity with a Hill-type yield surface that
    accounts for anisotropic yielding::

        f = F*(s22-s33)^2 + G*(s33-s11)^2 + H*(s11-s22)^2
           + 2*L*s23^2 + 2*M*s13^2 + 2*N*s12^2 - sigma_y^2

    where F, G, H, L, M, N are Hill parameters derived from
    directional yield stresses.

    Args:
        E1, E2, E3: Young's moduli along material axes (Pa).
        nu12, nu13, nu23: Poisson's ratios.
        G12, G13, G23: Shear moduli (Pa).
        yield_1, yield_2, yield_3: Yield stresses in each direction (Pa).
        yield_12, yield_13, yield_23: Shear yield stresses (Pa).
    """

    def __init__(
        self,
        E1: float = 210e9,
        E2: float = 210e9,
        E3: float = 210e9,
        nu12: float = 0.3,
        nu13: float = 0.3,
        nu23: float = 0.3,
        G12: float = 80.77e9,
        G13: float = 80.77e9,
        G23: float = 80.77e9,
        yield_1: float = 250e6,
        yield_2: float = 250e6,
        yield_3: float = 250e6,
        yield_12: float = 144.34e6,
        yield_13: float = 144.34e6,
        yield_23: float = 144.34e6,
    ) -> None:
        self._elastic = OrthotropicElasticModel(
            E1=E1, E2=E2, E3=E3,
            nu12=nu12, nu13=nu13, nu23=nu23,
            G12=G12, G13=G13, G23=G23,
        )
        self._sigma_y_ref = (yield_1 + yield_2 + yield_3) / 3.0

        # Hill parameters (normalised by reference yield stress)
        sy1, sy2, sy3 = yield_1, yield_2, yield_3
        sy12, sy13, sy23 = yield_12, yield_13, yield_23
        sy_ref_sq = self._sigma_y_ref ** 2

        self._F = 0.5 * (1.0 / sy2 ** 2 + 1.0 / sy3 ** 2 - 1.0 / sy1 ** 2)
        self._G = 0.5 * (1.0 / sy3 ** 2 + 1.0 / sy1 ** 2 - 1.0 / sy2 ** 2)
        self._H = 0.5 * (1.0 / sy1 ** 2 + 1.0 / sy2 ** 2 - 1.0 / sy3 ** 2)
        self._L = 0.5 / sy23 ** 2
        self._M = 0.5 / sy13 ** 2
        self._N = 0.5 / sy12 ** 2

        self._eq_plastic_strain: float = 0.0

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the 6x6 elastic stiffness matrix."""
        return self._elastic.elasticity_matrix

    @property
    def hill_parameters(self) -> dict:
        """Return Hill yield criterion parameters."""
        return {
            "F": self._F, "G": self._G, "H": self._H,
            "L": self._L, "M": self._M, "N": self._N,
        }

    @property
    def equivalent_plastic_strain(self) -> float:
        """Accumulated equivalent plastic strain."""
        return self._eq_plastic_strain

    def hill_yield_function(self, stress: torch.Tensor) -> float:
        """Evaluate the Hill yield function.

        Computes the Hill equivalent stress::

            sigma_hill = sqrt(F*(s22-s33)^2 + G*(s33-s11)^2 + H*(s11-s22)^2
                          + 2L*s23^2 + 2M*s13^2 + 2N*s12^2)

        Returns:
            Value of f = sigma_hill^2 - sigma_y_ref^2 (positive means yielding).
        """
        s = stress.to(dtype=torch.float64)
        s11, s22, s33 = s[0], s[1], s[2]
        s23, s13, s12 = s[3], s[4], s[5]

        sigma_hill_sq = (
            self._F * (s22 - s33) ** 2
            + self._G * (s33 - s11) ** 2
            + self._H * (s11 - s22) ** 2
            + 2.0 * self._L * s23 ** 2
            + 2.0 * self._M * s13 ** 2
            + 2.0 * self._N * s12 ** 2
        )
        return sigma_hill_sq.item() - 1.0

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress with Hill yield return-mapping.

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            ``(6,)`` stress.
        """
        strain = strain.to(dtype=torch.float64)
        trial = self._elastic.stress(strain)

        f = self.hill_yield_function(trial)
        if f <= 0:
            return trial

        # Scale trial stress to yield surface: sigma_hill_sq = f + 1.0
        # scale = 1/sqrt(sigma_hill_sq) brings stress to yield surface
        sigma_hill_sq = f + 1.0
        scale = 1.0 / torch.sqrt(
            torch.tensor(sigma_hill_sq, dtype=torch.float64)
        )
        corrected = trial * scale

        # Track plastic strain (approximate)
        delta_eps = max(0.0, strain.norm().item() * (1.0 - scale.item()))
        self._eq_plastic_strain += delta_eps

        return corrected

    def reset_state(self) -> None:
        """Reset plastic state."""
        self._eq_plastic_strain = 0.0

    def __repr__(self) -> str:
        E1 = self._elastic._E[0]
        return (
            f"OrthotropicPlasticModel(E1={E1:.2e}, "
            f"sigma_y={self._sigma_y_ref:.2e})"
        )


class ViscoelasticMaxwellModel:
    """Generalised Maxwell (Wiechert) viscoelastic model.

    Combines an equilibrium spring with ``n`` Maxwell elements
    (spring + dashpot in series)::

        sigma(t) = E_inf * eps(t) + sum_i { E_i * q_i(t) }
        q_i(t+dt) = exp(-dt/tau_i) * q_i(t) + (1 - exp(-dt/tau_i)) * eps(t)

    where tau_i = eta_i / E_i is the relaxation time.

    Args:
        E_inf: Long-term (equilibrium) modulus (Pa).
        E_1: Modulus of first Maxwell element (Pa).
        eta_1: Viscosity of first Maxwell element (Pa*s).
        E_2: Modulus of second Maxwell element (Pa, optional).
        eta_2: Viscosity of second Maxwell element (Pa*s, optional).
    """

    def __init__(
        self,
        E_inf: float = 1e9,
        E_1: float = 5e8,
        eta_1: float = 1e6,
        E_2: float = 0.0,
        eta_2: float = 1e6,
    ) -> None:
        self._E_inf = E_inf
        self._elements: list[tuple[float, float]] = [(E_1, eta_1)]
        if E_2 > 0:
            self._elements.append((E_2, eta_2))

        # Internal state: one relaxation variable per Maxwell element
        self._q = [0.0] * len(self._elements)

    @property
    def n_elements(self) -> int:
        """Number of Maxwell elements."""
        return len(self._elements)

    @property
    def relaxation_times(self) -> list[float]:
        """Return relaxation times tau_i = eta_i / E_i."""
        return [eta / E for E, eta in self._elements]

    def stress(
        self,
        strain: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute viscoelastic stress update.

        Args:
            strain: ``(6,)`` total strain in Voigt notation.
            dt: Time step (s).

        Returns:
            ``(6,)`` stress in Voigt notation.
        """
        strain = strain.to(dtype=torch.float64)

        # Equilibrium contribution (long-term response)
        stress = self._E_inf * strain

        # Maxwell elements: each element adds its own stress contribution
        # using exponential update of the relaxation variable
        for i, (E_i, eta_i) in enumerate(self._elements):
            tau_i = eta_i / max(E_i, 1e-30)
            exp_factor = torch.exp(
                torch.tensor(-dt / max(tau_i, 1e-30), dtype=torch.float64)
            ).item()

            # Update relaxation variable (memory of strain history)
            # At steady state, q -> strain, stress -> (E_inf + E_i) * strain
            self._q[i] = exp_factor * self._q[i] + (1.0 - exp_factor) * strain
            stress += E_i * self._q[i]

        return stress

    def reset_state(self) -> None:
        """Reset all relaxation variables."""
        self._q = [0.0] * len(self._elements)

    def __repr__(self) -> str:
        return (
            f"ViscoelasticMaxwellModel(E_inf={self._E_inf:.2e}, "
            f"n_elements={self.n_elements})"
        )


class DamageModel:
    """Isotropic damage model with stiffness degradation.

    The effective stress is computed as::

        sigma_eff = sigma / (1 - D)

    where D is the damage variable (0 = undamaged, 1 = fully damaged).
    Damage evolves as::

        D = 1 - exp(-eps / eps_d)

    where eps_d is the damage characteristic strain.

    Args:
        base_model: Base elastic model (e.g., LinearElasticModel).
        damage_strain: Characteristic strain for damage evolution.
        max_damage: Maximum allowable damage (0 < D < 1).
    """

    def __init__(
        self,
        base_model: LinearElasticModel,
        damage_strain: float = 0.01,
        max_damage: float = 0.99,
    ) -> None:
        self._model = base_model
        self._eps_d = damage_strain
        self._D_max = max_damage
        self._D: float = 0.0

    @property
    def damage(self) -> float:
        """Current damage variable."""
        return self._D

    @property
    def stiffness_reduction(self) -> float:
        """Current stiffness reduction factor (1 - D)."""
        return 1.0 - self._D

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the degraded 6x6 elasticity matrix."""
        return (1.0 - self._D) * self._model.elasticity_matrix

    def update_damage(self, strain: torch.Tensor) -> float:
        """Update damage based on current strain.

        Uses the equivalent strain (norm of strain vector) as the
        driving variable.

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            Updated damage value.
        """
        strain = strain.to(dtype=torch.float64)
        eps_eq = strain.norm().item()

        # Exponential damage evolution
        D_new = 1.0 - torch.exp(
            torch.tensor(-eps_eq / max(self._eps_d, 1e-30), dtype=torch.float64)
        ).item()

        # Damage can only grow
        self._D = max(self._D, min(D_new, self._D_max))
        return self._D

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute damaged stress.

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            ``(6,)`` degraded stress.
        """
        strain = strain.to(dtype=torch.float64)
        self.update_damage(strain)
        return (1.0 - self._D) * self._model.stress(strain)

    def reset_state(self) -> None:
        """Reset damage to zero."""
        self._D = 0.0

    def __repr__(self) -> str:
        return (
            f"DamageModel(eps_d={self._eps_d:.2e}, "
            f"D={self._D:.4f})"
        )
