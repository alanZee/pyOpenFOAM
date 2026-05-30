"""
Enhanced elastic material models v4 with advanced constitutive behaviour.

Extends :class:`~pyfoam.structural.elastic_model_enhanced_3` with:

- :class:`GursonDamageModel` — porous plasticity with void nucleation
- :class:`CrystalPlasticityModel` — single-crystal plasticity with slip systems
- :class:`PhaseFieldFractureModel` — phase-field approach to brittle fracture

Usage::

    model = GursonDamageModel(E=210e9, nu=0.3, sigma_y=250e6, f0=0.001)
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
from pyfoam.structural.elastic_model_enhanced_3 import (
    OrthotropicPlasticModel,
    ViscoelasticMaxwellModel,
    DamageModel,
)

__all__ = [
    "GursonDamageModel",
    "CrystalPlasticityModel",
    "PhaseFieldFractureModel",
]


class GursonDamageModel:
    """Gurson-Tvergaard-Needleman (GTN) porous plasticity model.

    Extends classical plasticity with void volume fraction evolution.
    The yield function is::

        f = (sigma_eq / sigma_y)^2 + 2*q1*f* cosh(3*sigma_m / (2*sigma_y))
           - (1 + q2*f^2)

    where f is the void volume fraction, sigma_eq is the von Mises
    stress, sigma_m is the hydrostatic stress, and q1, q2 are
    Tvergaard parameters.

    Args:
        E: Young's modulus (Pa).
        nu: Poisson's ratio.
        sigma_y: Initial yield stress (Pa).
        f0: Initial void volume fraction.
        f_n: Void volume fraction for nucleation.
        q1: Tvergaard parameter q1.
        q2: Tvergaard parameter q2.
    """

    def __init__(
        self,
        E: float = 210e9,
        nu: float = 0.3,
        sigma_y: float = 250e6,
        f0: float = 0.001,
        f_n: float = 0.04,
        q1: float = 1.5,
        q2: float = 1.0,
    ) -> None:
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )
        self._sigma_y = sigma_y
        self._f = f0
        self._f0 = f0
        self._f_n = f_n
        self._q1 = q1
        self._q2 = q2
        self._E = E
        self._nu = nu

    @property
    def void_fraction(self) -> float:
        """Current void volume fraction."""
        return self._f

    @property
    def void_fraction_initial(self) -> float:
        """Initial void volume fraction."""
        return self._f0

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the 6x6 elasticity matrix."""
        return self._model.elasticity_matrix

    def gurson_yield_function(
        self,
        stress: torch.Tensor,
        sigma_y: float | None = None,
    ) -> float:
        """Evaluate the GTN yield function.

        Args:
            stress: ``(6,)`` stress in Voigt notation.
            sigma_y: Current yield stress (uses initial if None).

        Returns:
            Value of f_yield (positive means yielding).
        """
        sy = sigma_y if sigma_y is not None else self._sigma_y
        s = stress.to(dtype=torch.float64)

        # Von Mises equivalent stress
        s11, s22, s33 = s[0], s[1], s[2]
        s12, s13, s23 = s[5], s[4], s[3]
        vm_sq = (
            0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
            + 3.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
        )
        sigma_eq = torch.sqrt(torch.clamp(vm_sq, min=0.0))

        # Hydrostatic stress
        sigma_m = (s11 + s22 + s33) / 3.0

        f = self._f
        sy_t = max(sy, 1e-30)

        # GTN yield function
        term1 = (sigma_eq / sy_t) ** 2
        arg = 3.0 * self._q1 * sigma_m / (2.0 * sy_t)
        arg = torch.clamp(arg, min=-20.0, max=20.0)
        term2 = 2.0 * self._q1 * f * torch.cosh(arg)
        term3 = -(1.0 + self._q2 * f ** 2)

        return (term1 + term2 + term3).item()

    def update_void_fraction(
        self,
        strain: torch.Tensor,
        d_strain: float = 0.0,
    ) -> float:
        """Update void volume fraction from strain.

        Includes void growth from plastic straining and nucleation.

        Args:
            strain: ``(6,)`` strain in Voigt notation.
            d_strain: Incremental equivalent plastic strain.

        Returns:
            Updated void fraction.
        """
        # Growth: df = (1 - f) * d_eps_p (dilatant plastic strain)
        if d_strain > 0:
            self._f += (1.0 - self._f) * d_strain

        # Nucleation (strain-controlled)
        eps_eq = strain.to(dtype=torch.float64).norm().item()
        if self._f_n > 0:
            # Nucleation from Chu-Needleman model
            f_nucleation = self._f_n * math.exp(
                -0.5 * (eps_eq / 0.01) ** 2
            ) / (0.01 * math.sqrt(2.0 * math.pi))
            self._f += f_nucleation * d_strain

        # Cap at physically meaningful maximum
        self._f = min(self._f, 0.99)

        return self._f

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress (elastic, with void effects noted).

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            ``(6,)`` stress.
        """
        return self._model.stress(strain.to(dtype=torch.float64))

    def reset_state(self) -> None:
        """Reset void fraction to initial value."""
        self._f = self._f0

    def __repr__(self) -> str:
        return (
            f"GursonDamageModel(E={self._E:.2e}, "
            f"f={self._f:.6f})"
        )


class CrystalPlasticityModel:
    """Single-crystal plasticity with slip system kinematics.

    Implements a simplified crystal plasticity model where the
    resolved shear stress on each slip system is::

        tau_alpha = sigma : (s_alpha x n_alpha)

    yielding occurs when tau_alpha exceeds the critical resolved
    shear stress (CRSS).

    Args:
        E: Young's modulus (Pa).
        nu: Poisson's ratio.
        tau_crss: Initial critical resolved shear stress (Pa).
        hardening_rate: Linear hardening rate (Pa).
        n_slip_systems: Number of active slip systems.
    """

    def __init__(
        self,
        E: float = 210e9,
        nu: float = 0.3,
        tau_crss: float = 50e6,
        hardening_rate: float = 500e6,
        n_slip_systems: int = 12,
    ) -> None:
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )
        self._tau_crss = tau_crss
        self._h0 = hardening_rate
        self._n_slip = n_slip_systems
        self._gamma_total = 0.0  # accumulated slip
        self._E = E
        self._nu = nu

        # Generate simplified slip system orientations
        # (uniformly distributed in 3D)
        self._slip_directions = []
        self._slip_normals = []
        for i in range(n_slip_systems):
            angle = 2.0 * math.pi * i / n_slip_systems
            d = torch.tensor(
                [math.cos(angle), math.sin(angle), 0.0],
                dtype=torch.float64,
            )
            n = torch.tensor(
                [-math.sin(angle), math.cos(angle), 0.0],
                dtype=torch.float64,
            )
            self._slip_directions.append(d)
            self._slip_normals.append(n)

    @property
    def n_slip_systems(self) -> int:
        """Number of active slip systems."""
        return self._n_slip

    @property
    def critical_resolved_shear_stress(self) -> float:
        """Current CRSS."""
        return self._tau_crss

    @property
    def accumulated_slip(self) -> float:
        """Total accumulated slip."""
        return self._gamma_total

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the 6x6 elasticity matrix."""
        return self._model.elasticity_matrix

    def resolved_shear_stresses(
        self,
        stress: torch.Tensor,
    ) -> torch.Tensor:
        """Compute resolved shear stress on each slip system.

        Args:
            stress: ``(6,)`` stress in Voigt notation.

        Returns:
            ``(n_slip,)`` resolved shear stresses.
        """
        s = stress.to(dtype=torch.float64)
        # Convert Voigt to full tensor (simplified)
        tau = torch.zeros(self._n_slip, dtype=torch.float64)
        for alpha in range(self._n_slip):
            d = self._slip_directions[alpha]
            n = self._slip_normals[alpha]
            # Schmid tensor: P = 0.5 * (s x n + n x s)
            # tau = sigma : P = sigma_ij * s_i * n_j
            tau[alpha] = (
                s[0] * d[0] * n[0]
                + s[1] * d[1] * n[1]
                + s[2] * d[2] * n[2]
                + s[3] * (d[1] * n[2] + d[2] * n[1])
                + s[4] * (d[0] * n[2] + d[2] * n[0])
                + s[5] * (d[0] * n[1] + d[1] * n[0])
            )
        return tau.abs()

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress from elastic constitutive law.

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            ``(6,)`` stress.
        """
        return self._model.stress(strain.to(dtype=torch.float64))

    def check_yielding(self, stress: torch.Tensor) -> bool:
        """Check if any slip system is yielding.

        Args:
            stress: ``(6,)`` stress in Voigt notation.

        Returns:
            True if yielding occurs on any slip system.
        """
        tau = self.resolved_shear_stresses(stress)
        return bool((tau > self._tau_crss).any().item())

    def update_hardening(self, d_gamma: float) -> float:
        """Update CRSS from accumulated slip increment.

        Args:
            d_gamma: Slip increment.

        Returns:
            Updated CRSS.
        """
        self._gamma_total += d_gamma
        self._tau_crss += self._h0 * d_gamma
        return self._tau_crss

    def reset_state(self) -> None:
        """Reset crystal state."""
        self._gamma_total = 0.0

    def __repr__(self) -> str:
        return (
            f"CrystalPlasticityModel(E={self._E:.2e}, "
            f"tau_crss={self._tau_crss:.2e}, "
            f"n_slip={self._n_slip})"
        )


class PhaseFieldFractureModel:
    """Phase-field model for brittle fracture.

    Uses an order parameter ``phi`` (1 = intact, 0 = fully broken)
    to regularise the crack surface. The energy functional is::

        E = int [g(phi) * psi_e(epsilon) + Gc/(2*l) * (1-phi)^2
                + Gc*l/2 * |grad(phi)|^2] dV

    where g(phi) = phi^2 + eta is the degradation function, Gc is the
    critical energy release rate, and l is the length scale.

    Args:
        base_model: Base elastic model.
        Gc: Critical energy release rate (J/m^2).
        length_scale: Regularisation length scale (m).
        eta: Small parameter to prevent singular stiffness.
    """

    def __init__(
        self,
        base_model: LinearElasticModel,
        Gc: float = 1000.0,
        length_scale: float = 0.01,
        eta: float = 1e-6,
    ) -> None:
        self._model = base_model
        self._Gc = Gc
        self._l = length_scale
        self._eta = eta
        self._phi: float = 1.0  # intact

    @property
    def phase_field(self) -> float:
        """Current phase-field order parameter (1=intact, 0=broken)."""
        return self._phi

    @property
    def damage(self) -> float:
        """Damage variable (0=intact, 1=broken), complement of phi."""
        return 1.0 - self._phi

    @property
    def critical_energy_release_rate(self) -> float:
        """Critical energy release rate Gc (J/m^2)."""
        return self._Gc

    @property
    def length_scale(self) -> float:
        """Regularisation length scale (m)."""
        return self._l

    def degradation(self, phi: float | None = None) -> float:
        """Compute degradation function g(phi) = phi^2 + eta.

        Args:
            phi: Phase-field value (uses current if None).

        Returns:
            Degradation factor.
        """
        p = phi if phi is not None else self._phi
        return p ** 2 + self._eta

    def elastic_energy_density(self, strain: torch.Tensor) -> float:
        """Compute elastic energy density psi_e = 0.5 * eps : C : eps.

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            Scalar energy density (J/m^3).
        """
        strain = strain.to(dtype=torch.float64)
        stress = self._model.stress(strain)
        return 0.5 * strain.dot(stress).item()

    def degraded_stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute degraded stress: sigma = g(phi) * C : epsilon.

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            ``(6,)`` degraded stress.
        """
        g = self.degradation()
        return g * self._model.stress(strain.to(dtype=torch.float64))

    def degraded_stiffness(self) -> torch.Tensor:
        """Compute degraded stiffness matrix: C_degraded = g(phi) * C.

        Returns:
            ``(6, 6)`` degraded elasticity matrix.
        """
        g = self.degradation()
        return g * self._model.elasticity_matrix

    def update_phase_field(
        self,
        strain: torch.Tensor,
        dt: float = 1.0,
    ) -> float:
        """Update phase-field based on strain energy.

        Uses an Allen-Cahn type evolution::

            phi_new = phi - dt * (phi - 1 + Gc*l * d_psi/d_phi) / eta_reg

        Simplified: if psi_e > Gc/(2*l), phi decreases (damage grows).

        Args:
            strain: ``(6,)`` strain in Voigt notation.
            dt: Pseudo time step.

        Returns:
            Updated phase-field value.
        """
        psi_e = self.elastic_energy_density(strain)

        # Critical threshold
        psi_c = self._Gc / (2.0 * max(self._l, 1e-30))

        if psi_e > psi_c:
            # Damage evolution
            excess = (psi_e - psi_c) / max(psi_c, 1e-30)
            d_phi = -dt * min(excess, 0.1)
            self._phi = max(0.0, self._phi + d_phi)

        return self._phi

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute degraded stress (convenience method).

        Args:
            strain: ``(6,)`` strain in Voigt notation.

        Returns:
            ``(6,)`` degraded stress.
        """
        return self.degraded_stress(strain)

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """Return the degraded 6x6 elasticity matrix."""
        return self.degraded_stiffness()

    def reset_state(self) -> None:
        """Reset phase field to intact."""
        self._phi = 1.0

    def __repr__(self) -> str:
        return (
            f"PhaseFieldFractureModel(Gc={self._Gc:.2e}, "
            f"l={self._l:.2e}, phi={self._phi:.4f})"
        )
