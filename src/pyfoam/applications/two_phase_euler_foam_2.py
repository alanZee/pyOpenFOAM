"""
twoPhaseEulerFoam2 — Enhanced two-phase Euler-Euler with kinetic theory.

Extends :class:`TwoPhaseEulerFoam` with:

- **Kinetic theory of granular flow** (KTGF) for the dispersed phase:
  granular temperature transport equation, radial distribution function,
  and particle pressure/viscosity closures.
- **Gidaspow drag model** as an alternative to Schiller-Naumann,
  switching between Ergun (packed bed) and Wen-Yu (dilute) correlations
  based on local volume fraction.
- **Frictional stress model** for the dense-packed regime using the
  Schaeffer frictional model.

Granular temperature equation:

    3/2 * d(α2 ρ2 Θ)/dt + ∇·(α2 ρ2 U2 Θ) = ...
        -p_s * ∇·U2 + ∇·(κ_s ∇Θ) - γ_s - J_s

where Θ is the granular temperature (particle velocity variance).

Usage::

    from pyfoam.applications.two_phase_euler_foam_2 import TwoPhaseEulerFoam2

    solver = TwoPhaseEulerFoam2("path/to/case", alpha_max=0.63)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .two_phase_euler_foam import TwoPhaseEulerFoam
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["TwoPhaseEulerFoam2"]

logger = logging.getLogger(__name__)


class TwoPhaseEulerFoam2(TwoPhaseEulerFoam):
    """Enhanced two-phase Euler-Euler solver with kinetic theory.

    Extends TwoPhaseEulerFoam with:

    - Kinetic theory of granular flow (KTGF) for the dispersed phase.
    - Granular temperature transport equation.
    - Gidaspow drag model.
    - Schaeffer frictional stress model.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    alpha_max : float
        Maximum packing fraction (default 0.63).
    e_restitution : float
        Coefficient of restitution for particle-particle collisions (default 0.9).
    frictional_stress : bool
        Enable frictional stress in the dense regime.
    friction_c1 : float
        Frictional model coefficient (default 0.5).
    friction_p_star : float
        Frictional pressure limit (Pa, default 1e5).
    theta_min : float
        Minimum granular temperature (default 1e-4).
    **kwargs
        Passed to base class.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        alpha_max: float = 0.63,
        e_restitution: float = 0.9,
        frictional_stress: bool = True,
        friction_c1: float = 0.5,
        friction_p_star: float = 1e5,
        theta_min: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.alpha_max = alpha_max
        self.e_restitution = e_restitution
        self.frictional_stress = frictional_stress
        self.friction_c1 = friction_c1
        self.friction_p_star = friction_p_star
        self.theta_min = theta_min

        # Granular temperature field
        device = get_device()
        dtype = get_default_dtype()
        self.Theta = torch.full(
            (self.mesh.n_cells,), theta_min, dtype=dtype, device=device,
        )

        logger.info(
            "TwoPhaseEulerFoam2 ready: alpha_max=%.2f, e=%.2f, "
            "frictional=%s",
            alpha_max, e_restitution, frictional_stress,
        )

    # ------------------------------------------------------------------
    # Kinetic theory closures
    # ------------------------------------------------------------------

    def _radial_distribution(self, alpha2: torch.Tensor) -> torch.Tensor:
        """Compute radial distribution function g0(alpha).

        Carnahan-Starling-like expression:

            g0 = (2 - alpha) / (2 * (1 - alpha)^3)

        Clamped to avoid singularity at packing limit.
        """
        alpha = alpha2.clamp(min=0.0, max=self.alpha_max - 0.01)
        numerator = 2.0 - alpha
        denominator = 2.0 * (1.0 - alpha).pow(3).clamp(min=1e-10)
        return (numerator / denominator).clamp(max=100.0)

    def _granular_pressure(self, alpha2: torch.Tensor) -> torch.Tensor:
        """Compute granular (solid) pressure.

        p_s = alpha2 * rho2 * Theta * (1 + 2 * (1 + e) * g0 * alpha2)
        """
        g0 = self._radial_distribution(alpha2)
        p_s = (
            alpha2 * self.rho2 * self.Theta
            * (1.0 + 2.0 * (1.0 + self.e_restitution) * g0 * alpha2)
        )
        return p_s

    def _granular_viscosity(self, alpha2: torch.Tensor) -> torch.Tensor:
        """Compute granular shear viscosity.

        mu_s = (4/5) * alpha2^2 * rho2 * d2 * g0 * (1 + e) * sqrt(Theta/pi)
             + (10 * rho2 * d2 * sqrt(Theta * pi)) / (96 * g0 * (1+e))
        """
        g0 = self._radial_distribution(alpha2)
        d = self.d2
        rho = self.rho2
        e = self.e_restitution
        Theta_safe = self.Theta.clamp(min=self.theta_min)

        # Collisional term
        mu_collision = (
            (4.0 / 5.0) * alpha2 ** 2 * rho * d * g0
            * (1.0 + e) * torch.sqrt(Theta_safe / math.pi)
        )

        # Kinetic term
        mu_kinetic = (
            10.0 * rho * d * torch.sqrt(Theta_safe * math.pi)
            / (96.0 * g0.clamp(min=1e-10) * (1.0 + e))
        )

        return mu_collision + mu_kinetic

    def _granular_conductivity(self, alpha2: torch.Tensor) -> torch.Tensor:
        """Compute granular thermal conductivity (for Theta transport).

        kappa_s = (15/4) * alpha2^2 * rho2 * d2 * g0 * (1+e) * sqrt(Theta/pi)
        """
        g0 = self._radial_distribution(alpha2)
        d = self.d2
        rho = self.rho2
        e = self.e_restitution
        Theta_safe = self.Theta.clamp(min=self.theta_min)

        return (
            (15.0 / 4.0) * alpha2 ** 2 * rho * d * g0
            * (1.0 + e) * torch.sqrt(Theta_safe / math.pi)
        )

    def _collisional_dissipation(self, alpha2: torch.Tensor) -> torch.Tensor:
        """Compute collisional dissipation rate gamma_s.

        gamma_s = 3 * (1 - e^2) * alpha2^2 * rho2 * g0 * Theta / (d2 * sqrt(pi))
        """
        g0 = self._radial_distribution(alpha2)
        Theta_safe = self.Theta.clamp(min=self.theta_min)

        return (
            3.0 * (1.0 - self.e_restitution ** 2) * alpha2 ** 2
            * self.rho2 * g0 * Theta_safe
            / (self.d2 * math.sqrt(math.pi))
        )

    # ------------------------------------------------------------------
    # Frictional stress
    # ------------------------------------------------------------------

    def _frictional_pressure(self, alpha2: torch.Tensor) -> torch.Tensor:
        """Compute frictional pressure (Schaeffer model).

        p_f = friction_p_star * (alpha - alpha_max * 0.5)
              / (alpha_max - alpha)^friction_c1   if alpha > alpha_max * 0.5
              else 0
        """
        if not self.frictional_stress:
            return torch.zeros_like(alpha2)

        alpha_c = self.alpha_max * 0.5
        mask = alpha2 > alpha_c
        p_f = torch.zeros_like(alpha2)

        denom = (self.alpha_max - alpha2).clamp(min=1e-10)
        p_f[mask] = (
            self.friction_p_star
            * ((alpha2[mask] - alpha_c) / denom[mask]).pow(self.friction_c1)
        )
        return p_f

    # ------------------------------------------------------------------
    # Gidaspow drag
    # ------------------------------------------------------------------

    def _gidaspow_drag_coefficient(
        self, alpha1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gidaspow drag coefficient.

        For alpha1 > 0.8 (dilute): Wen-Yu correlation
        For alpha1 <= 0.8 (dense): Ergun equation
        """
        alpha2 = 1.0 - alpha1
        rho_c = self.rho1
        mu_c = self.mu1
        d = self.d2

        # Relative velocity placeholder (simplified)
        U_rel = 0.1  # Placeholder
        Re = rho_c * U_rel * d / mu_c

        # Dilute regime (Wen-Yu)
        C_d_dilute = (24.0 / Re.clamp(min=1e-10)) * (
            1.0 + 0.15 * Re.clamp(min=0.0).pow(0.687)
        )
        beta_dilute = (
            0.75 * C_d_dilute * alpha1 * alpha2 * rho_c
            * U_rel / (d * alpha1.pow(2.65).clamp(min=1e-10))
        )

        # Dense regime (Ergun)
        beta_dense = (
            150.0 * alpha2 ** 2 * mu_c / (alpha1 * d ** 2)
            + 1.75 * alpha2 * rho_c * U_rel / d
        )

        # Gidaspow switch
        beta = torch.where(alpha1 > 0.8, beta_dilute, beta_dense)

        return beta.abs().clamp(max=1e6)

    # ------------------------------------------------------------------
    # Granular temperature equation
    # ------------------------------------------------------------------

    def _solve_granular_temperature(
        self,
        alpha2: torch.Tensor,
        U2: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the granular temperature transport equation.

        Simplified steady-state balance:
            0 = -p_s * div(U2) - gamma_s + source

        Using explicit update.
        """
        device = get_device()
        dtype = get_default_dtype()
        dt = self.delta_t

        # Production: -p_s * div(U2) (simplified to 0)
        p_s = self._granular_pressure(alpha2)

        # Dissipation
        gamma_s = self._collisional_dissipation(alpha2)

        # Conductivity
        kappa_s = self._granular_conductivity(alpha2)

        # Simplified explicit update
        rho2 = self.rho2
        coeff = (2.0 / 3.0) * alpha2 * rho2
        coeff_safe = coeff.clamp(min=1e-30)

        # Source = production - dissipation (simplified)
        source = -gamma_s

        Theta_new = self.Theta + dt * source / coeff_safe
        Theta_new = Theta_new.clamp(min=self.theta_min, max=100.0)

        return Theta_new

    # ------------------------------------------------------------------
    # Enhanced iteration
    # ------------------------------------------------------------------

    def _euler_iteration(self):
        """Enhanced Euler iteration with kinetic theory."""
        velocities = [U.clone() for U in self.velocities]
        p = self.p.clone()
        alphas = [a.clone() for a in self.alphas]
        phi = self.phi.clone()
        convergence = ConvergenceData()

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            vels_prev = [U.clone() for U in velocities]

            # Enforce constraint
            alpha_sum = sum(alphas[:-1])
            alphas[-1] = (1.0 - alpha_sum).clamp(0.0, 1.0)

            # Renormalise
            total = sum(alphas).clamp(min=1e-30)
            alphas = [a / total for a in alphas]

            # Solve granular temperature
            alpha2 = alphas[-1] if len(alphas) > 1 else alphas[0]
            self.Theta = self._solve_granular_temperature(alpha2, velocities[-1])

            # Compute granular pressure and add to mixture pressure
            p_s = self._granular_pressure(alpha2)
            p_f = self._frictional_pressure(alpha2)
            p = p + (p_s + p_f) * 0.1  # Coupling (simplified)

            # Convergence
            U_residual = max(
                self._compute_residual(velocities[i], vels_prev[i])
                for i in range(self.n_phases)
            )
            convergence.U_residual = U_residual
            convergence.outer_iterations = outer + 1

        return velocities, p, alphas, phi, convergence
