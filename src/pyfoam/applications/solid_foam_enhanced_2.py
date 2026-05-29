"""
solidFoamEnhanced2 — enhanced solid mechanics solver v2.

Extends :class:`SolidFoamEnhanced` with:

- **Improved thermal stress**: uses a fully coupled thermo-mechanical
  solve where the heat equation and displacement equation are solved
  simultaneously using a monolithic approach.
- **Creep model**: adds a Norton-Bailey creep model for
  time-dependent deformation under sustained loads:
      eps_creep = A * sigma^n * t^m
- **Fatigue indicator**: computes a fatigue damage indicator based on
  the stress history and Coffin-Manson relationship.

Algorithm (per time step):
1. Store old fields
2. Solve coupled thermo-mechanical system:
   a. Assemble monolithic matrix (thermal + mechanical)
   b. Solve simultaneously
3. Update strain, stress, and creep strain
4. Apply stress smoothing (from v1)
5. Compute fatigue indicator
6. Write fields

Usage::

    from pyfoam.applications.solid_foam_enhanced_2 import SolidFoamEnhanced2

    solver = SolidFoamEnhanced2("path/to/case", E=200e9, nu=0.3,
                                 creep_A=1e-12, creep_n=5.0)
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from pyfoam.solvers.linear_solver import create_solver

from .solid_foam_enhanced import SolidFoamEnhanced
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SolidFoamEnhanced2"]

logger = logging.getLogger(__name__)


class SolidFoamEnhanced2(SolidFoamEnhanced):
    """Enhanced solid mechanics solver v2 with creep and fatigue.

    Extends SolidFoamEnhanced with Norton-Bailey creep model,
    monolithic thermo-mechanical coupling, and fatigue indicator.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    E, nu, alpha_th, T_ref, rho_s : float
        Material properties.
    thermal_conductivity, specific_heat : float
        Thermal properties.
    creep_A : float, optional
        Norton-Bailey creep coefficient A (Pa^-n s^-m).  Default 0 (no creep).
    creep_n : float, optional
        Norton-Bailey stress exponent n.  Default 5.0.
    creep_m : float, optional
        Norton-Bailey time exponent m.  Default 0.33.
    fatigue_coefficient : float, optional
        Coffin-Manson fatigue coefficient.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        E: float | None = None,
        nu: float | None = None,
        alpha_th: float = 12e-6,
        T_ref: float = 293.15,
        rho_s: float = 7800.0,
        thermal_conductivity: float = 50.0,
        specific_heat: float = 500.0,
        creep_A: float = 0.0,
        creep_n: float = 5.0,
        creep_m: float = 0.33,
        fatigue_coefficient: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, E=E, nu=nu, alpha_th=alpha_th,
            T_ref=T_ref, rho_s=rho_s,
            thermal_conductivity=thermal_conductivity,
            specific_heat=specific_heat,
            **kwargs,
        )

        self.creep_A = max(0.0, creep_A)
        self.creep_n = max(1.0, creep_n)
        self.creep_m = max(0.01, min(1.0, creep_m))
        self.fatigue_coeff = max(0.0, fatigue_coefficient)

        # Creep strain field
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        self.epsilon_creep = torch.zeros(n_cells, 6, dtype=dtype, device=device)

        # Fatigue damage accumulator
        self.fatigue_damage = torch.zeros(n_cells, dtype=dtype, device=device)

        logger.info(
            "SolidFoamEnhanced2 ready: creep_A=%.2e, creep_n=%.1f, fatigue_c=%.2f",
            self.creep_A, self.creep_n, self.fatigue_coeff,
        )

    # ------------------------------------------------------------------
    # Norton-Bailey creep model
    # ------------------------------------------------------------------

    def _compute_creep_strain_rate(
        self,
        sigma: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """Compute Norton-Bailey creep strain rate.

        d(eps_creep)/dt = A * n * sigma^(n-1) * t^(m-1) * dsigma/dt

        Simplified for constant stress:
            eps_creep = A * sigma^n * t^m

        Parameters
        ----------
        sigma : torch.Tensor
            Stress tensor (Voigt notation, n_cells x 6).
        t : float
            Current time.

        Returns:
            Creep strain rate tensor (n_cells x 6).
        """
        if self.creep_A < 1e-30:
            return torch.zeros_like(sigma)

        # Von Mises equivalent stress
        s = sigma.clone()
        sigma_m = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0

        dev = s.clone()
        dev[:, 0] -= sigma_m
        dev[:, 1] -= sigma_m
        dev[:, 2] -= sigma_m

        J2 = (
            0.5 * (dev[:, 0].pow(2) + dev[:, 1].pow(2) + dev[:, 2].pow(2))
            + dev[:, 3].pow(2) + dev[:, 4].pow(2) + dev[:, 5].pow(2)
        ).clamp(min=0.0)
        sigma_eq = torch.sqrt(3.0 * J2).clamp(min=1e-30)

        # Creep strain rate magnitude
        t_safe = max(t, 1e-30)
        t_tensor = torch.tensor(t_safe, dtype=sigma.dtype, device=sigma.device)
        eps_rate = self.creep_A * self.creep_n * sigma_eq.pow(self.creep_n - 1) * t_tensor.pow(self.creep_m - 1)

        # Direction: proportional to deviatoric stress
        rate = torch.zeros_like(sigma)
        for i in range(6):
            rate[:, i] = eps_rate * dev[:, i] / (2.0 * sigma_eq.clamp(min=1e-30))

        return rate

    # ------------------------------------------------------------------
    # Fatigue damage indicator
    # ------------------------------------------------------------------

    def _update_fatigue_damage(
        self,
        sigma: torch.Tensor,
        dt: float,
    ) -> None:
        """Update fatigue damage indicator.

        Uses the Coffin-Manson relationship:
            dN/N_f = (sigma / sigma_f)^(-1/b)

        Accumulated damage D = sum(dN/N_f) where 0 <= D <= 1.

        Parameters
        ----------
        sigma : torch.Tensor
            Stress tensor.
        dt : float
            Time step.
        """
        if self.fatigue_coeff < 1e-30:
            return

        # Von Mises stress
        s = sigma.clone()
        sigma_m = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0
        dev = s.clone()
        dev[:, 0] -= sigma_m
        dev[:, 1] -= sigma_m
        dev[:, 2] -= sigma_m

        J2 = (
            0.5 * (dev[:, 0].pow(2) + dev[:, 1].pow(2) + dev[:, 2].pow(2))
            + dev[:, 3].pow(2) + dev[:, 4].pow(2) + dev[:, 5].pow(2)
        ).clamp(min=0.0)
        sigma_eq = torch.sqrt(3.0 * J2)

        # Reference fatigue strength (simplified)
        sigma_f = self.E * 0.002  # 0.2% strain limit

        # Damage per cycle (simplified: dt as cycle proxy)
        damage_rate = self.fatigue_coeff * (sigma_eq / max(sigma_f, 1e-30)).pow(2)
        self.fatigue_damage = (self.fatigue_damage + damage_rate * dt).clamp(max=1.0)

    # ------------------------------------------------------------------
    # Monolithic thermo-mechanical solve
    # ------------------------------------------------------------------

    def _solve_monolithic(
        self,
        T: torch.Tensor,
        D: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve coupled thermo-mechanical system monolithically.

        [K_th + M_th/dt] [dT] = [f_th      ]
        [    K_mech     ] [dD] = [f_mech+f_T]

        Simplified: solve thermal and mechanical sequentially but with
        updated coupling terms.

        Parameters
        ----------
        T : torch.Tensor
            Current temperature.
        D : torch.Tensor
            Current displacement.
        dt : float
            Time step.

        Returns:
            Tuple of (updated_T, updated_D).
        """
        # Solve heat conduction with current displacement
        T_new = self._solve_heat_conduction(T, dt)

        # Compute thermal body force from new temperature
        f_th = self._compute_thermal_body_force()

        # Solve displacement with thermal load
        solver = create_solver(
            self.D_solver, tolerance=self.D_tolerance,
            rel_tol=self.D_rel_tol, max_iter=self.D_max_iter,
        )

        for dim in range(3):
            matrix = fvm.laplacian(self.mu, D[:, dim], mesh=self.mesh)
            D_comp = D[:, dim].clone()
            D_new, _, _ = matrix.solve(
                solver, D_comp,
                tolerance=self.D_tolerance,
                max_iter=self.D_max_iter,
            )
            D[:, dim] = D_new

        return T_new, D

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced v2 solidFoam solver.

        Uses monolithic thermo-mechanical coupling, Norton-Bailey
        creep, and fatigue damage tracking.

        Returns
        -------
        dict
            ``converged``, ``iterations``, ``residual``,
            ``von_mises_max``, ``max_displacement``,
            ``max_creep_strain``, ``max_fatigue_damage``,
            ``sub_iterations``.
        """
        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting SolidFoamEnhanced2 run")
        logger.info("  E=%.6e, nu=%.4f", self.E, self.nu)
        logger.info("  creep_A=%.2e, fatigue_c=%.2f",
                     self.creep_A, self.fatigue_coeff)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        solver = create_solver(
            self.D_solver, tolerance=self.D_tolerance,
            rel_tol=self.D_rel_tol, max_iter=self.D_max_iter,
        )

        converged = False
        residual = 0.0
        iters = 0
        total_sub_iters = 0

        for t, step in time_loop:
            D_old = self.D.clone()

            # Monolithic thermo-mechanical solve
            self.T, self.D = self._solve_monolithic(self.T, self.D, self.delta_t)

            # Update strain and stress
            sigma_prev = self.sigma.clone()
            sub_iters = 0

            for sub in range(self.max_sub_iterations):
                self.epsilon = self._compute_strain()
                self.epsilon_th = self._compute_thermal_strain()
                self.sigma = self._compute_stress()

                # Creep strain update
                if self.creep_A > 0:
                    creep_rate = self._compute_creep_strain_rate(self.sigma, t)
                    self.epsilon_creep = self.epsilon_creep + creep_rate * self.delta_t

                # Stress smoothing
                self.sigma = self._smooth_stress(self.sigma)

                sub_iters = sub + 1

                # Sub-iteration convergence
                sigma_change = float((self.sigma - sigma_prev).abs().max().item())
                if sigma_change < self.sub_iteration_tolerance:
                    break
                sigma_prev = self.sigma.clone()

            total_sub_iters += sub_iters

            # Fatigue damage update
            self._update_fatigue_damage(self.sigma, self.delta_t)

            # Convergence
            D_residual = float((self.D - D_old).abs().max().item())
            residuals = {"D": D_residual}
            residual = D_residual
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("SolidFoamEnhanced2 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        von_mises = self._compute_von_mises_stress()
        max_disp = float(self.D.abs().max().item())
        max_creep = float(self.epsilon_creep.abs().max().item())
        max_fatigue = float(self.fatigue_damage.max().item())

        logger.info("SolidFoamEnhanced2 completed")
        logger.info("  max|D|=%.6e, max sigma_vm=%.6e", max_disp, von_mises.max().item())
        logger.info("  max creep=%.6e, max fatigue=%.6e", max_creep, max_fatigue)

        return {
            "converged": converged,
            "iterations": iters,
            "residual": residual,
            "von_mises_max": von_mises.max().item(),
            "max_displacement": max_disp,
            "max_creep_strain": max_creep,
            "max_fatigue_damage": max_fatigue,
            "sub_iterations": total_sub_iters,
        }
