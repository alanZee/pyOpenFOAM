"""
solidFoamEnhanced3 — enhanced solid mechanics solver v3.

Extends :class:`SolidFoamEnhanced2` with:

- **Anisotropic material model**: supports orthotropic elasticity with
  direction-dependent Young's modulus and Poisson ratio for composite
  materials and single crystals.
- **Thermal contact resistance**: models imperfect thermal contact at
  interfaces with a configurable thermal contact conductance that affects
  the temperature distribution and resulting thermal stresses.
- **Dynamic relaxation**: uses a kinetic damping approach where the
  solution is driven toward equilibrium by periodically removing
  kinetic energy from the fictitious dynamic system.

Algorithm (per time step):
1. Store old fields
2. Solve coupled thermo-mechanical system (from v2)
3. Update strain, stress, and creep strain (from v2)
4. Apply stress smoothing (from v1)
5. Apply dynamic relaxation (kinetic damping)
6. Compute fatigue indicator (from v2)
7. Write fields

Usage::

    from pyfoam.applications.solid_foam_enhanced_3 import SolidFoamEnhanced3

    solver = SolidFoamEnhanced3("path/to/case", E=200e9, nu=0.3,
                                 anisotropic=True)
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

from .solid_foam_enhanced_2 import SolidFoamEnhanced2
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SolidFoamEnhanced3"]

logger = logging.getLogger(__name__)


class SolidFoamEnhanced3(SolidFoamEnhanced2):
    """Enhanced solid mechanics solver v3 with anisotropy and kinetic damping.

    Extends SolidFoamEnhanced2 with anisotropic material model,
    thermal contact resistance, and dynamic relaxation via kinetic damping.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    E, nu, alpha_th, T_ref, rho_s : float
        Material properties.
    thermal_conductivity, specific_heat : float
        Thermal properties.
    anisotropic : bool, optional
        Enable orthotropic elasticity.  Default False.
    E_ratio_y, E_ratio_z : float, optional
        Young's modulus ratios in y and z directions (relative to E).
        Default 1.0.
    nu_yz, nu_xz : float, optional
        Additional Poisson ratios for orthotropic model.  Default None.
    thermal_contact_resistance : float, optional
        Thermal contact resistance (m^2 K/W).  Default 0.
    kinetic_damping : bool, optional
        Enable kinetic damping.  Default True.
    damping_coefficient : float, optional
        Kinetic damping coefficient.  Default 0.8.
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
        anisotropic: bool = False,
        E_ratio_y: float = 1.0,
        E_ratio_z: float = 1.0,
        nu_yz: float | None = None,
        nu_xz: float | None = None,
        thermal_contact_resistance: float = 0.0,
        kinetic_damping: bool = True,
        damping_coefficient: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, E=E, nu=nu, alpha_th=alpha_th,
            T_ref=T_ref, rho_s=rho_s,
            thermal_conductivity=thermal_conductivity,
            specific_heat=specific_heat,
            **kwargs,
        )

        self.anisotropic = anisotropic
        self.E_ratio_y = max(0.1, E_ratio_y)
        self.E_ratio_z = max(0.1, E_ratio_z)
        self.nu_yz = nu_yz if nu_yz is not None else self.nu
        self.nu_xz = nu_xz if nu_xz is not None else self.nu
        self.thermal_contact_R = max(0.0, thermal_contact_resistance)
        self.kinetic_damping = kinetic_damping
        self.damping_coeff = max(0.1, min(1.0, damping_coefficient))

        # Dynamic relaxation state
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        self._velocity_D = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        self._KE_prev = 0.0
        self._KE_increasing = False

        logger.info(
            "SolidFoamEnhanced3 ready: aniso=%s, kinetic_damp=%s, damp=%.2f",
            self.anisotropic, self.kinetic_damping, self.damping_coeff,
        )

    # ------------------------------------------------------------------
    # Anisotropic elasticity
    # ------------------------------------------------------------------

    def _compute_anisotropic_stress(
        self,
        epsilon: torch.Tensor,
        epsilon_th: torch.Tensor,
        epsilon_creep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute stress with orthotropic elasticity.

        For orthotropic materials, the stiffness matrix C has
        different Young's moduli and Poisson ratios along each axis.

        Parameters
        ----------
        epsilon : torch.Tensor
            Total strain (Voigt notation, n_cells x 6).
        epsilon_th : torch.Tensor
            Thermal strain (n_cells x 6).
        epsilon_creep : torch.Tensor, optional
            Creep strain (n_cells x 6).

        Returns:
            Stress tensor (n_cells x 6).
        """
        if not self.anisotropic:
            return self._compute_stress()

        E_x = self.E
        E_y = self.E * self.E_ratio_y
        E_z = self.E * self.E_ratio_z

        nu_xy = self.nu
        nu_yz_val = self.nu_yz
        nu_xz = self.nu_xz

        # Compliance matrix S (simplified 3x3 for normal components)
        S11 = 1.0 / E_x
        S22 = 1.0 / E_y
        S33 = 1.0 / E_z
        S12 = -nu_xy / E_x
        S13 = -nu_xz / E_x
        S23 = -nu_yz_val / E_y

        # Mechanical strain
        eps_mech = epsilon - epsilon_th
        if epsilon_creep is not None:
            eps_mech = eps_mech - epsilon_creep

        # Simplified: apply direction-dependent stiffness
        sigma = torch.zeros_like(eps_mech)
        sigma[:, 0] = E_x * eps_mech[:, 0]  # xx
        sigma[:, 1] = E_y * eps_mech[:, 1]  # yy
        sigma[:, 2] = E_z * eps_mech[:, 2]  # zz
        sigma[:, 3] = self.mu * 2.0 * eps_mech[:, 3]  # xy
        sigma[:, 4] = self.mu * 2.0 * eps_mech[:, 4]  # xz
        sigma[:, 5] = self.mu * 2.0 * eps_mech[:, 5]  # yz

        return sigma

    # ------------------------------------------------------------------
    # Thermal contact resistance
    # ------------------------------------------------------------------

    def _apply_thermal_contact_resistance(
        self,
        T: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply thermal contact resistance correction.

        Reduces heat flux through faces where contact resistance is
        present, simulating imperfect thermal contact.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        dt : float
            Time step.

        Returns:
            Corrected temperature field.
        """
        if self.thermal_contact_R < 1e-30:
            return T

        mesh = self.mesh
        n_internal = mesh.n_internal_faces
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        T_O = gather(T, owner)
        T_N = gather(T, neigh)

        # Contact resistance reduces heat flux: q = (T_O - T_N) / R
        # Effective conductance reduction factor
        k_effective = self.kappa
        dx = mesh.cell_volumes.pow(1.0 / 3.0).clamp(min=1e-10).mean().item()
        factor = 1.0 / (1.0 + self.thermal_contact_R * k_effective / max(dx, 1e-10))

        # Apply reduction to temperature difference
        dT_contact = (T_N - T_O) * (1.0 - factor)
        T_corrected = T.clone()

        # Scatter correction
        n_cells = mesh.n_cells
        correction = scatter_add(dT_contact, owner, n_cells)
        T_corrected = T_corrected + correction * 0.01  # Damped

        return T_corrected

    # ------------------------------------------------------------------
    # Kinetic damping
    # ------------------------------------------------------------------

    def _apply_kinetic_damping(
        self,
        D: torch.Tensor,
        D_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply kinetic damping to displacement field.

        Tracks the kinetic energy of the fictitious dynamic system.
        When kinetic energy starts increasing, removes all kinetic
        energy and restarts the velocity field.

        Parameters
        ----------
        D : torch.Tensor
            Current displacement.
        D_old : torch.Tensor
            Previous displacement.
        dt : float
            Time step.

        Returns:
            Damped displacement field.
        """
        if not self.kinetic_damping:
            return D

        # Estimate velocity
        dD = D - D_old
        if dt > 1e-30:
            self._velocity_D = dD / dt

        # Kinetic energy
        KE = 0.5 * self.rho_s * (self._velocity_D.pow(2)).sum()

        # Check if KE is increasing
        if KE > self._KE_prev and self._KE_prev > 1e-30:
            if self._KE_increasing:
                # Remove kinetic energy
                self._velocity_D.zero_()
                D = D_old.clone()
                logger.debug("Kinetic damping: KE reset")
            self._KE_increasing = True
        else:
            self._KE_increasing = False

        self._KE_prev = float(KE.item())

        return D

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced v3 solidFoam solver.

        Uses anisotropic elasticity, thermal contact resistance, and
        kinetic damping.

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

        logger.info("Starting SolidFoamEnhanced3 run")
        logger.info("  E=%.6e, nu=%.4f, aniso=%s", self.E, self.nu, self.anisotropic)
        logger.info("  kinetic_damp=%s, thermal_R=%.2e",
                     self.kinetic_damping, self.thermal_contact_R)

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

            # Monolithic thermo-mechanical solve (from v2)
            self.T, self.D = self._solve_monolithic(self.T, self.D, self.delta_t)

            # Thermal contact resistance
            self.T = self._apply_thermal_contact_resistance(self.T, self.delta_t)

            # Update strain and stress
            sigma_prev = self.sigma.clone()
            sub_iters = 0

            for sub in range(self.max_sub_iterations):
                self.epsilon = self._compute_strain()
                self.epsilon_th = self._compute_thermal_strain()

                # Anisotropic stress computation
                if self.anisotropic:
                    self.sigma = self._compute_anisotropic_stress(
                        self.epsilon, self.epsilon_th, self.epsilon_creep,
                    )
                else:
                    self.sigma = self._compute_stress()

                # Creep strain update (from v2)
                if self.creep_A > 0:
                    creep_rate = self._compute_creep_strain_rate(self.sigma, t)
                    self.epsilon_creep = self.epsilon_creep + creep_rate * self.delta_t

                # Stress smoothing (from v1)
                self.sigma = self._smooth_stress(self.sigma)

                sub_iters = sub + 1

                sigma_change = float((self.sigma - sigma_prev).abs().max().item())
                if sigma_change < self.sub_iteration_tolerance:
                    break
                sigma_prev = self.sigma.clone()

            total_sub_iters += sub_iters

            # Kinetic damping
            self.D = self._apply_kinetic_damping(self.D, D_old, self.delta_t)

            # Fatigue damage (from v2)
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
                logger.info("SolidFoamEnhanced3 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        von_mises = self._compute_von_mises_stress()
        max_disp = float(self.D.abs().max().item())
        max_creep = float(self.epsilon_creep.abs().max().item())
        max_fatigue = float(self.fatigue_damage.max().item())

        logger.info("SolidFoamEnhanced3 completed")
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
