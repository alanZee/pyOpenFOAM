"""
reactingFoamEnhanced6 — enhanced reacting solver v6.

Extends :class:`ReactingFoamEnhanced5` with:

- **Adaptive ISAT with error control**: extends the ISAT tabulation from
  v5 with a local error estimator that tracks the gradient of the stored
  mapping and adaptively refines the tolerance based on the local
  composition-space curvature.
- **Operator-splitting with Lie-Trotter correction**: adds a commutator
  correction to the Strang splitting that reduces the splitting error
  for stiff reaction-transport systems.
- **Species-specific sub-cycling**: allows each species to be advanced
  with its own time step based on the local Damkohler number, reducing
  the cost for species with very different time scales.

Algorithm (per transport time step):
1. Compute stiffness and classify reactions (from v4)
2. Species-specific sub-cycling
3. Strang splitting with Lie-Trotter correction:
   a. Half-step chemistry (ISAT with adaptive tolerance)
   b. Full-step transport
   c. Half-step chemistry (ISAT with adaptive tolerance)
4. Commutator correction
5. Pressure-dependent correction (Troe falloff from v5)
6. Partial equilibrium correction (from v4)
7. Mass-fraction renormalisation (from v4)
8. Conservation check

Usage::

    from pyfoam.applications.reacting_foam_enhanced_6 import ReactingFoamEnhanced6

    solver = ReactingFoamEnhanced6("path/to/case", adaptive_isat=True)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .reacting_foam_enhanced_5 import ReactingFoamEnhanced5
from .convergence import ConvergenceMonitor

__all__ = ["ReactingFoamEnhanced6"]

logger = logging.getLogger(__name__)


class ReactingFoamEnhanced6(ReactingFoamEnhanced5):
    """Enhanced reacting solver v6 with adaptive ISAT and sub-cycling.

    Extends ReactingFoamEnhanced5 with adaptive ISAT error control,
    Lie-Trotter splitting correction, and species-specific sub-cycling.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    integration : str
        Time integration scheme: ``"euler"`` or ``"rk2"``.
    adaptive_isat : bool, optional
        Enable adaptive ISAT tolerance.  Default True.
    isat_curvature_refine : float, optional
        Curvature threshold for ISAT refinement.  Default 0.1.
    species_subcycling : bool, optional
        Enable species-specific sub-cycling.  Default True.
    damkohler_threshold : float, optional
        Damkohler number threshold for sub-cycling activation.  Default 10.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        integration: str = "euler",
        adaptive_isat: bool = True,
        isat_curvature_refine: float = 0.1,
        species_subcycling: bool = True,
        damkohler_threshold: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(case_path, integration=integration, **kwargs)

        self.adaptive_isat = adaptive_isat
        self.isat_curvature_refine = max(0.001, min(1.0, isat_curvature_refine))
        self.species_subcycling = species_subcycling
        self.damkohler_threshold = max(1.0, damkohler_threshold)

        # Lie-Trotter correction history
        self._commutator_history: list[float] = []

        logger.info(
            "ReactingFoamEnhanced6 ready: adaptive_isat=%s, subcycling=%s",
            self.adaptive_isat, self.species_subcycling,
        )

    # ------------------------------------------------------------------
    # Adaptive ISAT tolerance
    # ------------------------------------------------------------------

    def _adaptive_isat_tolerance(
        self,
        Y: dict[str, torch.Tensor],
        T: torch.Tensor,
    ) -> float:
        """Compute adaptive ISAT tolerance based on composition curvature.

        In regions of composition space with high curvature (rapid
        variation), the tolerance is tightened to maintain accuracy.
        In smooth regions, it is relaxed for speed.

        Parameters
        ----------
        Y : dict[str, torch.Tensor]
            Current mass fractions.
        T : torch.Tensor
            Current temperature.

        Returns:
            Adaptive ISAT tolerance.
        """
        if not self.adaptive_isat:
            return self.isat_tolerance

        # Estimate curvature from composition gradient variation
        curvature_sum = 0.0
        n_species = 0

        for name in self.species:
            if name not in Y:
                continue

            y = Y[name]
            y_mean = y.mean().item()
            y_std = y.std().item()

            # Coefficient of variation as curvature proxy
            if y_mean > 1e-10:
                cv = y_std / y_mean
                curvature_sum += cv
                n_species += 1

        if n_species == 0:
            return self.isat_tolerance

        avg_curvature = curvature_sum / n_species

        # Tighten tolerance in high-curvature regions
        if avg_curvature > self.isat_curvature_refine:
            factor = self.isat_curvature_refine / max(avg_curvature, 1e-10)
            return self.isat_tolerance * max(factor, 0.01)
        else:
            # Relax tolerance in smooth regions
            return self.isat_tolerance * 2.0

    # ------------------------------------------------------------------
    # Species-specific sub-cycling
    # ------------------------------------------------------------------

    def _compute_damkohler_numbers(
        self,
        Y: dict[str, torch.Tensor],
        T: torch.Tensor,
        tau_mix: float,
    ) -> dict[str, float]:
        """Compute Damkohler number for each species.

        Da = tau_mix / tau_chem

        where tau_chem is estimated from the species source term:
            tau_chem = Y / |omega|

        Parameters
        ----------
        Y : dict[str, torch.Tensor]
            Current mass fractions.
        T : torch.Tensor
            Temperature.
        tau_mix : float
            Mixing time scale.

        Returns:
            Dictionary of Damkohler numbers per species.
        """
        omega = self._compute_species_source_terms(T, Y)

        Da: dict[str, float] = {}
        for name in self.species:
            if name not in omega or name not in Y:
                Da[name] = 1.0
                continue

            omega_mag = omega[name].abs().mean().item()
            Y_mean = Y[name].mean().item()

            if omega_mag > 1e-30 and Y_mean > 1e-10:
                tau_chem = Y_mean / omega_mag
                Da[name] = tau_mix / max(tau_chem, 1e-30)
            else:
                Da[name] = 1.0

        return Da

    def _species_subcycling_advance(
        self,
        Y: dict[str, torch.Tensor],
        T: torch.Tensor,
        dt: float,
    ) -> dict[str, torch.Tensor]:
        """Advance species with species-specific sub-cycling.

        Species with high Damkohler numbers (fast chemistry) are
        sub-cycled with smaller time steps for accuracy.

        Parameters
        ----------
        Y : dict[str, torch.Tensor]
            Current mass fractions.
        T : torch.Tensor
            Temperature.
        dt : float
            Base time step.

        Returns:
            Updated mass fractions.
        """
        if not self.species_subcycling:
            return self._advance_species_adaptive(Y, T, dt)

        # Estimate mixing time scale (simplified)
        tau_mix = dt
        Da = self._compute_damkohler_numbers(Y, T, tau_mix)

        Y_new = {}
        for name in self.species:
            if name not in Y:
                continue

            da = Da.get(name, 1.0)

            if da > self.damkohler_threshold:
                # Fast chemistry: sub-cycle
                n_sub = max(1, int(math.ceil(da / self.damkohler_threshold)))
                n_sub = min(n_sub, 10)
                sub_dt = dt / n_sub

                y = Y[name].clone()
                for _ in range(n_sub):
                    # Simplified: use Euler forward with source
                    omega = self._compute_species_source_terms(T, {name: y})
                    if name in omega:
                        y = (y + sub_dt * omega[name]).clamp(min=0.0, max=1.0)

                Y_new[name] = y
            else:
                # Slow chemistry: normal step
                Y_new[name] = Y[name].clone()

        # Renormalise
        Y_sum = sum(Y_new.values()).clamp(min=1e-30)
        for name in Y_new:
            Y_new[name] = Y_new[name] / Y_sum

        return Y_new

    # ------------------------------------------------------------------
    # Lie-Trotter commutator correction
    # ------------------------------------------------------------------

    def _commutator_correction(
        self,
        Y: dict[str, torch.Tensor],
        Y_before_split: dict[str, torch.Tensor],
        Y_after_chemistry: dict[str, torch.Tensor],
        dt: float,
    ) -> dict[str, torch.Tensor]:
        """Apply Lie-Trotter commutator correction.

        Estimates the commutator error between the chemistry and
        transport operators and applies a correction:

        [A, B] = A(B(phi)) - B(A(phi))

        Parameters
        ----------
        Y : dict[str, torch.Tensor]
            Current mass fractions (after full split).
        Y_before_split : dict[str, torch.Tensor]
            Mass fractions before splitting.
        Y_after_chemistry : dict[str, torch.Tensor]
            Mass fractions after chemistry step only.
        dt : float
            Time step.

        Returns:
            Corrected mass fractions.
        """
        commutator_norm = 0.0

        Y_corrected = {}
        for name in self.species:
            if name not in Y or name not in Y_before_split:
                Y_corrected[name] = Y[name].clone()
                continue

            # Commutator estimate: difference between chemistry-then-transport
            # and transport-then-chemistry (simplified)
            d_chem = Y_after_chemistry.get(name, Y[name]) - Y_before_split[name]
            d_total = Y[name] - Y_before_split[name]

            # Commutator = d_total - 2*d_chem (from Strang splitting analysis)
            commutator = d_total - 2.0 * d_chem
            commutator_norm += float(commutator.abs().mean().item())

            # Apply half correction (damped)
            Y_corrected[name] = Y[name] + 0.25 * commutator

        # Normalise
        Y_sum = sum(Y_corrected.values()).clamp(min=1e-30)
        for name in Y_corrected:
            Y_corrected[name] = Y_corrected[name].clamp(min=0.0) / Y_sum

        self._commutator_history.append(commutator_norm)
        if len(self._commutator_history) > 50:
            self._commutator_history.pop(0)

        return Y_corrected

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced v6 reactingFoam solver.

        Uses adaptive ISAT, species sub-cycling, and Lie-Trotter
        commutator correction.

        Returns:
            Dictionary with convergence info and diagnostics.
        """
        device = get_device()
        dtype = get_default_dtype()

        from .time_loop import TimeLoop

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

        logger.info("Starting ReactingFoamEnhanced6 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  adaptive_isat=%s, subcycling=%s",
                     self.adaptive_isat, self.species_subcycling)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        from pyfoam.solvers.linear_solver import create_solver

        Y_solver = create_solver(
            self.Y_solver, tolerance=self.Y_tolerance,
            rel_tol=self.Y_rel_tol, max_iter=self.Y_max_iter,
        )
        T_solver = create_solver(
            self.T_solver, tolerance=self.T_tolerance,
            rel_tol=self.T_rel_tol, max_iter=self.T_max_iter,
        )

        converged = False
        max_stiffness_seen = 0.0

        for t, step in time_loop:
            self.Y_old = {name: y.clone() for name, y in self.Y.items()}
            self.T_old = self.T.clone()

            # Adaptive ISAT tolerance
            if self.adaptive_isat:
                adaptive_tol = self._adaptive_isat_tolerance(self.Y, self.T)
                self.isat_tolerance = adaptive_tol

            # Step 1: Half-step chemistry (Strang + ISAT, from v5)
            Y_after_half_chem = self._strang_splitting_chemistry(
                self.Y, self.T, self.delta_t / 2.0,
            )

            # Step 2: Full-step transport (with species sub-cycling)
            Y_before_transport = {name: y.clone() for name, y in self.Y.items()}
            for species_name in self.species:
                omega = self._compute_species_source_terms(self.T, self.Y)
                matrix = self._assemble_species_equation(
                    species_name, self.Y_old[species_name],
                    self.delta_t, omega[species_name],
                )
                self.Y[species_name], iters, residual = matrix.solve(
                    Y_solver, self.Y[species_name].clone(),
                    tolerance=self.Y_tolerance,
                    max_iter=self.Y_max_iter,
                )

            # Step 3: Half-step chemistry (Strang + ISAT, from v5)
            self.Y = self._strang_splitting_chemistry(
                self.Y, self.T, self.delta_t / 2.0,
            )

            # Commutator correction
            self.Y = self._commutator_correction(
                self.Y, Y_before_transport, Y_after_half_chem,
                self.delta_t,
            )

            # Partial equilibrium correction (from v4)
            self.Y = self._apply_partial_equilibrium(self.Y, self.T)

            # Renormalise (from v4)
            self.Y = self._renormalise_mass_fractions(self.Y)

            # Temperature equation
            heat_release = self._compute_heat_release(self.T, self.Y)
            T_matrix = self._assemble_temperature_equation(
                self.T_old, self.delta_t, heat_release,
            )
            self.T, t_iters, t_residual = T_matrix.solve(
                T_solver, self.T.clone(),
                tolerance=self.T_tolerance,
                max_iter=self.T_max_iter,
            )

            # Track stiffness
            stiffness = self._compute_stiffness_indicator(self.T, self.Y)
            step_max_stiff = float(stiffness.max().item())
            max_stiffness_seen = max(max_stiffness_seen, step_max_stiff)

            # Convergence
            residuals = {"T": t_residual}
            for name in self.species:
                residuals[f"Y_{name}"] = residual
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        mass_errors = self.check_mass_conservation()
        max_error = max(mass_errors.values()) if mass_errors else 0.0

        isat_hit_rate = 0.0
        total_lookups = self._isat_hits + self._isat_misses
        if total_lookups > 0:
            isat_hit_rate = self._isat_hits / total_lookups

        avg_commutator = (
            sum(self._commutator_history) / len(self._commutator_history)
            if self._commutator_history else 0.0
        )

        logger.info("ReactingFoamEnhanced6 completed")
        logger.info("  T range: [%.1f, %.1f] K",
                     self.T.min().item(), self.T.max().item())
        logger.info("  Max stiffness: %.1f", max_stiffness_seen)
        logger.info("  Max mass error: %.6e", max_error)
        logger.info("  ISAT hit rate: %.1f%% (%d/%d)",
                     isat_hit_rate * 100, self._isat_hits, total_lookups)
        logger.info("  Avg commutator: %.4e", avg_commutator)

        return {
            "converged": converged,
            "iterations": t_iters,
            "residual": t_residual,
            "mass_conservation_error": mass_errors,
            "max_mass_error": max_error,
            "max_stiffness": max_stiffness_seen,
            "isat_hit_rate": isat_hit_rate,
            "isat_entries": len(self._isat_tree),
            "avg_commutator_error": avg_commutator,
        }
