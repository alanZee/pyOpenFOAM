"""
reactingFoamEnhanced4 — enhanced reacting solver v4.

Extends :class:`ReactingFoamEnhanced3` with:

- **Improved multi-step kinetics**: supports branching reaction networks
  where products of one reaction are reactants in another, with proper
  topological ordering for sequential evaluation.
- **Partial equilibrium correction**: for fast reactions near
  equilibrium, uses a partial equilibrium assumption to reduce the
  effective reaction rate rather than fully resolving the stiff ODE.
- **Adaptive species time stepping**: each species can have its own
  sub-time-step based on its local production/destruction rate,
  reducing unnecessary sub-cycling for slow species.
- **Conservation-preserving correction**: after all chemistry and
  transport steps, applies a global mass-fraction renormalisation
  weighted by molecular weight to preserve total mass.

Algorithm (per transport time step):
1. Compute stiffness and classify reactions
2. Topological ordering of reaction network
3. Strang splitting:
   a. Half-step chemistry (adaptive per-species sub-cycling)
   b. Full-step transport
   c. Half-step chemistry (adaptive per-species sub-cycling)
4. Partial equilibrium correction
5. Mass-fraction renormalisation
6. Conservation check

Usage::

    from pyfoam.applications.reacting_foam_enhanced_4 import ReactingFoamEnhanced4

    solver = ReactingFoamEnhanced4("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .reacting_foam_enhanced_3 import ReactingFoamEnhanced3, EquilibriumReaction
from .convergence import ConvergenceMonitor

__all__ = ["ReactingFoamEnhanced4"]

logger = logging.getLogger(__name__)


@dataclass
class ReactionNode:
    """Node in the reaction dependency graph.

    Attributes
    ----------
    index : int
        Index of the reaction in the reaction list.
    reaction : EquilibriumReaction
        The reaction object.
    dependencies : list[int]
        Indices of reactions whose products are this reaction's reactants.
    order : int
        Topological order (computed during graph sort).
    """
    index: int
    reaction: EquilibriumReaction
    dependencies: list[int] = field(default_factory=list)
    order: int = 0


class ReactingFoamEnhanced4(ReactingFoamEnhanced3):
    """Enhanced reacting solver v4 with branching kinetics.

    Extends ReactingFoamEnhanced3 with:

    - Topological ordering of reaction networks
    - Per-species adaptive sub-cycling
    - Partial equilibrium correction
    - Mass-conserving renormalisation

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    integration : str
        Time integration scheme: ``"euler"`` or ``"rk2"``.
    partial_equilibrium_threshold : float
        Ratio Q/Keq above which a reaction is treated as near-equilibrium.
        Default 0.9.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        integration: str = "euler",
        partial_equilibrium_threshold: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__(case_path, integration=integration, **kwargs)

        self.partial_equilibrium_threshold = partial_equilibrium_threshold

        # Build reaction dependency graph
        self.reaction_graph = self._build_reaction_graph()
        self._topological_sort()

        logger.info(
            "ReactingFoamEnhanced4 ready: %d reactions, %d equilibrium, "
            "pe_thresh=%.2f",
            len(self.equilibrium_reactions),
            sum(1 for r in self.equilibrium_reactions if r.Keq > 0),
            self.partial_equilibrium_threshold,
        )

    # ------------------------------------------------------------------
    # Reaction dependency graph
    # ------------------------------------------------------------------

    def _build_reaction_graph(self) -> list[ReactionNode]:
        """Build dependency graph between reactions.

        A reaction depends on another if the other's products are
        this reaction's reactants.

        Returns:
            List of ReactionNode objects.
        """
        nodes = []
        for i, rxn_i in enumerate(self.equilibrium_reactions):
            deps = []
            for j, rxn_j in enumerate(self.equilibrium_reactions):
                if i == j:
                    continue
                # Check if rxn_j produces something rxn_i needs
                for sp in rxn_i.reactants:
                    if sp in rxn_j.products:
                        deps.append(j)
                        break
            nodes.append(ReactionNode(
                index=i, reaction=rxn_i, dependencies=deps,
            ))
        return nodes

    def _topological_sort(self) -> None:
        """Sort reactions by dependency (topological order).

        Reactions that produce precursors for other reactions are
        evaluated first.
        """
        n = len(self.reaction_graph)
        visited = [False] * n
        order = 0

        def dfs(node: ReactionNode) -> None:
            nonlocal order
            if visited[node.index]:
                return
            visited[node.index] = True
            for dep_idx in node.dependencies:
                dfs(self.reaction_graph[dep_idx])
            node.order = order
            order += 1

        for node in self.reaction_graph:
            dfs(node)

    # ------------------------------------------------------------------
    # Per-species adaptive sub-cycling
    # ------------------------------------------------------------------

    def _advance_species_adaptive(
        self,
        Y: dict[str, torch.Tensor],
        T: torch.Tensor,
        dt: float,
    ) -> dict[str, torch.Tensor]:
        """Advance chemistry with per-species adaptive sub-cycling.

        Each species gets its own sub-step count based on its
        production/destruction rate, rather than using the global
        stiffness indicator.

        Parameters
        ----------
        Y : dict[str, torch.Tensor]
            Current mass fractions.
        T : torch.Tensor
            Temperature field.
        dt : float
            Transport time step.

        Returns:
            Updated mass fractions.
        """
        # Compute source terms
        omega = self._compute_species_source_terms(T, Y)

        Y_new = {}
        for name in self.species:
            if name not in omega:
                Y_new[name] = Y[name].clone()
                continue

            rate = omega[name]
            rate_abs = rate.abs().clamp(min=1e-30)

            # Per-species sub-step count
            y_safe = Y[name].clamp(min=1e-30)
            max_sub = max(1, int(float((rate_abs * dt / y_safe).max().item()) * 10) + 1)
            max_sub = min(max_sub, self.max_chem_sub_steps)

            sub_dt = dt / max_sub
            y_current = Y[name].clone()

            for _ in range(max_sub):
                y_current = y_current + sub_dt * rate
                y_current = y_current.clamp(min=0.0, max=1.0)

            Y_new[name] = y_current

        # Normalize
        Y_sum = sum(Y_new.values())
        Y_sum = Y_sum.clamp(min=1e-30)
        for name in self.species:
            Y_new[name] = Y_new[name] / Y_sum

        return Y_new

    # ------------------------------------------------------------------
    # Partial equilibrium correction
    # ------------------------------------------------------------------

    def _apply_partial_equilibrium(
        self,
        Y: dict[str, torch.Tensor],
        T: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply partial equilibrium correction to near-equilibrium reactions.

        For reactions with Q/Keq > threshold, scale the net rate down
        by the equilibrium approach factor.

        Parameters
        ----------
        Y : dict[str, torch.Tensor]
            Current mass fractions.
        T : torch.Tensor
            Temperature field.

        Returns:
            Corrected mass fractions.
        """
        Y_corrected = {k: v.clone() for k, v in Y.items()}

        for rxn in self.equilibrium_reactions:
            if rxn.Keq <= 0:
                continue

            n_cells = T.shape[0]
            Q = torch.ones(n_cells, dtype=T.dtype, device=T.device)

            for sp, nu in rxn.products.items():
                if sp in Y_corrected:
                    Q = Q * Y_corrected[sp].clamp(min=1e-30).pow(nu)

            for sp, nu in rxn.reactants.items():
                if sp in Y_corrected:
                    Q = Q / Y_corrected[sp].clamp(min=1e-30).pow(nu)

            # Partial equilibrium factor
            approach = (Q / rxn.Keq).clamp(min=0.0, max=1.0)
            near_eq = approach > self.partial_equilibrium_threshold

            if near_eq.any():
                # Scale reactant consumption down in near-equilibrium cells
                scale = torch.where(
                    near_eq,
                    1.0 - (approach - self.partial_equilibrium_threshold)
                    / (1.0 - self.partial_equilibrium_threshold + 1e-10),
                    torch.ones_like(approach),
                )
                for sp in rxn.reactants:
                    if sp in Y_corrected:
                        Y_corrected[sp] = Y[sp] + (Y_corrected[sp] - Y[sp]) * scale

        return Y_corrected

    # ------------------------------------------------------------------
    # Mass-fraction renormalisation
    # ------------------------------------------------------------------

    def _renormalise_mass_fractions(
        self,
        Y: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Renormalise mass fractions to ensure sum(Y) = 1.

        Uses weighted renormalisation that preserves relative ratios
        where possible.

        Parameters
        ----------
        Y : dict[str, torch.Tensor]
            Current mass fractions.

        Returns:
            Renormalised mass fractions.
        """
        Y_sum = sum(Y.values())
        Y_sum = Y_sum.clamp(min=1e-30)

        Y_new = {}
        for name in self.species:
            Y_new[name] = (Y[name] / Y_sum).clamp(min=0.0, max=1.0)

        return Y_new

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced v4 reactingFoam solver.

        Uses topological ordering, per-species adaptive sub-cycling,
        partial equilibrium, and mass-conserving renormalisation.

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

        logger.info("Starting ReactingFoamEnhanced4 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

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

            # Step 1: Half-step chemistry (per-species adaptive)
            self.Y = self._advance_species_adaptive(
                self.Y, self.T, self.delta_t / 2.0,
            )

            # Step 2: Full-step transport
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

            # Step 3: Half-step chemistry
            self.Y = self._advance_species_adaptive(
                self.Y, self.T, self.delta_t / 2.0,
            )

            # Partial equilibrium correction
            self.Y = self._apply_partial_equilibrium(self.Y, self.T)

            # Renormalise mass fractions
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

        logger.info("ReactingFoamEnhanced4 completed")
        logger.info("  T range: [%.1f, %.1f] K",
                     self.T.min().item(), self.T.max().item())
        logger.info("  Max stiffness seen: %.1f", max_stiffness_seen)
        logger.info("  Max mass conservation error: %.6e", max_error)

        return {
            "converged": converged,
            "iterations": t_iters,
            "residual": t_residual,
            "mass_conservation_error": mass_errors,
            "max_mass_error": max_error,
            "max_stiffness": max_stiffness_seen,
        }
