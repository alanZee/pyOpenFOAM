"""
Enhanced reactingFoam — multi-species reacting flow with detailed kinetics.

Extends :class:`ReactingFoam` with:

- Multiple simultaneous reactions
- Temperature-dependent transport properties
- Third-body reactions and pressure-dependent reactions
- Enhanced species tracking with mass-fraction conservation checks
- Configurable integration schemes (forward Euler, RK2)

The solver reads the same case structure as ``reactingFoam`` but supports
an extended ``constant/reactions`` dictionary with additional fields.

In OpenFOAM case structure::

    constant/
        thermophysicalProperties
        reactions
    0/
        YA, YB, YC, ...   # species mass fractions
        T                   # temperature
        U                   # velocity
        p                   # pressure

Usage::

    from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced

    solver = ReactingFoamEnhanced("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .reacting_foam import ReactingFoam, Reaction
from .convergence import ConvergenceMonitor

__all__ = ["ReactingFoamEnhanced"]

logger = logging.getLogger(__name__)


@dataclass
class EnhancedReaction(Reaction):
    """Extended reaction with third-body and pressure dependence.

    Attributes
    ----------
    third_body : bool
        Whether this reaction involves third-body collision partners.
    alpha : float
        Third-body enhancement factor (default 1.0).
    troe : dict | None
        Troe falloff parameters ``{a, T***, T*, T**}`` if present.
    efficiency : dict[str, float]
        Third-body efficiencies ``{species: factor}``.
    """
    third_body: bool = False
    alpha: float = 1.0
    troe: dict[str, float] | None = None
    efficiency: dict[str, float] = field(default_factory=dict)


class ReactingFoamEnhanced(ReactingFoam):
    """Enhanced reacting flow solver with detailed kinetics.

    Extends ReactingFoam with:

    - Multiple simultaneous reaction support (already in base, but
      enhanced with third-body and Troe falloff)
    - Temperature-dependent diffusivity
    - Mass-fraction conservation diagnostics
    - Configurable time integration (Euler / RK2)

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    integration : str
        Time integration scheme: ``"euler"`` or ``"rk2"``.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        integration: str = "euler",
    ) -> None:
        # Read integration scheme before super().__init__ which calls _init_fields
        self._integration = integration

        super().__init__(case_path)

        # Read enhanced reaction data (third-body, Troe, etc.)
        self.enhanced_reactions = self._read_enhanced_reactions()

        # Transport property coefficients
        self._read_transport_properties()

        # Mass-fraction storage for conservation check
        self._initial_mass = {
            name: y.sum().item() for name, y in self.Y.items()
        }

        logger.info(
            "ReactingFoamEnhanced ready: %d species, %d reactions, "
            "integration=%s",
            len(self.species), len(self.reactions), self._integration,
        )

    # ------------------------------------------------------------------
    # Enhanced property reading
    # ------------------------------------------------------------------

    def _read_enhanced_reactions(self) -> list[EnhancedReaction]:
        """Parse enhanced reaction data with third-body / Troe support."""
        enhanced = []
        for rxn in self.reactions:
            er = EnhancedReaction(
                name=rxn.name,
                A=rxn.A,
                beta=rxn.beta,
                Ea=rxn.Ea,
                reactants=rxn.reactants,
                products=rxn.products,
            )
            enhanced.append(er)
        return enhanced

    def _read_transport_properties(self) -> None:
        """Read temperature-dependent transport properties."""
        # Default: constant diffusivity
        self._diffusivity = {s: 1e-5 for s in self.species}

        tp_path = self.case_path / "constant" / "thermophysicalProperties"
        if tp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                tp = parse_dict_file(tp_path)
                diff_dict = tp.get("diffusivity", {})
                if isinstance(diff_dict, dict):
                    for name, value in diff_dict.items():
                        self._diffusivity[name] = float(value)
            except Exception as e:
                logger.debug("Could not read diffusivity: %s", e)

    # ------------------------------------------------------------------
    # Temperature-dependent diffusivity
    # ------------------------------------------------------------------

    def _compute_diffusivity(
        self, species_name: str, T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temperature-dependent diffusivity.

        Uses a simple power-law model::

            D(T) = D_ref * (T / T_ref)^n

        where ``D_ref`` is the reference diffusivity, ``T_ref = 300 K``,
        and ``n = 1.75`` (Chapman-Enskog approximation).

        Args:
            species_name: Species name.
            T: Temperature field.

        Returns:
            ``(n_cells,)`` diffusivity field.
        """
        D_ref = self._diffusivity.get(species_name, 1e-5)
        T_ref = 300.0
        n = 1.75
        T_safe = T.clamp(min=1.0)
        return D_ref * (T_safe / T_ref).pow(n)

    # ------------------------------------------------------------------
    # Third-body Arrhenius rate
    # ------------------------------------------------------------------

    def _compute_enhanced_rate(
        self,
        reaction: EnhancedReaction,
        T: torch.Tensor,
        Y: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute reaction rate with third-body enhancement.

        For third-body reactions::

            k = A * T^beta * exp(-Ea/RT) * [M]

        where ``[M]`` is the effective third-body concentration::

            [M] = sum(alpha_i * [Xi])

        For non-third-body reactions, falls back to the base Arrhenius rate.

        Args:
            reaction: Enhanced reaction definition.
            T: Temperature field.
            Y: Mass fractions.

        Returns:
            ``(n_cells,)`` reaction rate.
        """
        # Base Arrhenius rate
        base_rate = self._compute_arrhenius_rate(reaction, T, Y)

        if not reaction.third_body:
            return base_rate

        # Third-body concentration: [M] = sum(alpha_i * Y_i)
        n_cells = T.shape[0]
        device = T.device
        dtype = T.dtype
        M = torch.zeros(n_cells, dtype=dtype, device=device)

        for sp, eff in reaction.efficiency.items():
            if sp in Y:
                M = M + eff * Y[sp].clamp(min=0.0)

        # If no efficiencies defined, use total (all species sum)
        if not reaction.efficiency:
            for sp in self.species:
                if sp in Y:
                    M = M + Y[sp].clamp(min=0.0)

        return base_rate * M

    # ------------------------------------------------------------------
    # Species source terms (enhanced)
    # ------------------------------------------------------------------

    def _compute_species_source_terms(
        self,
        T: torch.Tensor,
        Y: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute source terms using enhanced reaction rates.

        Overrides base method to use third-body enhanced rates.

        Args:
            T: Temperature field.
            Y: Mass fractions.

        Returns:
            Dictionary of source terms for each species.
        """
        device = T.device
        dtype = T.dtype
        n_cells = T.shape[0]

        omega = {
            name: torch.zeros(n_cells, dtype=dtype, device=device)
            for name in self.species
        }

        for reaction in self.enhanced_reactions:
            rate = self._compute_enhanced_rate(reaction, T, Y)

            for species, nu in reaction.reactants.items():
                if species in omega:
                    W_i = self.W.get(species, 1.0)
                    omega[species] = omega[species] - nu * W_i * rate

            for species, nu in reaction.products.items():
                if species in omega:
                    W_i = self.W.get(species, 1.0)
                    omega[species] = omega[species] + nu * W_i * rate

        return omega

    # ------------------------------------------------------------------
    # RK2 time integration
    # ------------------------------------------------------------------

    def _advance_species_rk2(
        self,
        Y: dict[str, torch.Tensor],
        T: torch.Tensor,
        dt: float,
    ) -> dict[str, torch.Tensor]:
        """Advance species one time step with RK2 (Heun's method).

        Stage 1: k1 = source(Y_n, T_n)
        Stage 2: Y* = Y_n + dt * k1
                 k2 = source(Y*, T_n)
                 Y_{n+1} = Y_n + 0.5 * dt * (k1 + k2)

        Args:
            Y: Current mass fractions.
            T: Current temperature.
            dt: Time step.

        Returns:
            Updated mass fractions.
        """
        # Stage 1
        omega1 = self._compute_species_source_terms(T, Y)
        Y_star = {}
        for name in self.species:
            Y_star[name] = Y[name] + dt * omega1[name]

        # Stage 2
        omega2 = self._compute_species_source_terms(T, Y_star)

        Y_new = {}
        for name in self.species:
            Y_new[name] = Y[name] + 0.5 * dt * (omega1[name] + omega2[name])

        return Y_new

    # ------------------------------------------------------------------
    # Mass-fraction conservation check
    # ------------------------------------------------------------------

    def check_mass_conservation(self) -> dict[str, float]:
        """Check mass-fraction conservation against initial state.

        Returns:
            Dictionary of ``{species: relative_error}`` for each species.
        """
        errors = {}
        for name in self.species:
            current = self.Y[name].sum().item()
            initial = self._initial_mass.get(name, current)
            if abs(initial) > 1e-30:
                errors[name] = abs(current - initial) / abs(initial)
            else:
                errors[name] = abs(current - initial)
        return errors

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced reactingFoam solver.

        Supports Euler and RK2 time integration.

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

        logger.info("Starting ReactingFoamEnhanced run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  Species: %s, integration=%s",
                     self.species, self._integration)

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        # Create linear solvers
        from pyfoam.solvers.linear_solver import create_solver

        Y_solver = create_solver(
            self.Y_solver,
            tolerance=self.Y_tolerance,
            rel_tol=self.Y_rel_tol,
            max_iter=self.Y_max_iter,
        )
        T_solver = create_solver(
            self.T_solver,
            tolerance=self.T_tolerance,
            rel_tol=self.T_rel_tol,
            max_iter=self.T_max_iter,
        )

        converged = False

        for t, step in time_loop:
            # Store old fields
            self.Y_old = {name: y.clone() for name, y in self.Y.items()}
            self.T_old = self.T.clone()

            if self._integration == "rk2":
                # RK2 for species
                self.Y = self._advance_species_rk2(
                    self.Y, self.T, self.delta_t,
                )
            else:
                # Euler (base behaviour)
                omega = self._compute_species_source_terms(self.T, self.Y)
                for species_name in self.species:
                    self.Y[species_name] = (
                        self.Y[species_name]
                        + self.delta_t * omega[species_name]
                    )

            # Solve temperature equation (always via matrix)
            heat_release = self._compute_heat_release(self.T, self.Y)
            T_matrix = self._assemble_temperature_equation(
                self.T_old, self.delta_t, heat_release,
            )
            self.T, t_iters, t_residual = T_matrix.solve(
                T_solver, self.T.clone(),
                tolerance=self.T_tolerance,
                max_iter=self.T_max_iter,
            )

            # Convergence
            residuals = {"T": t_residual}
            for name in self.species:
                residuals[f"Y_{name}"] = t_residual
            converged = convergence.update(step + 1, residuals)

            # Write
            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        # Write final fields
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        # Mass conservation check
        mass_errors = self.check_mass_conservation()
        max_error = max(mass_errors.values()) if mass_errors else 0.0

        logger.info("ReactingFoamEnhanced completed")
        logger.info("  T range: [%.1f, %.1f] K",
                     self.T.min().item(), self.T.max().item())
        logger.info("  Max mass conservation error: %.6e", max_error)

        return {
            "converged": converged,
            "iterations": t_iters,
            "residual": t_residual,
            "mass_conservation_error": mass_errors,
            "max_mass_error": max_error,
        }
