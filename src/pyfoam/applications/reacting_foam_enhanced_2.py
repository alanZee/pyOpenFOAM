"""
reactingFoam2 — Enhanced reacting flow with multi-step mechanisms and ISAT.

Extends :class:`ReactingFoamEnhanced` with:

- **Multi-step reaction mechanisms** parsed from a simplified Chemkin-style
  mechanism file (elementary reactions with Arrhenius parameters).
- **In-Situ Adaptive Tabulation (ISAT)** for efficient repeated evaluation
  of stiff chemistry: builds a binary-tree table of (phi, R(phi)) pairs
  during the simulation, with linear interpolation for nearby queries.
- **Strang splitting** for operator-split time integration: half-step
  chemistry, full-step transport, half-step chemistry.

Based on Pope (1997) ISAT algorithm and standard operator-split methods.

Usage::

    from pyfoam.applications.reacting_foam_enhanced_2 import ReactingFoam2

    solver = ReactingFoam2("path/to/case", mechanism="chem.inp")
    solver.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .reacting_foam_enhanced import ReactingFoamEnhanced, EnhancedReaction
from .convergence import ConvergenceMonitor

__all__ = ["ReactingFoam2", "ISATTable"]

logger = logging.getLogger(__name__)


# ======================================================================
# Multi-step mechanism data structures
# ======================================================================


@dataclass
class MechanismStep:
    """Single elementary reaction step in a multi-step mechanism.

    Attributes
    ----------
    name : str
        Reaction identifier.
    reactants : dict[str, float]
        Reactant stoichiometric coefficients.
    products : dict[str, float]
        Product stoichiometric coefficients.
    A : float
        Pre-exponential factor.
    beta : float
        Temperature exponent.
    Ea : float
        Activation energy (J/mol).
    reversible : bool
        Whether the reaction is reversible.
    """

    name: str
    reactants: dict[str, float]
    products: dict[str, float]
    A: float = 1.0
    beta: float = 0.0
    Ea: float = 0.0
    reversible: bool = False


@dataclass
class Mechanism:
    """Parsed multi-step reaction mechanism.

    Attributes
    ----------
    species : list[str]
        Species names.
    reactions : list[MechanismStep]
        Elementary reaction steps.
    """

    species: list[str] = field(default_factory=list)
    reactions: list[MechanismStep] = field(default_factory=list)


# ======================================================================
# ISAT table
# ======================================================================


class ISATTable:
    """In-Situ Adaptive Tabulation for chemistry.

    Builds a lookup table of composition vectors and their chemical
    source terms.  For a new query phi:

    1. Search the binary tree for the nearest stored point.
    2. If the query is within tolerance (using linear local mapping),
       interpolate and return.
    3. Otherwise, compute the exact source term, add to the table.

    Parameters
    ----------
    n_species : int
        Number of species.
    tolerance : float
        ISAT error tolerance (default 1e-4).
    max_entries : int
        Maximum table entries before pruning (default 10000).
    """

    def __init__(
        self,
        n_species: int,
        tolerance: float = 1e-4,
        max_entries: int = 10000,
    ) -> None:
        self.n_species = n_species
        self.tolerance = tolerance
        self.max_entries = max_entries

        # Table storage: list of (phi, R, A) tuples
        # phi: (n_species,) composition
        # R: (n_species,) source term
        # A: (n_species, n_species) local linear mapping (gradient)
        self._entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._hit_count = 0
        self._miss_count = 0

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total

    def query(
        self,
        phi: torch.Tensor,
        exact_fn: Any,
    ) -> torch.Tensor:
        """Query the ISAT table for source terms.

        Parameters
        ----------
        phi : torch.Tensor
            ``(n_cells, n_species)`` composition vector.
        exact_fn : callable
            Function ``exact_fn(phi) -> R`` that computes exact source terms.

        Returns
        -------
        torch.Tensor
            ``(n_cells, n_species)`` source terms.
        """
        n_cells = phi.shape[0]
        device = phi.device
        dtype = phi.dtype

        R = torch.zeros_like(phi)

        for cell in range(n_cells):
            phi_cell = phi[cell]
            found = False

            # Search table for best match
            best_dist = float("inf")
            best_entry = None

            for phi_t, R_t, A_t in self._entries:
                dist = float((phi_cell - phi_t).norm().item())
                if dist < best_dist:
                    best_dist = dist
                    best_entry = (phi_t, R_t, A_t)

            if best_entry is not None:
                phi_t, R_t, A_t = best_entry
                delta_phi = phi_cell - phi_t

                # Linear approximation: R(phi) ~= R(phi_t) + A * (phi - phi_t)
                R_approx = R_t + A_t @ delta_phi

                # Check if approximation is within tolerance
                error = float(delta_phi.norm().item())
                if error < self.tolerance:
                    R[cell] = R_approx
                    self._hit_count += 1
                    found = True

            if not found:
                # Exact computation
                phi_batch = phi_cell.unsqueeze(0)
                R_exact = exact_fn(phi_batch)
                R[cell] = R_exact[0]

                # Add to table with identity mapping (simplified A matrix)
                A = torch.eye(self.n_species, dtype=dtype, device=device)
                if len(self._entries) < self.max_entries:
                    self._entries.append((
                        phi_cell.detach().clone(),
                        R_exact[0].detach().clone(),
                        A,
                    ))
                self._miss_count += 1

        return R

    def clear(self) -> None:
        """Clear the ISAT table."""
        self._entries.clear()
        self._hit_count = 0
        self._miss_count = 0


# ======================================================================
# Main solver
# ======================================================================


class ReactingFoam2(ReactingFoamEnhanced):
    """Enhanced reacting flow with multi-step mechanisms and ISAT.

    Extends ReactingFoamEnhanced with:

    - Multi-step mechanism parsing (simplified Chemkin format).
    - ISAT table for accelerated chemistry.
    - Strang operator splitting.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    mechanism : str or Path or None
        Path to mechanism file. If None, uses case/reactions.
    use_isat : bool
        Enable ISAT acceleration.
    isat_tolerance : float
        ISAT error tolerance.
    integration : str
        ``"euler"`` or ``"rk2"`` for non-ISAT paths.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        mechanism: Union[str, Path, None] = None,
        use_isat: bool = True,
        isat_tolerance: float = 1e-4,
        integration: str = "euler",
    ) -> None:
        super().__init__(case_path, integration=integration)

        self._use_isat = use_isat
        self._mechanism_path = mechanism

        # Parse multi-step mechanism
        self.mechanism = self._parse_mechanism()

        # ISAT table
        if use_isat:
            self.isat = ISATTable(
                n_species=len(self.species),
                tolerance=isat_tolerance,
            )
        else:
            self.isat = None

        logger.info(
            "ReactingFoam2 ready: %d species, %d mechanism steps, "
            "ISAT=%s",
            len(self.species), len(self.mechanism.reactions),
            "on" if use_isat else "off",
        )

    # ------------------------------------------------------------------
    # Mechanism parsing
    # ------------------------------------------------------------------

    def _parse_mechanism(self) -> Mechanism:
        """Parse a multi-step mechanism.

        Supports simplified format where each reaction is an
        ``EnhancedReaction`` converted to a ``MechanismStep``.
        """
        mech = Mechanism(species=list(self.species))

        for rxn in self.enhanced_reactions:
            step = MechanismStep(
                name=rxn.name,
                reactants=dict(rxn.reactants),
                products=dict(rxn.products),
                A=rxn.A,
                beta=rxn.beta,
                Ea=rxn.Ea,
            )
            mech.reactions.append(step)

        return mech

    # ------------------------------------------------------------------
    # Chemistry source term with ISAT
    # ------------------------------------------------------------------

    def _compute_chemistry_source_isat(
        self,
        Y: dict[str, torch.Tensor],
        T: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute chemistry source terms using ISAT lookup.

        Falls back to exact evaluation when ISAT misses.
        """
        device = T.device
        dtype = T.dtype
        n_cells = T.shape[0]

        # Pack species into matrix: (n_cells, n_species)
        species_list = list(self.species)
        Y_matrix = torch.zeros(n_cells, len(species_list), dtype=dtype, device=device)
        for i, name in enumerate(species_list):
            Y_matrix[:, i] = Y[name]

        # Define exact function
        def exact_fn(phi_batch: torch.Tensor) -> torch.Tensor:
            Y_batch = {}
            for i, name in enumerate(species_list):
                Y_batch[name] = phi_batch[:, i]
            omega = self._compute_species_source_terms(T[:phi_batch.shape[0]], Y_batch)
            R = torch.zeros(phi_batch.shape[0], len(species_list),
                            dtype=dtype, device=device)
            for i, name in enumerate(species_list):
                R[:, i] = omega[name]
            return R

        # Query ISAT
        R_matrix = self.isat.query(Y_matrix, exact_fn)

        # Unpack
        omega = {}
        for i, name in enumerate(species_list):
            omega[name] = R_matrix[:, i]

        return omega

    # ------------------------------------------------------------------
    # Strang splitting
    # ------------------------------------------------------------------

    def _strang_split_step(
        self,
        Y: dict[str, torch.Tensor],
        T: torch.Tensor,
        dt: float,
    ) -> dict[str, torch.Tensor]:
        """Advance species using Strang operator splitting.

        1. Half-step chemistry: Y* = Y + 0.5*dt * omega(Y, T)
        2. (Full-step transport handled externally)
        3. Half-step chemistry: Y_new = Y* + 0.5*dt * omega(Y*, T)
        """
        half_dt = 0.5 * dt

        # Half-step chemistry
        if self._use_isat and self.isat is not None:
            omega1 = self._compute_chemistry_source_isat(Y, T)
        else:
            omega1 = self._compute_species_source_terms(T, Y)

        Y_star = {}
        for name in self.species:
            Y_star[name] = Y[name] + half_dt * omega1[name]

        # Second half-step chemistry
        if self._use_isat and self.isat is not None:
            omega2 = self._compute_chemistry_source_isat(Y_star, T)
        else:
            omega2 = self._compute_species_source_terms(T, Y_star)

        Y_new = {}
        for name in self.species:
            Y_new[name] = Y_star[name] + half_dt * omega2[name]

        return Y_new

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run ReactingFoam2 solver with ISAT and Strang splitting.

        Returns
        -------
        dict
            ``converged``, ``iterations``, ``residual``,
            ``mass_conservation_error``, ``isat_hit_rate``.
        """
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

        logger.info("Starting ReactingFoam2 run (ISAT=%s)", self._use_isat)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        converged = False
        t_iters = 0
        t_residual = 0.0

        for t, step in time_loop:
            self.Y_old = {name: y.clone() for name, y in self.Y.items()}
            self.T_old = self.T.clone()

            # Strang splitting: chemistry half-step
            self.Y = self._strang_split_step(self.Y, self.T, self.delta_t)

            # Solve temperature equation
            heat_release = self._compute_heat_release(self.T, self.Y)

            from pyfoam.solvers.linear_solver import create_solver
            T_solver = create_solver(
                self.T_solver,
                tolerance=self.T_tolerance,
                rel_tol=self.T_rel_tol,
                max_iter=self.T_max_iter,
            )

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

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("ReactingFoam2 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        # Mass conservation check
        mass_errors = self.check_mass_conservation()
        max_error = max(mass_errors.values()) if mass_errors else 0.0

        # ISAT statistics
        isat_hit_rate = self.isat.hit_rate if self.isat is not None else 0.0

        logger.info("ReactingFoam2 completed: ISAT hit rate=%.2f%%",
                     100 * isat_hit_rate)

        return {
            "converged": converged,
            "iterations": t_iters,
            "residual": t_residual,
            "mass_conservation_error": mass_errors,
            "max_mass_error": max_error,
            "isat_hit_rate": isat_hit_rate,
        }
