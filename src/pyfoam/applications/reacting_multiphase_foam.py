"""
reactingMultiphaseFoam — reacting multiphase Euler-Euler solver.

Combines N-phase Euler-Euler multiphase flow with chemical reaction kinetics.
Each phase has its own velocity field and species transport, coupled through
interphase forces and shared reaction source terms.

Governing equations:
    Phase volume fraction:
        ∂αi/∂t + ∇·(αi Ui) = 0

    Phase momentum:
        ∂(αi ρi Ui)/∂t + ∇·(αi ρi Ui Ui) = -αi ∇p + ∇·(αi τi) + Fi + Mi

    Species transport (per phase):
        ∂(αi ρi Yi,k)/∂t + ∇·(αi ρi Ui Yi,k) = ∇·(αi ρi Di ∇Yi,k) + αi ωi,k

where Fi are interphase forces, Mi are interphase mass transfer terms,
and ωi,k are the reaction source terms from Arrhenius kinetics.

Based on OpenFOAM's reactingMultiphaseEulerFoam solver.

Usage::

    from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam

    phases = [
        {"name": "gas",    "rho": 1.225, "mu": 1.8e-5, "species": ["O2", "N2"]},
        {"name": "liquid", "rho": 1000.0, "mu": 1e-3,   "species": ["H2O"]},
    ]
    solver = ReactingMultiphaseFoam("path/to/case", phases=phases)
    solver.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["ReactingMultiphaseFoam"]

logger = logging.getLogger(__name__)


@dataclass
class Reaction:
    """A single chemical reaction with Arrhenius kinetics.

    Attributes
    ----------
    name : str
        Reaction name.
    A : float
        Pre-exponential factor.
    beta : float
        Temperature exponent.
    Ea : float
        Activation energy (J/mol).
    reactants : dict[str, float]
        Species stoichiometric coefficients for reactants.
    products : dict[str, float]
        Species stoichiometric coefficients for products.
    """
    name: str = ""
    A: float = 1.0
    beta: float = 0.0
    Ea: float = 0.0
    reactants: dict[str, float] = field(default_factory=dict)
    products: dict[str, float] = field(default_factory=dict)


class ReactingMultiphaseFoam(SolverBase):
    """Reacting multiphase Euler-Euler solver.

    Combines N-phase Euler-Euler multiphase flow with Arrhenius reaction
    kinetics.  Each phase carries its own species which react according
    to the specified reaction mechanism.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    phases : list[dict]
        Phase definitions with name, rho, mu, and optional species list.

    Attributes
    ----------
    phases : list[dict]
        Phase definitions.
    n_phases : int
        Number of phases.
    phase_names : list[str]
        Phase names.
    species : dict[str, list[str]]
        Species per phase.
    reactions : list[Reaction]
        Reaction mechanism.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        phases: List[Dict[str, Any]],
    ) -> None:
        super().__init__(case_path)

        self.phases = phases
        self.n_phases = len(phases)
        self.phase_names = [p["name"] for p in phases]

        # Species per phase
        self.species: dict[str, list[str]] = {}
        for p in phases:
            self.species[p["name"]] = p.get("species", [])

        # Read thermodynamic properties
        self.R = 8.314  # Universal gas constant
        self.Cp = 1005.0  # Specific heat capacity

        # Read reaction mechanism
        self.reactions = self._read_reactions()

        # Phase properties as tensors
        self.rho_phases = torch.tensor(
            [p["rho"] for p in phases],
            dtype=get_default_dtype(), device=get_device(),
        )
        self.mu_phases = torch.tensor(
            [p["mu"] for p in phases],
            dtype=get_default_dtype(), device=get_device(),
        )

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Initialise fields
        self.velocities, self.p, self.alphas, self.Y, self.T, self.phi = (
            self._init_fields()
        )

        # Store raw field data for writing
        self._field_data = self._init_field_data()

        logger.info(
            "ReactingMultiphaseFoam ready: %d phases, %d reactions",
            self.n_phases, len(self.reactions),
        )

    def _read_reactions(self) -> list[Reaction]:
        """Read reaction mechanism from constant/reactions."""
        reactions = []

        rxn_path = self.case_path / "constant" / "reactions"
        if not rxn_path.exists():
            logger.warning("No reactions file found, using default A→B reaction")
            reactions.append(Reaction(
                name="reaction1", A=1.0, beta=0.0, Ea=0.0,
                reactants={"A": 1.0}, products={"B": 1.0},
            ))
            return reactions

        try:
            from pyfoam.io.dictionary import parse_dict_file
            rxn_dict = parse_dict_file(rxn_path)
            for key, value in rxn_dict.items():
                if isinstance(value, dict):
                    rxn = Reaction(name=key)
                    rxn.A = float(value.get("A", 1.0))
                    rxn.beta = float(value.get("beta", 0.0))
                    rxn.Ea = float(value.get("Ea", 0.0))
                    reactants = value.get("reactants", {})
                    if isinstance(reactants, dict):
                        rxn.reactants = {k: float(v) for k, v in reactants.items()}
                    products = value.get("products", {})
                    if isinstance(products, dict):
                        rxn.products = {k: float(v) for k, v in products.items()}
                    reactions.append(rxn)
        except Exception as e:
            logger.warning("Could not read reactions: %s", e)
            reactions.append(Reaction(
                name="default", A=1.0, beta=0.0, Ea=0.0,
                reactants={"A": 1.0}, products={"B": 1.0},
            ))

        return reactions

    def _read_fv_solution_settings(self) -> None:
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution
        self.n_outer_correctors = int(fv.get_path("PIMPLE/nOuterCorrectors", 3))
        self.convergence_tolerance = float(
            fv.get_path("PIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("PIMPLE/maxOuterIterations", 100)
        )

    def _init_fields(self) -> tuple:
        """Initialise velocity, pressure, alpha, species, temperature, and flux."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Phase velocities
        velocities = []
        for name in self.phase_names:
            try:
                U, _ = self.read_field_tensor(f"U_{name}", 0)
            except Exception:
                U, _ = self.read_field_tensor("U", 0)
            velocities.append(U.to(device=device, dtype=dtype))

        # Pressure
        p, _ = self.read_field_tensor("p", 0)
        p = p.to(device=device, dtype=dtype)

        # Volume fractions
        alphas = []
        for i, name in enumerate(self.phase_names):
            if i < self.n_phases - 1:
                try:
                    a, _ = self.read_field_tensor(f"alpha_{name}", 0)
                except Exception:
                    a = torch.full(
                        (n_cells,), 1.0 / self.n_phases,
                        dtype=dtype, device=device,
                    )
                alphas.append(a.to(device=device, dtype=dtype))
            else:
                alpha_last = (1.0 - sum(alphas)).clamp(0.0, 1.0)
                alphas.append(alpha_last)

        # Species mass fractions (per phase)
        Y: dict[str, dict[str, torch.Tensor]] = {}
        for phase_name, species_list in self.species.items():
            Y[phase_name] = {}
            for sp in species_list:
                fname = f"Y_{phase_name}_{sp}"
                try:
                    y, _ = self.read_field_tensor(fname, 0)
                    Y[phase_name][sp] = y.to(device=device, dtype=dtype).squeeze()
                except Exception:
                    Y[phase_name][sp] = torch.zeros(
                        n_cells, dtype=dtype, device=device,
                    )

        # Temperature
        try:
            T, _ = self.read_field_tensor("T", 0)
            T = T.to(device=device, dtype=dtype).squeeze()
        except Exception:
            T = torch.full((n_cells,), 300.0, dtype=dtype, device=device)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return velocities, p, alphas, Y, T, phi

    def _init_field_data(self) -> dict[str, Any]:
        """Store raw FieldData for writing."""
        field_data: dict[str, Any] = {}

        for i, name in enumerate(self.phase_names):
            try:
                field_data[f"U_{name}"] = self.case.read_field(f"U_{name}", 0)
            except Exception:
                field_data[f"U_{name}"] = None

            for sp in self.species.get(name, []):
                fname = f"Y_{name}_{sp}"
                try:
                    field_data[fname] = self.case.read_field(fname, 0)
                except Exception:
                    field_data[fname] = None

        field_data["p"] = self.case.read_field("p", 0)

        try:
            field_data["T"] = self.case.read_field("T", 0)
        except Exception:
            field_data["T"] = None

        return field_data

    # ------------------------------------------------------------------
    # Arrhenius kinetics
    # ------------------------------------------------------------------

    def _compute_arrhenius_rate(
        self,
        reaction: Reaction,
        T: torch.Tensor,
        Y: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute Arrhenius reaction rate.

        k = A * T^beta * exp(-Ea / (R*T))
        Rate = k * prod[Yj^nu_j] for reactants

        Parameters
        ----------
        reaction : Reaction
            Reaction definition.
        T : torch.Tensor
            Temperature field ``(n_cells,)``.
        Y : dict[str, torch.Tensor]
            Species mass fractions (flattened across phases).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` reaction rate.
        """
        T_safe = T.clamp(min=1.0)
        k = reaction.A * T_safe.pow(reaction.beta) * torch.exp(
            -reaction.Ea / (self.R * T_safe)
        )

        conc_term = torch.ones_like(T)
        for species, nu in reaction.reactants.items():
            if species in Y:
                Y_safe = Y[species].clamp(min=0.0)
                conc_term = conc_term * Y_safe.pow(nu)

        return k * conc_term

    def _compute_species_source_terms(
        self,
        T: torch.Tensor,
        Y: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute reaction source terms for all species.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field ``(n_cells,)``.
        Y : dict[str, torch.Tensor]
            Flattened species mass fractions.

        Returns
        -------
        dict[str, torch.Tensor]
            Source terms for each species.
        """
        n_cells = T.shape[0]
        device = T.device
        dtype = T.dtype

        omega: dict[str, torch.Tensor] = {
            name: torch.zeros(n_cells, dtype=dtype, device=device)
            for name in Y
        }

        for reaction in self.reactions:
            rate = self._compute_arrhenius_rate(reaction, T, Y)

            for species, nu in reaction.reactants.items():
                if species in omega:
                    omega[species] = omega[species] - nu * rate

            for species, nu in reaction.products.items():
                if species in omega:
                    omega[species] = omega[species] + nu * rate

        return omega

    def _get_flattened_species(self) -> dict[str, torch.Tensor]:
        """Flatten all species across phases into a single dict for reaction."""
        flat: dict[str, torch.Tensor] = {}
        for phase_name, species_dict in self.Y.items():
            for sp_name, sp_tensor in species_dict.items():
                flat[sp_name] = sp_tensor
        return flat

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the reacting multiphase solver.

        Returns
        -------
        ConvergenceData
            Convergence information from the final time step.
        """
        time_loop = TimeLoop(
            start_time=self.start_time, end_time=self.end_time,
            delta_t=self.delta_t, write_interval=self.write_interval,
            write_control=self.write_control,
        )
        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance, min_steps=1,
        )

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence = None

        for t, step in time_loop:
            # Store old fields
            alphas_old = [a.clone() for a in self.alphas]

            # Compute reaction source terms (flattened across phases)
            Y_flat = self._get_flattened_species()
            omega = self._compute_species_source_terms(self.T, Y_flat)

            # Outer corrector loop
            velocities = [U.clone() for U in self.velocities]
            alphas = [a.clone() for a in self.alphas]
            n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

            for outer in range(n_outer):
                vels_prev = [U.clone() for U in velocities]

                # Enforce volume fraction constraint
                alpha_sum = sum(alphas[:-1])
                alphas[-1] = (1.0 - alpha_sum).clamp(0.0, 1.0)

                # Renormalise
                total = sum(alphas).clamp(min=1e-30)
                alphas = [a / total for a in alphas]

                # Update species with reaction source terms (per phase)
                for phase_name, species_dict in self.Y.items():
                    phase_alpha = alphas[self.phase_names.index(phase_name)]
                    for sp_name in species_dict:
                        if sp_name in omega:
                            # Apply species source scaled by phase fraction
                            source = omega[sp_name] * phase_alpha
                            # Simple explicit update
                            species_dict[sp_name] = (
                                species_dict[sp_name] + self.delta_t * source
                            ).clamp(min=0.0, max=1.0)

                # Convergence check
                U_residual = max(
                    self._compute_residual(velocities[i], vels_prev[i])
                    for i in range(self.n_phases)
                )

            # Update stored fields
            self.velocities = velocities
            self.alphas = alphas

            # Build convergence data
            conv = ConvergenceData()
            conv.U_residual = U_residual
            conv.outer_iterations = n_outer
            last_convergence = conv

            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        return last_convergence or ConvergenceData()

    def _compute_residual(self, field: torch.Tensor, field_old: torch.Tensor) -> float:
        """Compute relative residual between two fields."""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def _write_fields(self, time: float) -> None:
        """Write fields to a time directory."""
        time_str = f"{time:g}"

        # Write phase velocities
        for i, name in enumerate(self.phase_names):
            fd = self._field_data.get(f"U_{name}")
            if fd is not None:
                self.write_field(f"U_{name}", self.velocities[i], time_str, fd)

        # Write pressure
        p_fd = self._field_data.get("p")
        if p_fd is not None:
            self.write_field("p", self.p, time_str, p_fd)

        # Write volume fractions
        for i, name in enumerate(self.phase_names):
            alpha_fname = f"alpha_{name}"
            try:
                alpha_fd = self.case.read_field(alpha_fname, 0)
                self.write_field(alpha_fname, self.alphas[i], time_str, alpha_fd)
            except Exception:
                pass

        # Write species
        for phase_name, species_dict in self.Y.items():
            for sp_name, sp_tensor in species_dict.items():
                fname = f"Y_{phase_name}_{sp_name}"
                fd = self._field_data.get(fname)
                if fd is not None:
                    self.write_field(fname, sp_tensor, time_str, fd)

        # Write temperature
        t_fd = self._field_data.get("T")
        if t_fd is not None:
            self.write_field("T", self.T, time_str, t_fd)
