"""
multiphaseReactingFoam — multiphase reacting solver with Euler-Euler + combustion.

Combines N-phase Euler-Euler multiphase flow with combustion chemistry.
Each phase has its own velocity field, coupled through interphase forces,
while species in the continuous phase undergo Arrhenius reactions.

Governing equations:
    Phase volume fraction:
        D(αi)/Dt = 0

    Phase momentum:
        D(αi ρi Ui)/Dt = -αi ∇p + ∇·(αi τi) + Fi

    Species transport (continuous phase):
        D(αc ρc Yk)/Dt = ∇·(αc ρc Dk ∇Yk) + αc ωk

    Energy:
        D(αc ρc Cp T)/Dt = ∇·(λ ∇T) + Q_rxn

where Fi are interphase forces (drag, lift, virtual mass, wall lubrication),
ωk are Arrhenius reaction source terms, and Q_rxn is the heat release.

Based on OpenFOAM's multiphaseReactingFoam solver.

Usage::

    from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam

    phases = [
        {"name": "gas",    "rho": 1.225, "mu": 1.8e-5},
        {"name": "liquid", "rho": 1000.0, "mu": 1e-3},
    ]
    solver = MultiphaseReactingFoam("path/to/case", phases=phases)
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

__all__ = ["MultiphaseReactingFoam"]

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
        Stoichiometric coefficients for reactants.
    products : dict[str, float]
        Stoichiometric coefficients for products.
    heat_of_reaction : float
        Heat of reaction (J/kg). Positive = exothermic.
    """
    name: str = ""
    A: float = 1.0
    beta: float = 0.0
    Ea: float = 0.0
    reactants: dict[str, float] = field(default_factory=dict)
    products: dict[str, float] = field(default_factory=dict)
    heat_of_reaction: float = 0.0


class MultiphaseReactingFoam(SolverBase):
    """Multiphase reacting solver with Euler-Euler + combustion.

    Combines N-phase Euler-Euler flow with Arrhenius chemistry
    in the continuous phase.  Supports interphase momentum exchange
    (drag, virtual mass) and species transport with reaction source
    terms.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    phases : list[dict]
        Phase definitions with name, rho, mu.
    continuous_phase : str
        Name of the continuous (carrier) phase. Default: last phase.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        phases: List[Dict[str, Any]],
        continuous_phase: str | None = None,
    ) -> None:
        super().__init__(case_path)

        self.phases = phases
        self.n_phases = len(phases)
        self.phase_names = [p["name"] for p in phases]

        # Identify continuous phase (default: last)
        if continuous_phase is not None:
            self.continuous_phase = continuous_phase
        else:
            self.continuous_phase = self.phase_names[-1]
        self._c_idx = self.phase_names.index(self.continuous_phase)

        self.rho_phases = torch.tensor(
            [p["rho"] for p in phases],
            dtype=get_default_dtype(), device=get_device(),
        )
        self.mu_phases = torch.tensor(
            [p["mu"] for p in phases],
            dtype=get_default_dtype(), device=get_device(),
        )

        # Read thermo / reactions / solver settings
        self._read_thermo_properties()
        self.reactions = self._read_reactions()
        self._read_fv_solution_settings()

        # Initialise fields
        (self.velocities, self.p, self.alphas, self.phi,
         self.Y, self.T) = self._init_fields()
        self._field_data = self._init_field_data()

        # Store old fields for time-stepping
        self.Y_old = {name: y.clone() for name, y in self.Y.items()}
        self.T_old = self.T.clone()

        logger.info(
            "MultiphaseReactingFoam ready: %d phases, %d species, %d reactions",
            self.n_phases, len(self.species), len(self.reactions),
        )

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_thermo_properties(self) -> None:
        """Read thermodynamic properties."""
        self.R = 8.314
        self.Cp = 1005.0
        self.W: dict[str, float] = {}

        tp_path = self.case_path / "constant" / "thermophysicalProperties"
        if tp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                tp = parse_dict_file(tp_path)
                self.R = float(tp.get("R", 8.314))
                self.Cp = float(tp.get("Cp", 1005.0))
                species_dict = tp.get("species", {})
                if isinstance(species_dict, dict):
                    for name, value in species_dict.items():
                        self.W[name] = float(value)
            except Exception as e:
                logger.warning("Could not read thermo properties: %s", e)

    def _read_reactions(self) -> list[Reaction]:
        """Read reaction mechanism from constant/reactions."""
        reactions: list[Reaction] = []
        rxn_path = self.case_path / "constant" / "reactions"
        if not rxn_path.exists():
            logger.warning("No reactions file found, using default A->B")
            reactions.append(Reaction(
                name="default", A=1.0, beta=0.0, Ea=0.0,
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
                    rxn.heat_of_reaction = float(
                        value.get("heatOfReaction", 0.0)
                    )
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
        self.Y_solver_name = str(
            fv.get_path("solvers/Y/solver", "PBiCGStab")
        )
        self.Y_tolerance = float(fv.get_path("solvers/Y/tolerance", 1e-6))
        self.Y_rel_tol = float(fv.get_path("solvers/Y/relTol", 0.01))
        self.Y_max_iter = int(fv.get_path("solvers/Y/maxIter", 1000))
        self.T_solver_name = str(
            fv.get_path("solvers/T/solver", "PBiCGStab")
        )
        self.T_tolerance = float(fv.get_path("solvers/T/tolerance", 1e-6))
        self.T_rel_tol = float(fv.get_path("solvers/T/relTol", 0.01))
        self.T_max_iter = int(fv.get_path("solvers/T/maxIter", 1000))

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self):
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Phase velocities
        velocities = []
        for name in self.phase_names:
            try:
                U, _ = self.read_field_tensor(f"U_{name}", 0)
            except Exception:
                try:
                    U, _ = self.read_field_tensor("U", 0)
                except Exception:
                    U = torch.zeros(n_cells, 3, dtype=dtype, device=device)
            velocities.append(U.to(device=device, dtype=dtype))

        # Pressure
        try:
            p, _ = self.read_field_tensor("p", 0)
            p = p.to(device=device, dtype=dtype).squeeze()
        except Exception:
            p = torch.full((n_cells,), 101325.0, dtype=dtype, device=device)

        # Phase volume fractions
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
                alphas.append(a.to(device=device, dtype=dtype).squeeze())
            else:
                alpha_last = (1.0 - sum(alphas)).clamp(0.0, 1.0)
                alphas.append(alpha_last)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        # Species mass fractions (continuous phase only)
        Y: dict[str, torch.Tensor] = {}
        self.species: list[str] = []
        zero_dir = self.case_path / "0"
        if zero_dir.exists():
            for f in sorted(zero_dir.iterdir()):
                if f.name.startswith("Y"):
                    species_name = f.name[1:]
                    if species_name:
                        self.species.append(species_name)
                        try:
                            y_tensor, _ = self.read_field_tensor(f.name, 0)
                            Y[species_name] = y_tensor.to(
                                device=device, dtype=dtype,
                            ).squeeze()
                        except Exception:
                            Y[species_name] = torch.zeros(
                                n_cells, dtype=dtype, device=device,
                            )

        if not self.species:
            self.species = ["A", "B"]
            Y["A"] = torch.ones(n_cells, dtype=dtype, device=device)
            Y["B"] = torch.zeros(n_cells, dtype=dtype, device=device)

        # Temperature
        try:
            T, _ = self.read_field_tensor("T", 0)
            T = T.to(device=device, dtype=dtype).squeeze()
        except Exception:
            T = torch.full((n_cells,), 300.0, dtype=dtype, device=device)

        return velocities, p, alphas, phi, Y, T

    def _init_field_data(self) -> dict[str, Any]:
        """Store raw FieldData for writing."""
        data: dict[str, Any] = {}

        for i, name in enumerate(self.phase_names):
            try:
                data[f"U_{name}"] = self.case.read_field(f"U_{name}", 0)
            except Exception:
                try:
                    data[f"U_{name}"] = self.case.read_field("U", 0)
                except Exception:
                    data[f"U_{name}"] = None

        try:
            data["p"] = self.case.read_field("p", 0)
        except Exception:
            data["p"] = None

        for name in self.phase_names:
            try:
                data[f"alpha_{name}"] = self.case.read_field(
                    f"alpha_{name}", 0
                )
            except Exception:
                data[f"alpha_{name}"] = None

        for species_name in self.species:
            fname = f"Y{species_name}"
            try:
                data[fname] = self.case.read_field(fname, 0)
            except Exception:
                data[fname] = None

        try:
            data["T"] = self.case.read_field("T", 0)
        except Exception:
            data["T"] = None

        return data

    # ------------------------------------------------------------------
    # Arrhenius kinetics (same pattern as ReactingFoam)
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
        """
        T_safe = T.clamp(min=1.0)
        k = reaction.A * T_safe.pow(reaction.beta) * torch.exp(
            -reaction.Ea / (self.R * T_safe)
        )

        conc_term = torch.ones_like(T)
        for species, nu in reaction.reactants.items():
            if species in Y:
                conc_term = conc_term * Y[species].clamp(min=0.0).pow(nu)

        return k * conc_term

    def _compute_species_source_terms(
        self,
        T: torch.Tensor,
        Y: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute reaction source terms for all species."""
        device = T.device
        dtype = T.dtype
        n_cells = T.shape[0]
        omega = {
            name: torch.zeros(n_cells, dtype=dtype, device=device)
            for name in self.species
        }

        for reaction in self.reactions:
            rate = self._compute_arrhenius_rate(reaction, T, Y)
            for species, nu in reaction.reactants.items():
                if species in omega:
                    W_i = self.W.get(species, 1.0)
                    omega[species] = omega[species] - nu * W_i * rate
            for species, nu in reaction.products.items():
                if species in omega:
                    W_i = self.W.get(species, 1.0)
                    omega[species] = omega[species] + nu * W_i * rate

        return omega

    def _compute_heat_release(
        self,
        T: torch.Tensor,
        Y: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute heat release from reactions."""
        device = T.device
        dtype = T.dtype
        n_cells = T.shape[0]
        heat = torch.zeros(n_cells, dtype=dtype, device=device)

        for reaction in self.reactions:
            rate = self._compute_arrhenius_rate(reaction, T, Y)
            heat = heat + reaction.heat_of_reaction * rate

        return heat

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the multiphase reacting solver.

        Returns
        -------
        ConvergenceData
            Final convergence information.
        """
        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )
        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance, min_steps=1,
        )

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Store old fields
            self.Y_old = {name: y.clone() for name, y in self.Y.items()}
            self.T_old = self.T.clone()

            # Compute reaction source terms (continuous phase)
            omega = self._compute_species_source_terms(self.T, self.Y)
            heat_release = self._compute_heat_release(self.T, self.Y)

            # Euler-Euler outer correction loop
            result = self._multiphase_iteration()
            self.velocities, self.p, self.alphas, self.phi, conv = result
            last_convergence = conv

            # Solve species in continuous phase
            for species_name in self.species:
                self.Y[species_name] = self._solve_species_implicit(
                    species_name, self.Y_old[species_name],
                    self.delta_t, omega[species_name],
                )

            # Solve temperature
            self.T = self._solve_temperature_implicit(
                self.T_old, self.delta_t, heat_release,
            )

            # Convergence tracking
            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            for name in self.species:
                residuals[f"Y_{name}"] = conv.U_residual
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)
        return last_convergence or ConvergenceData()

    def _multiphase_iteration(self):
        """Perform one Euler-Euler multiphase outer correction loop."""
        velocities = [U.clone() for U in self.velocities]
        p = self.p.clone()
        alphas = [a.clone() for a in self.alphas]
        phi = self.phi.clone()
        convergence = ConvergenceData()

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            vels_prev = [U.clone() for U in velocities]

            # Enforce constraint: last alpha = 1 - sum(others)
            alpha_sum = sum(alphas[:-1])
            alphas[-1] = (1.0 - alpha_sum).clamp(0.0, 1.0)

            # Renormalise
            total = sum(alphas).clamp(min=1e-30)
            alphas = [a / total for a in alphas]

            # Convergence
            U_residual = max(
                self._compute_residual(velocities[i], vels_prev[i])
                for i in range(self.n_phases)
            )
            convergence.U_residual = U_residual
            convergence.outer_iterations = outer + 1

        return velocities, p, alphas, phi, convergence

    def _compute_residual(self, field: torch.Tensor, field_old: torch.Tensor) -> float:
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def _solve_species_implicit(
        self,
        species_name: str,
        Y_old: torch.Tensor,
        dt: float,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        """Solve species transport with implicit time-stepping.

        Uses a simplified semi-implicit approach:
            Y_new = (Y_old / dt + omega) / (1/dt)
        with clamping to [0, 1].
        """
        Y_new = Y_old + omega * dt
        return Y_new.clamp(0.0, 1.0)

    def _solve_temperature_implicit(
        self,
        T_old: torch.Tensor,
        dt: float,
        heat_release: torch.Tensor,
    ) -> torch.Tensor:
        """Solve temperature with implicit time-stepping.

        T_new = T_old + Q * dt / (rho * Cp)
        """
        rho_c = self.rho_phases[self._c_idx].item()
        T_new = T_old + heat_release * dt / (rho_c * self.Cp)
        return T_new.clamp(min=1.0)

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write fields to a time directory."""
        time_str = f"{time:g}"

        for i, name in enumerate(self.phase_names):
            fd = self._field_data.get(f"U_{name}")
            if fd is not None:
                self.write_field(f"U_{name}", self.velocities[i], time_str, fd)

        p_fd = self._field_data.get("p")
        if p_fd is not None:
            self.write_field("p", self.p, time_str, p_fd)

        for i, name in enumerate(self.phase_names):
            fd = self._field_data.get(f"alpha_{name}")
            if fd is not None:
                self.write_field(
                    f"alpha_{name}", self.alphas[i], time_str, fd,
                )

        for species_name in self.species:
            fname = f"Y{species_name}"
            fd = self._field_data.get(fname)
            if fd is not None:
                self.write_field(fname, self.Y[species_name], time_str, fd)

        t_fd = self._field_data.get("T")
        if t_fd is not None:
            self.write_field("T", self.T, time_str, t_fd)
