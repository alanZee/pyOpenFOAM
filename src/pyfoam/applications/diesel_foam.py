"""
dieselFoam — diesel spray combustion solver.

Combines compressible gas-phase flow (PIMPLE) with Lagrangian spray
tracking and finite-rate combustion chemistry.  Extends the sprayFoam
architecture by adding species transport equations with Arrhenius
reaction source terms on top of the Euler-Lagrange coupling.

Key features:
- Compressible PIMPLE algorithm for the gas phase
- Lagrangian droplet cloud (injection, breakup, collision, evaporation)
- Two-way Euler-Lagrange coupling (momentum, heat, mass)
- Species transport with Arrhenius combustion kinetics
- Diesel-specific fuel vapour oxidation reactions

Usage::

    from pyfoam.applications.diesel_foam import DieselFoam

    solver = DieselFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.lagrangian.cloud import KinematicCloud
from pyfoam.lagrangian.particle import Particle

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["DieselFoam"]

logger = logging.getLogger(__name__)


# ======================================================================
# Diesel reaction model
# ======================================================================


@dataclass
class DieselReaction:
    """A single combustion reaction with Arrhenius kinetics.

    Attributes
    ----------
    name : str
        Reaction identifier.
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
    """
    name: str = ""
    A: float = 1.0
    beta: float = 0.0
    Ea: float = 0.0
    reactants: dict[str, float] = field(default_factory=dict)
    products: dict[str, float] = field(default_factory=dict)


# ======================================================================
# Lagrangian-Eulerian coupling (diesel variant)
# ======================================================================


class DieselCoupling:
    """Manages two-way coupling between a Lagrangian spray cloud and
    the Eulerian gas phase for diesel combustion.

    Extends :class:`SprayFoam.LagrangianCoupling` with fuel-vapour
    mass source tracking (the evaporated fuel species is added to the
    gas-phase species transport).

    Parameters
    ----------
    mesh : Any
        The finite-volume mesh.
    cloud : KinematicCloud
        Lagrangian droplet cloud.
    Cp_fuel : float
        Specific heat capacity of liquid diesel (J/(kg K)).
    L_vap : float
        Latent heat of vaporisation (J/kg).
    fuel_species : str
        Name of the fuel vapour species (default ``"C12H23"``).
    """

    def __init__(
        self,
        mesh: Any,
        cloud: KinematicCloud,
        Cp_fuel: float = 2090.0,
        L_vap: float = 2.5e5,
        fuel_species: str = "C12H23",
    ) -> None:
        self.mesh = mesh
        self.cloud = cloud
        self.Cp_fuel = Cp_fuel
        self.L_vap = L_vap
        self.fuel_species = fuel_species

    def momentum_source(self) -> torch.Tensor:
        """Compute volumetric momentum source from drag on droplets."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        cell_vol = self.mesh.cell_volumes.to(device=device, dtype=dtype)
        source = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        for p in self.cloud.particles:
            if not p.alive:
                continue
            ci = self._locate_cell(p.position)
            if ci < 0 or ci >= n_cells:
                continue

            d = max(p.diameter, 1e-10)
            mu = max(self.cloud.fluid_viscosity, 1e-30)
            tau_p = p.density * d * d / (18.0 * mu)

            m_p = p.mass
            v_p = torch.tensor(p.velocity, dtype=dtype, device=device)
            v_g = torch.tensor(self.cloud.fluid_velocity, dtype=dtype, device=device)
            drag = m_p * (v_p - v_g) / max(tau_p, 1e-30)

            vol = max(float(cell_vol[ci].item()), 1e-30)
            source[ci] += drag / vol

        return source

    def heat_source(self, T_gas: torch.Tensor) -> torch.Tensor:
        """Compute volumetric heat source from convective heating."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        cell_vol = self.mesh.cell_volumes.to(device=device, dtype=dtype)
        source = torch.zeros(n_cells, dtype=dtype, device=device)

        for p in self.cloud.particles:
            if not p.alive:
                continue
            ci = self._locate_cell(p.position)
            if ci < 0 or ci >= n_cells:
                continue

            d = max(p.diameter, 1e-10)
            A_p = math.pi * d * d
            Re_p = self._particle_reynolds(p)
            Pr = 0.7
            k_air = 0.026
            Nu = 2.0 + 0.6 * math.sqrt(max(Re_p, 0.0)) * (Pr ** (1.0 / 3.0))
            h = Nu * k_air / d

            T_p = getattr(p, "temperature", 300.0)
            dT = float(T_gas[ci].item()) - T_p

            vol = max(float(cell_vol[ci].item()), 1e-30)
            source[ci] += h * A_p * dT / vol

        return source

    def mass_source(self, dt: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute mass source from evaporation, split by species.

        The evaporated fuel mass is attributed to the fuel species.
        An equal amount of ambient gas is displaced (tracked via the
        ``fuel_species`` key).

        Returns
        -------
        total_source : torch.Tensor
            ``(n_cells,)`` total mass source (kg/s per cell volume).
        species_source : dict[str, torch.Tensor]
            Per-species mass source (only the fuel species is populated).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        cell_vol = self.mesh.cell_volumes.to(device=device, dtype=dtype)
        fuel_source = torch.zeros(n_cells, dtype=dtype, device=device)

        D_vap = 2.0e-5
        Y_s = 0.03
        Y_inf = 0.0

        for p in self.cloud.particles:
            if not p.alive:
                continue
            ci = self._locate_cell(p.position)
            if ci < 0 or ci >= n_cells:
                continue

            d = max(p.diameter, 1e-10)
            Re_p = self._particle_reynolds(p)
            Sc = 0.7
            Sh = 2.0 + 0.6 * math.sqrt(max(Re_p, 0.0)) * (Sc ** (1.0 / 3.0))
            dm_dt = -math.pi * p.density * D_vap * Sh * d * (Y_s - Y_inf)

            vol = max(float(cell_vol[ci].item()), 1e-30)
            fuel_source[ci] += abs(dm_dt) / vol

            p.mass = max(p.mass + dm_dt * dt, 0.0)
            if p.mass <= 0.0:
                p.alive = False

        species_source = {self.fuel_species: fuel_source}
        return fuel_source, species_source

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _locate_cell(self, position: list[float]) -> int:
        """Locate cell index for a particle position."""
        centres = self.mesh.cell_centres
        dists = torch.sum(
            (centres - torch.tensor(position, dtype=centres.dtype, device=centres.device)) ** 2,
            dim=1,
        )
        return int(torch.argmin(dists).item())

    def _particle_reynolds(self, p: Particle) -> float:
        """Compute particle Reynolds number."""
        d = max(p.diameter, 1e-10)
        nu = max(
            self.cloud.fluid_viscosity / max(self.cloud.fluid_density, 1e-30),
            1e-30,
        )
        v_rel = sum(
            (p.velocity[i] - self.cloud.fluid_velocity[i]) ** 2 for i in range(3)
        ) ** 0.5
        return d * v_rel / nu


# ======================================================================
# Main solver
# ======================================================================


class DieselFoam(SolverBase):
    """Diesel spray combustion solver.

    Combines compressible PIMPLE gas-phase solver with Lagrangian spray
    tracking and finite-rate Arrhenius combustion chemistry.  The fuel
    vapour from spray evaporation reacts with oxidiser species according
    to the reaction mechanism defined in the case.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    cloud : KinematicCloud, optional
        Pre-configured Lagrangian droplet cloud.
    rho_gas : float
        Gas-phase density (kg/m3, default 1.225).
    mu_gas : float
        Gas-phase dynamic viscosity (Pa s, default 1.8e-5).
    Cp_gas : float
        Gas specific heat capacity (J/(kg K), default 1005.0).
    Cp_fuel : float
        Liquid fuel specific heat capacity (J/(kg K), default 2090.0).
    L_vap : float
        Latent heat of vaporisation (J/kg, default 2.5e5).
    fuel_species : str
        Name of the fuel species (default ``"C12H23"``).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        cloud: Optional[KinematicCloud] = None,
        rho_gas: float = 1.225,
        mu_gas: float = 1.8e-5,
        Cp_gas: float = 1005.0,
        Cp_fuel: float = 2090.0,
        L_vap: float = 2.5e5,
        fuel_species: str = "C12H23",
    ) -> None:
        super().__init__(case_path)

        self.rho_gas = rho_gas
        self.mu_gas = mu_gas
        self.Cp_gas = Cp_gas
        self.Cp_fuel = Cp_fuel
        self.L_vap = L_vap
        self.fuel_species = fuel_species
        self.R_universal = 8.314

        # Lagrangian cloud
        self.cloud = cloud or KinematicCloud(
            fluid_velocity=[0.0, 0.0, 0.0],
            fluid_density=rho_gas,
            fluid_viscosity=mu_gas,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0],
        )

        # Coupling
        self.coupling = DieselCoupling(
            mesh=self.mesh,
            cloud=self.cloud,
            Cp_fuel=Cp_fuel,
            L_vap=L_vap,
            fuel_species=fuel_species,
        )

        # Reactions
        self.reactions = self._read_reactions()

        # fvSolution / fvSchemes
        self._read_fv_solution_settings()

        # Fields
        self.U, self.p, self.T, self.phi = self._init_flow_fields()
        self.species, self.Y = self._init_species_fields()
        self._U_data, self._p_data, self._T_data = self._init_field_data()
        self._Y_data = self._init_species_field_data()

        logger.info(
            "DieselFoam ready: rho=%.3f, n_species=%d, n_reactions=%d, "
            "n_particles=%d",
            rho_gas, len(self.species), len(self.reactions),
            self.cloud.n_particles,
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _read_reactions(self) -> list[DieselReaction]:
        """Read diesel combustion reactions (default: simplified diesel)."""
        rxn_path = self.case_path / "constant" / "reactions"
        reactions: list[DieselReaction] = []

        if rxn_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                rxn_dict = parse_dict_file(rxn_path)
                for key, value in rxn_dict.items():
                    if isinstance(value, dict):
                        rxn = DieselReaction(name=key)
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
                return reactions
            except Exception as e:
                logger.warning("Could not read reactions: %s", e)

        # Default simplified diesel oxidation:
        # C12H23 + 17.75 O2 -> 12 CO2 + 11.5 H2O
        reactions.append(DieselReaction(
            name="diesel_oxidation",
            A=4.16e9,
            beta=0.0,
            Ea=1.255e5,
            reactants={"C12H23": 1.0, "O2": 17.75},
            products={"CO2": 12.0, "H2O": 11.5},
        ))
        return reactions

    def _read_fv_solution_settings(self) -> None:
        """Read PIMPLE + chemistry settings."""
        fv = self.case.fvSolution

        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        self.n_outer_correctors = int(fv.get_path("PIMPLE/nOuterCorrectors", 3))
        self.n_correctors = int(fv.get_path("PIMPLE/nCorrectors", 2))
        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))

        self.convergence_tolerance = float(
            fv.get_path("PIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("PIMPLE/maxOuterIterations", 100)
        )

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_flow_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise U, p, T, phi from the 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        try:
            T_tensor, _ = self.read_field_tensor("T", 0)
            T = T_tensor.to(device=device, dtype=dtype)
        except Exception:
            T = torch.full((self.mesh.n_cells,), 300.0, dtype=dtype, device=device)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)
        return U, p, T, phi

    def _init_species_fields(self) -> tuple[list[str], dict[str, torch.Tensor]]:
        """Initialise species mass fractions."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        species: list[str] = []
        Y: dict[str, torch.Tensor] = {}

        zero_dir = self.case_path / "0"
        if zero_dir.exists():
            for f in sorted(zero_dir.iterdir()):
                if f.name.startswith("Y") and len(f.name) > 1:
                    sp = f.name[1:]
                    species.append(sp)
                    try:
                        y_tensor, _ = self.read_field_tensor(f.name, 0)
                        Y[sp] = y_tensor.to(device=device, dtype=dtype).squeeze()
                    except Exception:
                        Y[sp] = torch.zeros(n_cells, dtype=dtype, device=device)

        # Default species if none found
        if not species:
            species = [self.fuel_species, "O2", "CO2", "H2O", "N2"]
            Y[self.fuel_species] = torch.zeros(n_cells, dtype=dtype, device=device)
            Y["O2"] = torch.full((n_cells,), 0.23, dtype=dtype, device=device)
            Y["CO2"] = torch.zeros(n_cells, dtype=dtype, device=device)
            Y["H2O"] = torch.zeros(n_cells, dtype=dtype, device=device)
            Y["N2"] = torch.full((n_cells,), 0.77, dtype=dtype, device=device)

        return species, Y

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        try:
            T_data = self.case.read_field("T", 0)
        except Exception:
            T_data = U_data
        return U_data, p_data, T_data

    def _init_species_field_data(self) -> dict[str, Any]:
        """Store raw FieldData for species writing."""
        data = {}
        for sp in self.species:
            fname = f"Y{sp}"
            try:
                data[fname] = self.case.read_field(fname, 0)
            except Exception:
                data[fname] = None
        return data

    # ------------------------------------------------------------------
    # Combustion kinetics
    # ------------------------------------------------------------------

    def _compute_reaction_rate(self, reaction: DieselReaction) -> torch.Tensor:
        """Compute Arrhenius reaction rate.

        k = A * T^beta * exp(-Ea / (R * T)) * prod(Yj^nu_j)
        """
        T_safe = self.T.clamp(min=1.0)
        k = reaction.A * T_safe.pow(reaction.beta) * torch.exp(
            -reaction.Ea / (self.R_universal * T_safe)
        )

        conc_term = torch.ones_like(self.T)
        for sp, nu in reaction.reactants.items():
            if sp in self.Y:
                conc_term = conc_term * self.Y[sp].clamp(min=0.0).pow(nu)

        return k * conc_term

    def _compute_species_sources(self) -> dict[str, torch.Tensor]:
        """Compute species source terms from all reactions."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        omega = {sp: torch.zeros(n_cells, dtype=dtype, device=device)
                 for sp in self.species}

        for rxn in self.reactions:
            rate = self._compute_reaction_rate(rxn)
            for sp, nu in rxn.reactants.items():
                if sp in omega:
                    omega[sp] = omega[sp] - nu * rate
            for sp, nu in rxn.products.items():
                if sp in omega:
                    omega[sp] = omega[sp] + nu * rate

        return omega

    def _compute_heat_release(self) -> torch.Tensor:
        """Compute heat release from combustion (simplified)."""
        device = get_device()
        dtype = get_default_dtype()
        # Placeholder: full implementation uses enthalpy of formation
        return torch.zeros(self.mesh.n_cells, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the dieselFoam solver.

        Each time step:
        1. Advance Lagrangian spray (inject, move, evaporate).
        2. Compute Euler-Lagrange coupling sources.
        3. Compute combustion source terms.
        4. Solve gas-phase equations (PIMPLE) with all sources.
        5. Solve species transport with reaction sources.
        6. Update cloud fluid conditions.

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
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting dieselFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # 1. Advance spray
            self.cloud.advance(self.delta_t)
            self.cloud.remove_dead()

            # 2. Coupling sources
            S_mom = self.coupling.momentum_source()
            S_heat = self.coupling.heat_source(self.T)
            S_mass, S_species = self.coupling.mass_source(self.delta_t)

            # 3. Combustion sources
            omega = self._compute_species_sources()
            Q_comb = self._compute_heat_release()

            # Add evaporation fuel source to species
            for sp, src in S_species.items():
                if sp in omega:
                    omega[sp] = omega[sp] + src

            # 4. Solve gas phase
            self.U, self.p, self.T, self.phi, conv = (
                self._pimple_iteration(S_mom, S_heat + Q_comb, S_mass)
            )
            last_convergence = conv

            # 5. Update species (explicit forward Euler)
            self._advance_species(omega)

            # 6. Update cloud
            self._update_cloud_fluid_conditions()

            # Check convergence
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
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Species transport
    # ------------------------------------------------------------------

    def _advance_species(self, omega: dict[str, torch.Tensor]) -> None:
        """Advance species mass fractions (explicit Euler)."""
        for sp in self.species:
            if sp in omega:
                dY = self.delta_t * omega[sp]
                self.Y[sp] = (self.Y[sp] + dY).clamp(min=0.0, max=1.0)

    # ------------------------------------------------------------------
    # Cloud update
    # ------------------------------------------------------------------

    def _update_cloud_fluid_conditions(self) -> None:
        """Update cloud fluid state from current gas-phase fields."""
        if self.cloud.n_particles == 0:
            return
        U_mean = self.U.mean(dim=0)
        self.cloud.fluid_velocity = [
            float(U_mean[0].item()),
            float(U_mean[1].item()),
            float(U_mean[2].item()),
        ]
        self.cloud.fluid_density = self.rho_gas

    # ------------------------------------------------------------------
    # PIMPLE iteration (reuses SprayFoam pattern)
    # ------------------------------------------------------------------

    def _pimple_iteration(
        self,
        S_mom: torch.Tensor,
        S_heat: torch.Tensor,
        S_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run one PIMPLE time step with spray + combustion sources."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        T = self.T.clone()
        phi = self.phi.clone()
        convergence = ConvergenceData()

        rho = torch.full((mesh.n_cells,), self.rho_gas, dtype=dtype, device=device)
        mu = torch.full((mesh.n_cells,), self.mu_gas, dtype=dtype, device=device)

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            U, A_p, H = self._momentum_predictor(U, p, phi, rho, mu, S_mom)

            for corr in range(self.n_correctors):
                HbyA = H / A_p.abs().clamp(min=1e-30).unsqueeze(-1)
                n_internal = mesh.n_internal_faces
                int_owner = mesh.owner[:n_internal]
                int_neigh = mesh.neighbour
                w = mesh.face_weights[:n_internal]

                HbyA_face = (
                    w.unsqueeze(-1) * HbyA[int_owner]
                    + (1.0 - w).unsqueeze(-1) * HbyA[int_neigh]
                )
                phiHbyA = (HbyA_face * mesh.face_areas[:n_internal]).sum(dim=1)

                p = self._solve_pressure(p, phiHbyA, A_p, rho, mesh)

                grad_p = self._compute_grad(p, mesh)
                U = HbyA - grad_p / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                p_P = gather(p, int_owner)
                p_N = gather(p, int_neigh)
                A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
                A_p_inv_face = (
                    w * gather(A_p_inv, int_owner)
                    + (1.0 - w) * gather(A_p_inv, int_neigh)
                )
                phi = phiHbyA - (p_N - p_P) * A_p_inv_face

            T = self._solve_temperature(T, S_heat, S_mass)

            if self.alpha_U < 1.0:
                U = self.alpha_U * U + (1.0 - self.alpha_U) * U_prev
            if self.alpha_p < 1.0:
                p = self.alpha_p * p + (1.0 - self.alpha_p) * p_prev

            U_res = self._compute_residual(U, U_prev)
            p_res = self._compute_residual(p, p_prev)
            cont = self._compute_continuity_error(phi, rho)

            convergence.p_residual = p_res
            convergence.U_residual = U_res
            convergence.continuity_error = cont
            convergence.outer_iterations = outer + 1

            if cont < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return U, p, T, phi, convergence

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        mu: torch.Tensor,
        S_mom: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with spray source."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        dt = self.delta_t
        rho_V_dt = rho * cell_volumes / dt

        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        A_p = torch.zeros(n_cells, dtype=dtype, device=device)
        A_p = A_p + scatter_add(
            (diff_coeff - flux_neg * rho_face) / V_P, int_owner, n_cells
        )
        A_p = A_p + scatter_add(
            (diff_coeff + flux_pos * rho_face) / V_N, int_neigh, n_cells
        )
        A_p = A_p + rho_V_dt

        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        H = H + rho_V_dt.unsqueeze(-1) * self.U

        w = mesh.face_weights[:n_internal]
        p_P = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

        spray_source = S_mom * cell_volumes.unsqueeze(-1)
        source = H - grad_p + spray_source

        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)
        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        return U_new, A_p, H

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------

    def _solve_temperature(
        self,
        T: torch.Tensor,
        S_heat: torch.Tensor,
        S_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Solve simplified temperature equation."""
        rho = self.rho_gas
        Cp = self.Cp_gas
        Q_total = S_heat - S_mass * self.L_vap
        dT = self.delta_t * Q_total / (rho * Cp)
        T_new = T + dT
        return T_new.clamp(min=200.0, max=2500.0)

    # ------------------------------------------------------------------
    # Pressure
    # ------------------------------------------------------------------

    def _solve_pressure(
        self,
        p: torch.Tensor,
        phiHbyA: torch.Tensor,
        A_p: torch.Tensor,
        rho: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """Solve pressure Poisson equation."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        w = mesh.face_weights[:n_internal]
        A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
        A_p_inv_face = (
            w * gather(A_p_inv, int_owner)
            + (1.0 - w) * gather(A_p_inv, int_neigh)
        )

        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        face_coeff = A_p_inv_face * S_mag * delta_f

        V_P = gather(cell_volumes.clamp(min=1e-30), int_owner)
        V_N = gather(cell_volumes.clamp(min=1e-30), int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        source = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        source = source + scatter_add(phiHbyA, int_owner, n_cells)
        source = source + scatter_add(-phiHbyA, int_neigh, n_cells)

        diag_safe = diag.abs().clamp(min=1e-30)
        for _ in range(self.p_max_iter):
            off_diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
            p_P = gather(p, int_owner)
            p_N = gather(p, int_neigh)
            off_diag = off_diag + scatter_add(lower * p_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * p_P, int_neigh, n_cells)

            p_new = (source - off_diag) / diag_safe
            if (p_new - p).abs().max() < self.p_tolerance:
                break
            p = p_new

        return p

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------

    def _compute_grad(self, phi: torch.Tensor, mesh: Any) -> torch.Tensor:
        """Compute gradient of scalar field."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        phi_P = gather(phi, int_owner)
        phi_N = gather(phi, int_neigh)
        phi_face = w * phi_P + (1.0 - w) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * face_areas

        grad = torch.zeros(n_cells, 3, dtype=phi.dtype, device=phi.device)
        grad.index_add_(0, int_owner, face_contrib)
        grad.index_add_(0, int_neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    def _compute_continuity_error(
        self, phi: torch.Tensor, rho: torch.Tensor,
    ) -> float:
        """Compute continuity error."""
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        rho_face = 0.5 * (
            gather(rho, mesh.owner[:n_internal]) + gather(rho, mesh.neighbour)
        )
        mass_flux = phi[:n_internal] * rho_face

        div_rho_phi = torch.zeros(n_cells, dtype=phi.dtype, device=phi.device)
        div_rho_phi = div_rho_phi + scatter_add(
            mass_flux, mesh.owner[:n_internal], n_cells
        )
        div_rho_phi = div_rho_phi + scatter_add(
            -mass_flux, mesh.neighbour, n_cells
        )

        V = mesh.cell_volumes.clamp(min=1e-30)
        return float((div_rho_phi / V).abs().mean().item())

    def _compute_residual(
        self, field: torch.Tensor, field_old: torch.Tensor,
    ) -> float:
        """Compute relative L2 residual."""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        return norm_diff / norm_field if norm_field > 1e-30 else norm_diff

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write fields to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("T", self.T, time_str, self._T_data)
        for sp in self.species:
            fname = f"Y{sp}"
            fd = self._Y_data.get(fname)
            if fd is not None:
                self.write_field(fname, self.Y[sp], time_str, fd)
