"""
reactingFoam — reacting flow solver with Arrhenius chemistry.

Implements the OpenFOAM reactingFoam solver for reactive flows with
chemical kinetics.  Solves the species transport equations with
Arrhenius reaction source terms:

    ∂(ρYi)/∂t + ∇·(ρUYi) = ∇·(ρDi∇Yi) + ωi

where ωi is the reaction source term computed from Arrhenius kinetics:

    ωi = νi * A * T^β * exp(-Ea/RT) * ∏[Yj^νj]

The solver reads:
- ``0/Y`` — mass fractions for each species
- ``0/T`` — temperature field
- ``0/U`` — velocity field
- ``0/p`` — pressure field
- ``constant/polyMesh`` — mesh
- ``constant/thermophysicalProperties`` — thermodynamic properties
- ``constant/reactions`` — reaction mechanism
- ``system/controlDict`` — endTime, deltaT
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — linear solver tolerances

Usage::

    from pyfoam.applications.reacting_foam import ReactingFoam

    solver = ReactingFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from pyfoam.solvers.linear_solver import create_solver

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["ReactingFoam"]

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


class ReactingFoam(SolverBase):
    """Reacting flow solver with Arrhenius chemistry.

    Solves species transport equations with Arrhenius reaction source terms.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.

    Attributes
    ----------
    Y : dict[str, torch.Tensor]
        Mass fractions for each species.
    T : torch.Tensor
        ``(n_cells,)`` temperature field.
    U : torch.Tensor
        ``(n_cells, 3)`` velocity field.
    p : torch.Tensor
        ``(n_cells,)`` pressure field.
    species : list[str]
        List of species names.
    reactions : list[Reaction]
        List of reactions.
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Read thermodynamic properties
        self._read_thermo_properties()

        # Read reaction mechanism
        self.reactions = self._read_reactions()

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes
        self._read_fv_schemes_settings()

        # Initialise fields
        self.Y, self.T, self.U, self.p = self._init_fields()

        # Store raw field data for writing
        self._field_data = self._init_field_data()

        # Store old fields
        self.Y_old = {name: y.clone() for name, y in self.Y.items()}
        self.T_old = self.T.clone()

        logger.info("ReactingFoam ready: %d species, %d reactions",
                    len(self.species), len(self.reactions))

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_thermo_properties(self) -> None:
        """Read thermodynamic properties."""
        self.R = 8.314  # Universal gas constant (J/(mol·K))
        self.Cp = 1005.0  # Specific heat capacity (J/(kg·K))
        self.W = {}  # Molecular weights for each species

        tp_path = self.case_path / "constant" / "thermophysicalProperties"
        if tp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                tp = parse_dict_file(tp_path)
                self.R = float(tp.get("R", 8.314))
                self.Cp = float(tp.get("Cp", 1005.0))

                # Read molecular weights
                species_dict = tp.get("species", {})
                if isinstance(species_dict, dict):
                    for name, value in species_dict.items():
                        self.W[name] = float(value)
            except Exception as e:
                logger.warning("Could not read thermo properties: %s", e)

    def _read_reactions(self) -> list[Reaction]:
        """Read reaction mechanism from constant/reactions."""
        reactions = []

        rxn_path = self.case_path / "constant" / "reactions"
        if not rxn_path.exists():
            # Default: single reaction A → B
            logger.warning("No reactions file found, using default A→B reaction")
            reactions.append(Reaction(
                name="reaction1",
                A=1.0,
                beta=0.0,
                Ea=0.0,
                reactants={"A": 1.0},
                products={"B": 1.0},
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
                        rxn.reactants = {
                            k: float(v) for k, v in reactants.items()
                        }

                    products = value.get("products", {})
                    if isinstance(products, dict):
                        rxn.products = {
                            k: float(v) for k, v in products.items()
                        }

                    reactions.append(rxn)
        except Exception as e:
            logger.warning("Could not read reactions: %s", e)
            reactions.append(Reaction(
                name="default",
                A=1.0,
                beta=0.0,
                Ea=0.0,
                reactants={"A": 1.0},
                products={"B": 1.0},
            ))

        return reactions

    def _read_fv_solution_settings(self) -> None:
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.Y_solver = str(fv.get_path("solvers/Y/solver", "PBiCGStab"))
        self.Y_tolerance = float(fv.get_path("solvers/Y/tolerance", 1e-6))
        self.Y_rel_tol = float(fv.get_path("solvers/Y/relTol", 0.01))
        self.Y_max_iter = int(fv.get_path("solvers/Y/maxIter", 1000))

        self.T_solver = str(fv.get_path("solvers/T/solver", "PBiCGStab"))
        self.T_tolerance = float(fv.get_path("solvers/T/tolerance", 1e-6))
        self.T_rel_tol = float(fv.get_path("solvers/T/relTol", 0.01))
        self.T_max_iter = int(fv.get_path("solvers/T/maxIter", 1000))

        self.convergence_tolerance = float(
            fv.get_path("reactingFoam/convergenceTolerance", 1e-4)
        )

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings (for logging)."""
        fs = self.case.fvSchemes
        self.ddt_scheme = str(fs.get_path("ddtSchemes/default", "Euler"))
        self.div_scheme = str(fs.get_path("divSchemes/default", "Gauss linear"))

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise Y, T, U, p from the 0/ directory.

        Returns:
            Tuple of ``(Y, T, U, p)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Read species mass fractions
        Y = {}
        self.species = []

        # Try to read species from 0/ directory
        zero_dir = self.case_path / "0"
        if zero_dir.exists():
            for f in sorted(zero_dir.iterdir()):
                if f.name.startswith("Y"):
                    species_name = f.name[1:]  # Remove 'Y' prefix
                    if species_name:
                        self.species.append(species_name)
                        try:
                            y_tensor, _ = self.read_field_tensor(f.name, 0)
                            Y[species_name] = y_tensor.to(
                                device=device, dtype=dtype
                            ).squeeze()
                        except Exception:
                            Y[species_name] = torch.zeros(
                                n_cells, dtype=dtype, device=device
                            )

        # If no species found, create default A and B
        if not self.species:
            self.species = ["A", "B"]
            Y["A"] = torch.ones(n_cells, dtype=dtype, device=device)
            Y["B"] = torch.zeros(n_cells, dtype=dtype, device=device)

        # Read temperature
        try:
            T_tensor, _ = self.read_field_tensor("T", 0)
            T = T_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            T = torch.full((n_cells,), 300.0, dtype=dtype, device=device)

        # Read velocity
        try:
            U_tensor, _ = self.read_field_tensor("U", 0)
            U = U_tensor.to(device=device, dtype=dtype)
        except Exception:
            U = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        # Read pressure
        try:
            p_tensor, _ = self.read_field_tensor("p", 0)
            p = p_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            p = torch.full((n_cells,), 101325.0, dtype=dtype, device=device)

        return Y, T, U, p

    def _init_field_data(self) -> dict[str, Any]:
        """Store raw FieldData for writing."""
        field_data = {}

        for species_name in self.species:
            fname = f"Y{species_name}"
            try:
                field_data[fname] = self.case.read_field(fname, 0)
            except Exception:
                field_data[fname] = None

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

        k = A * T^β * exp(-Ea / (R*T))

        Rate = k * ∏[Yj^νj] for reactants

        Args:
            reaction: Reaction definition.
            T: Temperature field.
            Y: Mass fractions.

        Returns:
            ``(n_cells,)`` reaction rate.
        """
        device = T.device
        dtype = T.dtype

        # Avoid division by zero
        T_safe = T.clamp(min=1.0)

        # Arrhenius rate constant
        k = reaction.A * T_safe.pow(reaction.beta) * torch.exp(
            -reaction.Ea / (self.R * T_safe)
        )

        # Concentration term: ∏[Yj^νj]
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
        """Compute source terms for all species.

        Args:
            T: Temperature field.
            Y: Mass fractions.

        Returns:
            Dictionary of source terms for each species.
        """
        device = T.device
        dtype = T.dtype
        n_cells = T.shape[0]

        # Initialize source terms
        omega = {name: torch.zeros(n_cells, dtype=dtype, device=device)
                 for name in self.species}

        for reaction in self.reactions:
            rate = self._compute_arrhenius_rate(reaction, T, Y)

            # Reactants: negative source (consumed)
            for species, nu in reaction.reactants.items():
                if species in omega:
                    W_i = self.W.get(species, 1.0)
                    omega[species] = omega[species] - nu * W_i * rate

            # Products: positive source (produced)
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
        """Compute heat release from reactions.

        Q = -Σ(ΔH_reaction * rate)

        For simplicity, assume constant heat of reaction.

        Args:
            T: Temperature field.
            Y: Mass fractions.

        Returns:
            ``(n_cells,)`` heat release rate.
        """
        device = T.device
        dtype = T.dtype
        n_cells = T.shape[0]

        heat_release = torch.zeros(n_cells, dtype=dtype, device=device)

        for reaction in self.reactions:
            rate = self._compute_arrhenius_rate(reaction, T, Y)

            # Simplified: assume ΔH = 0 for now
            # In a full implementation, this would read from thermo data
            # heat_release += delta_H * rate

        return heat_release

    # ------------------------------------------------------------------
    # Transport equation assembly
    # ------------------------------------------------------------------

    def _assemble_species_equation(
        self,
        species_name: str,
        Y_old: torch.Tensor,
        dt: float,
        omega: torch.Tensor,
    ) -> Any:
        """Assemble species transport equation.

        ∂(ρYi)/∂t + ∇·(ρUYi) = ∇·(ρDi∇Yi) + ωi

        Args:
            species_name: Species name.
            Y_old: Old mass fraction.
            dt: Time step.
            omega: Reaction source term.

        Returns:
            FvMatrix for the species equation.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Compute face flux from velocity
        n_faces = self.mesh.n_faces
        n_internal = self.mesh.n_internal_faces
        face_areas = self.mesh.face_areas
        owner = self.mesh.owner
        int_owner = owner[:n_internal]
        int_neigh = self.mesh.neighbour

        U_face = torch.zeros(n_faces, 3, dtype=dtype, device=device)
        U_P = self.U[int_owner]
        U_N = self.U[int_neigh]
        U_face[:n_internal] = 0.5 * U_P + 0.5 * U_N

        if n_faces > n_internal:
            bnd_owner = owner[n_internal:]
            U_face[n_internal:] = self.U[bnd_owner]

        phi = (U_face * face_areas).sum(dim=1)

        # Build combined matrix
        from pyfoam.core.fv_matrix import FvMatrix
        from pyfoam.core.backend import scatter_add

        matrix = FvMatrix(
            n_cells, int_owner, int_neigh,
            device=device, dtype=dtype,
        )

        cell_volumes = self.mesh.cell_volumes

        # Time derivative
        matrix._diag = matrix._diag + cell_volumes / dt
        matrix._source = matrix._source + cell_volumes * Y_old / dt

        # Convection (upwind)
        flux = phi[:n_internal]
        is_positive = flux >= 0.0
        flux_pos = torch.where(is_positive, flux, torch.zeros_like(flux))
        flux_neg = torch.where(~is_positive, flux, torch.zeros_like(flux))

        V_P = cell_volumes[int_owner]
        V_N = cell_volumes[int_neigh]

        matrix._lower = matrix._lower + flux_neg / V_P
        matrix._upper = matrix._upper + flux_pos / V_N

        diag_conv = torch.zeros(n_cells, dtype=dtype, device=device)
        diag_conv = diag_conv + scatter_add(-flux_pos / V_P, int_owner, n_cells)
        diag_conv = diag_conv + scatter_add(flux_neg.abs() / V_N, int_neigh, n_cells)
        matrix._diag = matrix._diag + diag_conv

        # Diffusion
        D = 1e-5
        delta_coeffs = self.mesh.delta_coefficients
        S_mag = face_areas[:n_internal].norm(dim=1)
        D_face = torch.full((n_faces,), D, dtype=dtype, device=device)
        face_coeff = D_face[:n_internal] * S_mag * delta_coeffs[:n_internal]

        matrix._lower = matrix._lower - face_coeff / V_P
        matrix._upper = matrix._upper - face_coeff / V_N

        diag_diff = torch.zeros(n_cells, dtype=dtype, device=device)
        diag_diff = diag_diff + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag_diff = diag_diff + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        if n_faces > n_internal:
            bnd_areas = face_areas[n_internal:]
            bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
            bnd_delta = delta_coeffs[n_internal:]
            bnd_D = D_face[n_internal:]
            bnd_coeff = bnd_D * bnd_S_mag * bnd_delta
            bnd_V = cell_volumes[owner[n_internal:]]
            diag_diff = diag_diff + scatter_add(bnd_coeff / bnd_V, owner[n_internal:], n_cells)

        matrix._diag = matrix._diag + diag_diff

        # Reaction source (explicit)
        matrix._source = matrix._source + omega * cell_volumes

        return matrix

    def _assemble_temperature_equation(
        self,
        T_old: torch.Tensor,
        dt: float,
        heat_release: torch.Tensor,
    ) -> Any:
        """Assemble temperature transport equation.

        ρCp ∂T/∂t + ρCp ∇·(UT) = ∇·(λ∇T) + Q

        Args:
            T_old: Old temperature.
            dt: Time step.
            heat_release: Heat release from reactions.

        Returns:
            FvMatrix for the temperature equation.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Compute face flux from velocity
        n_faces = self.mesh.n_faces
        n_internal = self.mesh.n_internal_faces
        face_areas = self.mesh.face_areas
        owner = self.mesh.owner
        int_owner = owner[:n_internal]
        int_neigh = self.mesh.neighbour

        U_face = torch.zeros(n_faces, 3, dtype=dtype, device=device)
        U_P = self.U[int_owner]
        U_N = self.U[int_neigh]
        U_face[:n_internal] = 0.5 * U_P + 0.5 * U_N

        if n_faces > n_internal:
            bnd_owner = owner[n_internal:]
            U_face[n_internal:] = self.U[bnd_owner]

        phi = (U_face * face_areas).sum(dim=1)

        # Build combined matrix
        from pyfoam.core.fv_matrix import FvMatrix
        from pyfoam.core.backend import scatter_add

        matrix = FvMatrix(
            n_cells, int_owner, int_neigh,
            device=device, dtype=dtype,
        )

        cell_volumes = self.mesh.cell_volumes
        Cp = self.Cp
        lambda_ = 0.026  # Thermal conductivity

        # Time derivative (ρCp * dT/dt)
        matrix._diag = matrix._diag + Cp * cell_volumes / dt
        matrix._source = matrix._source + Cp * cell_volumes * T_old / dt

        # Convection (ρCp * ∇·(UT)) - upwind
        flux = Cp * phi[:n_internal]
        is_positive = flux >= 0.0
        flux_pos = torch.where(is_positive, flux, torch.zeros_like(flux))
        flux_neg = torch.where(~is_positive, flux, torch.zeros_like(flux))

        V_P = cell_volumes[int_owner]
        V_N = cell_volumes[int_neigh]

        matrix._lower = matrix._lower + flux_neg / V_P
        matrix._upper = matrix._upper + flux_pos / V_N

        diag_conv = torch.zeros(n_cells, dtype=dtype, device=device)
        diag_conv = diag_conv + scatter_add(-flux_pos / V_P, int_owner, n_cells)
        diag_conv = diag_conv + scatter_add(flux_neg.abs() / V_N, int_neigh, n_cells)
        matrix._diag = matrix._diag + diag_conv

        # Diffusion (λ * ∇²T)
        delta_coeffs = self.mesh.delta_coefficients
        S_mag = face_areas[:n_internal].norm(dim=1)
        D_face = torch.full((n_faces,), lambda_, dtype=dtype, device=device)
        face_coeff = D_face[:n_internal] * S_mag * delta_coeffs[:n_internal]

        matrix._lower = matrix._lower - face_coeff / V_P
        matrix._upper = matrix._upper - face_coeff / V_N

        diag_diff = torch.zeros(n_cells, dtype=dtype, device=device)
        diag_diff = diag_diff + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag_diff = diag_diff + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        if n_faces > n_internal:
            bnd_areas = face_areas[n_internal:]
            bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
            bnd_delta = delta_coeffs[n_internal:]
            bnd_D = D_face[n_internal:]
            bnd_coeff = bnd_D * bnd_S_mag * bnd_delta
            bnd_V = cell_volumes[owner[n_internal:]]
            diag_diff = diag_diff + scatter_add(bnd_coeff / bnd_V, owner[n_internal:], n_cells)

        matrix._diag = matrix._diag + diag_diff

        # Heat release source term
        matrix._source = matrix._source + heat_release * cell_volumes

        return matrix

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the reactingFoam solver.

        Solves species transport with Arrhenius chemistry in a
        time-stepping loop.

        Returns:
            Dictionary with convergence information.
        """
        device = get_device()
        dtype = get_default_dtype()

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

        logger.info("Starting reactingFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  Species: %s", self.species)

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        # Create linear solvers
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

        last_convergence = None

        for t, step in time_loop:
            # Store old fields
            self.Y_old = {name: y.clone() for name, y in self.Y.items()}
            self.T_old = self.T.clone()

            # Compute reaction source terms
            omega = self._compute_species_source_terms(self.T, self.Y)
            heat_release = self._compute_heat_release(self.T, self.Y)

            # Solve species equations
            for species_name in self.species:
                matrix = self._assemble_species_equation(
                    species_name, self.Y_old[species_name],
                    self.delta_t, omega[species_name],
                )

                self.Y[species_name], iters, residual = matrix.solve(
                    Y_solver, self.Y[species_name].clone(),
                    tolerance=self.Y_tolerance,
                    max_iter=self.Y_max_iter,
                )

            # Solve temperature equation
            T_matrix = self._assemble_temperature_equation(
                self.T_old, self.delta_t, heat_release,
            )

            self.T, t_iters, t_residual = T_matrix.solve(
                T_solver, self.T.clone(),
                tolerance=self.T_tolerance,
                max_iter=self.T_max_iter,
            )

            # Track convergence
            residuals = {"T": t_residual}
            for name in self.species:
                residuals[f"Y_{name}"] = residual  # Use last species residual
            converged = convergence.update(step + 1, residuals)

            # Write fields if needed
            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        # Write final fields
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info("reactingFoam completed")
        logger.info("  T range: [%.1f, %.1f] K",
                    self.T.min().item(), self.T.max().item())

        return {
            "converged": converged if converged else False,
            "iterations": t_iters,
            "residual": t_residual,
        }

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write fields to a time directory."""
        time_str = f"{time:g}"

        # Write species mass fractions
        for species_name in self.species:
            fname = f"Y{species_name}"
            if fname in self._field_data and self._field_data[fname] is not None:
                self.write_field(fname, self.Y[species_name], time_str,
                               self._field_data[fname])

        # Write temperature
        if "T" in self._field_data and self._field_data["T"] is not None:
            self.write_field("T", self.T, time_str, self._field_data["T"])
