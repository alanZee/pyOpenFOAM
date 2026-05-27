"""
electrostaticFoam — electrostatics solver.

Solves the Laplace equation for electric potential V:

    ∇²V = -ρ_e / ε

where:
- V is the electric potential [V]
- ρ_e is the free charge density [C/m³]
- ε is the permittivity [F/m]

After solving, the electric field is computed as:

    E = -∇V

and the charge density can be recovered via:

    ρ_e = -ε ∇·E

For a source-free region (ρ_e = 0), this reduces to Laplace's equation:

    ∇²V = 0

Algorithm:
1. Assemble ∇²V = -ρ_e/ε (steady-state)
2. Solve the linear system for V
3. Compute E = -∇V
4. Write fields

The solver reads:
- ``0/V`` — initial/boundary conditions for electric potential
- ``0/rhoE`` — charge density field (optional, defaults to zero)
- ``constant/polyMesh`` — mesh
- ``constant/electricProperties`` — permittivity (epsilon)
- ``system/controlDict`` — endTime, deltaT
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — linear solver tolerances

Usage::

    from pyfoam.applications.electrostatic_foam import ElectrostaticFoam

    solver = ElectrostaticFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.discretisation.operators import fvm, fvc
from pyfoam.solvers.linear_solver import create_solver
from pyfoam.core.backend import scatter_add, gather

from .solver_base import SolverBase
from .convergence import ConvergenceMonitor

__all__ = ["ElectrostaticFoam"]

logger = logging.getLogger(__name__)


class ElectrostaticFoam(SolverBase):
    """Electrostatics solver (∇²V = -ρ_e/ε).

    Solves the Poisson equation for the electric potential V, then
    computes E = -∇V.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    epsilon : float | None, optional
        Permittivity.  If None, reads from ``constant/electricProperties``.

    Attributes
    ----------
    V : torch.Tensor
        ``(n_cells,)`` electric potential field.
    E : torch.Tensor
        ``(n_cells, 3)`` electric field.
    rho_e : torch.Tensor
        ``(n_cells,)`` charge density field.
    epsilon : float
        Permittivity of the medium.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        epsilon: float | None = None,
    ) -> None:
        super().__init__(case_path)

        # Read properties
        self.epsilon = epsilon if epsilon is not None else self._read_permittivity()

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes
        self._read_fv_schemes_settings()

        # Initialise fields
        self.V, self.E, self.rho_e = self._init_fields()

        # Store raw field data for writing
        self._V_data, self._E_data, self._rho_e_data = self._init_field_data()

        # Parse boundary conditions
        self._bc_values = self._parse_boundary_conditions()

        logger.info("ElectrostaticFoam ready: epsilon=%.6e", self.epsilon)

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_permittivity(self) -> float:
        """Read permittivity from constant/electricProperties."""
        ep_path = self.case_path / "constant" / "electricProperties"
        if not ep_path.exists():
            logger.warning(
                "constant/electricProperties not found, using epsilon=1.0"
            )
            return 1.0

        try:
            from pyfoam.io.dictionary import parse_dict_file
            ep = parse_dict_file(ep_path)
            eps = ep.get("epsilon", ep.get("epsilon0", 1.0))
            if isinstance(eps, dict):
                eps = eps.get("value", 1.0)
            return float(eps)
        except Exception as e:
            logger.warning(
                "Could not parse electricProperties: %s, using epsilon=1.0", e
            )
            return 1.0

    def _read_fv_solution_settings(self) -> None:
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.V_solver = str(fv.get_path("solvers/V/solver", "PCG"))
        self.V_tolerance = float(fv.get_path("solvers/V/tolerance", 1e-6))
        self.V_rel_tol = float(fv.get_path("solvers/V/relTol", 0.01))
        self.V_max_iter = int(fv.get_path("solvers/V/maxIter", 1000))

        self.n_non_orth_correctors = int(
            fv.get_path("electrostatic/nNonOrthogonalCorrectors", 0)
        )
        self.convergence_tolerance = float(
            fv.get_path("electrostatic/convergenceTolerance", 1e-5)
        )

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings."""
        fs = self.case.fvSchemes
        self.grad_scheme = str(fs.get_path("gradSchemes/default", "Gauss linear"))
        self.lap_scheme = str(
            fs.get_path("laplacianSchemes/default", "Gauss linear corrected")
        )

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise V, E, rho_e from the 0/ directory.

        Returns:
            Tuple of ``(V, E, rho_e)`` tensors.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Read electric potential
        try:
            V_tensor, _ = self.read_field_tensor("V", 0)
            V = V_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            V = torch.zeros(n_cells, dtype=dtype, device=device)

        # Read charge density
        try:
            rho_tensor, _ = self.read_field_tensor("rhoE", 0)
            rho_e = rho_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            rho_e = torch.zeros(n_cells, dtype=dtype, device=device)

        # Electric field (computed, not read)
        E = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        return V, E, rho_e

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        try:
            V_data = self.case.read_field("V", 0)
        except Exception:
            V_data = None
        try:
            E_data = self.case.read_field("E", 0)
        except Exception:
            E_data = None
        try:
            rho_e_data = self.case.read_field("rhoE", 0)
        except Exception:
            rho_e_data = None
        return V_data, E_data, rho_e_data

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _parse_boundary_conditions(self) -> dict[str, dict[str, Any]]:
        """Parse boundary conditions from the field data.

        Returns a dict mapping patch name to its BC info.
        """
        bc_values = {}

        if self._V_data is None:
            return bc_values

        boundary = self._V_data.boundary_field
        mesh_boundary = self.case.boundary

        if boundary is None:
            return bc_values

        for i, patch in enumerate(boundary.patches):
            bc_info = {"type": patch.patch_type, "value": None}

            if patch.patch_type == "fixedValue" and patch.value is not None:
                val = patch.value
                if isinstance(val, str):
                    val = val.strip()
                    if val.startswith("uniform"):
                        val = val[len("uniform"):].strip()
                    try:
                        bc_info["value"] = float(val)
                    except ValueError:
                        logger.warning("Could not parse BC value: %s", val)
                elif isinstance(val, (int, float)):
                    bc_info["value"] = float(val)
                elif isinstance(val, (list, tuple)) and len(val) == 1:
                    bc_info["value"] = float(val[0])

            if i < len(mesh_boundary):
                bp = mesh_boundary[i]
                bc_info["start_face"] = bp.start_face
                bc_info["n_faces"] = bp.n_faces

            bc_values[patch.name] = bc_info

        return bc_values

    def _build_V_boundary_conditions(self) -> torch.Tensor:
        """Build boundary conditions for electric potential.

        Returns:
            ``(n_cells,)`` — prescribed potential (NaN where no BC).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        V_bc = torch.full((n_cells,), float('nan'), dtype=dtype, device=device)

        for patch_name, bc_info in self._bc_values.items():
            if bc_info["type"] != "fixedValue" or bc_info["value"] is None:
                continue

            start_face = bc_info.get("start_face", 0)
            n_faces = bc_info.get("n_faces", 0)
            if n_faces == 0:
                continue

            owner = self.mesh.owner
            for i in range(n_faces):
                face_idx = start_face + i
                cell_idx = owner[face_idx].item()
                V_bc[cell_idx] = bc_info["value"]

        return V_bc

    # ------------------------------------------------------------------
    # Equation assembly
    # ------------------------------------------------------------------

    def _assemble_equation(self) -> FvMatrix:
        """Assemble the Poisson equation ∇²V = -ρ_e/ε.

        Returns:
            :class:`FvMatrix` for the potential equation.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # ---- Laplacian matrix coefficients (internal faces) ----
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        face_coeff = S_mag * delta_f  # coefficient = 1 * |S| * δ (no diffusion coeff)

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        lap_diag = torch.zeros(n_cells, dtype=dtype, device=device)
        lap_diag = lap_diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        lap_diag = lap_diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # ---- Boundary contributions ----
        bc_source = torch.zeros(n_cells, dtype=dtype, device=device)

        for patch_name, bc_info in self._bc_values.items():
            start_face = bc_info.get("start_face", 0)
            n_faces = bc_info.get("n_faces", 0)
            if n_faces == 0:
                continue
            if bc_info["type"] in ("empty", "wedge"):
                continue
            if bc_info["type"] == "zeroGradient":
                continue

            bnd_faces = slice(start_face, start_face + n_faces)
            bnd_areas = mesh.face_areas[bnd_faces]
            bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
            bnd_cells = mesh.owner[bnd_faces]
            bnd_V = gather(cell_volumes, bnd_cells)

            bnd_face_centres = mesh.face_centres[bnd_faces]
            bnd_cell_centres = mesh.cell_centres[bnd_cells]
            d_P = bnd_face_centres - bnd_cell_centres
            safe_area = torch.where(
                bnd_S_mag.unsqueeze(-1) > 1e-30,
                bnd_areas,
                torch.ones_like(bnd_areas),
            )
            n_f = safe_area / safe_area.norm(dim=1, keepdim=True)
            d_dot_n = (d_P * n_f).sum(dim=1).abs()
            bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)

            bnd_coeff = bnd_S_mag * bnd_delta

            lap_diag = lap_diag + scatter_add(bnd_coeff / bnd_V, bnd_cells, n_cells)

            if bc_info["type"] == "fixedValue" and bc_info["value"] is not None:
                V_bc = bc_info["value"]
                bc_source = bc_source + scatter_add(
                    bnd_coeff * V_bc / bnd_V, bnd_cells, n_cells
                )

        # ---- Source: -ρ_e/ε ----
        source = -self.rho_e / self.epsilon * cell_volumes + bc_source

        # Build FvMatrix manually (same pattern as ScalarTransportFoam)
        matrix = FvMatrix(
            n_cells,
            mesh.owner[:n_internal],
            mesh.neighbour,
            device=device, dtype=dtype,
        )
        matrix._diag = lap_diag
        matrix._lower = lower
        matrix._upper = upper
        matrix._source = source

        return matrix

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the electrostaticFoam solver.

        Solves ∇²V = -ρ_e/ε, then computes E = -∇V.

        Returns:
            Dictionary with convergence information.
        """
        logger.info("Starting electrostaticFoam run")
        logger.info("  epsilon=%.6e", self.epsilon)

        # Build boundary conditions
        V_bc = self._build_V_boundary_conditions()

        # Apply boundary conditions to initial potential
        bc_mask = ~torch.isnan(V_bc)
        self.V[bc_mask] = V_bc[bc_mask]

        # Solve with non-orthogonal corrections
        iters = 0
        residual = 0.0
        for non_orth in range(self.n_non_orth_correctors + 1):
            logger.info("Non-orthogonal correction %d/%d",
                       non_orth, self.n_non_orth_correctors)

            matrix = self._assemble_equation()

            # Apply BCs via penalty method
            if bc_mask.any():
                large_coeff = matrix.diag.abs().clamp(min=1.0) * 1e10
                matrix._diag[bc_mask] += large_coeff[bc_mask]
                matrix._source[bc_mask] += large_coeff[bc_mask] * V_bc[bc_mask]

            # Create linear solver
            solver = create_solver(
                self.V_solver,
                tolerance=self.V_tolerance,
                rel_tol=self.V_rel_tol,
                max_iter=self.V_max_iter,
            )

            self.V, iters, residual = matrix.solve(
                solver, self.V.clone(),
                tolerance=self.V_tolerance,
                max_iter=self.V_max_iter,
            )

            logger.info("  Potential equation: iters=%d, residual=%.6e",
                       iters, residual)

        # Compute electric field: E = -∇V
        grad_V = fvc.grad(self.V, mesh=self.mesh)
        self.E = -grad_V

        # Compute charge density: ρ_e = -ε ∇·E
        div_E = fvc.div(self.E, mesh=self.mesh)
        self.rho_e = -self.epsilon * div_E

        # Write final fields
        self._write_fields(self.end_time)

        logger.info("electrostaticFoam completed")
        logger.info("  V range: [%.6e, %.6e]",
                    self.V.min().item(), self.V.max().item())
        logger.info("  max|E| = %.6e",
                    (self.E ** 2).sum(dim=1).sqrt().max().item())

        return {
            "converged": residual < self.convergence_tolerance,
            "iterations": iters,
            "residual": residual,
        }

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write V, E, and rho_e to a time directory."""
        time_str = f"{time:g}"
        if self._V_data is not None:
            self.write_field("V", self.V, time_str, self._V_data)
        if self._E_data is not None:
            self.write_field("E", self.E, time_str, self._E_data)
        if self._rho_e_data is not None:
            self.write_field("rhoE", self.rho_e, time_str, self._rho_e_data)
