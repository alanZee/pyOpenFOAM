"""
magneticFoam — magnetostatics solver.

Solves the Poisson equation for the magnetic vector potential A:

    ∇²A = -μ₀ J

using the Coulomb gauge (∇·A = 0), where:
- A is the magnetic vector potential [T·m]
- μ₀ is the permeability of free space [H/m]
- J is the current density [A/m²]

After solving, the magnetic field is computed as:

    B = ∇ × A

This is the magnetostatic analogue of electrostaticFoam.  The vector
Poisson equation is solved component-by-component (x, y, z).

Algorithm:
1. For each component A_i: assemble ∇²A_i = -μ₀ J_i
2. Solve the three scalar linear systems
3. Compute B = curl(A)
4. Write fields

The solver reads:
- ``0/A`` — initial/boundary conditions for vector potential
- ``0/J`` — current density field
- ``constant/polyMesh`` — mesh
- ``constant/magneticProperties`` — permeability (mu0)
- ``system/controlDict`` — endTime, deltaT
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — linear solver tolerances

Usage::

    from pyfoam.applications.magnetic_foam import MagneticFoam

    solver = MagneticFoam("path/to/case")
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
from pyfoam.discretisation.operators import fvc
from pyfoam.solvers.linear_solver import create_solver
from pyfoam.core.backend import scatter_add, gather

from .solver_base import SolverBase
from .convergence import ConvergenceMonitor

__all__ = ["MagneticFoam"]

logger = logging.getLogger(__name__)


class MagneticFoam(SolverBase):
    """Magnetostatics solver (∇²A = -μ₀ J, Coulomb gauge).

    Solves the vector Poisson equation for the magnetic vector potential A,
    then computes B = curl(A).

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    mu0 : float | None, optional
        Permeability.  If None, reads from ``constant/magneticProperties``.

    Attributes
    ----------
    A : torch.Tensor
        ``(n_cells, 3)`` magnetic vector potential field.
    B : torch.Tensor
        ``(n_cells, 3)`` magnetic flux density field.
    J : torch.Tensor
        ``(n_cells, 3)`` current density field.
    mu0 : float
        Permeability of the medium.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        mu0: float | None = None,
    ) -> None:
        super().__init__(case_path)

        # Read properties
        self.mu0 = mu0 if mu0 is not None else self._read_permeability()

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes
        self._read_fv_schemes_settings()

        # Initialise fields
        self.A, self.B, self.J = self._init_fields()

        # Store raw field data for writing
        self._A_data, self._B_data, self._J_data = self._init_field_data()

        # Parse boundary conditions
        self._bc_values = self._parse_boundary_conditions()

        logger.info("MagneticFoam ready: mu0=%.6e", self.mu0)

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_permeability(self) -> float:
        """Read permeability from constant/magneticProperties."""
        mp_path = self.case_path / "constant" / "magneticProperties"
        if not mp_path.exists():
            logger.warning(
                "constant/magneticProperties not found, using mu0=1.0"
            )
            return 1.0

        try:
            from pyfoam.io.dictionary import parse_dict_file
            mp = parse_dict_file(mp_path)
            mu = mp.get("mu0", mp.get("mu", 1.0))
            if isinstance(mu, dict):
                mu = mu.get("value", 1.0)
            return float(mu)
        except Exception as e:
            logger.warning(
                "Could not parse magneticProperties: %s, using mu0=1.0", e
            )
            return 1.0

    def _read_fv_solution_settings(self) -> None:
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.A_solver = str(fv.get_path("solvers/A/solver", "PCG"))
        self.A_tolerance = float(fv.get_path("solvers/A/tolerance", 1e-6))
        self.A_rel_tol = float(fv.get_path("solvers/A/relTol", 0.01))
        self.A_max_iter = int(fv.get_path("solvers/A/maxIter", 1000))

        self.n_non_orth_correctors = int(
            fv.get_path("magnetic/nNonOrthogonalCorrectors", 0)
        )
        self.convergence_tolerance = float(
            fv.get_path("magnetic/convergenceTolerance", 1e-5)
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
        """Initialise A, B, J from the 0/ directory.

        Returns:
            Tuple of ``(A, B, J)`` tensors.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Read vector potential
        try:
            A_tensor, _ = self.read_field_tensor("A", 0)
            A = A_tensor.to(device=device, dtype=dtype)
        except Exception:
            A = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        # Read current density
        try:
            J_tensor, _ = self.read_field_tensor("J", 0)
            J = J_tensor.to(device=device, dtype=dtype)
        except Exception:
            J = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        # Magnetic field (computed, not read)
        B = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        return A, B, J

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        try:
            A_data = self.case.read_field("A", 0)
        except Exception:
            A_data = None
        try:
            B_data = self.case.read_field("B", 0)
        except Exception:
            B_data = None
        try:
            J_data = self.case.read_field("J", 0)
        except Exception:
            J_data = None
        return A_data, B_data, J_data

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _parse_boundary_conditions(self) -> dict[str, dict[str, Any]]:
        """Parse boundary conditions from the field data."""
        bc_values: dict[str, dict[str, Any]] = {}

        if self._A_data is None:
            return bc_values

        boundary = self._A_data.boundary_field
        mesh_boundary = self.case.boundary

        if boundary is None:
            return bc_values

        for i, patch in enumerate(boundary.patches):
            bc_info: dict[str, Any] = {
                "type": patch.patch_type,
                "value": None,
            }

            if patch.patch_type == "fixedValue" and patch.value is not None:
                val = patch.value
                if isinstance(val, str):
                    val = val.strip()
                    if val.startswith("uniform"):
                        val = val[len("uniform"):].strip()
                    # Parse vector value like (1 0 0)
                    vec_match = re.match(
                        r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)",
                        val,
                    )
                    if vec_match:
                        bc_info["value"] = (
                            float(vec_match.group(1)),
                            float(vec_match.group(2)),
                            float(vec_match.group(3)),
                        )
                    else:
                        try:
                            bc_info["value"] = float(val)
                        except ValueError:
                            logger.warning("Could not parse BC value: %s", val)
                elif isinstance(val, (list, tuple)) and len(val) == 3:
                    bc_info["value"] = tuple(float(v) for v in val)

            if i < len(mesh_boundary):
                bp = mesh_boundary[i]
                bc_info["start_face"] = bp.start_face
                bc_info["n_faces"] = bp.n_faces

            bc_values[patch.name] = bc_info

        return bc_values

    def _build_component_boundary_conditions(
        self, comp: int
    ) -> torch.Tensor:
        """Build boundary conditions for one component of A.

        Parameters
        ----------
        comp : int
            Component index (0=x, 1=y, 2=z).

        Returns:
            ``(n_cells,)`` — prescribed value (NaN where no BC).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        A_bc = torch.full((n_cells,), float('nan'), dtype=dtype, device=device)

        for patch_name, bc_info in self._bc_values.items():
            if bc_info["type"] != "fixedValue" or bc_info["value"] is None:
                continue

            start_face = bc_info.get("start_face", 0)
            n_faces = bc_info.get("n_faces", 0)
            if n_faces == 0:
                continue

            val = bc_info["value"]
            if isinstance(val, (list, tuple)) and len(val) == 3:
                comp_val = float(val[comp])
            elif isinstance(val, (int, float)):
                comp_val = float(val)
            else:
                continue

            owner = self.mesh.owner
            for i in range(n_faces):
                face_idx = start_face + i
                cell_idx = owner[face_idx].item()
                A_bc[cell_idx] = comp_val

        return A_bc

    # ------------------------------------------------------------------
    # Equation assembly (one component)
    # ------------------------------------------------------------------

    def _assemble_component_equation(self, comp: int) -> FvMatrix:
        """Assemble ∇²A_comp = -μ₀ J_comp for one component.

        Parameters
        ----------
        comp : int
            Component index (0=x, 1=y, 2=z).

        Returns:
            :class:`FvMatrix` for the scalar equation.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # Laplacian matrix coefficients (same as electrostaticFoam)
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        face_coeff = S_mag * delta_f

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        lap_diag = torch.zeros(n_cells, dtype=dtype, device=device)
        lap_diag = lap_diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        lap_diag = lap_diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # Boundary contributions
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
                val = bc_info["value"]
                if isinstance(val, (list, tuple)) and len(val) == 3:
                    comp_val = float(val[comp])
                elif isinstance(val, (int, float)):
                    comp_val = float(val)
                else:
                    continue
                bc_source = bc_source + scatter_add(
                    bnd_coeff * comp_val / bnd_V, bnd_cells, n_cells
                )

        # Source: -μ₀ J_comp * V_cell
        source = -self.mu0 * self.J[:, comp] * cell_volumes + bc_source

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
        """Run the magneticFoam solver.

        Solves ∇²A = -μ₀ J component-by-component, then computes B = curl(A).

        Returns:
            Dictionary with convergence information.
        """
        logger.info("Starting magneticFoam run")
        logger.info("  mu0=%.6e", self.mu0)

        total_iters = 0
        max_residual = 0.0

        # Solve each component of A
        for comp in range(3):
            comp_name = ["Ax", "Ay", "Az"][comp]
            logger.info("Solving for %s", comp_name)

            # Build and apply BCs
            A_bc = self._build_component_boundary_conditions(comp)
            bc_mask = ~torch.isnan(A_bc)
            self.A[bc_mask, comp] = A_bc[bc_mask]

            for non_orth in range(self.n_non_orth_correctors + 1):
                matrix = self._assemble_component_equation(comp)

                if bc_mask.any():
                    large_coeff = matrix.diag.abs().clamp(min=1.0) * 1e10
                    matrix._diag[bc_mask] += large_coeff[bc_mask]
                    matrix._source[bc_mask] += large_coeff[bc_mask] * A_bc[bc_mask]

                solver = create_solver(
                    self.A_solver,
                    tolerance=self.A_tolerance,
                    rel_tol=self.A_rel_tol,
                    max_iter=self.A_max_iter,
                )

                A_comp, iters, residual = matrix.solve(
                    solver, self.A[:, comp].clone(),
                    tolerance=self.A_tolerance,
                    max_iter=self.A_max_iter,
                )

                self.A[:, comp] = A_comp
                total_iters += iters
                max_residual = max(max_residual, residual)

                logger.info("  %s: iters=%d, residual=%.6e",
                           comp_name, iters, residual)

        # Compute magnetic field: B = curl(A)
        self.B = fvc.curl(self.A, mesh=self.mesh)

        # Write final fields
        self._write_fields(self.end_time)

        logger.info("magneticFoam completed")
        logger.info("  max|B| = %.6e",
                    (self.B ** 2).sum(dim=1).sqrt().max().item())

        return {
            "converged": max_residual < self.convergence_tolerance,
            "iterations": total_iters,
            "residual": max_residual,
        }

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write A, B, and J to a time directory."""
        time_str = f"{time:g}"
        if self._A_data is not None:
            self.write_field("A", self.A, time_str, self._A_data)
        if self._B_data is not None:
            self.write_field("B", self.B, time_str, self._B_data)
        if self._J_data is not None:
            self.write_field("J", self.J, time_str, self._J_data)
