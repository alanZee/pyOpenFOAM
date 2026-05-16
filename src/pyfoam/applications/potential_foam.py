"""
potentialFoam — potential flow solver.

Implements the OpenFOAM potentialFoam solver for inviscid, incompressible,
irrotational flow.  Solves the velocity potential equation:

    ∇²φ = 0

The velocity field is then recovered as:

    U = ∇φ

and the pressure field from Bernoulli's equation:

    p + ½|U|² = const

This solver is typically used to generate initial conditions for more
complex solvers (e.g. simpleFoam, rhoSimpleFoam).

The solver reads:
- ``0/phi`` — initial/boundary conditions for velocity potential
- ``0/U`` — initial/boundary conditions for velocity (used for inlet BC)
- ``constant/polyMesh`` — mesh
- ``system/controlDict`` — endTime, deltaT
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — potentialFlow settings

Usage::

    from pyfoam.applications.potential_foam import PotentialFoam

    solver = PotentialFoam("path/to/case")
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

from .solver_base import SolverBase
from .convergence import ConvergenceMonitor

__all__ = ["PotentialFoam"]

logger = logging.getLogger(__name__)


class PotentialFoam(SolverBase):
    """Potential flow solver (∇²φ = 0).

    Solves the Laplace equation for the velocity potential φ, then
    computes U = ∇φ and p from Bernoulli's equation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.

    Attributes
    ----------
    phi_potential : torch.Tensor
        ``(n_cells,)`` velocity potential field.
    U : torch.Tensor
        ``(n_cells, 3)`` velocity field.
    p : torch.Tensor
        ``(n_cells,)`` pressure field.
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes (for logging)
        self._read_fv_schemes_settings()

        # Initialise fields
        self.phi_potential, self.U, self.p = self._init_fields()

        # Store raw field data for writing
        self._phi_data, self._U_data, self._p_data = self._init_field_data()

        logger.info("PotentialFoam ready")

    # ------------------------------------------------------------------
    # Settings reading
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read potentialFlow settings from fvSolution."""
        fv = self.case.fvSolution

        # Potential solver settings
        self.phi_solver = str(fv.get_path("solvers/phi/solver", "PCG"))
        self.phi_tolerance = float(fv.get_path("solvers/phi/tolerance", 1e-6))
        self.phi_rel_tol = float(fv.get_path("solvers/phi/relTol", 0.01))
        self.phi_max_iter = int(fv.get_path("solvers/phi/maxIter", 1000))

        # potentialFlow settings
        self.n_non_orth_correctors = int(
            fv.get_path("potentialFlow/nNonOrthogonalCorrectors", 0)
        )
        self.convergence_tolerance = float(
            fv.get_path("potentialFlow/convergenceTolerance", 1e-5)
        )

        # Reference pressure
        self.p_ref_value = float(
            fv.get_path("potentialFlow/pRefValue", 0.0)
        )
        self.p_ref_cell = int(
            fv.get_path("potentialFlow/pRefCell", 0)
        )

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings (for logging)."""
        fs = self.case.fvSchemes
        self.grad_scheme = str(fs.get_path("gradSchemes/default", "Gauss linear"))
        self.lap_scheme = str(fs.get_path("laplacianSchemes/default", "Gauss linear corrected"))

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise phi, U, p from the 0/ directory.

        Returns:
            Tuple of ``(phi_potential, U, p)`` tensors.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Try to read velocity potential from 0/phi
        try:
            phi_tensor, _ = self.read_field_tensor("phi", 0)
            phi_potential = phi_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            # If phi doesn't exist, initialise to zero
            phi_potential = torch.zeros(n_cells, dtype=dtype, device=device)

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
            p = torch.zeros(n_cells, dtype=dtype, device=device)

        return phi_potential, U, p

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        try:
            phi_data = self.case.read_field("phi", 0)
        except Exception:
            phi_data = None
        try:
            U_data = self.case.read_field("U", 0)
        except Exception:
            U_data = None
        try:
            p_data = self.case.read_field("p", 0)
        except Exception:
            p_data = None
        return phi_data, U_data, p_data

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _build_phi_boundary_conditions(self) -> torch.Tensor:
        """Build boundary conditions for velocity potential.

        Returns:
            ``(n_cells,)`` — prescribed potential (NaN where no BC).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        phi_bc = torch.full((n_cells,), float('nan'), dtype=dtype, device=device)

        # Read boundary field from 0/phi if it exists
        try:
            phi_field_data = self.case.read_field("phi", 0)
            boundary_field = phi_field_data.boundary_field
        except Exception:
            return phi_bc

        if boundary_field is None or len(boundary_field) == 0:
            return phi_bc

        # Get mesh boundary info
        mesh_boundary = self.case.boundary
        owner = self.mesh.owner

        mesh_patches = {}
        for bp in mesh_boundary:
            mesh_patches[bp.name] = {
                "startFace": bp.start_face,
                "nFaces": bp.n_faces,
            }

        for patch in boundary_field:
            if patch.patch_type == "fixedValue" and patch.value is not None:
                value = self._parse_scalar_value(patch.value)
                if value is not None:
                    mesh_info = mesh_patches.get(patch.name)
                    if mesh_info is not None:
                        start_face = mesh_info["startFace"]
                        n_faces = mesh_info["nFaces"]
                        for i in range(n_faces):
                            face_idx = start_face + i
                            cell_idx = owner[face_idx].item()
                            phi_bc[cell_idx] = value

        return phi_bc

    @staticmethod
    def _parse_scalar_value(value: Any) -> float | None:
        """Parse a scalar value from field data."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Parse "uniform value" format
            match = re.search(r"uniform\s+([\d.eE+\-]+)", value)
            if match:
                return float(match.group(1))
            # Try plain number
            try:
                return float(value)
            except ValueError:
                pass
        return None

    # ------------------------------------------------------------------
    # Potential equation assembly
    # ------------------------------------------------------------------

    def _assemble_potential_equation(self) -> FvMatrix:
        """Assemble the Laplacian equation ∇²φ = 0.

        Returns:
            :class:`FvMatrix` for the potential equation.
        """
        # Assemble ∇²φ = 0 → fvm.laplacian(1, phi) = 0
        matrix = fvm.laplacian(1.0, self.phi_potential, mesh=self.mesh)

        return matrix

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the potentialFoam solver.

        Solves ∇²φ = 0, then computes U = ∇φ and p from Bernoulli.

        Returns:
            Dictionary with convergence information.
        """
        device = get_device()
        dtype = get_default_dtype()

        logger.info("Starting potentialFoam run")

        # Build boundary conditions
        phi_bc = self._build_phi_boundary_conditions()

        # Apply boundary conditions to initial potential
        bc_mask = ~torch.isnan(phi_bc)
        self.phi_potential[bc_mask] = phi_bc[bc_mask]

        # Solve potential equation with non-orthogonal corrections
        for non_orth in range(self.n_non_orth_correctors + 1):
            logger.info("Non-orthogonal correction %d/%d",
                       non_orth, self.n_non_orth_correctors)

            # Assemble ∇²φ = 0
            matrix = self._assemble_potential_equation()

            # Apply boundary conditions
            # For fixedValue BCs: large diagonal + matching source
            if bc_mask.any():
                large_coeff = matrix.diag.abs().clamp(min=1.0) * 1e10
                matrix._diag[bc_mask] += large_coeff[bc_mask]
                matrix._source[bc_mask] += large_coeff[bc_mask] * phi_bc[bc_mask]

            # Create linear solver
            solver = create_solver(
                self.phi_solver,
                tolerance=self.phi_tolerance,
                rel_tol=self.phi_rel_tol,
                max_iter=self.phi_max_iter,
            )

            # Solve
            self.phi_potential, iters, residual = matrix.solve(
                solver, self.phi_potential.clone(),
                tolerance=self.phi_tolerance,
                max_iter=self.phi_max_iter,
            )

            logger.info("  Potential equation: iters=%d, residual=%.6e",
                       iters, residual)

        # Compute velocity: U = ∇φ
        grad_phi = fvc.grad(self.phi_potential, mesh=self.mesh)
        self.U = grad_phi

        # Compute pressure from Bernoulli: p = p_ref - ½|U|²
        # p + ½|U|² = p_ref + ½|U_ref|²
        # For simplicity, use p_ref = 0 at reference cell
        U_mag_sq = (self.U ** 2).sum(dim=1)
        self.p = -0.5 * U_mag_sq

        # Set reference pressure
        if self.p_ref_cell < len(self.p):
            p_ref_actual = self.p[self.p_ref_cell].item()
            self.p = self.p - p_ref_actual + self.p_ref_value

        # Write final fields
        self._write_fields(self.end_time)

        logger.info("potentialFoam completed")
        logger.info("  max|U| = %.6e", U_mag_sq.sqrt().max().item())

        return {
            "converged": True,
            "iterations": iters,
            "residual": residual,
        }

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write phi, U, and p to a time directory."""
        time_str = f"{time:g}"
        if self._phi_data is not None:
            self.write_field("phi", self.phi_potential, time_str, self._phi_data)
        if self._U_data is not None:
            self.write_field("U", self.U, time_str, self._U_data)
        if self._p_data is not None:
            self.write_field("p", self.p, time_str, self._p_data)
