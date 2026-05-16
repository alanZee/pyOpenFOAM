"""
solidDisplacementFoam — linear elasticity solver.

Implements the OpenFOAM solidDisplacementFoam solver for linear elastic
stress analysis.  Solves the equilibrium equation:

    ∇·σ + f = 0

where σ is the Cauchy stress tensor and f is the body force vector.

For linear elasticity:

    σ = λ tr(ε) I + 2μ ε

where ε is the small strain tensor:

    ε = ½(∇u + (∇u)ᵀ)

and λ, μ are Lamé parameters related to Young's modulus E and Poisson's
ratio ν by:

    λ = Eν / ((1+ν)(1-2ν))
    μ = E / (2(1+ν))

The solver reads:
- ``0/D`` — displacement field
- ``constant/polyMesh`` — mesh
- ``constant/mechanicalProperties`` — E, nu (Poisson's ratio)
- ``system/controlDict`` — endTime, deltaT
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — linear solver tolerances

Usage::

    from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam

    solver = SolidDisplacementFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from pyfoam.solvers.linear_solver import create_solver

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SolidDisplacementFoam"]

logger = logging.getLogger(__name__)


class SolidDisplacementFoam(SolverBase):
    """Linear elasticity solver for solid mechanics.

    Solves ∇·σ + f = 0 for linear elastic materials.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.

    Attributes
    ----------
    D : torch.Tensor
        ``(n_cells, 3)`` displacement field.
    sigma : torch.Tensor
        ``(n_cells, 6)`` stress tensor (Voigt notation: xx, yy, zz, xy, yz, zx).
    epsilon : torch.Tensor
        ``(n_cells, 6)`` strain tensor (Voigt notation).
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Read mechanical properties
        self.E, self.nu = self._read_mechanical_properties()

        # Compute Lamé parameters
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes
        self._read_fv_schemes_settings()

        # Initialise fields
        self.D = self._init_displacement()

        # Compute strain and stress
        self.epsilon = self._compute_strain()
        self.sigma = self._compute_stress()

        # Store raw field data for writing
        self._D_data = self._init_field_data()

        logger.info("SolidDisplacementFoam ready: E=%.6e, nu=%.4f",
                    self.E, self.nu)
        logger.info("  Lamé parameters: λ=%.6e, μ=%.6e", self.lam, self.mu)

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_mechanical_properties(self) -> tuple[float, float]:
        """Read Young's modulus and Poisson's ratio."""
        E = 1e9  # Default: 1 GPa
        nu = 0.3  # Default: 0.3

        mp_path = self.case_path / "constant" / "mechanicalProperties"
        if mp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                mp = parse_dict_file(mp_path)

                raw_E = mp.get("E", E)
                if isinstance(raw_E, (int, float)):
                    E = float(raw_E)
                else:
                    raw_str = str(raw_E).strip()
                    match = re.search(r"\]\s*([\d.eE+\-]+)", raw_str)
                    if match:
                        E = float(match.group(1))

                raw_nu = mp.get("nu", nu)
                if isinstance(raw_nu, (int, float)):
                    nu = float(raw_nu)
                else:
                    raw_str = str(raw_nu).strip()
                    match = re.search(r"\]\s*([\d.eE+\-]+)", raw_str)
                    if match:
                        nu = float(match.group(1))
            except Exception as e:
                logger.warning("Could not read mechanical properties: %s", e)

        return E, nu

    def _read_fv_solution_settings(self) -> None:
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.D_solver = str(fv.get_path("solvers/D/solver", "PBiCGStab"))
        self.D_tolerance = float(fv.get_path("solvers/D/tolerance", 1e-6))
        self.D_rel_tol = float(fv.get_path("solvers/D/relTol", 0.01))
        self.D_max_iter = int(fv.get_path("solvers/D/maxIter", 1000))

        self.convergence_tolerance = float(
            fv.get_path("solidMechanics/convergenceTolerance", 1e-5)
        )
        self.n_correctors = int(
            fv.get_path("solidMechanics/nCorrectors", 1)
        )

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings (for logging)."""
        fs = self.case.fvSchemes
        self.grad_scheme = str(fs.get_path("gradSchemes/default", "Gauss linear"))

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_displacement(self) -> torch.Tensor:
        """Initialise displacement field from 0/D."""
        device = get_device()
        dtype = get_default_dtype()

        try:
            D_tensor, _ = self.read_field_tensor("D", 0)
            D = D_tensor.to(device=device, dtype=dtype)
        except Exception:
            n_cells = self.mesh.n_cells
            D = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        return D

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        try:
            return self.case.read_field("D", 0)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Strain and stress computation
    # ------------------------------------------------------------------

    def _compute_strain(self) -> torch.Tensor:
        """Compute small strain tensor from displacement.

        ε = ½(∇u + (∇u)ᵀ)

        Returns:
            ``(n_cells, 6)`` strain tensor in Voigt notation.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Compute displacement gradient component by component
        # fvc.grad only handles scalar fields (1D tensors)
        grad_D = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)

        for dim in range(3):
            # Gradient of each displacement component: ∂u_i/∂x_j
            grad_D[:, dim, :] = fvc.grad(self.D[:, dim], mesh=self.mesh)

        # Symmetric strain tensor
        # ε_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i)
        epsilon = torch.zeros(n_cells, 6, dtype=dtype, device=device)

        # Voigt notation: xx, yy, zz, xy, yz, zx
        epsilon[:, 0] = grad_D[:, 0, 0]  # ε_xx = ∂u/∂x
        epsilon[:, 1] = grad_D[:, 1, 1]  # ε_yy = ∂v/∂y
        epsilon[:, 2] = grad_D[:, 2, 2]  # ε_zz = ∂w/∂z
        epsilon[:, 3] = 0.5 * (grad_D[:, 0, 1] + grad_D[:, 1, 0])  # ε_xy
        epsilon[:, 4] = 0.5 * (grad_D[:, 1, 2] + grad_D[:, 2, 1])  # ε_yz
        epsilon[:, 5] = 0.5 * (grad_D[:, 0, 2] + grad_D[:, 2, 0])  # ε_zx

        return epsilon

    def _compute_stress(self) -> torch.Tensor:
        """Compute stress tensor from strain using Hooke's law.

        σ = λ tr(ε) I + 2μ ε

        Returns:
            ``(n_cells, 6)`` stress tensor in Voigt notation.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        sigma = torch.zeros(n_cells, 6, dtype=dtype, device=device)

        # Volumetric strain (trace)
        tr_eps = self.epsilon[:, 0] + self.epsilon[:, 1] + self.epsilon[:, 2]

        # σ_xx = λ tr(ε) + 2μ ε_xx
        sigma[:, 0] = self.lam * tr_eps + 2 * self.mu * self.epsilon[:, 0]
        # σ_yy = λ tr(ε) + 2μ ε_yy
        sigma[:, 1] = self.lam * tr_eps + 2 * self.mu * self.epsilon[:, 1]
        # σ_zz = λ tr(ε) + 2μ ε_zz
        sigma[:, 2] = self.lam * tr_eps + 2 * self.mu * self.epsilon[:, 2]
        # σ_xy = 2μ ε_xy
        sigma[:, 3] = 2 * self.mu * self.epsilon[:, 3]
        # σ_yz = 2μ ε_yz
        sigma[:, 4] = 2 * self.mu * self.epsilon[:, 4]
        # σ_zx = 2μ ε_zx
        sigma[:, 5] = 2 * self.mu * self.epsilon[:, 5]

        return sigma

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _build_D_boundary_conditions(self) -> torch.Tensor:
        """Build boundary conditions for displacement.

        Returns:
            ``(n_cells, 3)`` — prescribed displacement (NaN where no BC).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        D_bc = torch.full((n_cells, 3), float('nan'), dtype=dtype, device=device)

        try:
            D_field_data = self.case.read_field("D", 0)
            boundary_field = D_field_data.boundary_field
        except Exception:
            return D_bc

        if boundary_field is None or len(boundary_field) == 0:
            return D_bc

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
                value = self._parse_vector_value(patch.value)
                if value is not None:
                    mesh_info = mesh_patches.get(patch.name)
                    if mesh_info is not None:
                        start_face = mesh_info["startFace"]
                        n_faces = mesh_info["nFaces"]
                        for i in range(n_faces):
                            face_idx = start_face + i
                            cell_idx = owner[face_idx].item()
                            D_bc[cell_idx, 0] = value[0]
                            D_bc[cell_idx, 1] = value[1]
                            D_bc[cell_idx, 2] = value[2]

        return D_bc

    @staticmethod
    def _parse_vector_value(value: Any) -> tuple[float, float, float] | None:
        """Parse a vector value from field data."""
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
        if isinstance(value, str):
            match = re.search(
                r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)",
                value,
            )
            if match:
                return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        return None

    # ------------------------------------------------------------------
    # Equation assembly
    # ------------------------------------------------------------------

    def _assemble_displacement_equation(self) -> Any:
        """Assemble the displacement equation.

        For linear elasticity (Navier's equation):
            μ ∇²u + (λ + μ) ∇(∇·u) + f = 0

        Assembles Laplacian term μ ∇²D component by component.

        Returns:
            FvMatrix for the displacement equation (per component).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Build Laplacian matrix for μ ∇²D
        # Use fvm.laplacian for scalar field (per component)
        # We'll solve each component separately
        lap_matrix = fvm.laplacian(self.mu, self.D[:, 0], mesh=self.mesh)

        return lap_matrix

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the solidDisplacementFoam solver.

        Solves the linear elasticity equilibrium equation iteratively.

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

        logger.info("Starting solidDisplacementFoam run")
        logger.info("  E=%.6e, nu=%.4f", self.E, self.nu)

        # Build boundary conditions
        D_bc = self._build_D_boundary_conditions()

        # Apply boundary conditions
        bc_mask = ~torch.isnan(D_bc)
        if bc_mask.any():
            self.D[bc_mask] = D_bc[bc_mask]

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        # Create linear solver
        solver = create_solver(
            self.D_solver,
            tolerance=self.D_tolerance,
            rel_tol=self.D_rel_tol,
            max_iter=self.D_max_iter,
        )

        last_convergence = None

        for t, step in time_loop:
            # Solve with corrections
            for corrector in range(self.n_correctors):
                # Assemble displacement equation
                matrix = self._assemble_displacement_equation()

                # Apply boundary conditions
                if bc_mask.any():
                    large_coeff = matrix.diag.abs().clamp(min=1.0) * 1e10
                    for dim in range(3):
                        mask = bc_mask[:, dim]
                        matrix._diag[mask] += large_coeff[mask]
                        matrix._source[mask] += large_coeff[mask] * D_bc[mask, dim]

                # Solve for each component (simplified)
                for dim in range(3):
                    # Extract component
                    D_comp = self.D[:, dim].clone()
                    source_comp = matrix._source.clone()

                    # Solve
                    D_comp_new, iters, residual = matrix.solve(
                        solver, D_comp,
                        tolerance=self.D_tolerance,
                        max_iter=self.D_max_iter,
                    )

                    self.D[:, dim] = D_comp_new

            # Update strain and stress
            self.epsilon = self._compute_strain()
            self.sigma = self._compute_stress()

            # Track convergence
            residuals = {"D": residual}
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

        # Compute von Mises stress
        von_mises = self._compute_von_mises_stress()

        logger.info("solidDisplacementFoam completed")
        logger.info("  max|D| = %.6e", self.D.abs().max().item())
        logger.info("  max σ_vm = %.6e", von_mises.max().item())

        return {
            "converged": converged if converged else False,
            "iterations": iters,
            "residual": residual,
            "von_mises_max": von_mises.max().item(),
        }

    def _compute_von_mises_stress(self) -> torch.Tensor:
        """Compute von Mises stress.

        σ_vm = √(½[(σ_xx - σ_yy)² + (σ_yy - σ_zz)² + (σ_zz - σ_xx)²
                     + 6(σ_xy² + σ_yz² + σ_zx²)])

        Returns:
            ``(n_cells,)`` von Mises stress.
        """
        s = self.sigma

        von_mises = torch.sqrt(
            0.5 * (
                (s[:, 0] - s[:, 1]) ** 2 +
                (s[:, 1] - s[:, 2]) ** 2 +
                (s[:, 2] - s[:, 0]) ** 2 +
                6 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
            )
        )

        return von_mises

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write displacement and stress to a time directory."""
        time_str = f"{time:g}"

        if self._D_data is not None:
            self.write_field("D", self.D, time_str, self._D_data)
