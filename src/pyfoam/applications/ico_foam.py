"""
icoFoam — transient incompressible laminar flow solver.

Implements the classic OpenFOAM icoFoam solver for transient
incompressible laminar Navier-Stokes equations using the PISO
(Pressure-Implicit with Splitting of Operators) algorithm.

This is the simplest transient incompressible solver:

    ∂U/∂t + ∇·(UU) - ∇²(νU) = -∇p
    ∇·U = 0

The solver reads:
- ``0/U``, ``0/p`` — initial/boundary conditions
- ``constant/polyMesh`` — mesh
- ``constant/transportProperties`` — kinematic viscosity (nu)
- ``system/controlDict`` — endTime, deltaT, writeControl, writeInterval
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — PISO settings, linear solver tolerances

Usage::

    from pyfoam.applications.ico_foam import IcoFoam

    solver = IcoFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.piso import PISOSolver, PISOConfig
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["IcoFoam"]

logger = logging.getLogger(__name__)


class IcoFoam(SolverBase):
    """Transient incompressible laminar PISO solver.

    Reads an OpenFOAM case directory and solves the transient
    incompressible Navier-Stokes equations using the PISO algorithm.

    This is the classic icoFoam solver — the simplest transient
    incompressible flow solver in OpenFOAM.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.

    Attributes
    ----------
    U : torch.Tensor
        ``(n_cells, 3)`` velocity field.
    p : torch.Tensor
        ``(n_cells,)`` pressure field.
    phi : torch.Tensor
        ``(n_faces,)`` face flux field.
    nu : float
        Kinematic viscosity.
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Read transport properties
        self.nu = self._read_nu()

        # Read PISO/fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes (for logging)
        self._read_fv_schemes_settings()

        # Initialise fields
        self.U, self.p, self.phi = self._init_fields()

        # Store raw field data for writing
        self._U_data, self._p_data = self._init_field_data()

        # Store old fields for time derivative
        self.U_old = self.U.clone()
        self.p_old = self.p.clone()

        logger.info("IcoFoam ready: nu=%.6e, Re~%.0f", self.nu, 1.0 / max(self.nu, 1e-30))

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_nu(self) -> float:
        """Read kinematic viscosity from transportProperties.

        The dictionary value may be a dimensioned scalar like
        ``[0 2 -1 0 0 0 0] 0.01`` or a plain number ``0.01``.
        """
        tp_path = self.case_path / "constant" / "transportProperties"
        if tp_path.exists():
            try:
                import re
                from pyfoam.io.dictionary import parse_dict_file
                tp = parse_dict_file(tp_path)
                raw = tp.get("nu", 1.0)
                if isinstance(raw, (int, float)):
                    return float(raw)
                # Parse dimensioned scalar: "[dims] value"
                raw_str = str(raw).strip()
                match = re.search(r"\]\s*([\d.eE+\-]+)", raw_str)
                if match:
                    return float(match.group(1))
                # Try plain number
                return float(raw_str)
            except Exception:
                pass
        return 1.0

    def _read_fv_solution_settings(self) -> None:
        """Read PISO settings from fvSolution."""
        fv = self.case.fvSolution

        # Pressure solver settings
        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_rel_tol = float(fv.get_path("solvers/p/relTol", 0.01))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        # Velocity solver settings
        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_rel_tol = float(fv.get_path("solvers/U/relTol", 0.01))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        # PISO algorithm settings
        self.n_piso_correctors = int(
            fv.get_path("PISO/nCorrectors", 2)
        )
        self.n_non_orth_correctors = int(
            fv.get_path("PISO/nNonOrthogonalCorrectors", 0)
        )

        # Convergence tolerance
        self.convergence_tolerance = float(
            fv.get_path("PISO/convergenceTolerance", 1e-4)
        )

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings (for logging)."""
        fs = self.case.fvSchemes
        self.ddt_scheme = str(fs.get_path("ddtSchemes/default", "Euler"))
        self.grad_scheme = str(fs.get_path("gradSchemes/default", "Gauss linear"))
        self.div_scheme = str(fs.get_path("divSchemes/default", "Gauss linear"))
        self.lap_scheme = str(fs.get_path("laplacianSchemes/default", "Gauss linear corrected"))

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise U, p, phi from the 0/ directory.

        Returns:
            Tuple of ``(U, p, phi)`` tensors.
        """
        device = get_device()
        dtype = get_default_dtype()

        # Read velocity
        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        # Read pressure
        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        # Compute initial flux from velocity field
        n_internal = self.mesh.n_internal_faces
        owner = self.mesh.owner[:n_internal]
        neighbour = self.mesh.neighbour
        w = self.mesh.face_weights[:n_internal]
        fa = self.mesh.face_areas[:n_internal]

        U_f = w.unsqueeze(-1) * U[owner] + (1 - w).unsqueeze(-1) * U[neighbour]
        phi_internal = (U_f * fa).sum(dim=1)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)
        phi[:n_internal] = phi_internal

        return U, p, phi

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        return U_data, p_data

    def _read_fv_options_body_force(self) -> 'torch.Tensor | None':
        """Read body force from constant/fvOptions."""
        fv_path = self.case_path / "constant" / "fvOptions"
        if not fv_path.exists():
            return None
        try:
            from pyfoam.io.dictionary import parse_dict_file
            fv_data = parse_dict_file(fv_path)
        except Exception:
            return None
        for key, block in fv_data.items():
            if isinstance(block, dict) and block.get("type") == "vectorSemiImplicitSource":
                su = block.get("Su", block.get("sources", {}).get("U"))
                if isinstance(su, (list, tuple)) and len(su) >= 3:
                    import torch
                    force = torch.tensor([float(su[0]), float(su[1]), float(su[2])],
                                         dtype=self.U.dtype, device=self.U.device)
                    return force.unsqueeze(0).expand(self.mesh.n_cells, -1).clone()
        return None

    # ------------------------------------------------------------------
    # PISO solver construction
    # ------------------------------------------------------------------

    def _build_solver(self) -> PISOSolver:
        """Build a PISOSolver with settings from fvSolution."""
        config = PISOConfig(
            n_correctors=self.n_piso_correctors,
            nu=self.nu,
            dt=self.delta_t,
            p_solver=self.p_solver,
            U_solver=self.U_solver,
            p_tolerance=self.p_tolerance,
            U_tolerance=self.U_tolerance,
            p_max_iter=self.p_max_iter,
            U_max_iter=self.U_max_iter,
            n_non_orthogonal_correctors=self.n_non_orth_correctors,
            relaxation_factor_U=1.0,  # PISO: no under-relaxation
            relaxation_factor_p=1.0,
        )
        return PISOSolver(self.mesh, config)

    # ------------------------------------------------------------------
    # Boundary condition setup
    # ------------------------------------------------------------------

    def _build_boundary_conditions(self) -> torch.Tensor:
        """Build the velocity BC tensor from the 0/U boundary field.

        Reads the boundary conditions from the ``0/U`` file and constructs
        a tensor of prescribed velocities.  Cells with fixed-value BCs get
        their prescribed values; cells without BCs get NaN.

        Returns:
            ``(n_cells, 3)`` — prescribed velocity (NaN where no BC).
        """
        import re

        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        U_bc = torch.full((n_cells, 3), float('nan'), dtype=dtype, device=device)

        # Read boundary field from 0/U
        U_field_data = self.case.read_field("U", 0)
        boundary_field = U_field_data.boundary_field

        if boundary_field is None or len(boundary_field) == 0:
            return U_bc

        # Get mesh boundary info (has startFace and nFaces)
        mesh_boundary = self.case.boundary

        owner = self.mesh.owner

        # Build a lookup from patch name to mesh boundary info
        mesh_patches = {}
        for bp in mesh_boundary:
            mesh_patches[bp.name] = {
                "startFace": bp.start_face,
                "nFaces": bp.n_faces,
            }

        # Iterate over BoundaryPatch objects from the field file.
        # Process non-zero BCs LAST so they take priority over zero BCs
        # at shared cells (e.g., corner cells touching both movingWall
        # and fixedWalls).
        patches_by_priority = []
        for patch in boundary_field:
            if patch.patch_type == "fixedValue" and patch.value is not None:
                value = self._parse_vector_value(patch.value)
                if value is not None:
                    is_zero = abs(value[0]) < 1e-30 and abs(value[1]) < 1e-30 and abs(value[2]) < 1e-30
                    patches_by_priority.append((0 if is_zero else 1, patch, value))

        # Sort: zero BCs first (priority 0), non-zero BCs last (priority 1)
        patches_by_priority.sort(key=lambda x: x[0])

        for _, patch, value in patches_by_priority:
            mesh_info = mesh_patches.get(patch.name)
            if mesh_info is not None:
                start_face = mesh_info["startFace"]
                n_faces = mesh_info["nFaces"]
                for i in range(n_faces):
                    face_idx = start_face + i
                    cell_idx = owner[face_idx].item()
                    U_bc[cell_idx, 0] = value[0]
                    U_bc[cell_idx, 1] = value[1]
                    U_bc[cell_idx, 2] = value[2]

        return U_bc

    @staticmethod
    def _parse_vector_value(value: Any) -> tuple[float, float, float] | None:
        """Parse a vector value from field data.

        Handles both tuple/list and string formats like ``'uniform ( 1 0 0 )'``.

        Args:
            value: Raw value from BoundaryPatch.

        Returns:
            Tuple of (x, y, z) floats, or None if unparseable.
        """
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))

        if isinstance(value, str):
            # Parse "uniform ( x y z )" format
            import re
            match = re.search(r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)", value)
            if match:
                return (float(match.group(1)), float(match.group(2)), float(match.group(3)))

        return None

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the icoFoam solver.

        Executes the PISO algorithm in a time-stepping loop until
        ``endTime`` is reached.

        Returns:
            Final :class:`ConvergenceData`.
        """
        solver = self._build_solver()

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

        logger.info("Starting icoFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  PISO correctors=%d", self.n_piso_correctors)
        logger.info("  ddtScheme=%s, divScheme=%s", self.ddt_scheme, self.div_scheme)

        # Build boundary conditions
        U_bc = self._build_boundary_conditions()

        # Read body force from fvOptions
        body_force = self._read_fv_options_body_force()

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Store old fields for time derivative
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            # Run one PISO time step
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                U_old=self.U_old,
                p_old=self.p_old,
                body_force=body_force,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Track convergence
            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
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

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("icoFoam completed successfully (converged)")
            else:
                logger.warning("icoFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, and phi to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
