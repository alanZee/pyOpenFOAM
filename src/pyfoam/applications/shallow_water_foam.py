"""
shallowWaterFoam — 2D shallow water equations solver.

Solves the depth-averaged shallow water equations with Coriolis force
and quadratic bottom friction using a PISO-like algorithm:

    Continuity:   ∂h/∂t + ∇·(hU) = 0
    Momentum:     ∂(hU)/∂t + ∇·(hUU) + (g/2)∇(h²) = f k×(hU) - Cf|U|U

Fields:
- ``U``  — depth-averaged velocity (vector)
- ``h``  — water depth (scalar)
- ``phi`` — volume flux h*U·Sf across faces

Physical parameters (from ``constant/shallowWaterProperties``):
- ``g``  — gravitational acceleration (default 9.81)
- ``f``  — Coriolis parameter (default 0)
- ``Cf`` — bottom friction coefficient (default 0)

The solver reads:
- ``0/U``, ``0/h`` — initial/boundary conditions
- ``constant/polyMesh`` — mesh
- ``constant/shallowWaterProperties`` — physical parameters
- ``system/controlDict`` — endTime, deltaT, writeControl, writeInterval
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — PISO settings, linear solver tolerances

Usage::

    from pyfoam.applications.shallow_water_foam import ShallowWaterFoam

    solver = ShallowWaterFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import re
from typing import Any, Union

from pathlib import Path

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.piso import PISOSolver, PISOConfig
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["ShallowWaterFoam"]

logger = logging.getLogger(__name__)


class ShallowWaterFoam(SolverBase):
    """2D shallow water equations solver using PISO algorithm.

    Reads an OpenFOAM case directory and solves the depth-averaged
    shallow water equations with Coriolis and bottom friction source
    terms.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.

    Attributes
    ----------
    U : torch.Tensor
        ``(n_cells, 3)`` depth-averaged velocity field.
    h : torch.Tensor
        ``(n_cells,)`` water depth field.
    phi : torch.Tensor
        ``(n_faces,)`` face volume flux field.
    g : float
        Gravitational acceleration.
    f : float
        Coriolis parameter.
    Cf : float
        Bottom friction coefficient.
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Read physical properties
        self.g, self.f, self.Cf = self._read_shallow_water_properties()

        # Read PISO/fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes (for logging)
        self._read_fv_schemes_settings()

        # Initialise fields
        self.U, self.h, self.phi = self._init_fields()

        # Store raw field data for writing
        self._U_data, self._h_data = self._init_field_data()

        # Store old fields for time derivative
        self.U_old = self.U.clone()
        self.h_old = self.h.clone()

        logger.info(
            "ShallowWaterFoam ready: g=%.4f, f=%.4e, Cf=%.4e",
            self.g, self.f, self.Cf,
        )

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_shallow_water_properties(self) -> tuple[float, float, float]:
        """Read shallow water physical properties.

        Reads ``constant/shallowWaterProperties`` for:
        - ``g``  — gravitational acceleration (default 9.81 m/s²)
        - ``f``  — Coriolis parameter (default 0, no rotation)
        - ``Cf`` — bottom friction coefficient (default 0, frictionless)

        Returns:
            Tuple of ``(g, f, Cf)``.
        """
        props_path = self.case_path / "constant" / "shallowWaterProperties"
        g, f, Cf = 9.81, 0.0, 0.0

        if props_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                props = parse_dict_file(props_path)
                g = self._parse_scalar(props.get("g", 9.81), 9.81)
                f = self._parse_scalar(props.get("f", 0.0), 0.0)
                Cf = self._parse_scalar(props.get("Cf", 0.0), 0.0)
            except Exception:
                pass

        return g, f, Cf

    @staticmethod
    def _parse_scalar(raw: Any, default: float) -> float:
        """Parse a scalar value that may be dimensioned.

        Handles formats like ``[0 2 -1 0 0 0 0] 0.01`` or plain numbers.
        """
        if isinstance(raw, (int, float)):
            return float(raw)
        raw_str = str(raw).strip()
        match = re.search(r"\]\s*([\d.eE+\-]+)", raw_str)
        if match:
            return float(match.group(1))
        try:
            return float(raw_str)
        except (ValueError, TypeError):
            return default

    def _read_fv_solution_settings(self) -> None:
        """Read PISO settings from fvSolution."""
        fv = self.case.fvSolution

        # Pressure (water depth h) solver settings
        self.h_solver = str(fv.get_path("solvers/h/solver", "PCG"))
        self.h_tolerance = float(fv.get_path("solvers/h/tolerance", 1e-6))
        self.h_rel_tol = float(fv.get_path("solvers/h/relTol", 0.01))
        self.h_max_iter = int(fv.get_path("solvers/h/maxIter", 1000))

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
        self.lap_scheme = str(
            fs.get_path("laplacianSchemes/default", "Gauss linear corrected")
        )

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise U, h, phi from the 0/ directory.

        Returns:
            Tuple of ``(U, h, phi)`` tensors.
        """
        device = get_device()
        dtype = get_default_dtype()

        # Read velocity
        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        # Read water depth
        h_tensor, _ = self.read_field_tensor("h", 0)
        h = h_tensor.to(device=device, dtype=dtype)

        # Ensure h has positive values for numerical stability
        h_min = 1e-6
        h = torch.clamp(h, min=h_min)

        # Initialise volume flux to zero
        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return U, h, phi

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        h_data = self.case.read_field("h", 0)
        return U_data, h_data

    # ------------------------------------------------------------------
    # Source terms
    # ------------------------------------------------------------------

    def _coriolis_source(self, U: torch.Tensor) -> torch.Tensor:
        """Compute Coriolis source term: f * k x U.

        In 2D shallow water, the Coriolis force acts perpendicular
        to the velocity in the horizontal plane:
            S_Coriolis_x = f * U_y
            S_Coriolis_y = -f * U_x

        Args:
            U: ``(n_cells, 3)`` velocity field.

        Returns:
            ``(n_cells, 3)`` Coriolis acceleration.
        """
        device = U.device
        dtype = U.dtype
        n_cells = U.shape[0]

        S = torch.zeros_like(U)
        if abs(self.f) > 1e-30:
            # f * k x U: (f*Uy, -f*Ux, 0)
            S[:, 0] = self.f * U[:, 1]
            S[:, 1] = -self.f * U[:, 0]
        return S

    def _friction_source(
        self, U: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bottom friction source term: -Cf * |U| * U / h.

        The quadratic bottom drag is:
            S_friction = -Cf * |U| * U / h

        For numerical stability, a small depth floor is applied.

        Args:
            U: ``(n_cells, 3)`` velocity field.
            h: ``(n_cells,)`` water depth.

        Returns:
            ``(n_cells, 3)`` friction acceleration.
        """
        if abs(self.Cf) < 1e-30:
            return torch.zeros_like(U)

        # |U| = sqrt(Ux² + Uy²), only horizontal components
        U_mag = torch.sqrt(U[:, 0] ** 2 + U[:, 1] ** 2 + 1e-30)
        h_safe = torch.clamp(h, min=1e-6)

        S = torch.zeros_like(U)
        S[:, 0] = -self.Cf * U_mag * U[:, 0] / h_safe
        S[:, 1] = -self.Cf * U_mag * U[:, 1] / h_safe
        return S

    def _total_source(
        self, U: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total source terms for the momentum equation.

        Combines Coriolis and bottom friction source terms:
            S = f k×U - Cf|U|U/h

        Args:
            U: ``(n_cells, 3)`` velocity field.
            h: ``(n_cells,)`` water depth.

        Returns:
            ``(n_cells, 3)`` total source acceleration.
        """
        return self._coriolis_source(U) + self._friction_source(U, h)

    # ------------------------------------------------------------------
    # PISO solver construction
    # ------------------------------------------------------------------

    def _build_solver(self) -> PISOSolver:
        """Build a PISOSolver with settings from fvSolution."""
        config = PISOConfig(
            n_correctors=self.n_piso_correctors,
            p_solver=self.h_solver,
            U_solver=self.U_solver,
            p_tolerance=self.h_tolerance,
            U_tolerance=self.U_tolerance,
            p_max_iter=self.h_max_iter,
            U_max_iter=self.U_max_iter,
            n_non_orthogonal_correctors=self.n_non_orth_correctors,
            relaxation_factor_U=1.0,  # PISO: no under-relaxation
            relaxation_factor_p=1.0,
        )
        return PISOSolver(self.mesh, config)

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _build_boundary_conditions(self) -> torch.Tensor:
        """Build the velocity BC tensor from the 0/U boundary field.

        Reads the boundary conditions from the ``0/U`` file and constructs
        a tensor of prescribed velocities.  Cells with fixed-value BCs get
        their prescribed values; cells without BCs get NaN.

        Returns:
            ``(n_cells, 3)`` — prescribed velocity (NaN where no BC).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        U_bc = torch.full((n_cells, 3), float('nan'), dtype=dtype, device=device)

        # Read boundary field from 0/U
        U_field_data = self.case.read_field("U", 0)
        boundary_field = U_field_data.boundary_field

        if boundary_field is None or len(boundary_field) == 0:
            return U_bc

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
                value = self._parse_vector_value(patch.value)
                if value is not None:
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

    def _build_h_boundary_conditions(self) -> torch.Tensor:
        """Build the water depth BC tensor from the 0/h boundary field.

        Returns:
            ``(n_cells,)`` — prescribed water depth (NaN where no BC).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        h_bc = torch.full((n_cells,), float('nan'), dtype=dtype, device=device)

        h_field_data = self.case.read_field("h", 0)
        boundary_field = h_field_data.boundary_field

        if boundary_field is None or len(boundary_field) == 0:
            return h_bc

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
                            h_bc[cell_idx] = value

        return h_bc

    @staticmethod
    def _parse_vector_value(value: Any) -> tuple[float, float, float] | None:
        """Parse a vector value from field data.

        Handles both tuple/list and string formats like ``'uniform ( 1 0 0 )'``.
        """
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

    @staticmethod
    def _parse_scalar_value(value: Any) -> float | None:
        """Parse a scalar value from field data.

        Handles formats like ``'uniform 1.0'`` or plain numbers.
        """
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Parse "uniform value" format
            match = re.search(r"uniform\s+([\d.eE+\-]+)", value)
            if match:
                return float(match.group(1))
            # Try plain number
            try:
                return float(value.strip())
            except (ValueError, TypeError):
                pass

        return None

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the shallowWaterFoam solver.

        Executes the PISO-like algorithm adapted for shallow water
        equations in a time-stepping loop until ``endTime`` is reached.

        Each time step:
        1. Store old fields
        2. Compute source terms (Coriolis + friction)
        3. Predictor: solve momentum equation for U
        4. Corrector: solve continuity equation for h (pressure-like)
        5. Update face flux phi

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

        logger.info("Starting shallowWaterFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  PISO correctors=%d", self.n_piso_correctors)
        logger.info("  g=%.4f, f=%.4e, Cf=%.4e", self.g, self.f, self.Cf)

        # Build boundary conditions
        U_bc = self._build_boundary_conditions()
        _h_bc = self._build_h_boundary_conditions()

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Store old fields for time derivative
            self.U_old = self.U.clone()
            self.h_old = self.h.clone()

            # Compute source terms
            source = self._total_source(self.U, self.h)

            # Run one PISO time step
            # The h field is treated as "pressure" in the PISO algorithm
            self.U, self.h, self.phi, conv = solver.solve(
                self.U, self.h, self.phi,
                U_bc=U_bc,
                U_old=self.U_old,
                p_old=self.h_old,
                tolerance=self.convergence_tolerance,
            )

            # Apply source term corrections as explicit update
            # Semi-implicit: U_new = U + dt * source (after PISO solve)
            if source.abs().max() > 1e-30:
                U_source = self.delta_t * source
                # Only apply to interior cells (not BC cells)
                bc_mask = ~torch.isnan(U_bc[:, 0])
                U_source[bc_mask] = 0.0
                self.U = self.U + U_source

            # Clamp water depth to positive values
            self.h = torch.clamp(self.h, min=1e-6)

            last_convergence = conv

            # Track convergence
            residuals = {
                "U": conv.U_residual,
                "h": conv.p_residual,
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
                logger.info("shallowWaterFoam completed successfully (converged)")
            else:
                logger.warning("shallowWaterFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, h, and phi to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("h", self.h, time_str, self._h_data)
