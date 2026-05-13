"""
pimpleFoam — transient incompressible solver with PIMPLE algorithm.

Implements the PIMPLE algorithm for transient incompressible Navier-Stokes
equations with optional RANS turbulence modelling.  PIMPLE merges PISO and
SIMPLE: outer SIMPLE-like iterations with under-relaxation, each containing
inner PISO pressure corrections.

This is the recommended transient incompressible solver when:
- The time step is too large for pure PISO
- Under-relaxation is needed for stability
- Both transient accuracy and convergence are required

The solver reads:
- ``0/U``, ``0/p`` — initial/boundary conditions
- ``constant/polyMesh`` — mesh
- ``constant/transportProperties`` — kinematic viscosity (nu)
- ``constant/turbulenceProperties`` — turbulence model selection (optional)
- ``system/controlDict`` — endTime, deltaT, writeControl, writeInterval
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — PIMPLE settings, linear solver tolerances

Usage::

    from pyfoam.applications.pimple_foam import PimpleFoam

    solver = PimpleFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.pimple import PIMPLESolver, PIMPLEConfig
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.turbulence.ras_model import RASModel, RASConfig

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PimpleFoam"]

logger = logging.getLogger(__name__)


class PimpleFoam(SolverBase):
    """Transient incompressible PIMPLE solver with turbulence support.

    Reads an OpenFOAM case directory and solves the transient
    incompressible Navier-Stokes equations using the PIMPLE algorithm.

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
    turbulence : RASModel
        Turbulence model wrapper (may be disabled).
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Read transport properties
        self.nu = self._read_nu()

        # Read PIMPLE/fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes (for logging)
        self._read_fv_schemes_settings()

        # Initialise fields
        self.U, self.p, self.phi = self._init_fields()

        # Store raw field data for writing
        self._U_data, self._p_data = self._init_field_data()

        # Old fields for time derivative
        self.U_old = self.U.clone()
        self.p_old = self.p.clone()

        # Turbulence model
        self.turbulence = self._init_turbulence()

        logger.info("PimpleFoam ready: nu=%.6e, Re~%.0f", self.nu, 1.0 / max(self.nu, 1e-30))
        if self.turbulence.enabled:
            logger.info("  Turbulence: %s", self.turbulence.config.model_name)
        else:
            logger.info("  Turbulence: laminar")

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
                from pyfoam.io.dictionary import parse_dict_file
                tp = parse_dict_file(tp_path)
                raw = tp.get("nu", 1.0)
                if isinstance(raw, (int, float)):
                    return float(raw)
                raw_str = str(raw).strip()
                match = re.search(r"\]\s*([\d.eE+\-]+)", raw_str)
                if match:
                    return float(match.group(1))
                return float(raw_str)
            except Exception:
                pass
        return 1.0

    def _read_fv_solution_settings(self) -> None:
        """Read PIMPLE settings from fvSolution."""
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

        # PIMPLE algorithm settings
        self.n_outer_correctors = int(
            fv.get_path("PIMPLE/nOuterCorrectors", 3)
        )
        self.n_correctors = int(
            fv.get_path("PIMPLE/nCorrectors", 2)
        )
        self.n_non_orth_correctors = int(
            fv.get_path("PIMPLE/nNonOrthogonalCorrectors", 0)
        )

        # Relaxation factors
        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))

        # Convergence tolerance
        self.convergence_tolerance = float(
            fv.get_path("PIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("PIMPLE/maxOuterIterations", 100)
        )

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings (for logging)."""
        fs = self.case.fvSchemes
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

        # Initialise flux to zero
        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return U, p, phi

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        return U_data, p_data

    # ------------------------------------------------------------------
    # Turbulence model
    # ------------------------------------------------------------------

    def _init_turbulence(self) -> RASModel:
        """Initialise turbulence model from turbulenceProperties.

        Reads ``constant/turbulenceProperties`` to determine:
        - Whether RAS turbulence modelling is enabled
        - Which model to use (e.g. kEpsilon, kOmegaSST)

        If the file doesn't exist or RAS is disabled, returns a
        disabled :class:`RASModel` (laminar mode).
        """
        turb_path = self.case_path / "constant" / "turbulenceProperties"
        model_name = "kEpsilon"
        enabled = False

        if turb_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                turb = parse_dict_file(turb_path)
                simulation_type = str(turb.get("simulationType", "laminar")).strip()
                if simulation_type == "RAS":
                    enabled = True
                    ras_dict = turb.get("RAS", {})
                    if isinstance(ras_dict, dict):
                        model_name = str(ras_dict.get("model", "kEpsilon"))
                    else:
                        model_name = "kEpsilon"
            except Exception:
                pass

        config = RASConfig(
            model_name=model_name,
            enabled=enabled,
            nu=self.nu,
        )

        ras = RASModel(self.mesh, self.U, self.phi, config)
        return ras

    # ------------------------------------------------------------------
    # PIMPLE solver construction
    # ------------------------------------------------------------------

    def _build_solver(self) -> PIMPLESolver:
        """Build a PIMPLESolver with settings from fvSolution."""
        config = PIMPLEConfig(
            n_outer_correctors=self.n_outer_correctors,
            n_correctors=self.n_correctors,
            p_solver=self.p_solver,
            U_solver=self.U_solver,
            p_tolerance=self.p_tolerance,
            U_tolerance=self.U_tolerance,
            p_max_iter=self.p_max_iter,
            U_max_iter=self.U_max_iter,
            n_non_orthogonal_correctors=self.n_non_orth_correctors,
            relaxation_factor_p=self.alpha_p,
            relaxation_factor_U=self.alpha_U,
        )
        return PIMPLESolver(self.mesh, config)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the pimpleFoam solver.

        Executes the PIMPLE algorithm in a transient time-stepping loop
        until ``endTime`` is reached.

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

        logger.info("Starting pimpleFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  nOuterCorrectors=%d, nCorrectors=%d",
                     self.n_outer_correctors, self.n_correctors)
        logger.info("  relaxation: alpha_U=%.2f, alpha_p=%.2f",
                     self.alpha_U, self.alpha_p)

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Store old fields for time derivative
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            # Update turbulence (solve transport equations)
            if self.turbulence.enabled:
                self.turbulence.correct()

            # Build boundary condition tensor
            U_bc = self._build_boundary_conditions()

            # Run one PIMPLE time step
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                U_old=self.U_old,
                p_old=self.p_old,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Check convergence
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
                logger.info("pimpleFoam completed successfully (converged)")
            else:
                logger.warning("pimpleFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _build_boundary_conditions(self) -> torch.Tensor | None:
        """Build the U_bc tensor from the 0/U boundary field.

        Reads the boundary conditions from the ``0/U`` file and constructs
        a tensor of prescribed velocities.  Cells with fixed-value BCs get
        their prescribed values; cells without BCs get NaN.

        Returns:
            ``(n_cells, 3)`` — prescribed velocity (NaN where no BC).
            Returns ``None`` if no fixed-value velocity BCs are found.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        U_bc = torch.full((n_cells, 3), float('nan'), dtype=dtype, device=device)

        # Read boundary field from 0/U
        U_field_data = self._U_data
        boundary_field = U_field_data.boundary_field

        if boundary_field is None or len(boundary_field) == 0:
            return None

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

        has_bc = False

        # Iterate over BoundaryPatch objects from the field file
        for patch in boundary_field:
            if patch.patch_type == "fixedValue" and patch.value is not None:
                value = self._parse_vector_value(patch.value)
                if value is not None:
                    # Get face range from mesh boundary info
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
                        has_bc = True

        return U_bc if has_bc else None

    @staticmethod
    def _parse_vector_value(value):
        """Parse a vector value from field data.

        Handles both tuple/list and string formats like ``'uniform ( 1 0 0 )'``.

        Args:
            value: Raw value from BoundaryPatch.

        Returns:
            Tuple of (x, y, z) floats, or None if parsing fails.
        """
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))

        if isinstance(value, str):
            match = re.search(r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)", value)
            if match:
                return (float(match.group(1)), float(match.group(2)), float(match.group(3)))

        return None

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, and phi to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
