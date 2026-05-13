"""
simpleFoam — steady-state incompressible solver.

Implements the SIMPLE algorithm for steady-state incompressible
Navier-Stokes equations.  This is the first complete application-level
solver demonstrating the full pyOpenFOAM pipeline:

    read case → build mesh → initialise fields → SIMPLE loop → write results

The solver reads:
- ``0/U``, ``0/p`` — initial/boundary conditions
- ``constant/polyMesh`` — mesh
- ``constant/transportProperties`` — kinematic viscosity (nu)
- ``constant/turbulenceProperties`` — turbulence model selection (optional)
- ``system/controlDict`` — endTime, deltaT, writeControl, writeInterval
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — SIMPLE settings, linear solver tolerances

Usage::

    from pyfoam.applications.simple_foam import SimpleFoam

    solver = SimpleFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SimpleFoam"]

logger = logging.getLogger(__name__)


class SimpleFoam(SolverBase):
    """Steady-state incompressible SIMPLE solver.

    Reads an OpenFOAM case directory and solves the steady-state
    incompressible Navier-Stokes equations using the SIMPLE algorithm.

    Supports optional RANS turbulence modelling via
    ``constant/turbulenceProperties``.  When enabled, the effective
    viscosity ``ν_eff = ν + ν_t`` is passed to the SIMPLE solver
    each outer iteration.

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
    turbulence_enabled : bool
        Whether RANS turbulence modelling is active.
    ras : RASModel or None
        The RAS turbulence model wrapper (None if turbulence is disabled).
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Read transport properties
        self.nu = self._read_nu()

        # Read SIMPLE/fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes (for logging; actual schemes handled by solver)
        self._read_fv_schemes_settings()

        # Initialise fields
        self.U, self.p, self.phi = self._init_fields()

        # Store raw field data for writing
        self._U_data, self._p_data = self._init_field_data()

        # Turbulence model (optional)
        self.ras, self.turbulence_enabled = self._init_turbulence()

        logger.info("SimpleFoam ready: nu=%.6e, Re~%.0f", self.nu, 1.0 / max(self.nu, 1e-30))
        if self.turbulence_enabled:
            logger.info("  Turbulence: %s", self.ras)

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
        """Read SIMPLE settings from fvSolution."""
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

        # SIMPLE algorithm settings
        self.n_non_orth_correctors = int(
            fv.get_path("SIMPLE/nNonOrthogonalCorrectors", 0)
        )
        self.alpha_p = float(fv.get_path("SIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("SIMPLE/relaxationFactors/U", 0.7))

        # Convergence tolerance
        self.convergence_tolerance = float(
            fv.get_path("SIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("SIMPLE/maxOuterIterations", 100)
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

    def _init_turbulence(self) -> tuple[Any, bool]:
        """Initialise RAS turbulence model from turbulenceProperties.

        Reads ``constant/turbulenceProperties`` to determine:
        - ``simulationType``: ``laminar`` (default) or ``RAS``
        - ``RAS/model``: model name (e.g. ``kEpsilon``, ``kOmegaSST``)

        Returns:
            Tuple of ``(RASModel | None, enabled)``.
        """
        tp_path = self.case_path / "constant" / "turbulenceProperties"
        if not tp_path.exists():
            return None, False

        try:
            from pyfoam.io.dictionary import parse_dict_file
            from pyfoam.turbulence.ras_model import RASModel, RASConfig

            tp = parse_dict_file(tp_path)
            sim_type = str(tp.get("simulationType", "laminar")).strip()

            if sim_type != "RAS":
                return None, False

            # Read RAS settings
            ras_dict = tp.get("RAS", {})
            if isinstance(ras_dict, dict):
                model_name = str(ras_dict.get("model", "kEpsilon")).strip()
                ras_enabled = str(ras_dict.get("enabled", "true")).strip().lower() == "true"
            else:
                model_name = "kEpsilon"
                ras_enabled = True

            if not ras_enabled:
                return None, False

            config = RASConfig(
                model_name=model_name,
                enabled=True,
                nu=self.nu,
            )
            ras = RASModel(self.mesh, self.U, self.phi, config)
            logger.info("Turbulence model: %s", model_name)
            return ras, True

        except Exception as e:
            logger.warning("Could not initialise turbulence model: %s", e)
            return None, False

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
            match = re.search(
                r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)",
                value,
            )
            if match:
                return (float(match.group(1)), float(match.group(2)), float(match.group(3)))

        return None

    # ------------------------------------------------------------------
    # SIMPLE solver construction
    # ------------------------------------------------------------------

    def _build_solver(self) -> SIMPLESolver:
        """Build a SIMPLESolver with settings from fvSolution."""
        config = SIMPLEConfig(
            n_correctors=1,
            p_solver=self.p_solver,
            U_solver=self.U_solver,
            p_tolerance=self.p_tolerance,
            U_tolerance=self.U_tolerance,
            p_max_iter=self.p_max_iter,
            U_max_iter=self.U_max_iter,
            n_non_orthogonal_correctors=self.n_non_orth_correctors,
            relaxation_factor_p=self.alpha_p,
            relaxation_factor_U=self.alpha_U,
            nu=self.nu,
        )
        return SIMPLESolver(self.mesh, config)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the simpleFoam solver.

        Executes the SIMPLE algorithm in a time-stepping loop until
        convergence or ``endTime`` is reached.  When a turbulence model
        is active, it is corrected each outer iteration and the effective
        viscosity ``ν + ν_t`` is passed to the SIMPLE solver.

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

        logger.info("Starting simpleFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  relaxation: alpha_U=%.2f, alpha_p=%.2f", self.alpha_U, self.alpha_p)

        # Build boundary conditions
        U_bc = self._build_boundary_conditions()

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Update turbulence model (if active)
            nu_field = self._update_turbulence()

            # Run one SIMPLE outer iteration
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
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

        # Write final fields at the next time after the last completed step
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("simpleFoam completed successfully (converged)")
            else:
                logger.warning("simpleFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    def _update_turbulence(self) -> torch.Tensor | None:
        """Update the turbulence model and return the effective viscosity.

        If turbulence is disabled, returns ``None`` (solver uses scalar ν).

        Returns:
            ``(n_cells,)`` effective viscosity ``ν + ν_t``, or ``None``.
        """
        if not self.turbulence_enabled or self.ras is None:
            return None

        # Update velocity and flux references in the turbulence model
        self.ras._model._U = self.U
        self.ras._model._phi = self.phi

        # Correct turbulence (solve transport equations)
        self.ras.correct()

        # Return effective viscosity: ν + ν_t
        nu_eff = self.ras.mu_eff()
        return nu_eff

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, and phi to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
