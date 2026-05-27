"""
incompressibleFluid — unified incompressible solver.

Implements the modern OpenFOAM unified incompressible solver that dynamically
selects the pressure-velocity coupling algorithm based on ``fvSolution``
settings:

- **SIMPLE** sub-dict present → steady-state SIMPLE algorithm
- **PISO** sub-dict present → transient PISO algorithm
- **PIMPLE** sub-dict present → transient PIMPLE algorithm (default)

This replaces the need to choose between simpleFoam, pisoFoam, and
pimpleFoam — a single solver adapts to the case configuration.

The solver reads:
- ``0/U``, ``0/p`` — initial/boundary conditions
- ``constant/polyMesh`` — mesh
- ``constant/transportProperties`` — kinematic viscosity (nu)
- ``constant/turbulenceProperties`` — turbulence model selection (optional)
- ``system/controlDict`` — endTime, deltaT, writeControl, writeInterval
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — algorithm selection and settings

Usage::

    from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

    solver = IncompressibleFluidFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.solvers.piso import PISOSolver, PISOConfig
from pyfoam.solvers.pimple import PIMPLESolver, PIMPLEConfig
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.turbulence.ras_model import RASModel, RASConfig

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["IncompressibleFluidFoam", "Algorithm"]

logger = logging.getLogger(__name__)


class Algorithm(Enum):
    """Pressure-velocity coupling algorithm selection."""

    SIMPLE = "SIMPLE"
    PISO = "PISO"
    PIMPLE = "PIMPLE"


class IncompressibleFluidFoam(SolverBase):
    """Unified incompressible solver with automatic algorithm selection.

    Reads an OpenFOAM case directory and determines the appropriate
    pressure-velocity coupling algorithm from ``system/fvSolution``:

    - ``SIMPLE`` sub-dict → :class:`SIMPLESolver` (steady-state)
    - ``PISO`` sub-dict → :class:`PISOSolver` (transient)
    - ``PIMPLE`` sub-dict → :class:`PIMPLESolver` (transient, default)

    Supports optional RANS turbulence modelling via
    ``constant/turbulenceProperties``.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.

    Attributes
    ----------
    algorithm : Algorithm
        Detected pressure-velocity coupling algorithm.
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

        # Detect algorithm from fvSolution
        self.algorithm = self._detect_algorithm()

        # Read algorithm-specific fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes (for logging)
        self._read_fv_schemes_settings()

        # Initialise fields
        self.U, self.p, self.phi = self._init_fields()

        # Store raw field data for writing
        self._U_data, self._p_data = self._init_field_data()

        # Old fields for transient algorithms
        if self.algorithm in (Algorithm.PISO, Algorithm.PIMPLE):
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

        # Turbulence model
        self.ras, self.turbulence_enabled = self._init_turbulence()

        logger.info(
            "IncompressibleFluidFoam ready: algorithm=%s, nu=%.6e, Re~%.0f",
            self.algorithm.value,
            self.nu,
            1.0 / max(self.nu, 1e-30),
        )
        if self.turbulence_enabled:
            logger.info("  Turbulence: %s", self.ras)

    # ------------------------------------------------------------------
    # Algorithm detection
    # ------------------------------------------------------------------

    def _detect_algorithm(self) -> Algorithm:
        """Detect the pressure-velocity algorithm from fvSolution.

        Checks for the presence of ``SIMPLE``, ``PISO``, and ``PIMPLE``
        sub-dicts.  Priority: PIMPLE > PISO > SIMPLE.
        Falls back to PIMPLE if none found.
        """
        fv = self.case.fvSolution

        # Check in reverse priority order so highest-priority wins
        has_simple = self._has_sub_dict(fv, "SIMPLE")
        has_piso = self._has_sub_dict(fv, "PISO")
        has_pimple = self._has_sub_dict(fv, "PIMPLE")

        if has_pimple:
            return Algorithm.PIMPLE
        if has_piso:
            return Algorithm.PISO
        if has_simple:
            return Algorithm.SIMPLE

        # Default: PIMPLE (matches OpenFOAM behaviour)
        logger.warning(
            "No SIMPLE/PISO/PIMPLE sub-dict found in fvSolution; "
            "defaulting to PIMPLE"
        )
        return Algorithm.PIMPLE

    @staticmethod
    def _has_sub_dict(fv: Any, name: str) -> bool:
        """Check if fvSolution contains a named sub-dictionary."""
        try:
            value = fv.get(name)
            # A sub-dict should be a dict-like object (not None, not scalar)
            if value is None:
                return False
            if isinstance(value, dict):
                return True
            # For non-dict types, check if it has keys (dict-like)
            if hasattr(value, "keys"):
                return True
            return False
        except (KeyError, AttributeError, TypeError):
            return False

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
        """Read algorithm-specific settings from fvSolution."""
        fv = self.case.fvSolution

        # Pressure solver settings (shared across all algorithms)
        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_rel_tol = float(fv.get_path("solvers/p/relTol", 0.01))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        # Velocity solver settings (shared across all algorithms)
        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_rel_tol = float(fv.get_path("solvers/U/relTol", 0.01))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        # Algorithm-specific settings
        section = self.algorithm.value

        self.n_non_orth_correctors = int(
            fv.get_path(f"{section}/nNonOrthogonalCorrectors", 0)
        )

        # SIMPLE and PIMPLE use under-relaxation
        if self.algorithm in (Algorithm.SIMPLE, Algorithm.PIMPLE):
            self.alpha_p = float(fv.get_path(f"{section}/relaxationFactors/p", 0.3))
            self.alpha_U = float(fv.get_path(f"{section}/relaxationFactors/U", 0.7))
        else:
            # PISO: no under-relaxation (transient)
            self.alpha_p = 1.0
            self.alpha_U = 1.0

        # PISO and PIMPLE: nCorrectors
        if self.algorithm in (Algorithm.PISO, Algorithm.PIMPLE):
            self.n_correctors = int(fv.get_path(f"{section}/nCorrectors", 2))
        else:
            self.n_correctors = 1

        # PIMPLE: nOuterCorrectors
        if self.algorithm == Algorithm.PIMPLE:
            self.n_outer_correctors = int(
                fv.get_path("PIMPLE/nOuterCorrectors", 3)
            )
        else:
            self.n_outer_correctors = 1

        # Convergence tolerance
        self.convergence_tolerance = float(
            fv.get_path(f"{section}/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path(f"{section}/maxOuterIterations", 100)
        )

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings (for logging)."""
        fs = self.case.fvSchemes
        self.grad_scheme = str(fs.get_path("gradSchemes/default", "Gauss linear"))
        self.div_scheme = str(fs.get_path("divSchemes/default", "Gauss linear"))
        self.lap_scheme = str(
            fs.get_path("laplacianSchemes/default", "Gauss linear corrected")
        )

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

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

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

        Returns:
            Tuple of ``(RASModel | None, enabled)``.
        """
        tp_path = self.case_path / "constant" / "turbulenceProperties"
        if not tp_path.exists():
            return None, False

        try:
            from pyfoam.io.dictionary import parse_dict_file

            tp = parse_dict_file(tp_path)
            sim_type = str(tp.get("simulationType", "laminar")).strip()

            if sim_type != "RAS":
                return None, False

            ras_dict = tp.get("RAS", {})
            if isinstance(ras_dict, dict):
                model_name = str(ras_dict.get("model", "kEpsilon")).strip()
                ras_enabled = (
                    str(ras_dict.get("enabled", "true")).strip().lower() == "true"
                )
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

    def _update_turbulence(self) -> torch.Tensor | None:
        """Update the turbulence model and return effective viscosity.

        Returns:
            ``(n_cells,)`` effective viscosity ``nu + nu_t``, or ``None``.
        """
        if not self.turbulence_enabled or self.ras is None:
            return None

        self.ras._model._U = self.U
        self.ras._model._phi = self.phi
        self.ras.correct()
        return self.ras.mu_eff()

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _build_boundary_conditions(self) -> torch.Tensor | None:
        """Build the U_bc tensor from the 0/U boundary field.

        Returns:
            ``(n_cells, 3)`` — prescribed velocity (NaN where no BC).
            Returns ``None`` if no fixed-value velocity BCs are found.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        U_bc = torch.full((n_cells, 3), float("nan"), dtype=dtype, device=device)

        U_field_data = self._U_data
        boundary_field = U_field_data.boundary_field

        if boundary_field is None or len(boundary_field) == 0:
            return None

        mesh_boundary = self.case.boundary
        owner = self.mesh.owner

        mesh_patches = {}
        for bp in mesh_boundary:
            mesh_patches[bp.name] = {
                "startFace": bp.start_face,
                "nFaces": bp.n_faces,
            }

        has_bc = False

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
                        has_bc = True

        return U_bc if has_bc else None

    @staticmethod
    def _parse_vector_value(value: Any) -> tuple[float, float, float] | None:
        """Parse a vector value from field data.

        Handles both tuple/list and string formats like
        ``'uniform ( 1 0 0 )'``.
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

    # ------------------------------------------------------------------
    # Solver construction
    # ------------------------------------------------------------------

    def _build_simple_solver(self) -> SIMPLESolver:
        """Build a SIMPLESolver with settings from fvSolution."""
        config = SIMPLEConfig(
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
            nu=self.nu,
        )
        return SIMPLESolver(self.mesh, config)

    def _build_piso_solver(self) -> PISOSolver:
        """Build a PISOSolver with settings from fvSolution."""
        config = PISOConfig(
            n_correctors=self.n_correctors,
            p_solver=self.p_solver,
            U_solver=self.U_solver,
            p_tolerance=self.p_tolerance,
            U_tolerance=self.U_tolerance,
            p_max_iter=self.p_max_iter,
            U_max_iter=self.U_max_iter,
            n_non_orthogonal_correctors=self.n_non_orth_correctors,
            relaxation_factor_U=1.0,
            relaxation_factor_p=1.0,
        )
        return PISOSolver(self.mesh, config)

    def _build_pimple_solver(self) -> PIMPLESolver:
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
        """Run the incompressibleFluid solver.

        Dispatches to the appropriate algorithm based on fvSolution
        settings detected during initialisation.

        Returns:
            Final :class:`ConvergenceData`.
        """
        dispatch = {
            Algorithm.SIMPLE: self._run_simple,
            Algorithm.PISO: self._run_piso,
            Algorithm.PIMPLE: self._run_pimple,
        }

        runner = dispatch[self.algorithm]
        logger.info(
            "Starting incompressibleFluid run (algorithm=%s)",
            self.algorithm.value,
        )
        return runner()

    def _run_simple(self) -> ConvergenceData:
        """Execute the SIMPLE algorithm."""
        solver = self._build_simple_solver()

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

        logger.info(
            "  endTime=%.6g, deltaT=%.6g, relaxation: alpha_U=%.2f, alpha_p=%.2f",
            self.end_time, self.delta_t, self.alpha_U, self.alpha_p,
        )

        U_bc = self._build_boundary_conditions()
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            nu_field = self._update_turbulence()

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

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
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None and last_convergence.converged:
            logger.info("incompressibleFluid (SIMPLE) completed successfully")
        elif last_convergence is not None:
            logger.warning("incompressibleFluid (SIMPLE) completed without full convergence")

        return last_convergence or ConvergenceData()

    def _run_piso(self) -> ConvergenceData:
        """Execute the PISO algorithm."""
        solver = self._build_piso_solver()

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

        logger.info(
            "  endTime=%.6g, deltaT=%.6g, nCorrectors=%d",
            self.end_time, self.delta_t, self.n_correctors,
        )

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            if self.turbulence_enabled and self.ras is not None:
                self.ras._model._U = self.U
                self.ras._model._phi = self.phi
                self.ras.correct()

            U_bc = self._build_boundary_conditions()

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                U_old=self.U_old,
                p_old=self.p_old,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            residuals = {
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None and last_convergence.converged:
            logger.info("incompressibleFluid (PISO) completed successfully")
        elif last_convergence is not None:
            logger.warning("incompressibleFluid (PISO) completed without full convergence")

        return last_convergence or ConvergenceData()

    def _run_pimple(self) -> ConvergenceData:
        """Execute the PIMPLE algorithm."""
        solver = self._build_pimple_solver()

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

        logger.info(
            "  endTime=%.6g, deltaT=%.6g, nOuterCorrectors=%d, nCorrectors=%d",
            self.end_time, self.delta_t,
            self.n_outer_correctors, self.n_correctors,
        )
        logger.info(
            "  relaxation: alpha_U=%.2f, alpha_p=%.2f",
            self.alpha_U, self.alpha_p,
        )

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            if self.turbulence_enabled and self.ras is not None:
                self.ras._model._U = self.U
                self.ras._model._phi = self.phi
                self.ras.correct()

            U_bc = self._build_boundary_conditions()

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                U_old=self.U_old,
                p_old=self.p_old,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

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
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None and last_convergence.converged:
            logger.info("incompressibleFluid (PIMPLE) completed successfully")
        elif last_convergence is not None:
            logger.warning("incompressibleFluid (PIMPLE) completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, and phi to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
