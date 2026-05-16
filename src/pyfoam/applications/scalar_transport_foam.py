"""
scalarTransportFoam — passive scalar transport solver.

Implements the OpenFOAM scalarTransportFoam solver for passive scalar
transport with diffusion and convection:

    ∂C/∂t + ∇·(UC) = ∇·(D∇C)

where:
- C is the scalar concentration
- U is the velocity field (read from file, not solved)
- D is the diffusion coefficient

The solver reads:
- ``0/C`` — initial/boundary conditions for scalar
- ``0/U`` — velocity field (frozen, read from file)
- ``constant/polyMesh`` — mesh
- ``constant/transportProperties`` — diffusion coefficient (D)
- ``system/controlDict`` — endTime, deltaT, writeControl
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — linear solver tolerances

Usage::

    from pyfoam.applications.scalar_transport_foam import ScalarTransportFoam

    solver = ScalarTransportFoam("path/to/case")
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
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["ScalarTransportFoam"]

logger = logging.getLogger(__name__)


class ScalarTransportFoam(SolverBase):
    """Passive scalar transport solver.

    Solves ∂C/∂t + ∇·(UC) = ∇·(D∇C) with a frozen velocity field.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.

    Attributes
    ----------
    C : torch.Tensor
        ``(n_cells,)`` scalar concentration field.
    U : torch.Tensor
        ``(n_cells, 3)`` velocity field (frozen).
    D : float
        Diffusion coefficient.
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Read transport properties
        self.D = self._read_diffusion_coefficient()

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes (for logging)
        self._read_fv_schemes_settings()

        # Initialise fields
        self.C, self.U = self._init_fields()

        # Store raw field data for writing
        self._C_data, self._U_data = self._init_field_data()

        # Store old field for time derivative
        self.C_old = self.C.clone()

        logger.info("ScalarTransportFoam ready: D=%.6e", self.D)

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_diffusion_coefficient(self) -> float:
        """Read diffusion coefficient from transportProperties."""
        tp_path = self.case_path / "constant" / "transportProperties"
        if tp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                tp = parse_dict_file(tp_path)
                raw = tp.get("D", 1.0)
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
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.C_solver = str(fv.get_path("solvers/C/solver", "PCG"))
        self.C_tolerance = float(fv.get_path("solvers/C/tolerance", 1e-6))
        self.C_rel_tol = float(fv.get_path("solvers/C/relTol", 0.01))
        self.C_max_iter = int(fv.get_path("solvers/C/maxIter", 1000))

        self.convergence_tolerance = float(
            fv.get_path("scalarTransport/convergenceTolerance", 1e-4)
        )

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings (for logging)."""
        fs = self.case.fvSchemes
        self.ddt_scheme = str(fs.get_path("ddtSchemes/default", "Euler"))
        self.div_scheme = str(fs.get_path("divSchemes/default", "Gauss linear"))
        self.lap_scheme = str(fs.get_path("laplacianSchemes/default", "Gauss linear corrected"))

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialise C and U from the 0/ directory.

        Returns:
            Tuple of ``(C, U)`` tensors.
        """
        device = get_device()
        dtype = get_default_dtype()

        # Read scalar concentration
        C_tensor, _ = self.read_field_tensor("C", 0)
        C = C_tensor.to(device=device, dtype=dtype).squeeze()

        # Read velocity field
        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        return C, U

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        C_data = self.case.read_field("C", 0)
        U_data = self.case.read_field("U", 0)
        return C_data, U_data

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _build_C_boundary_conditions(self) -> torch.Tensor:
        """Build boundary conditions for scalar concentration.

        Returns:
            ``(n_cells,)`` — prescribed concentration (NaN where no BC).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        C_bc = torch.full((n_cells,), float('nan'), dtype=dtype, device=device)

        try:
            C_field_data = self.case.read_field("C", 0)
            boundary_field = C_field_data.boundary_field
        except Exception:
            return C_bc

        if boundary_field is None or len(boundary_field) == 0:
            return C_bc

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
                            C_bc[cell_idx] = value

        return C_bc

    @staticmethod
    def _parse_scalar_value(value: Any) -> float | None:
        """Parse a scalar value from field data."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"uniform\s+([\d.eE+\-]+)", value)
            if match:
                return float(match.group(1))
            try:
                return float(value)
            except ValueError:
                pass
        return None

    # ------------------------------------------------------------------
    # Transport equation assembly
    # ------------------------------------------------------------------

    def _assemble_transport_equation(
        self,
        C_old: torch.Tensor,
        dt: float,
    ) -> FvMatrix:
        """Assemble the scalar transport equation.

        ∂C/∂t + ∇·(UC) = ∇·(D∇C)

        Args:
            C_old: Old field values.
            dt: Time step.

        Returns:
            :class:`FvMatrix` for the transport equation.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Compute face flux: U · S
        n_faces = self.mesh.n_faces
        n_internal = self.mesh.n_internal_faces
        face_areas = self.mesh.face_areas
        owner = self.mesh.owner

        # Linearly interpolate velocity to faces
        U_face = torch.zeros(n_faces, 3, dtype=dtype, device=device)
        int_owner = owner[:n_internal]
        int_neigh = self.mesh.neighbour
        U_P = self.U[int_owner]
        U_N = self.U[int_neigh]
        U_face[:n_internal] = 0.5 * U_P + 0.5 * U_N

        if n_faces > n_internal:
            bnd_owner = owner[n_internal:]
            U_face[n_internal:] = self.U[bnd_owner]

        # Face flux = U_face · S_face
        phi = (U_face * face_areas).sum(dim=1)

        # Build combined matrix manually
        # Start with a fresh FvMatrix
        from pyfoam.core.fv_matrix import FvMatrix
        matrix = FvMatrix(
            n_cells,
            self.mesh.owner[:n_internal],
            self.mesh.neighbour,
            device=device, dtype=dtype,
        )

        # Time derivative: diag += V/dt, source += V*C_old/dt
        cell_volumes = self.mesh.cell_volumes
        matrix._diag = matrix._diag + cell_volumes / dt
        matrix._source = matrix._source + cell_volumes * C_old / dt

        # Convection: ∇·(φC)
        # Upwind scheme for stability
        flux = phi[:n_internal]
        is_positive = flux >= 0.0
        flux_pos = torch.where(is_positive, flux, torch.zeros_like(flux))
        flux_neg = torch.where(~is_positive, flux, torch.zeros_like(flux))

        V_P = cell_volumes[int_owner]
        V_N = cell_volumes[int_neigh]

        matrix._lower = matrix._lower + flux_neg / V_P
        matrix._upper = matrix._upper + flux_pos / V_N

        diag_conv = torch.zeros(n_cells, dtype=dtype, device=device)
        from pyfoam.core.backend import scatter_add
        diag_conv = diag_conv + scatter_add(-flux_pos / V_P, int_owner, n_cells)
        diag_conv = diag_conv + scatter_add(flux_neg.abs() / V_N, int_neigh, n_cells)
        matrix._diag = matrix._diag + diag_conv

        # Diffusion: ∇·(D∇C)
        delta_coeffs = self.mesh.delta_coefficients
        S_mag = face_areas[:n_internal].norm(dim=1)
        D_face = torch.full((n_faces,), float(self.D), dtype=dtype, device=device)
        face_coeff = D_face[:n_internal] * S_mag * delta_coeffs[:n_internal]

        matrix._lower = matrix._lower - face_coeff / V_P
        matrix._upper = matrix._upper - face_coeff / V_N

        diag_diff = torch.zeros(n_cells, dtype=dtype, device=device)
        diag_diff = diag_diff + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag_diff = diag_diff + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # Boundary diffusion
        if n_faces > n_internal:
            bnd_areas = face_areas[n_internal:]
            bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
            bnd_delta = delta_coeffs[n_internal:]
            bnd_D = D_face[n_internal:]
            bnd_coeff = bnd_D * bnd_S_mag * bnd_delta
            bnd_V = cell_volumes[owner[n_internal:]]
            diag_diff = diag_diff + scatter_add(bnd_coeff / bnd_V, owner[n_internal:], n_cells)

        matrix._diag = matrix._diag + diag_diff

        return matrix

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the scalarTransportFoam solver.

        Solves ∂C/∂t + ∇·(UC) = ∇·(D∇C) in a time-stepping loop.

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

        logger.info("Starting scalarTransportFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  D=%.6e", self.D)

        # Build boundary conditions
        C_bc = self._build_C_boundary_conditions()

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        # Create linear solver
        solver = create_solver(
            self.C_solver,
            tolerance=self.C_tolerance,
            rel_tol=self.C_rel_tol,
            max_iter=self.C_max_iter,
        )

        last_convergence = None

        for t, step in time_loop:
            # Store old field
            self.C_old = self.C.clone()

            # Assemble transport equation
            matrix = self._assemble_transport_equation(self.C_old, self.delta_t)

            # Apply boundary conditions
            bc_mask = ~torch.isnan(C_bc)
            if bc_mask.any():
                large_coeff = matrix.diag.abs().clamp(min=1.0) * 1e10
                matrix._diag[bc_mask] += large_coeff[bc_mask]
                matrix._source[bc_mask] += large_coeff[bc_mask] * C_bc[bc_mask]

            # Solve
            self.C, iters, residual = matrix.solve(
                solver, self.C.clone(),
                tolerance=self.C_tolerance,
                max_iter=self.C_max_iter,
            )

            # Track convergence
            residuals = {"C": residual}
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

        logger.info("scalarTransportFoam completed")
        logger.info("  C range: [%.6e, %.6e]",
                    self.C.min().item(), self.C.max().item())

        return {
            "converged": converged if converged else False,
            "iterations": iters,
            "residual": residual,
        }

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write C and U to a time directory."""
        time_str = f"{time:g}"
        self.write_field("C", self.C, time_str, self._C_data)
