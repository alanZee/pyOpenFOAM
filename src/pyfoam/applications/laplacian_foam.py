"""
laplacianFoam — transient scalar diffusion solver.

Solves the diffusion equation for a scalar field T:

    ∂T/∂t = ∇·(D∇T)

where D is the diffusion coefficient (can be a scalar constant or a
spatially-varying field).

This is the simplest OpenFOAM solver — no convection, no pressure-
velocity coupling, no turbulence.  It demonstrates the basic time-
stepping loop with the implicit Laplacian operator.

Algorithm (per time step):
1. Assemble: [1/Δt + ∇·(D∇)] T^{n+1} = T^n / Δt + BC sources
2. Solve the linear system for T^{n+1}
3. Write fields at output intervals

Usage::

    from pyfoam.applications.laplacian_foam import LaplacianFoam

    solver = LaplacianFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["LaplacianFoam"]

logger = logging.getLogger(__name__)


class LaplacianFoam(SolverBase):
    """Transient scalar diffusion solver.

    Solves ∂T/∂t = ∇·(D∇T) using Euler implicit time-stepping and
    the finite volume method.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    D : float, optional
        Diffusion coefficient.  If None, reads from ``constant/transportProperties``.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        D: float | None = None,
    ) -> None:
        super().__init__(case_path)

        # Read settings
        self._read_fv_solution_settings()
        self._read_fv_schemes_settings()

        # Diffusion coefficient
        self.D = D if D is not None else self._read_diffusion_coeff()

        # Initialise scalar field T
        self.T, self._T_data = self._init_field()

        # Parse boundary conditions
        self._bc_values = self._parse_boundary_conditions()

        logger.info("LaplacianFoam ready: D=%.6g", self.D)

    # ------------------------------------------------------------------
    # Settings reading
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.T_solver = str(fv.get_path("solvers/T/solver", "PCG"))
        self.T_tolerance = float(fv.get_path("solvers/T/tolerance", 1e-6))
        self.T_rel_tol = float(fv.get_path("solvers/T/relTol", 0.01))
        self.T_max_iter = int(fv.get_path("solvers/T/maxIter", 1000))

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings."""
        fs = self.case.fvSchemes
        self.lap_scheme = str(
            fs.get_path("laplacianSchemes/default", "Gauss linear corrected")
        )

    def _read_diffusion_coeff(self) -> float:
        """Read diffusion coefficient from constant/transportProperties.

        Looks for ``DT`` or ``D`` in the file.  Defaults to 1.0 if not found.
        """
        tp_path = self.case_path / "constant" / "transportProperties"
        if not tp_path.exists():
            logger.warning(
                "constant/transportProperties not found, using D=1.0"
            )
            return 1.0

        try:
            from pyfoam.io.dictionary import parse_dict_file
            tp = parse_dict_file(tp_path)
            # Try DT first (OpenFOAM convention for thermal diffusivity)
            D = tp.get("DT", tp.get("D", 1.0))
            if isinstance(D, dict):
                # Handle dimensionedScalar format: { value 0.01; }
                D = D.get("value", 1.0)
            return float(D)
        except Exception as e:
            logger.warning(
                "Could not parse transportProperties: %s, using D=1.0", e
            )
            return 1.0

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_field(self) -> tuple[torch.Tensor, Any]:
        """Initialise T from the 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()

        T_tensor, T_data = self.read_field_tensor("T", 0)
        T = T_tensor.to(device=device, dtype=dtype)

        return T, T_data

    # ------------------------------------------------------------------
    # Boundary condition parsing
    # ------------------------------------------------------------------

    def _parse_boundary_conditions(self) -> dict[str, dict[str, Any]]:
        """Parse boundary conditions from the field data.

        Returns a dict mapping patch name to its BC info:
            {patch_name: {"type": "fixedValue", "value": 300.0, ...}}

        For fixedValue patches, stores the uniform value.
        For zeroGradient patches, stores None (no source contribution).
        """
        bc_values = {}
        boundary = self._T_data.boundary_field
        mesh_boundary = self.case.boundary

        for i, patch in enumerate(boundary.patches):
            bc_info = {"type": patch.patch_type, "value": None}

            if patch.patch_type == "fixedValue" and patch.value is not None:
                # Handle "uniform X" format (string)
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

            # Get face indices from mesh boundary (not from field data)
            if i < len(mesh_boundary):
                bp = mesh_boundary[i]
                bc_info["start_face"] = bp.start_face
                bc_info["n_faces"] = bp.n_faces

            bc_values[patch.name] = bc_info

        return bc_values

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the laplacianFoam solver.

        Returns:
            Final :class:`ConvergenceData`.
        """
        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=1e-10,  # Diffusion always converges
            min_steps=1,
        )

        logger.info("Starting laplacianFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  D=%.6g", self.D)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            T_prev = self.T.clone()

            # Solve: (1/Δt) T^{n+1} - ∇·(D∇T^{n+1}) = (1/Δt) T^n + BC
            self.T = self._solve_timestep(self.T, T_prev)

            # Compute residual
            T_residual = self._compute_residual(self.T, T_prev)
            conv = ConvergenceData()
            conv.T_residual = T_residual
            conv.converged = T_residual < 1e-10
            last_convergence = conv

            residuals = {"T": T_residual}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            logger.info("laplacianFoam completed: T_res=%.6e",
                        last_convergence.T_residual)

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # Time-step solve
    # ------------------------------------------------------------------

    def _solve_timestep(
        self,
        T: torch.Tensor,
        T_old: torch.Tensor,
    ) -> torch.Tensor:
        """Solve one time step: (1/Δt + ∇·(D∇)) T = T_old / Δt + BC sources.

        Uses Euler implicit time discretisation with Jacobi iteration
        for the linear system solve.

        Boundary conditions are applied as follows:
        - fixedValue: adds D * |S_f| * δ_f * T_boundary / V to source
        - zeroGradient: no additional source (diagonal contribution only)

        Parameters
        ----------
        T : torch.Tensor
            Current temperature field (initial guess).
        T_old : torch.Tensor
            Temperature from previous time step.

        Returns
        -------
        torch.Tensor
            Updated temperature field.
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
        face_coeff = self.D * S_mag * delta_f

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        # Off-diagonal coefficients
        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        # Diagonal from Laplacian (internal faces)
        lap_diag = torch.zeros(n_cells, dtype=dtype, device=device)
        lap_diag = lap_diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        lap_diag = lap_diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # ---- Boundary contributions ----
        # For each boundary face:
        #   - Add D * |S_f| * δ_f to diagonal (already done for all BCs)
        #   - For fixedValue: add D * |S_f| * δ_f * T_bc to source
        bc_source = torch.zeros(n_cells, dtype=dtype, device=device)

        for patch_name, bc_info in self._bc_values.items():
            start_face = bc_info.get("start_face", 0)
            n_faces = bc_info.get("n_faces", 0)

            if n_faces == 0:
                continue

            # Skip empty and wedge patches (2D/axi-symmetric placeholders)
            if bc_info["type"] in ("empty", "wedge"):
                continue

            # Skip zeroGradient patches (no flux = no contribution)
            if bc_info["type"] == "zeroGradient":
                continue

            bnd_faces = slice(start_face, start_face + n_faces)
            bnd_areas = mesh.face_areas[bnd_faces]
            bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
            bnd_cells = mesh.owner[bnd_faces]
            bnd_V = gather(cell_volumes, bnd_cells)

            # Compute boundary delta coefficient:
            # delta = 1 / |d_P · n_f| where d_P = face_centre - cell_centre
            bnd_face_centres = mesh.face_centres[bnd_faces]
            bnd_cell_centres = mesh.cell_centres[bnd_cells]
            d_P = bnd_face_centres - bnd_cell_centres  # (n_bnd, 3)
            safe_area = torch.where(
                bnd_S_mag.unsqueeze(-1) > 1e-30,
                bnd_areas,
                torch.ones_like(bnd_areas),
            )
            n_f = safe_area / safe_area.norm(dim=1, keepdim=True)  # unit normal
            d_dot_n = (d_P * n_f).sum(dim=1).abs()
            bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)

            bnd_coeff = self.D * bnd_S_mag * bnd_delta

            # Add to diagonal (for all BC types)
            lap_diag = lap_diag + scatter_add(bnd_coeff / bnd_V, bnd_cells, n_cells)

            # For fixedValue, add source contribution
            if bc_info["type"] == "fixedValue" and bc_info["value"] is not None:
                T_bc = bc_info["value"]
                bc_source = bc_source + scatter_add(
                    bnd_coeff * T_bc / bnd_V, bnd_cells, n_cells
                )

        # ---- Add time derivative: 1/Δt on diagonal ----
        dt = self.delta_t
        dt_inv = 1.0 / dt
        V = cell_volumes
        diag = lap_diag + dt_inv * V

        # ---- Source: T_old / Δt * V + BC sources ----
        source = dt_inv * V * T_old + bc_source

        # ---- Solve using Jacobi iteration ----
        diag_safe = diag.abs().clamp(min=1e-30)

        for _ in range(self.T_max_iter):
            # Off-diagonal contributions (internal faces only)
            off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
            T_P = gather(T, int_owner)
            T_N = gather(T, int_neigh)
            off_diag = off_diag + scatter_add(lower * T_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * T_P, int_neigh, n_cells)

            T_new = (source - off_diag) / diag_safe

            # Check convergence
            if (T_new - T).abs().max() < self.T_tolerance:
                break
            T = T_new

        return T

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_residual(
        field: torch.Tensor,
        field_old: torch.Tensor,
    ) -> float:
        """Compute L2 residual normalised by field magnitude."""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write T to a time directory."""
        time_str = f"{time:g}"
        self.write_field("T", self.T, time_str, self._T_data)
