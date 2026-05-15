"""
boundaryFoam — 1D turbulent boundary layer solver.

Implements a 1D steady-state boundary layer solver following OpenFOAM's
``boundaryFoam`` application.  Solves the 1D boundary layer momentum
equation with a prescribed pressure gradient and optional RANS turbulence
modelling.

The 1D momentum equation (in the y-direction normal to the wall) is::

    d/dy( ν_eff dU/dy ) = -dp/dx

where:

- ``U`` is the streamwise velocity component (function of ``y`` only)
- ``ν_eff = ν + ν_t`` is the effective (molecular + turbulent) viscosity
- ``dp/dx`` is the prescribed (constant) streamwise pressure gradient

Boundary conditions:

- **Wall** (``y = 0``): no-slip, ``U = 0``
- **Far-field** (``y = y_max``): freestream, ``U = U_inf``

The solver uses SIMPLE-like outer iterations to couple the momentum
equation with the turbulence model (when enabled).

Usage::

    from pyfoam.applications.boundary_foam import BoundaryFoam

    solver = BoundaryFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.core.backend import scatter_add, gather
from pyfoam.solvers.linear_solver import create_solver

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BoundaryFoam"]

logger = logging.getLogger(__name__)


class BoundaryFoam(SolverBase):
    """1D steady-state turbulent boundary layer solver.

    Reads an OpenFOAM case directory with a 1D mesh (single column of
    cells in the y/wall-normal direction) and solves the boundary layer
    momentum equation with optional RANS turbulence modelling.

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
    dp_dx : float
        Prescribed streamwise pressure gradient.
    U_inf : float
        Freestream (far-field) velocity.
    turbulence_enabled : bool
        Whether RANS turbulence modelling is active.
    ras : RASModel or None
        The RAS turbulence model wrapper.
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Read transport properties
        self.nu = self._read_nu()

        # Read boundaryFoam-specific settings
        self._read_boundary_foam_settings()

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes
        self._read_fv_schemes_settings()

        # Create linear solver object
        self._linear_solver = create_solver(
            self.U_solver,
            tolerance=self.U_tolerance,
            max_iter=self.U_max_iter,
        )

        # Initialise fields
        self.U, self.p, self.phi = self._init_fields()

        # Store raw field data for writing
        self._U_data, self._p_data = self._init_field_data()

        # Turbulence model (optional)
        self.ras, self.turbulence_enabled = self._init_turbulence()

        logger.info(
            "BoundaryFoam ready: nu=%.6e, dp/dx=%.6e, U_inf=%.6f",
            self.nu, self.dp_dx, self.U_inf,
        )
        if self.turbulence_enabled:
            logger.info("  Turbulence: %s", self.ras)

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_nu(self) -> float:
        """Read kinematic viscosity from transportProperties."""
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

    def _read_boundary_foam_settings(self) -> None:
        """Read boundaryFoam-specific settings from fvSolution."""
        fv = self.case.fvSolution

        # Prescribed pressure gradient (source term for momentum)
        self.dp_dx = float(fv.get_path("boundaryFoam/dpdx", 0.0))

        # Freestream velocity (far-field boundary value)
        self.U_inf = float(fv.get_path("boundaryFoam/UInf", 1.0))

        # Relaxation factors for SIMPLE-like iteration
        self.alpha_U = float(
            fv.get_path("boundaryFoam/relaxationFactors/U", 0.7)
        )

        # Convergence settings
        self.convergence_tolerance = float(
            fv.get_path("boundaryFoam/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("boundaryFoam/maxOuterIterations", 200)
        )

    def _read_fv_solution_settings(self) -> None:
        """Read linear solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_rel_tol = float(fv.get_path("solvers/U/relTol", 0.01))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings."""
        fs = self.case.fvSchemes
        self.grad_scheme = str(
            fs.get_path("gradSchemes/default", "Gauss linear")
        )
        self.lap_scheme = str(
            fs.get_path("laplacianSchemes/default", "Gauss linear corrected")
        )

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise U, p, phi from the 0/ directory."""
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
        """Initialise RAS turbulence model from turbulenceProperties."""
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

            ras_dict = tp.get("RAS", {})
            if isinstance(ras_dict, dict):
                model_name = str(ras_dict.get("model", "kEpsilon")).strip()
                ras_enabled = (
                    str(ras_dict.get("enabled", "true")).strip().lower()
                    == "true"
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

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _build_boundary_conditions(self) -> torch.Tensor:
        """Build velocity BC tensor for the 1D boundary layer.

        For the 1D boundary layer:
        - Wall (y=0): U = (0, 0, 0) — no-slip
        - Far-field (y=y_max): U = (U_inf, 0, 0) — freestream

        Returns:
            ``(n_cells, 3)`` — prescribed velocity (NaN where no BC).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        U_bc = torch.full(
            (n_cells, 3), float("nan"), dtype=dtype, device=device
        )

        # Read boundary field from 0/U
        U_field_data = self.case.read_field("U", 0)
        boundary_field = U_field_data.boundary_field

        if boundary_field is None or len(boundary_field) == 0:
            return U_bc

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
                return (
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                )

        return None

    # ------------------------------------------------------------------
    # 1D Momentum equation
    # ------------------------------------------------------------------

    def _build_1d_momentum_matrix(
        self,
        U: torch.Tensor,
        nu_field: torch.Tensor | None = None,
    ) -> tuple[FvMatrix, torch.Tensor]:
        """Assemble the 1D momentum equation matrix.

        For the 1D boundary layer, the momentum equation is::

            d/dy( ν_eff dU/dy ) = -dp/dx

        This is discretised as a diffusion equation with a prescribed
        pressure gradient source term.

        Args:
            U: ``(n_cells, 3)`` — current velocity.
            nu_field: ``(n_cells,)`` — effective viscosity field.
                If ``None``, uses scalar ``self.nu``.

        Returns:
            Tuple of ``(matrix, source)`` where matrix is an
            :class:`FvMatrix` and source is ``(n_cells, 3)``.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        mat = FvMatrix(
            n_cells, owner[:n_internal], neighbour,
            device=device, dtype=dtype,
        )

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        # Diffusion coefficient
        S_mag = mesh.face_areas[:n_internal].norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]

        if nu_field is not None:
            nu_field = nu_field.to(device=device, dtype=dtype)
            nu_face = 0.5 * (
                gather(nu_field, int_owner) + gather(nu_field, int_neigh)
            )
            diff_coeff = nu_face * S_mag * delta_f
        else:
            diff_coeff = self.nu * S_mag * delta_f

        V_P = gather(mesh.cell_volumes, int_owner)
        V_N = gather(mesh.cell_volumes, int_neigh)

        # Off-diagonal coefficients (diffusion only, no convection in 1D BL)
        mat.lower = -diff_coeff / V_P
        mat.upper = -diff_coeff / V_N

        # Diagonal: sum of absolute off-diagonal
        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(diff_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(diff_coeff / V_N, int_neigh, n_cells)

        # Boundary condition enforcement (implicit BC method)
        bc_mask_wall = torch.zeros(n_cells, dtype=torch.bool, device=device)
        bc_mask_farfield = torch.zeros(n_cells, dtype=torch.bool, device=device)

        if n_cells > 0:
            # Wall cell (first cell, owner of first boundary face)
            if mesh.n_faces > n_internal:
                wall_face = n_internal  # first boundary face
                wall_cell = owner[wall_face].item()
                bc_mask_wall[wall_cell] = True

                # Far-field cell (last cell, owner of last boundary face)
                farfield_face = mesh.n_faces - 1
                farfield_cell = owner[farfield_face].item()
                bc_mask_farfield[farfield_cell] = True

                # Boundary face diffusion coefficients
                for bnd_face_idx in [n_internal, mesh.n_faces - 1]:
                    bnd_cell = owner[bnd_face_idx].item()
                    bnd_area = mesh.face_areas[bnd_face_idx].norm()
                    bnd_centre = mesh.face_centres[bnd_face_idx]
                    cell_centre = mesh.cell_centres[bnd_cell]
                    d_P = (bnd_centre - cell_centre).norm()
                    bnd_delta = 1.0 / d_P.clamp(min=1e-30)

                    if nu_field is not None:
                        bnd_nu = nu_field[bnd_cell]
                    else:
                        bnd_nu = self.nu

                    bnd_coeff = bnd_nu * bnd_area * bnd_delta
                    bnd_V = mesh.cell_volumes[bnd_cell]

                    diag[bnd_cell] = diag[bnd_cell] + bnd_coeff / bnd_V

        mat.diag = diag.clone()

        # Source term: prescribed pressure gradient
        # dp/dx acts as a body force driving the flow
        source = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        source[:, 0] = -self.dp_dx * mesh.cell_volumes  # Force in x-direction

        return mat, source

    def _enforce_boundary_conditions(
        self,
        U: torch.Tensor,
        U_bc: torch.Tensor,
        source: torch.Tensor,
        mat: FvMatrix,
        nu_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add boundary condition contributions to the matrix and source.

        For fixed-value BCs:
        - Add face_coeff to diagonal (internalCoeffs)
        - Add face_coeff * U_bc to source (boundaryCoeffs)

        Args:
            U: ``(n_cells, 3)`` — current velocity.
            U_bc: ``(n_cells, 3)`` — prescribed velocity (NaN where no BC).
            source: ``(n_cells, 3)`` — current source term.
            mat: The FvMatrix to modify.
            nu_field: ``(n_cells,)`` — effective viscosity field.

        Returns:
            Updated source ``(n_cells, 3)``.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_internal = mesh.n_internal_faces

        bc_mask = ~torch.isnan(U_bc[:, 0])
        if not bc_mask.any() or mesh.n_faces <= n_internal:
            return source

        owner = mesh.owner

        for bnd_face_idx in range(n_internal, mesh.n_faces):
            bnd_cell = owner[bnd_face_idx].item()
            if not bc_mask[bnd_cell]:
                continue

            bnd_area = mesh.face_areas[bnd_face_idx].norm()
            bnd_centre = mesh.face_centres[bnd_face_idx]
            cell_centre = mesh.cell_centres[bnd_cell]
            d_P = (bnd_centre - cell_centre).norm()
            bnd_delta = 1.0 / d_P.clamp(min=1e-30)

            if nu_field is not None:
                bnd_nu = nu_field[bnd_cell]
            else:
                bnd_nu = self.nu

            bnd_coeff = bnd_nu * bnd_area * bnd_delta
            bnd_V = mesh.cell_volumes[bnd_cell]
            coeff_pv = bnd_coeff / bnd_V

            # Add to source: boundaryCoeffs = face_coeff * U_bc / V
            for comp in range(3):
                source[bnd_cell, comp] = (
                    source[bnd_cell, comp]
                    + coeff_pv * U_bc[bnd_cell, comp]
                )

        return source

    # ------------------------------------------------------------------
    # Under-relaxation
    # ------------------------------------------------------------------

    def _apply_under_relaxation(
        self,
        mat: FvMatrix,
        source: torch.Tensor,
        U: torch.Tensor,
        alpha: float,
    ) -> tuple[FvMatrix, torch.Tensor, torch.Tensor]:
        """Apply implicit under-relaxation to the momentum equation.

        Following OpenFOAM's approach:
        1. Compute sum of off-diagonal magnitudes
        2. Ensure diagonal dominance: D_dominant = max(|D|, Σ|off-diag|)
        3. Relax: D_new = D_dominant / alpha
        4. Add relaxation source: source += (D_new - D_old) * U_old

        Args:
            mat: The FvMatrix.
            source: ``(n_cells, 3)`` — current source.
            U: ``(n_cells, 3)`` — current velocity.
            alpha: Under-relaxation factor.

        Returns:
            Tuple of ``(modified_mat, modified_source, A_p_eff)``.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        diag_old = mat.diag.clone()

        # Sum of off-diagonal magnitudes
        sum_off = torch.zeros(n_cells, dtype=dtype, device=device)
        sum_off = sum_off + scatter_add(
            mat.lower.abs(), mesh.owner[:n_internal], n_cells
        )
        sum_off = sum_off + scatter_add(
            mat.upper.abs(), mesh.neighbour, n_cells
        )

        # Ensure diagonal dominance
        D_dominant = torch.max(diag_old.abs(), sum_off)

        # Apply relaxation
        D_new = D_dominant / alpha
        mat.diag = D_new

        # Add relaxation contribution to source
        source = source + (D_new - diag_old).unsqueeze(-1) * U

        A_p_eff = D_new.clone()

        return mat, source, A_p_eff

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the boundaryFoam solver.

        Executes SIMPLE-like outer iterations coupling the 1D momentum
        equation with the turbulence model until convergence.

        Returns:
            Dict with convergence information:
            - ``converged``: bool
            - ``iterations``: int
            - ``U_residual``: float
            - ``continuity_error``: float
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting boundaryFoam run")
        logger.info("  maxOuterIterations=%d", self.max_outer_iterations)
        logger.info("  convergenceTolerance=%.6e", self.convergence_tolerance)
        logger.info("  dp/dx=%.6e, U_inf=%.6f", self.dp_dx, self.U_inf)
        logger.info("  relaxation: alpha_U=%.2f", self.alpha_U)

        # Build boundary conditions
        U_bc = self._build_boundary_conditions()

        # Write initial fields
        self._write_fields(0.0)

        converged = False
        last_iteration = 0

        for outer in range(self.max_outer_iterations):
            U_prev = self.U.clone()

            # ---- Update turbulence model ----
            nu_field = self._update_turbulence()

            # ---- Assemble 1D momentum equation ----
            mat, source = self._build_1d_momentum_matrix(
                self.U, nu_field=nu_field
            )

            # ---- Apply boundary conditions ----
            source = self._enforce_boundary_conditions(
                self.U, U_bc, source, mat, nu_field=nu_field
            )

            # ---- Apply under-relaxation ----
            mat, source, A_p_eff = self._apply_under_relaxation(
                mat, source, self.U, self.alpha_U
            )
            mat.source = source

            # ---- Solve momentum equation ----
            U_new = torch.zeros_like(self.U)
            for comp in range(3):
                mat.source = source[:, comp]
                U_comp, _, _ = mat.solve(
                    self._linear_solver,
                    self.U[:, comp],
                    tolerance=self.U_tolerance,
                    max_iter=self.U_max_iter,
                )
                U_new[:, comp] = U_comp

            # ---- Re-apply boundary conditions ----
            bc_mask = ~torch.isnan(U_bc[:, 0])
            if bc_mask.any():
                U_new[bc_mask] = U_bc[bc_mask]

            self.U = U_new

            # ---- Compute velocity residual ----
            U_diff = self.U - U_prev
            U_norm = self.U.abs().max().clamp(min=1e-30)
            U_residual = float((U_diff.abs() / U_norm).max().item())

            # ---- Compute continuity error ----
            continuity_error = self._compute_continuity_error(self.U)

            last_iteration = outer + 1

            # ---- Check convergence ----
            residuals = {
                "U": U_residual,
                "cont": continuity_error,
            }
            converged = convergence.update(outer + 1, residuals)

            if converged:
                logger.info(
                    "boundaryFoam converged at iteration %d "
                    "(U_res=%.6e, cont=%.6e)",
                    outer + 1, U_residual, continuity_error,
                )
                break

            # NaN detection
            if torch.isnan(self.U).any():
                logger.error(
                    "boundaryFoam diverged at iteration %d (NaN detected)",
                    outer + 1,
                )
                break

        if not converged:
            logger.warning(
                "boundaryFoam did not converge in %d iterations",
                self.max_outer_iterations,
            )

        # Write final fields
        self._write_fields(float(last_iteration))

        return {
            "converged": converged,
            "iterations": last_iteration,
            "U_residual": convergence.history[-1].residuals.get("U", 0.0)
            if convergence.history
            else 0.0,
            "continuity_error": convergence.history[-1].residuals.get(
                "cont", 0.0
            )
            if convergence.history
            else 0.0,
            "convergence_history": convergence.history,
        }

    def _update_turbulence(self) -> torch.Tensor | None:
        """Update turbulence model and return effective viscosity field."""
        if not self.turbulence_enabled or self.ras is None:
            return None

        # Update velocity and flux references
        self.ras._model._U = self.U
        self.ras._model._phi = self.phi

        # Correct turbulence
        self.ras.correct()

        # Return effective viscosity
        return self.ras.mu_eff()

    def _compute_continuity_error(self, U: torch.Tensor) -> float:
        """Compute the global continuity/momentum residual error.

        For 1D boundary layer, this measures how well the diffusion
        equation is satisfied: |d/dy(ν_eff dU/dy) + dp/dx| / V

        Args:
            U: ``(n_cells, 3)`` velocity field.

        Returns:
            Scalar continuity error.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        # Compute flux through internal faces
        diff_flux = torch.zeros(
            mesh.n_faces, dtype=dtype, device=device
        )

        w = mesh.face_weights[:n_internal]
        int_owner = owner[:n_internal]
        int_neigh = neighbour

        # Velocity gradient at faces (x-component only)
        U_x = U[:, 0]
        U_own = gather(U_x, int_owner)
        U_neigh = gather(U_x, int_neigh)
        delta_f = mesh.delta_coefficients[:n_internal]
        S_mag = mesh.face_areas[:n_internal].norm(dim=1)

        # Viscous flux: ν * S * dU/dy
        diff_flux[:n_internal] = self.nu * S_mag * delta_f * (U_neigh - U_own)

        # Sum flux per cell (divergence)
        div_phi = torch.zeros(n_cells, dtype=dtype, device=device)
        div_phi = div_phi + scatter_add(
            diff_flux[:n_internal], int_owner, n_cells
        )
        div_phi = div_phi + scatter_add(
            -diff_flux[:n_internal], int_neigh, n_cells
        )

        # Add boundary fluxes
        if mesh.n_faces > n_internal:
            bnd_owner = owner[n_internal:]
            for i in range(n_internal, mesh.n_faces):
                bnd_cell = owner[i].item()
                bnd_area = mesh.face_areas[i].norm()
                bnd_centre = mesh.face_centres[i]
                cell_centre = mesh.cell_centres[bnd_cell]
                d_P = (bnd_centre - cell_centre).norm()
                bnd_delta = 1.0 / d_P.clamp(min=1e-30)

                # Boundary velocity (from BC)
                bc_val = U[bnd_cell, 0].item()
                U_cell = U[bnd_cell, 0]

                visc_flux = self.nu * bnd_area * bnd_delta * (bc_val - U_cell)
                div_phi[bnd_cell] = div_phi[bnd_cell] + visc_flux

        # Add pressure gradient source
        div_phi = div_phi + self.dp_dx * mesh.cell_volumes

        # Normalise by cell volume
        V = mesh.cell_volumes.clamp(min=1e-30)
        div_phi = div_phi / V

        return float(div_phi.abs().mean().item())

    # ------------------------------------------------------------------
    # Velocity profile extraction
    # ------------------------------------------------------------------

    def get_velocity_profile(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the velocity profile (U_x vs y).

        Returns:
            Tuple of ``(y_coords, U_x)`` where:
            - ``y_coords``: ``(n_cells,)`` cell-centre y-coordinates
            - ``U_x``: ``(n_cells,)`` streamwise velocity component
        """
        y_coords = self.mesh.cell_centres[:, 1]
        U_x = self.U[:, 0]
        return y_coords.clone(), U_x.clone()

    def get_turbulent_viscosity_profile(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the turbulent viscosity profile (ν_t vs y).

        Returns:
            Tuple of ``(y_coords, nu_t)`` where:
            - ``y_coords``: ``(n_cells,)`` cell-centre y-coordinates
            - ``nu_t``: ``(n_cells,)`` turbulent viscosity
        """
        y_coords = self.mesh.cell_centres[:, 1]
        if self.turbulence_enabled and self.ras is not None:
            nu_t = self.ras.nut()
        else:
            nu_t = torch.zeros(
                self.mesh.n_cells,
                dtype=self.U.dtype,
                device=self.U.device,
            )
        return y_coords.clone(), nu_t.clone()

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, and phi to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
