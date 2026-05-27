"""
adjointFoam — adjoint shape optimization solver.

Implements the continuous adjoint method for gradient-based shape
optimization.  Solves the adjoint Navier-Stokes equations to compute
shape sensitivity fields that drive mesh morphing for design
optimization.

The continuous adjoint equations for a generic objective function J(U, p)
are obtained by taking the variation of the Lagrangian:

    Momentum adjoint:
        -(U·∇)Ua - (∇U)^T · Ua + ∇pa - ∇·(ν∇Ua) = -dJ/dU

    Continuity adjoint:
        ∇·Ua = 0

where Ua is the adjoint velocity, pa is the adjoint pressure, and
dJ/dU is the functional derivative of the objective with respect to
the primal velocity.

After solving, the shape sensitivity is computed on boundary faces:

    sensitivity = n · [ν(∇Ua + (∇Ua)^T) - pa I] · n - ν n · ∇(U·Ua)

This gives the steepest-descent direction for boundary deformation.

Algorithm (per outer iteration, SIMPLE-like):
1. Solve adjoint momentum: linearise convective term using primal U
2. Solve adjoint pressure correction: enforce adjoint continuity
3. Correct adjoint velocity from adjoint pressure gradient
4. Check convergence

The solver reads:
- ``0/Ua`` — adjoint velocity (initialised to zero if missing)
- ``0/pa`` — adjoint pressure (initialised to zero if missing)
- ``0/U`` — primal velocity field (frozen)
- ``0/p`` — primal pressure field (frozen)
- ``constant/polyMesh`` — mesh
- ``constant/transportProperties`` — kinematic viscosity (nu)
- ``system/controlDict`` — endTime, deltaT, writeControl
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — SIMPLE settings, linear solver tolerances

Usage::

    from pyfoam.applications.adjoint_foam import AdjointFoam

    solver = AdjointFoam("path/to/case")
    result = solver.run()

    # Access shape sensitivity
    sensitivity = solver.sensitivity
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["AdjointFoam"]

logger = logging.getLogger(__name__)


class AdjointFoam(SolverBase):
    """Continuous adjoint shape optimization solver.

    Solves the adjoint Navier-Stokes equations to compute boundary
    shape sensitivities for gradient-based design optimization.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    objective : str, optional
        Objective function type.  ``"drag"`` (default) minimises drag
        force in the x-direction.

    Attributes
    ----------
    U : torch.Tensor
        ``(n_cells, 3)`` primal velocity (frozen from case).
    p : torch.Tensor
        ``(n_cells,)`` primal pressure (frozen from case).
    phi_primal : torch.Tensor
        ``(n_faces,)`` primal face flux.
    Ua : torch.Tensor
        ``(n_cells, 3)`` adjoint velocity.
    pa : torch.Tensor
        ``(n_cells,)`` adjoint pressure.
    sensitivity : torch.Tensor
        ``(n_cells,)`` shape sensitivity field (negative = beneficial
        deformation direction).
    nu : float
        Kinematic viscosity.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        objective: str = "drag",
    ) -> None:
        super().__init__(case_path)

        # Read transport properties
        self.nu = self._read_nu()
        self.objective = objective

        # Read fvSolution settings
        self._read_fv_solution_settings()
        self._read_fv_schemes_settings()

        # Initialise primal fields (frozen)
        self.U, self.p, self.phi_primal = self._init_primal_fields()

        # Initialise adjoint fields
        self.Ua, self.pa = self._init_adjoint_fields()

        # Sensitivity field (per cell)
        device = get_device()
        dtype = get_default_dtype()
        self.sensitivity = torch.zeros(
            self.mesh.n_cells, dtype=dtype, device=device,
        )

        # Store raw field data for writing
        self._U_data, self._p_data, self._Ua_data, self._pa_data = \
            self._init_field_data()

        logger.info("AdjointFoam ready: nu=%.6e, objective=%s", self.nu, objective)

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

    def _read_fv_solution_settings(self) -> None:
        """Read adjoint solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.Ua_solver = str(fv.get_path("solvers/Ua/solver", "PBiCGStab"))
        self.Ua_tolerance = float(fv.get_path("solvers/Ua/tolerance", 1e-6))
        self.Ua_rel_tol = float(fv.get_path("solvers/Ua/relTol", 0.01))
        self.Ua_max_iter = int(fv.get_path("solvers/Ua/maxIter", 1000))

        self.pa_solver = str(fv.get_path("solvers/pa/solver", "PCG"))
        self.pa_tolerance = float(fv.get_path("solvers/pa/tolerance", 1e-6))
        self.pa_rel_tol = float(fv.get_path("solvers/pa/relTol", 0.01))
        self.pa_max_iter = int(fv.get_path("solvers/pa/maxIter", 1000))

        # SIMPLE settings for adjoint
        self.n_non_orth_correctors = int(
            fv.get_path("SIMPLE/nNonOrthogonalCorrectors", 0)
        )
        self.alpha_pa = float(fv.get_path("adjoint/relaxationFactors/pa", 0.3))
        self.alpha_Ua = float(fv.get_path("adjoint/relaxationFactors/Ua", 0.7))
        self.convergence_tolerance = float(
            fv.get_path("adjoint/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("adjoint/maxOuterIterations", 100)
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

    def _init_primal_fields(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise frozen primal fields U, p, phi from 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype).squeeze()

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return U, p, phi

    def _init_adjoint_fields(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialise adjoint fields Ua, pa (zero or from file)."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        try:
            Ua_tensor, _ = self.read_field_tensor("Ua", 0)
            Ua = Ua_tensor.to(device=device, dtype=dtype)
        except Exception:
            Ua = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        try:
            pa_tensor, _ = self.read_field_tensor("pa", 0)
            pa = pa_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            pa = torch.zeros(n_cells, dtype=dtype, device=device)

        return Ua, pa

    def _init_field_data(self) -> tuple[Any, Any, Any, Any]:
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)

        try:
            Ua_data = self.case.read_field("Ua", 0)
        except Exception:
            Ua_data = None
        try:
            pa_data = self.case.read_field("pa", 0)
        except Exception:
            pa_data = None

        return U_data, p_data, Ua_data, pa_data

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _objective_source(self) -> torch.Tensor:
        """Compute the RHS source term from the objective function.

        For drag minimisation (default), the source is the unit vector
        in the flow direction (x), acting on boundary-adjacent cells.
        The adjoint source is -dJ/dU where J = F_x (drag force).

        Returns:
            ``(n_cells, 3)`` — objective gradient w.r.t. velocity.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        source = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        if self.objective == "drag":
            # dJ/dU = n_x * deltaS on walls (drag = integral of p * n_x)
            # Simplified: source on boundary-adjacent cells proportional
            # to outward normal x-component
            mesh_boundary = self.case.boundary
            owner = self.mesh.owner
            face_areas = self.mesh.face_areas

            for bp in mesh_boundary:
                if bp.patch_type in ("empty", "wedge", "symmetryPlane", "symmetry"):
                    continue
                start = bp.start_face
                n_f = bp.n_faces
                faces_slice = slice(start, start + n_f)
                bnd_owners = owner[faces_slice]
                bnd_areas = face_areas[faces_slice]

                # Normal x-component (area-weighted)
                if bnd_areas.dim() > 1:
                    nx = bnd_areas[:, 0]  # x-component of area vector
                else:
                    nx = bnd_areas  # scalar area (fallback)

                # Source on boundary-adjacent cells
                source[:, 0] = source[:, 0] + scatter_add(
                    nx, bnd_owners, n_cells,
                )

        return source

    # ------------------------------------------------------------------
    # Adjoint equation assembly (SIMPLE-like)
    # ------------------------------------------------------------------

    def _solve_adjoint_momentum(
        self,
        Ua_bc: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the adjoint momentum equation.

        -(U·∇)Ua - (∇U)^T · Ua + ∇pa - ∇·(ν∇Ua) = -dJ/dU

        Simplified to Laplacian + source for each component:

        -∇·(ν∇Ua_i) = source_i - convection_i - pressure_gradient_i

        Returns:
            ``(n_cells, 3)`` updated adjoint velocity.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # Compute primal face flux for convection
        face_areas = mesh.face_areas
        n_faces = mesh.n_faces

        U_face = torch.zeros(n_faces, 3, dtype=dtype, device=device)
        U_P = self.U[int_owner]
        U_N = self.U[int_neigh]
        U_face[:n_internal] = 0.5 * U_P + 0.5 * U_N
        if n_faces > n_internal:
            bnd_owner = mesh.owner[n_internal:]
            U_face[n_internal:] = self.U[bnd_owner]
        phi = (U_face * face_areas).sum(dim=1)

        # Objective source
        obj_source = self._objective_source()

        Ua_new = self.Ua.clone()

        for dim in range(3):
            # Assemble Laplacian term: -∇·(ν∇Ua_i)
            S_mag = face_areas[:n_internal].norm(dim=1)
            delta_f = mesh.delta_coefficients[:n_internal]
            face_coeff = self.nu * S_mag * delta_f

            V_P = gather(cell_volumes, int_owner)
            V_N = gather(cell_volumes, int_neigh)

            # Off-diagonal (diffusion)
            lower = -face_coeff / V_P
            upper = -face_coeff / V_N

            # Diagonal from diffusion (internal faces)
            diag = torch.zeros(n_cells, dtype=dtype, device=device)
            diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
            diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

            # Boundary diffusion contribution
            if n_faces > n_internal:
                bnd_areas_f = face_areas[n_internal:]
                bnd_S_mag = (
                    bnd_areas_f.norm(dim=1)
                    if bnd_areas_f.dim() > 1
                    else bnd_areas_f.abs()
                )
                bnd_delta = mesh.delta_coefficients[n_internal:]
                bnd_coeff = self.nu * bnd_S_mag * bnd_delta
                bnd_V = gather(cell_volumes, mesh.owner[n_internal:])
                diag = diag + scatter_add(
                    bnd_coeff / bnd_V, mesh.owner[n_internal:], n_cells,
                )

            # Convection contribution: upwind
            flux = phi[:n_internal]
            is_pos = flux >= 0.0
            flux_pos = torch.where(is_pos, flux, torch.zeros_like(flux))
            flux_neg = torch.where(~is_pos, flux, torch.zeros_like(flux))

            conv_lower = flux_neg / V_P
            conv_upper = flux_pos / V_N

            diag_conv = torch.zeros(n_cells, dtype=dtype, device=device)
            diag_conv = diag_conv + scatter_add(
                -flux_pos / V_P, int_owner, n_cells,
            )
            diag_conv = diag_conv + scatter_add(
                flux_neg.abs() / V_N, int_neigh, n_cells,
            )
            diag = diag + diag_conv

            # Source: objective + pressure gradient + convection from
            # adjoint (initialised with previous iteration values)
            source = obj_source[:, dim].clone()

            # Add pressure gradient contribution (simplified)
            if dim == 0:
                grad_pa = self._compute_gradient(self.pa)
                source = source + grad_pa[:, 0] * cell_volumes
            elif dim == 1:
                if 'grad_pa' not in dir():
                    grad_pa = self._compute_gradient(self.pa)
                source = source + grad_pa[:, 1] * cell_volumes
            else:
                if 'grad_pa' not in dir():
                    grad_pa = self._compute_gradient(self.pa)
                source = source + grad_pa[:, 2] * cell_volumes

            # Jacobi iteration to solve
            diag_safe = diag.abs().clamp(min=1e-30)
            Ua_comp = Ua_new[:, dim].clone()

            for _ in range(self.Ua_max_iter):
                off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
                Ua_P = gather(Ua_comp, int_owner)
                Ua_N = gather(Ua_comp, int_neigh)
                combined_lower = lower + conv_lower
                combined_upper = upper + conv_upper
                off_diag = off_diag + scatter_add(
                    combined_lower * Ua_N, int_owner, n_cells,
                )
                off_diag = off_diag + scatter_add(
                    combined_upper * Ua_P, int_neigh, n_cells,
                )

                Ua_new_comp = (source - off_diag) / diag_safe

                # Apply boundary conditions
                bc_mask = ~torch.isnan(Ua_bc[:, dim])
                if bc_mask.any():
                    Ua_new_comp[bc_mask] = Ua_bc[bc_mask, dim]

                if (Ua_new_comp - Ua_comp).abs().max() < self.Ua_tolerance:
                    Ua_comp = Ua_new_comp
                    break
                Ua_comp = Ua_new_comp

            Ua_new[:, dim] = Ua_comp

        return Ua_new

    def _solve_adjoint_pressure(self, Ua_bc: torch.Tensor) -> torch.Tensor:
        """Solve the adjoint pressure correction equation.

        ∇²pa = ∇·(ν∇·Ua*) - ∇·(source)

        where Ua* is the intermediate adjoint velocity.

        Returns:
            ``(n_cells,)`` updated adjoint pressure.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # Divergence of adjoint velocity
        face_areas = mesh.face_areas
        n_faces = mesh.n_faces

        Ua_face = torch.zeros(n_faces, 3, dtype=dtype, device=device)
        Ua_P = self.Ua[int_owner]
        Ua_N = self.Ua[int_neigh]
        Ua_face[:n_internal] = 0.5 * Ua_P + 0.5 * Ua_N
        if n_faces > n_internal:
            bnd_owner = mesh.owner[n_internal:]
            Ua_face[n_internal:] = self.Ua[bnd_owner]

        div_Ua = (Ua_face * face_areas).sum(dim=1)

        # Assemble Laplacian for pressure correction
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        face_coeff = S_mag * delta_f

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # Boundary
        if n_faces > n_internal:
            bnd_areas_f = face_areas[n_internal:]
            bnd_S_mag = (
                bnd_areas_f.norm(dim=1)
                if bnd_areas_f.dim() > 1
                else bnd_areas_f.abs()
            )
            bnd_delta = mesh.delta_coefficients[n_internal:]
            bnd_coeff = bnd_S_mag * bnd_delta
            diag = diag + scatter_add(
                bnd_coeff / V_P[:n_faces - n_internal]
                if len(V_P) >= n_faces - n_internal
                else bnd_coeff / gather(cell_volumes, mesh.owner[n_internal:]),
                mesh.owner[n_internal:],
                n_cells,
            )

        # Source: divergence of adjoint velocity
        source = torch.zeros(n_cells, dtype=dtype, device=device)
        source = source + scatter_add(
            div_Ua[:n_internal] / V_P, int_owner, n_cells,
        )

        # Jacobi iteration
        diag_safe = diag.abs().clamp(min=1e-30)
        pa = self.pa.clone()

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        for _ in range(self.pa_max_iter):
            off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
            pa_P = gather(pa, int_owner)
            pa_N = gather(pa, int_neigh)
            off_diag = off_diag + scatter_add(lower * pa_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * pa_P, int_neigh, n_cells)

            pa_new = (source - off_diag) / diag_safe

            if (pa_new - pa).abs().max() < self.pa_tolerance:
                pa = pa_new
                break
            pa = pa_new

        return pa

    def _correct_adjoint_velocity(self) -> None:
        """Correct adjoint velocity from adjoint pressure gradient.

        Ua = Ua* - ∇pa / diag_coeff
        """
        grad_pa = self._compute_gradient(self.pa)
        # Simplified correction: subtract pressure gradient
        correction = grad_pa * self.alpha_Ua
        self.Ua = self.Ua - correction

    def _compute_gradient(self, field: torch.Tensor) -> torch.Tensor:
        """Compute cell-centred gradient of a scalar field.

        Uses least-squares or Gauss linear gradient.

        Returns:
            ``(n_cells, 3)`` gradient vector.
        """
        try:
            from pyfoam.discretisation.operators import fvc
            return fvc.grad(field, mesh=self.mesh)
        except Exception:
            # Fallback: finite difference approximation
            return self._gradient_fallback(field)

    def _gradient_fallback(self, field: torch.Tensor) -> torch.Tensor:
        """Fallback gradient computation using face-based method."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        grad_field = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        face_areas = mesh.face_areas

        # Internal face contribution
        df = field[int_neigh] - field[int_owner]  # (n_internal,)
        S_f = face_areas[:n_internal]  # (n_internal, 3)

        # grad_P += (df * S_f) / V_P
        cell_volumes = mesh.cell_volumes
        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        for d in range(3):
            contrib = df * S_f[:, d]
            grad_field[:, d] = grad_field[:, d] + scatter_add(
                contrib / V_P, int_owner, n_cells,
            )
            grad_field[:, d] = grad_field[:, d] + scatter_add(
                -contrib / V_N, int_neigh, n_cells,
            )

        return grad_field

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _build_adjoint_boundary_conditions(self) -> torch.Tensor:
        """Build boundary conditions for adjoint velocity.

        For wall boundaries: Ua = 0 (no-slip adjoint).
        For inlet/outlet: zeroGradient (default).

        Returns:
            ``(n_cells, 3)`` — prescribed Ua (NaN where no BC).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        Ua_bc = torch.full((n_cells, 3), float('nan'), dtype=dtype, device=device)

        mesh_boundary = self.case.boundary
        owner = self.mesh.owner

        for bp in mesh_boundary:
            # Walls get zero adjoint velocity (no-slip adjoint)
            if bp.patch_type == "wall":
                start = bp.start_face
                n_f = bp.n_faces
                for i in range(n_f):
                    face_idx = start + i
                    cell_idx = owner[face_idx].item()
                    Ua_bc[cell_idx, :] = 0.0

        return Ua_bc

    # ------------------------------------------------------------------
    # Sensitivity computation
    # ------------------------------------------------------------------

    def _compute_sensitivity(self) -> torch.Tensor:
        """Compute shape sensitivity on boundary cells.

        sensitivity_i = n · [ν(∇Ua + (∇Ua)^T) - pa I] · n

        Simplified: uses normal stress from adjoint field on
        boundary-adjacent cells.

        Returns:
            ``(n_cells,)`` sensitivity (negative = beneficial deformation).
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells

        sensitivity = torch.zeros(n_cells, dtype=dtype, device=device)

        # Compute gradient of adjoint velocity
        grad_Ua = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        for dim in range(3):
            grad_Ua[:, dim, :] = self._compute_gradient(self.Ua[:, dim])

        # On boundary cells: sensitivity = ν * (∂Ua_i/∂x_i) - pa
        # (trace of viscous stress minus pressure)
        for i in range(3):
            sensitivity = sensitivity + self.nu * grad_Ua[:, i, i]
        sensitivity = sensitivity - self.pa

        # Only meaningful on boundary-adjacent cells
        # Mask interior cells to zero
        is_boundary = torch.zeros(n_cells, dtype=torch.bool, device=device)
        mesh_boundary = self.case.boundary
        owner = self.mesh.owner

        for bp in mesh_boundary:
            if bp.patch_type in ("empty", "wedge"):
                continue
            start = bp.start_face
            n_f = bp.n_faces
            bnd_owners = owner[start:start + n_f]
            is_boundary[bnd_owners] = True

        sensitivity = sensitivity * is_boundary.to(dtype=dtype)

        return sensitivity

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the adjoint solver.

        Solves the adjoint Navier-Stokes equations using a SIMPLE-like
        iterative algorithm.  After convergence, computes the shape
        sensitivity field.

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
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting adjointFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  relaxation: alpha_Ua=%.2f, alpha_pa=%.2f",
                     self.alpha_Ua, self.alpha_pa)

        # Build boundary conditions
        Ua_bc = self._build_adjoint_boundary_conditions()

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Solve adjoint momentum
            self.Ua = self._solve_adjoint_momentum(Ua_bc)

            # Relax adjoint velocity
            Ua_old = self.Ua.clone()

            # Solve adjoint pressure
            self.pa = self._solve_adjoint_pressure(Ua_bc)

            # Relax adjoint pressure
            pa_old = self.pa.clone()
            self.pa = self.alpha_pa * self.pa + (1 - self.alpha_pa) * pa_old

            # Correct adjoint velocity
            self._correct_adjoint_velocity()

            # Compute residuals
            Ua_residual = float(
                (self.Ua - Ua_old).norm() / (self.Ua.norm().clamp(min=1e-30))
            )
            pa_residual = float(
                (self.pa - pa_old).norm() / (self.pa.norm().clamp(min=1e-30))
            )

            conv = ConvergenceData()
            conv.U_residual = Ua_residual
            conv.p_residual = pa_residual

            residuals = {"Ua": Ua_residual, "pa": pa_residual}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Adjoint converged at step %d (t=%.6g)", step + 1, t)
                break

        # Compute shape sensitivity
        self.sensitivity = self._compute_sensitivity()

        # Write final fields
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info("adjointFoam completed")
        logger.info("  sensitivity range: [%.6e, %.6e]",
                     self.sensitivity.min().item(),
                     self.sensitivity.max().item())

        conv.converged = converged if converged else False
        return conv

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write Ua, pa, and sensitivity to a time directory."""
        time_str = f"{time:g}"

        if self._Ua_data is not None:
            self.write_field("Ua", self.Ua, time_str, self._Ua_data)
        else:
            # Create synthetic field data for writing
            from pyfoam.io.field_io import FieldData, BoundaryField
            from pyfoam.io.foam_file import FoamFileHeader, FileFormat
            Ua_data = FieldData(
                header=FoamFileHeader(
                    version="2.0",
                    format=FileFormat.ASCII,
                    class_name="volVectorField",
                    location=str(time),
                    object="Ua",
                ),
                dimensions=[0, 1, -1, 0, 0, 0, 0],
                internal_field=self.Ua.detach().cpu(),
                boundary_field=BoundaryField(),
                is_uniform=False,
                scalar_type="vector",
            )
            self.write_field("Ua", self.Ua, time_str, Ua_data)

        if self._pa_data is not None:
            self.write_field("pa", self.pa, time_str, self._pa_data)
        else:
            from pyfoam.io.field_io import FieldData, BoundaryField
            from pyfoam.io.foam_file import FoamFileHeader, FileFormat
            pa_data = FieldData(
                header=FoamFileHeader(
                    version="2.0",
                    format=FileFormat.ASCII,
                    class_name="volScalarField",
                    location=str(time),
                    object="pa",
                ),
                dimensions=[0, 2, -2, 0, 0, 0, 0],
                internal_field=self.pa.detach().cpu(),
                boundary_field=BoundaryField(),
                is_uniform=False,
                scalar_type="scalar",
            )
            self.write_field("pa", self.pa, time_str, pa_data)
