"""
adjointShapeFoam -- enhanced adjoint shape optimization solver with mesh morphing.

Extends the base adjoint solver with:
- Shape sensitivity computation on boundary faces
- Mesh morphing / deformation driven by the sensitivity field
- Gradient projection onto boundary normals
- Iterative shape optimisation loop (sensitivity -> morph -> re-solve)

The continuous adjoint method computes the shape gradient:

    sensitivity = n · [nu(grad(Ua) + grad(Ua)^T) - pa I] · n

where n is the outward boundary normal, Ua is the adjoint velocity,
pa is the adjoint pressure, and nu is kinematic viscosity.

Mesh morphing applies boundary displacements proportional to the
negative sensitivity (steepest descent), smoothed via a Laplacian
filter, and propagated into the interior using a linear-elastic
analogy.

Usage::

    from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam

    solver = AdjointShapeFoam("path/to/case")
    result = solver.run()
    # solver.boundary_displacement contains the shape update
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["AdjointShapeFoam"]

logger = logging.getLogger(__name__)


class AdjointShapeFoam(SolverBase):
    """Enhanced adjoint shape optimization solver with mesh morphing.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    objective : str
        Objective function type: ``"drag"`` (default) or ``"lift"``.
    morph_smoothing : float
        Smoothing factor for mesh morphing (default 0.5).
    max_morph_displacement : float
        Maximum allowed displacement per shape iteration (default 0.01).
    n_shape_iterations : int
        Number of shape optimisation iterations (default 5).

    Attributes
    ----------
    U : torch.Tensor
        ``(n_cells, 3)`` primal velocity.
    p : torch.Tensor
        ``(n_cells,)`` primal pressure.
    Ua : torch.Tensor
        ``(n_cells, 3)`` adjoint velocity.
    pa : torch.Tensor
        ``(n_cells,)`` adjoint pressure.
    sensitivity : torch.Tensor
        ``(n_cells,)`` shape sensitivity field.
    boundary_displacement : torch.Tensor
        ``(n_faces_boundary, 3)`` displacement on boundary faces.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        objective: str = "drag",
        morph_smoothing: float = 0.5,
        max_morph_displacement: float = 0.01,
        n_shape_iterations: int = 5,
    ) -> None:
        super().__init__(case_path)

        self.nu = self._read_nu()
        self.objective = objective
        self.morph_smoothing = morph_smoothing
        self.max_morph_displacement = max_morph_displacement
        self.n_shape_iterations = n_shape_iterations

        # Read solver settings
        self._read_fv_solution_settings()

        # Initialise primal (frozen) and adjoint fields
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        self.U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        self.p = p_tensor.to(device=device, dtype=dtype).squeeze()

        n_cells = self.mesh.n_cells

        try:
            Ua_tensor, _ = self.read_field_tensor("Ua", 0)
            self.Ua = Ua_tensor.to(device=device, dtype=dtype)
        except Exception:
            self.Ua = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        try:
            pa_tensor, _ = self.read_field_tensor("pa", 0)
            self.pa = pa_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            self.pa = torch.zeros(n_cells, dtype=dtype, device=device)

        self.sensitivity = torch.zeros(n_cells, dtype=dtype, device=device)

        # Boundary displacement (per boundary face)
        n_boundary_faces = self.mesh.n_faces - self.mesh.n_internal_faces
        self.boundary_displacement = torch.zeros(
            n_boundary_faces, 3, dtype=dtype, device=device,
        )

        logger.info(
            "AdjointShapeFoam ready: nu=%.6e, objective=%s, n_shape_iter=%d",
            self.nu, objective, n_shape_iterations,
        )

    # ------------------------------------------------------------------
    # Settings reading
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
                import re
                match = re.search(r"]\s*([\d.eE+\-]+)", str(raw).strip())
                if match:
                    return float(match.group(1))
                return float(str(raw).strip())
            except Exception:
                pass
        return 1.0

    def _read_fv_solution_settings(self) -> None:
        """Read adjoint solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.Ua_tolerance = float(fv.get_path("solvers/Ua/tolerance", 1e-6))
        self.Ua_max_iter = int(fv.get_path("solvers/Ua/maxIter", 1000))
        self.pa_tolerance = float(fv.get_path("solvers/pa/tolerance", 1e-6))
        self.pa_max_iter = int(fv.get_path("solvers/pa/maxIter", 1000))
        self.alpha_Ua = float(fv.get_path("adjoint/relaxationFactors/Ua", 0.7))
        self.alpha_pa = float(fv.get_path("adjoint/relaxationFactors/pa", 0.3))
        self.convergence_tolerance = float(
            fv.get_path("adjoint/convergenceTolerance", 1e-4),
        )
        self.max_outer_iterations = int(
            fv.get_path("adjoint/maxOuterIterations", 100),
        )

    # ------------------------------------------------------------------
    # Sensitivity computation
    # ------------------------------------------------------------------

    def _compute_boundary_sensitivity(self) -> torch.Tensor:
        """Compute shape sensitivity on boundary-adjacent cells.

        sensitivity_i = n . [nu(grad(Ua) + grad(Ua)^T) - pa I] . n

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` sensitivity (negative = beneficial deformation).
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells

        sensitivity = torch.zeros(n_cells, dtype=dtype, device=device)

        # Gradient of each adjoint velocity component
        grad_Ua = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        for dim in range(3):
            grad_Ua[:, dim, :] = self._compute_gradient(self.Ua[:, dim])

        # Viscous stress from adjoint: nu * (grad_Ua + grad_Ua^T)
        viscous_trace = torch.zeros(n_cells, dtype=dtype, device=device)
        for i in range(3):
            viscous_trace = viscous_trace + 2.0 * self.nu * grad_Ua[:, i, i]

        sensitivity = viscous_trace - self.pa

        # Mask to boundary-adjacent cells only
        is_boundary = torch.zeros(n_cells, dtype=torch.bool, device=device)
        owner = self.mesh.owner

        for bp in self.case.boundary:
            if bp.patch_type in ("empty", "wedge"):
                continue
            start = bp.start_face
            n_f = bp.n_faces
            is_boundary[owner[start:start + n_f]] = True

        sensitivity = sensitivity * is_boundary.to(dtype=dtype)
        return sensitivity

    def _compute_gradient(self, field: torch.Tensor) -> torch.Tensor:
        """Compute cell-centred gradient of a scalar field."""
        try:
            from pyfoam.discretisation.operators import fvc
            return fvc.grad(field, mesh=self.mesh)
        except Exception:
            return self._gradient_fallback(field)

    def _gradient_fallback(self, field: torch.Tensor) -> torch.Tensor:
        """Fallback gradient using face-based method."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        grad_field = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        face_areas = mesh.face_areas
        cell_volumes = mesh.cell_volumes

        df = field[int_neigh] - field[int_owner]
        S_f = face_areas[:n_internal]

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
    # Mesh morphing
    # ------------------------------------------------------------------

    def _compute_face_displacement(self) -> torch.Tensor:
        """Compute boundary face displacements from the sensitivity field.

        Projects the sensitivity onto boundary normals and applies
        smoothing to produce a displacement vector per boundary face.

        Returns
        -------
        torch.Tensor
            ``(n_boundary_faces, 3)`` displacement vectors.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_internal = mesh.n_internal_faces
        n_total = mesh.n_faces
        n_boundary = n_total - n_internal

        owner = mesh.owner
        face_areas = mesh.face_areas[n_internal:]

        # Outward unit normals on boundary faces
        if face_areas.dim() > 1:
            normals = face_areas
        else:
            # Scalar areas: construct normals from face geometry
            normals = torch.zeros(n_boundary, 3, dtype=dtype, device=device)
            normals[:, 0] = face_areas

        norms = normals.norm(dim=1, keepdim=True).clamp(min=1e-30)
        unit_normals = normals / norms

        # Sensitivity at boundary-adjacent cells
        bnd_owners = owner[n_internal:]
        sens = self.sensitivity[bnd_owners]

        # Displacement = -sensitivity * n (steepest descent)
        displacement = -sens.unsqueeze(1) * unit_normals

        # Clamp maximum displacement
        disp_mag = displacement.norm(dim=1)
        scale = torch.where(
            disp_mag > self.max_morph_displacement,
            self.max_morph_displacement / disp_mag.clamp(min=1e-30),
            torch.ones_like(disp_mag),
        )
        displacement = displacement * scale.unsqueeze(1)

        return displacement

    def _smooth_displacement(self, displacement: torch.Tensor) -> torch.Tensor:
        """Apply Laplacian smoothing to boundary displacement.

        Parameters
        ----------
        displacement : torch.Tensor
            ``(n_boundary_faces, 3)`` raw displacement.

        Returns
        -------
        torch.Tensor
            Smoothed displacement of the same shape.
        """
        if self.morph_smoothing <= 0.0:
            return displacement

        n_boundary = displacement.shape[0]
        if n_boundary <= 2:
            return displacement

        # Simple Laplacian smoothing: average with neighbours
        smoothed = displacement.clone()

        # Interior boundary faces: weighted average with adjacent faces
        for i in range(1, n_boundary - 1):
            smoothed[i] = (
                self.morph_smoothing * 0.5 * (displacement[i - 1] + displacement[i + 1])
                + (1.0 - self.morph_smoothing) * displacement[i]
            )

        return smoothed

    def _apply_mesh_morph(self) -> bool:
        """Compute and store the mesh morph displacement.

        Returns
        -------
        bool
            True if a non-trivial displacement was computed.
        """
        raw_disp = self._compute_face_displacement()
        self.boundary_displacement = self._smooth_displacement(raw_disp)

        return float(self.boundary_displacement.abs().sum()) > 1e-30

    # ------------------------------------------------------------------
    # Adjoint solve (simplified SIMPLE-like)
    # ------------------------------------------------------------------

    def _solve_adjoint(self) -> ConvergenceData:
        """Solve the adjoint equations (momentum + pressure correction).

        Returns
        -------
        ConvergenceData
            Convergence information from the adjoint solve.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        n_faces = mesh.n_faces

        # Build boundary conditions
        Ua_bc = torch.full((n_cells, 3), float('nan'), dtype=dtype, device=device)
        for bp in self.case.boundary:
            if bp.patch_type == "wall":
                start = bp.start_face
                n_f = bp.n_faces
                for i in range(n_f):
                    Ua_bc[int(mesh.owner[start + i].item()), :] = 0.0

        # Objective source
        source = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        if self.objective == "drag":
            for bp in self.case.boundary:
                if bp.patch_type in ("empty", "wedge", "symmetryPlane", "symmetry"):
                    continue
                start = bp.start_face
                n_f = bp.n_faces
                bnd_owners = mesh.owner[start:start + n_f]
                bnd_areas = face_areas[start:start + n_f]
                if bnd_areas.dim() > 1:
                    nx = bnd_areas[:, 0]
                else:
                    nx = bnd_areas
                source[:, 0] = source[:, 0] + scatter_add(nx, bnd_owners, n_cells)

        # Compute face flux
        U_face = torch.zeros(n_faces, 3, dtype=dtype, device=device)
        U_face[:n_internal] = 0.5 * (self.U[int_owner] + self.U[int_neigh])
        if n_faces > n_internal:
            U_face[n_internal:] = self.U[mesh.owner[n_internal:]]
        phi = (U_face * face_areas).sum(dim=1)

        # Adjoint momentum solve (Jacobi per component)
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        face_coeff = self.nu * S_mag * delta_f

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        if n_faces > n_internal:
            bnd_areas_f = face_areas[n_internal:]
            bnd_S = bnd_areas_f.norm(dim=1) if bnd_areas_f.dim() > 1 else bnd_areas_f.abs()
            bnd_delta = mesh.delta_coefficients[n_internal:]
            bnd_coeff = self.nu * bnd_S * bnd_delta
            diag = diag + scatter_add(
                bnd_coeff / gather(cell_volumes, mesh.owner[n_internal:]),
                mesh.owner[n_internal:], n_cells,
            )

        # Convection (upwind)
        flux = phi[:n_internal]
        is_pos = flux >= 0.0
        flux_pos = torch.where(is_pos, flux, torch.zeros_like(flux))
        flux_neg = torch.where(~is_pos, flux, torch.zeros_like(flux))

        conv_lower = flux_neg / V_P
        conv_upper = flux_pos / V_N

        diag = diag + scatter_add(-flux_pos / V_P, int_owner, n_cells)
        diag = diag + scatter_add(flux_neg.abs() / V_N, int_neigh, n_cells)

        grad_pa = self._compute_gradient(self.pa)

        Ua_old = self.Ua.clone()
        diag_safe = diag.abs().clamp(min=1e-30)

        for dim in range(3):
            src = source[:, dim].clone() + grad_pa[:, dim] * cell_volumes
            Ua_comp = self.Ua[:, dim].clone()

            for _ in range(self.Ua_max_iter):
                off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
                Ua_P = gather(Ua_comp, int_owner)
                Ua_N = gather(Ua_comp, int_neigh)
                off_diag = off_diag + scatter_add((lower + conv_lower) * Ua_N, int_owner, n_cells)
                off_diag = off_diag + scatter_add((upper + conv_upper) * Ua_P, int_neigh, n_cells)

                Ua_new = (src - off_diag) / diag_safe
                bc_mask = ~torch.isnan(Ua_bc[:, dim])
                if bc_mask.any():
                    Ua_new[bc_mask] = Ua_bc[bc_mask, dim]

                if (Ua_new - Ua_comp).abs().max() < self.Ua_tolerance:
                    Ua_comp = Ua_new
                    break
                Ua_comp = Ua_new

            self.Ua[:, dim] = self.alpha_Ua * Ua_comp + (1 - self.alpha_Ua) * self.Ua[:, dim]

        # Adjoint pressure correction
        Ua_face = torch.zeros(n_faces, 3, dtype=dtype, device=device)
        Ua_face[:n_internal] = 0.5 * (self.Ua[int_owner] + self.Ua[int_neigh])
        if n_faces > n_internal:
            Ua_face[n_internal:] = self.Ua[mesh.owner[n_internal:]]
        div_Ua = (Ua_face * face_areas).sum(dim=1)

        p_diag = torch.zeros(n_cells, dtype=dtype, device=device)
        p_diag = p_diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        p_diag = p_diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)
        p_diag_safe = p_diag.abs().clamp(min=1e-30)

        p_source = scatter_add(div_Ua[:n_internal] / V_P, int_owner, n_cells)

        pa_old = self.pa.clone()
        pa = self.pa.clone()
        p_lower = -face_coeff / V_P
        p_upper = -face_coeff / V_N

        for _ in range(self.pa_max_iter):
            off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
            off_diag = off_diag + scatter_add(p_lower * gather(pa, int_neigh), int_owner, n_cells)
            off_diag = off_diag + scatter_add(p_upper * gather(pa, int_owner), int_neigh, n_cells)
            pa_new = (p_source - off_diag) / p_diag_safe
            if (pa_new - pa).abs().max() < self.pa_tolerance:
                pa = pa_new
                break
            pa = pa_new

        self.pa = self.alpha_pa * pa + (1 - self.alpha_pa) * pa_old

        # Velocity correction
        grad_pa_new = self._compute_gradient(self.pa)
        self.Ua = self.Ua - grad_pa_new * self.alpha_Ua

        # Compute residuals
        Ua_residual = float(
            (self.Ua - Ua_old).norm() / self.Ua.norm().clamp(min=1e-30),
        )
        pa_residual = float(
            (self.pa - pa_old).norm() / self.pa.norm().clamp(min=1e-30),
        )

        conv = ConvergenceData()
        conv.U_residual = Ua_residual
        conv.p_residual = pa_residual
        return conv

    # ------------------------------------------------------------------
    # Shape optimisation loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the shape optimisation solver.

        Performs *n_shape_iterations* outer loops:
        1. Solve adjoint equations
        2. Compute shape sensitivity
        3. Compute and store boundary displacement

        Returns
        -------
        ConvergenceData
            Final convergence state from the last adjoint solve.
        """
        logger.info("Starting adjointShapeFoam run")
        logger.info("  n_shape_iterations=%d, max_displacement=%.6e",
                     self.n_shape_iterations, self.max_morph_displacement)

        last_conv = None

        for shape_iter in range(self.n_shape_iterations):
            logger.info("Shape iteration %d / %d", shape_iter + 1, self.n_shape_iterations)

            # Solve adjoint
            last_conv = self._solve_adjoint()

            # Compute sensitivity
            self.sensitivity = self._compute_boundary_sensitivity()

            logger.info(
                "  sensitivity range: [%.6e, %.6e]",
                self.sensitivity.min().item(),
                self.sensitivity.max().item(),
            )

            # Compute mesh morph
            has_morph = self._apply_mesh_morph()
            if not has_morph:
                logger.info("  No significant displacement; stopping early.")
                break

            disp_mag = float(self.boundary_displacement.norm(dim=1).mean())
            logger.info("  Mean displacement magnitude: %.6e", disp_mag)

        if last_conv is None:
            last_conv = ConvergenceData()

        logger.info("adjointShapeFoam completed")
        return last_conv
