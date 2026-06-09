"""
mhdFoam — magnetohydrodynamics solver.

Solves the coupled Navier-Stokes + induction equations for
magnetohydrodynamic (MHD) flow:

    Momentum:  ∂U/∂t + ∇·(UU) = -∇p + ν∇²U + (1/ρ)(J × B)
    Induction: ∂B/∂t = ∇ × (U × B) + η∇²B
    Continuity: ∇·U = 0

where:
- U is the velocity field
- B is the magnetic flux density
- J = (1/μ₀) ∇ × B is the current density
- F_Lorentz = J × B is the Lorentz force
- η = 1/(μ₀ σ) is the magnetic diffusivity
- σ is the electrical conductivity

Uses a SIMPLE-like iteration with electromagnetic coupling:
1. Solve momentum equation (with Lorentz force from old B)
2. Solve pressure correction (continuity)
3. Solve induction equation (with convected B from U)
4. Update J = curl(B)/μ₀
5. Repeat until convergence

The solver reads:
- ``0/U``, ``0/p``, ``0/B`` — initial/boundary conditions
- ``constant/polyMesh`` — mesh
- ``constant/transportProperties`` — kinematic viscosity (nu)
- ``constant/mhdProperties`` — mu0, sigma (electrical conductivity)
- ``system/controlDict`` — endTime, deltaT
- ``system/fvSchemes`` — discretisation schemes
- ``system/fvSolution`` — SIMPLE settings, linear solver tolerances

Usage::

    from pyfoam.applications.mhd_foam import MhdFoam

    solver = MhdFoam("path/to/case")
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
from pyfoam.discretisation.operators import fvc
from pyfoam.solvers.linear_solver import create_solver
from pyfoam.core.backend import scatter_add, gather

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["MhdFoam"]

logger = logging.getLogger(__name__)


class MhdFoam(SolverBase):
    """Magnetohydrodynamics solver (Navier-Stokes + induction).

    Solves the coupled MHD equations with SIMPLE-like iteration.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    nu : float | None, optional
        Kinematic viscosity.  If None, reads from transportProperties.
    mu0 : float | None, optional
        Permeability.  If None, reads from mhdProperties.
    sigma : float | None, optional
        Electrical conductivity.  If None, reads from mhdProperties.

    Attributes
    ----------
    U : torch.Tensor
        ``(n_cells, 3)`` velocity field.
    p : torch.Tensor
        ``(n_cells,)`` pressure field.
    B : torch.Tensor
        ``(n_cells, 3)`` magnetic flux density field.
    J : torch.Tensor
        ``(n_cells, 3)`` current density field.
    nu : float
        Kinematic viscosity.
    mu0 : float
        Magnetic permeability.
    sigma : float
        Electrical conductivity.
    eta : float
        Magnetic diffusivity (1 / (mu0 * sigma)).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        nu: float | None = None,
        mu0: float | None = None,
        sigma: float | None = None,
    ) -> None:
        super().__init__(case_path)

        # Read transport properties
        self.nu = nu if nu is not None else self._read_viscosity()

        # Read MHD properties
        props = self._read_mhd_properties()
        self.mu0 = mu0 if mu0 is not None else props.get("mu0", 1.0)
        self.sigma = sigma if sigma is not None else props.get("sigma", 1.0)
        self.eta = 1.0 / (self.mu0 * self.sigma) if (self.mu0 * self.sigma) > 0 else 1.0

        # Read fvSolution settings
        self._read_fv_solution_settings()

        # Read fvSchemes
        self._read_fv_schemes_settings()

        # Initialise fields
        self.U, self.p, self.B, self.J = self._init_fields()

        # Store raw field data for writing
        self._U_data, self._p_data, self._B_data = self._init_field_data()

        # Parse boundary conditions
        self._U_bc = self._parse_vector_bc("U")
        self._p_bc = self._parse_scalar_bc("p")
        self._B_bc = self._parse_vector_bc("B")

        logger.info("MhdFoam ready: nu=%.6e, mu0=%.6e, sigma=%.6e, eta=%.6e",
                    self.nu, self.mu0, self.sigma, self.eta)

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_viscosity(self) -> float:
        """Read kinematic viscosity from transportProperties."""
        tp_path = self.case_path / "constant" / "transportProperties"
        if not tp_path.exists():
            return 1.0
        try:
            from pyfoam.io.dictionary import parse_dict_file
            tp = parse_dict_file(tp_path)
            nu = tp.get("nu", tp.get("transportModel", 1.0))
            if isinstance(nu, dict):
                nu = nu.get("value", 1.0)
            return float(nu)
        except Exception:
            return 1.0

    def _read_mhd_properties(self) -> dict[str, float]:
        """Read MHD properties from constant/mhdProperties."""
        mp_path = self.case_path / "constant" / "mhdProperties"
        if not mp_path.exists():
            logger.warning("constant/mhdProperties not found, using defaults")
            return {"mu0": 1.0, "sigma": 1.0}
        try:
            from pyfoam.io.dictionary import parse_dict_file
            mp = parse_dict_file(mp_path)
            mu0 = mp.get("mu0", 1.0)
            sigma = mp.get("sigma", 1.0)
            if isinstance(mu0, dict):
                mu0 = mu0.get("value", 1.0)
            if isinstance(sigma, dict):
                sigma = sigma.get("value", 1.0)
            return {"mu0": float(mu0), "sigma": float(sigma)}
        except Exception as e:
            logger.warning("Could not parse mhdProperties: %s", e)
            return {"mu0": 1.0, "sigma": 1.0}

    def _read_fv_solution_settings(self) -> None:
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution

        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_rel_tol = float(fv.get_path("solvers/p/relTol", 0.01))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_rel_tol = float(fv.get_path("solvers/U/relTol", 0.01))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        self.B_solver = str(fv.get_path("solvers/B/solver", "PBiCGStab"))
        self.B_tolerance = float(fv.get_path("solvers/B/tolerance", 1e-6))
        self.B_rel_tol = float(fv.get_path("solvers/B/relTol", 0.01))
        self.B_max_iter = int(fv.get_path("solvers/B/maxIter", 1000))

        # SIMPLE settings
        self.n_outer_correctors = int(
            fv.get_path("SIMPLE/nOuterCorrectors", 3)
        )
        self.n_non_orth_correctors = int(
            fv.get_path("SIMPLE/nNonOrthogonalCorrectors", 0)
        )
        self.convergence_tolerance = float(
            fv.get_path("SIMPLE/convergenceTolerance", 1e-5)
        )

        # Under-relaxation
        self.alpha_U = float(fv.get_path("relaxationFactors/U", 0.7))
        self.alpha_p = float(fv.get_path("relaxationFactors/p", 0.3))
        self.alpha_B = float(fv.get_path("relaxationFactors/B", 0.7))

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings."""
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

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise U, p, B, J from the 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # Velocity
        try:
            U_tensor, _ = self.read_field_tensor("U", 0)
            U = U_tensor.to(device=device, dtype=dtype)
        except Exception:
            U = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        # Pressure
        try:
            p_tensor, _ = self.read_field_tensor("p", 0)
            p = p_tensor.to(device=device, dtype=dtype).squeeze()
        except Exception:
            p = torch.zeros(n_cells, dtype=dtype, device=device)

        # Magnetic field
        try:
            B_tensor, _ = self.read_field_tensor("B", 0)
            B = B_tensor.to(device=device, dtype=dtype)
            if B.dim() < 2:
                B = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        except Exception:
            B = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        # Current density (computed from B)
        J = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        return U, p, B, J

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        try:
            U_data = self.case.read_field("U", 0)
        except Exception:
            U_data = None
        try:
            p_data = self.case.read_field("p", 0)
        except Exception:
            p_data = None
        try:
            B_data = self.case.read_field("B", 0)
            if B_data.scalar_type == "scalar":
                B_data = None
        except Exception:
            B_data = None
        return U_data, p_data, B_data

    # ------------------------------------------------------------------
    # Boundary condition parsing
    # ------------------------------------------------------------------

    def _parse_vector_bc(self, field_name: str) -> dict[str, dict[str, Any]]:
        """Parse boundary conditions for a vector field."""
        bc_values: dict[str, dict[str, Any]] = {}

        try:
            field_data = self.case.read_field(field_name, 0)
        except Exception:
            return bc_values

        boundary = field_data.boundary_field
        mesh_boundary = self.case.boundary
        if boundary is None:
            return bc_values

        for i, patch in enumerate(boundary.patches):
            bc_info: dict[str, Any] = {"type": patch.patch_type, "value": None}

            if patch.patch_type == "fixedValue" and patch.value is not None:
                val = patch.value
                if isinstance(val, str):
                    val = val.strip()
                    if val.startswith("uniform"):
                        val = val[len("uniform"):].strip()
                    vec_match = re.match(
                        r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)",
                        val,
                    )
                    if vec_match:
                        bc_info["value"] = (
                            float(vec_match.group(1)),
                            float(vec_match.group(2)),
                            float(vec_match.group(3)),
                        )
                elif isinstance(val, (list, tuple)) and len(val) == 3:
                    bc_info["value"] = tuple(float(v) for v in val)

            if i < len(mesh_boundary):
                bp = mesh_boundary[i]
                bc_info["start_face"] = bp.start_face
                bc_info["n_faces"] = bp.n_faces

            bc_values[patch.name] = bc_info

        return bc_values

    def _parse_scalar_bc(self, field_name: str) -> dict[str, dict[str, Any]]:
        """Parse boundary conditions for a scalar field."""
        bc_values: dict[str, dict[str, Any]] = {}

        try:
            field_data = self.case.read_field(field_name, 0)
        except Exception:
            return bc_values

        boundary = field_data.boundary_field
        mesh_boundary = self.case.boundary
        if boundary is None:
            return bc_values

        for i, patch in enumerate(boundary.patches):
            bc_info: dict[str, Any] = {"type": patch.patch_type, "value": None}

            if patch.patch_type == "fixedValue" and patch.value is not None:
                val = patch.value
                if isinstance(val, str):
                    val = val.strip()
                    if val.startswith("uniform"):
                        val = val[len("uniform"):].strip()
                    try:
                        bc_info["value"] = float(val)
                    except ValueError:
                        pass
                elif isinstance(val, (int, float)):
                    bc_info["value"] = float(val)

            if i < len(mesh_boundary):
                bp = mesh_boundary[i]
                bc_info["start_face"] = bp.start_face
                bc_info["n_faces"] = bp.n_faces

            bc_values[patch.name] = bc_info

        return bc_values

    def _build_vector_bc_tensor(
        self, bc_dict: dict[str, dict[str, Any]], comp: int | None = None,
    ) -> torch.Tensor:
        """Build BC tensor for a vector (or scalar) field.

        Parameters
        ----------
        bc_dict : dict
            Boundary condition dictionary.
        comp : int | None
            If not None, extract this component from vector BCs.

        Returns:
            ``(n_cells,)`` — prescribed value (NaN where no BC).
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        bc_tensor = torch.full((n_cells,), float('nan'), dtype=dtype, device=device)
        owner = self.mesh.owner

        for patch_name, bc_info in bc_dict.items():
            if bc_info["type"] != "fixedValue" or bc_info["value"] is None:
                continue

            start_face = bc_info.get("start_face", 0)
            n_faces = bc_info.get("n_faces", 0)
            if n_faces == 0:
                continue

            val = bc_info["value"]
            if comp is not None and isinstance(val, (list, tuple)) and len(val) == 3:
                cell_val = float(val[comp])
            elif comp is None and isinstance(val, (int, float)):
                cell_val = float(val)
            else:
                continue

            for i in range(n_faces):
                face_idx = start_face + i
                cell_idx = owner[face_idx].item()
                bc_tensor[cell_idx] = cell_val

        return bc_tensor

    # ------------------------------------------------------------------
    # Lorentz force computation
    # ------------------------------------------------------------------

    def _compute_lorentz_force(self) -> torch.Tensor:
        """Compute the Lorentz force F = J × B.

        Returns:
            ``(n_cells, 3)`` Lorentz force per unit volume.
        """
        # J = curl(B) / mu0
        self.J = fvc.curl(self.B, mesh=self.mesh) / self.mu0

        # F = J × B (cross product)
        F = torch.cross(self.J, self.B, dim=1)
        return F

    # ------------------------------------------------------------------
    # Momentum equation (simplified diffusion + source)
    # ------------------------------------------------------------------

    def _solve_momentum(self, F_lorentz: torch.Tensor) -> None:
        """Solve the momentum equation for each component of U.

        Uses a simplified diffusion operator with Lorentz force source.
        Applies under-relaxation.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        face_coeff = self.nu * S_mag * delta_f

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        lap_diag = torch.zeros(n_cells, dtype=dtype, device=device)
        lap_diag = lap_diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        lap_diag = lap_diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # Boundary diffusion
        bc_diag = torch.zeros(n_cells, dtype=dtype, device=device)
        bc_source = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        for patch_name, bc_info in self._U_bc.items():
            start_face = bc_info.get("start_face", 0)
            n_faces = bc_info.get("n_faces", 0)
            if n_faces == 0 or bc_info["type"] in ("empty", "wedge"):
                continue

            bnd_faces = slice(start_face, start_face + n_faces)
            bnd_areas = mesh.face_areas[bnd_faces]
            bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
            bnd_cells = mesh.owner[bnd_faces]
            bnd_V = gather(cell_volumes, bnd_cells)

            bnd_face_centres = mesh.face_centres[bnd_faces]
            bnd_cell_centres = mesh.cell_centres[bnd_cells]
            d_P = bnd_face_centres - bnd_cell_centres
            safe_area = torch.where(
                bnd_S_mag.unsqueeze(-1) > 1e-30,
                bnd_areas,
                torch.ones_like(bnd_areas),
            )
            n_f = safe_area / safe_area.norm(dim=1, keepdim=True)
            d_dot_n = (d_P * n_f).sum(dim=1).abs()
            bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)
            bnd_coeff = self.nu * bnd_S_mag * bnd_delta

            bc_diag = bc_diag + scatter_add(bnd_coeff / bnd_V, bnd_cells, n_cells)

            if bc_info["type"] == "fixedValue" and bc_info["value"] is not None:
                val = bc_info["value"]
                if isinstance(val, (list, tuple)) and len(val) == 3:
                    for c in range(3):
                        bc_source[:, c] = bc_source[:, c] + scatter_add(
                            bnd_coeff * float(val[c]) / bnd_V, bnd_cells, n_cells
                        )

        diag = lap_diag + bc_diag

        # Add time derivative contribution (pseudo-time for steady state)
        # Use cell volume / dt as diagonal enhancement
        diag_safe = diag.abs().clamp(min=1e-30)

        # Solve each component with Jacobi iteration
        for comp in range(3):
            source = F_lorentz[:, comp] * cell_volumes + bc_source[:, comp]

            # Under-relaxation: blend old and new
            U_old_comp = self.U[:, comp].clone()

            for _ in range(self.U_max_iter):
                off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
                U_P = gather(self.U[:, comp], int_owner)
                U_N = gather(self.U[:, comp], int_neigh)
                off_diag = off_diag + scatter_add(lower * U_N, int_owner, n_cells)
                off_diag = off_diag + scatter_add(upper * U_P, int_neigh, n_cells)

                U_new = (source - off_diag) / diag_safe

                # Under-relaxation
                U_relaxed = self.alpha_U * U_new + (1.0 - self.alpha_U) * U_old_comp

                if (U_relaxed - self.U[:, comp]).abs().max() < self.U_tolerance:
                    self.U[:, comp] = U_relaxed
                    break
                self.U[:, comp] = U_relaxed

    # ------------------------------------------------------------------
    # Pressure correction (simplified)
    # ------------------------------------------------------------------

    def _solve_pressure(self) -> None:
        """Solve the pressure correction equation.

        Simple pressure-velocity coupling: solve ∇²p = -ρ ∇·U
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # Compute divergence of U (continuity error)
        div_U = fvc.div(self.U, mesh=self.mesh)

        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        face_coeff = S_mag * delta_f

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        lap_diag = torch.zeros(n_cells, dtype=dtype, device=device)
        lap_diag = lap_diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        lap_diag = lap_diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # Source: -div(U) * V (pressure correction drives flow toward continuity)
        source = -div_U * cell_volumes

        diag_safe = lap_diag.abs().clamp(min=1e-30)

        # Jacobi iteration
        p_corr = torch.zeros(n_cells, dtype=dtype, device=device)
        for _ in range(self.p_max_iter):
            off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
            p_P = gather(p_corr, int_owner)
            p_N = gather(p_corr, int_neigh)
            off_diag = off_diag + scatter_add(lower * p_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * p_P, int_neigh, n_cells)

            p_new = (source - off_diag) / diag_safe

            if (p_new - p_corr).abs().max() < self.p_tolerance:
                p_corr = p_new
                break
            p_corr = p_new

        # Under-relax and update pressure
        self.p = self.p + self.alpha_p * p_corr

    # ------------------------------------------------------------------
    # Induction equation
    # ------------------------------------------------------------------

    def _solve_induction(self) -> None:
        """Solve the induction equation for B.

        ∂B/∂t = ∇ × (U × B) + η ∇²B

        Uses implicit diffusion and explicit convection.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        dt = self.delta_t

        # Compute curl(U × B) — the induction source term
        # U × B cross product
        U_cross_B = torch.cross(self.U, self.B, dim=1)
        # curl of cross product
        induction_source = fvc.curl(U_cross_B, mesh=self.mesh)

        # Diffusion matrix for η ∇²B
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        face_coeff = self.eta * S_mag * delta_f

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        lap_diag = torch.zeros(n_cells, dtype=dtype, device=device)
        lap_diag = lap_diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        lap_diag = lap_diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # Add time derivative: V/dt on diagonal
        diag = lap_diag + cell_volumes / dt

        # Boundary contributions for diffusion
        bc_diag = torch.zeros(n_cells, dtype=dtype, device=device)
        bc_source = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        for patch_name, bc_info in self._B_bc.items():
            start_face = bc_info.get("start_face", 0)
            n_faces = bc_info.get("n_faces", 0)
            if n_faces == 0 or bc_info["type"] in ("empty", "wedge"):
                continue

            bnd_faces = slice(start_face, start_face + n_faces)
            bnd_areas = mesh.face_areas[bnd_faces]
            bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
            bnd_cells = mesh.owner[bnd_faces]
            bnd_V = gather(cell_volumes, bnd_cells)

            bnd_face_centres = mesh.face_centres[bnd_faces]
            bnd_cell_centres = mesh.cell_centres[bnd_cells]
            d_P = bnd_face_centres - bnd_cell_centres
            safe_area = torch.where(
                bnd_S_mag.unsqueeze(-1) > 1e-30,
                bnd_areas,
                torch.ones_like(bnd_areas),
            )
            n_f = safe_area / safe_area.norm(dim=1, keepdim=True)
            d_dot_n = (d_P * n_f).sum(dim=1).abs()
            bnd_delta = 1.0 / d_dot_n.clamp(min=1e-30)
            bnd_coeff = self.eta * bnd_S_mag * bnd_delta

            bc_diag = bc_diag + scatter_add(bnd_coeff / bnd_V, bnd_cells, n_cells)

            if bc_info["type"] == "fixedValue" and bc_info["value"] is not None:
                val = bc_info["value"]
                if isinstance(val, (list, tuple)) and len(val) == 3:
                    for c in range(3):
                        bc_source[:, c] = bc_source[:, c] + scatter_add(
                            bnd_coeff * float(val[c]) / bnd_V, bnd_cells, n_cells
                        )

        diag = diag + bc_diag
        diag_safe = diag.abs().clamp(min=1e-30)

        B_old = self.B.clone()

        # Solve each component
        for comp in range(3):
            source = (
                cell_volumes * B_old[:, comp] / dt
                + induction_source[:, comp] * cell_volumes
                + bc_source[:, comp]
            )

            for _ in range(self.B_max_iter):
                off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
                B_P = gather(self.B[:, comp], int_owner)
                B_N = gather(self.B[:, comp], int_neigh)
                off_diag = off_diag + scatter_add(lower * B_N, int_owner, n_cells)
                off_diag = off_diag + scatter_add(upper * B_P, int_neigh, n_cells)

                B_new = (source - off_diag) / diag_safe

                # Under-relaxation
                B_relaxed = self.alpha_B * B_new + (1.0 - self.alpha_B) * B_old[:, comp]

                if (B_relaxed - self.B[:, comp]).abs().max() < self.B_tolerance:
                    self.B[:, comp] = B_relaxed
                    break
                self.B[:, comp] = B_relaxed

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_residual(field: torch.Tensor, field_old: torch.Tensor) -> float:
        """Compute L2 residual normalised by field magnitude."""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the mhdFoam solver.

        Solves coupled MHD equations with SIMPLE-like iteration.

        Returns:
            Dictionary with convergence information.
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

        logger.info("Starting mhdFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  nu=%.6e, mu0=%.6e, sigma=%.6e, eta=%.6e",
                    self.nu, self.mu0, self.sigma, self.eta)

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        converged = False
        last_residual = 0.0

        for t, step in time_loop:
            U_old = self.U.clone()
            B_old = self.B.clone()

            # SIMPLE-like outer iterations
            for outer in range(self.n_outer_correctors):
                # 1. Compute Lorentz force from current B
                F_lorentz = self._compute_lorentz_force()

                # 2. Solve momentum equation
                self._solve_momentum(F_lorentz)

                # 3. Solve pressure correction
                self._solve_pressure()

                # 4. Solve induction equation
                self._solve_induction()

            # Compute residuals
            U_residual = self._compute_residual(self.U, U_old)
            B_residual = self._compute_residual(self.B, B_old)

            residuals = {
                "U": U_residual,
                "B": B_residual,
            }

            converged = convergence.update(step + 1, residuals)
            last_residual = max(U_residual, B_residual)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        # Write final fields
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info("mhdFoam completed")
        logger.info("  max|U| = %.6e",
                    (self.U ** 2).sum(dim=1).sqrt().max().item())
        logger.info("  max|B| = %.6e",
                    (self.B ** 2).sum(dim=1).sqrt().max().item())

        return {
            "converged": converged,
            "residual": last_residual,
        }

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, and B to a time directory."""
        time_str = f"{time:g}"
        if self._U_data is not None:
            self.write_field("U", self.U, time_str, self._U_data)
        if self._p_data is not None:
            self.write_field("p", self.p, time_str, self._p_data)
        if self._B_data is not None:
            self.write_field("B", self.B, time_str, self._B_data)
