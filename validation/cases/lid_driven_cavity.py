"""
Lid-driven cavity validation case.

Classic CFD benchmark: square cavity with a moving top wall.

- All walls: no-slip (u=0, v=0)
- Top wall (y=H): u=U_lid, v=0 (moving lid)
- No pressure gradient (closed cavity)

This case uses the SIMPLE solver from pyfoam.solvers to solve the
steady-state Navier-Stokes equations (including convection).

Validation compares against Ghia, Ghia & Shin (1982) benchmark data:
  Ghia, U., Ghia, K.N. and Shin, C.T. (1982). "High-Re solutions for
  incompressible flow using the Navier-Stokes equations and a multigrid
  method." Journal of Computational Physics, 48(3), 387-411.
  DOI: 10.1016/0021-9991(82)90058-4

Usage::

    case = LidDrivenCavityCase(n_cells=32, Re=100.0)
    case.setup()
    case.run()
    ref = case.get_reference()
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.mesh import PolyMesh, FvMesh
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from validation.runner import ValidationCaseBase

logger = logging.getLogger(__name__)


# Ghia et al. (1982) benchmark data for Re=100
GHIA_RE100_U_VCENTRELINE = [
    (0.0000, 0.0000), (0.0547, -0.03717), (0.0625, -0.04192),
    (0.0703, -0.04775), (0.1016, -0.06434), (0.1719, -0.10150),
    (0.2813, -0.15662), (0.4531, -0.21090), (0.5000, -0.20581),
    (0.6172, -0.13641), (0.7344, -0.00332), (0.8516, 0.23151),
    (0.9531, 0.68717), (0.9609, 0.73722), (0.9688, 0.78871),
    (0.9766, 0.84123), (1.0000, 1.00000),
]

GHIA_RE100_V_HCENTRELINE = [
    (0.0000, 0.0000), (0.0625, 0.09233), (0.0703, 0.10091),
    (0.0781, 0.10890), (0.0938, 0.12317), (0.1563, 0.16077),
    (0.2266, 0.17507), (0.2344, 0.17527), (0.5000, 0.05454),
    (0.8047, -0.24533), (0.8594, -0.22445), (0.9063, -0.16914),
    (0.9453, -0.10313), (0.9531, -0.08864), (0.9609, -0.07391),
    (0.9688, -0.05906), (1.0000, 0.00000),
]


class LidDrivenCavityCase(ValidationCaseBase):
    """Lid-driven cavity validation case using SIMPLE solver."""

    def __init__(
        self,
        n_cells: int = 32,
        Re: float = 100.0,
        U_lid: float = 1.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-4,
    ) -> None:
        self.n_cells = n_cells
        self.Re = Re
        self.U_lid = U_lid
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.nu = U_lid * 1.0 / Re

        self._mesh = None
        self._U_computed = None
        self._p_computed = None
        self._solver_info = {}

    @property
    def name(self) -> str:
        return "Lid-Driven Cavity"

    @property
    def description(self) -> str:
        return f"Lid-driven cavity: Re={self.Re:.0f}, U_lid={self.U_lid}, mesh={self.n_cells}x{self.n_cells}"

    def setup(self) -> None:
        """Build a proper 3D mesh for the cavity."""
        N = self.n_cells
        dtype = torch.float64
        device = torch.device("cpu")

        nx = ny = N
        dx = dy = 1.0 / N
        dz = 0.1  # Small thickness in z

        # Create points for a 3D extruded mesh
        # We need (nx+1)*(ny+1)*2 points (front and back planes)
        n_points = (nx + 1) * (ny + 1) * 2
        points = torch.zeros(n_points, 3, dtype=dtype, device=device)

        # Front plane (z=0)
        for j in range(ny + 1):
            for i in range(nx + 1):
                idx = j * (nx + 1) + i
                points[idx, 0] = i * dx
                points[idx, 1] = j * dy
                points[idx, 2] = 0.0

        # Back plane (z=dz)
        offset = (nx + 1) * (ny + 1)
        for j in range(ny + 1):
            for i in range(nx + 1):
                idx = offset + j * (nx + 1) + i
                points[idx, 0] = i * dx
                points[idx, 1] = j * dy
                points[idx, 2] = dz

        # Helper function to add faces with proper vertex ordering
        def add_face(p0, p1, p2, p3, own, neigh=None):
            """Add a quadrilateral face with counter-clockwise ordering when viewed from outside."""
            faces_list.append(torch.tensor([p0, p1, p2, p3], dtype=torch.long, device=device))
            owner_list.append(own)
            if neigh is not None:
                neighbour_list.append(neigh)

        faces_list = []
        owner_list = []
        neighbour_list = []

        # Internal vertical faces (between columns)
        # Face between cell (i,j) and (i+1,j) at x = (i+1)*dx
        for j in range(ny):
            for i in range(nx - 1):
                cell_left = j * nx + i
                cell_right = j * nx + (i + 1)

                # Vertices for vertical face at x = (i+1)*dx
                # Bottom-front, top-front, top-back, bottom-back
                p0 = j * (nx + 1) + (i + 1)  # front-bottom
                p1 = (j + 1) * (nx + 1) + (i + 1)  # front-top
                p2 = offset + (j + 1) * (nx + 1) + (i + 1)  # back-top
                p3 = offset + j * (nx + 1) + (i + 1)  # back-bottom

                add_face(p0, p1, p2, p3, cell_left, cell_right)

        # Internal horizontal faces (between rows)
        # Face between cell (i,j) and (i,j+1) at y = (j+1)*dy
        for j in range(ny - 1):
            for i in range(nx):
                cell_bottom = j * nx + i
                cell_top = (j + 1) * nx + i

                # Vertices for horizontal face at y = (j+1)*dy
                p0 = (j + 1) * (nx + 1) + i  # front-left
                p1 = (j + 1) * (nx + 1) + i + 1  # front-right
                p2 = offset + (j + 1) * (nx + 1) + i + 1  # back-right
                p3 = offset + (j + 1) * (nx + 1) + i  # back-left

                add_face(p0, p1, p2, p3, cell_bottom, cell_top)

        n_internal = len(faces_list)

        # Boundary faces
        boundary_patches = []

        # Bottom wall (y=0, j=0)
        start_face = len(faces_list)
        for i in range(nx):
            p0 = i
            p1 = offset + i
            p2 = offset + i + 1
            p3 = i + 1
            add_face(p0, p1, p2, p3, i)
        boundary_patches.append({"name": "bottomWall", "type": "wall", "startFace": start_face, "nFaces": nx})

        # Top wall (y=1, j=ny) - moving lid
        start_face = len(faces_list)
        for i in range(nx):
            cell_idx = (ny - 1) * nx + i
            p0 = ny * (nx + 1) + i
            p1 = ny * (nx + 1) + i + 1
            p2 = offset + ny * (nx + 1) + i + 1
            p3 = offset + ny * (nx + 1) + i
            add_face(p0, p1, p2, p3, cell_idx)
        boundary_patches.append({"name": "topWall", "type": "wall", "startFace": start_face, "nFaces": nx})

        # Left wall (x=0, i=0)
        start_face = len(faces_list)
        for j in range(ny):
            cell_idx = j * nx
            p0 = j * (nx + 1)
            p1 = offset + j * (nx + 1)
            p2 = offset + (j + 1) * (nx + 1)
            p3 = (j + 1) * (nx + 1)
            add_face(p0, p1, p2, p3, cell_idx)
        boundary_patches.append({"name": "leftWall", "type": "wall", "startFace": start_face, "nFaces": ny})

        # Right wall (x=1, i=nx)
        start_face = len(faces_list)
        for j in range(ny):
            cell_idx = j * nx + (nx - 1)
            p0 = j * (nx + 1) + nx
            p1 = (j + 1) * (nx + 1) + nx
            p2 = offset + (j + 1) * (nx + 1) + nx
            p3 = offset + j * (nx + 1) + nx
            add_face(p0, p1, p2, p3, cell_idx)
        boundary_patches.append({"name": "rightWall", "type": "wall", "startFace": start_face, "nFaces": ny})

        # Front wall (z=0)
        start_face = len(faces_list)
        for j in range(ny):
            for i in range(nx):
                cell_idx = j * nx + i
                p0 = j * (nx + 1) + i
                p1 = j * (nx + 1) + i + 1
                p2 = (j + 1) * (nx + 1) + i + 1
                p3 = (j + 1) * (nx + 1) + i
                add_face(p0, p1, p2, p3, cell_idx)
        boundary_patches.append({"name": "frontWall", "type": "wall", "startFace": start_face, "nFaces": nx * ny})

        # Back wall (z=dz)
        start_face = len(faces_list)
        for j in range(ny):
            for i in range(nx):
                cell_idx = j * nx + i
                p0 = offset + j * (nx + 1) + i
                p1 = offset + (j + 1) * (nx + 1) + i
                p2 = offset + (j + 1) * (nx + 1) + i + 1
                p3 = offset + j * (nx + 1) + i + 1
                add_face(p0, p1, p2, p3, cell_idx)
        boundary_patches.append({"name": "backWall", "type": "wall", "startFace": start_face, "nFaces": nx * ny})

        # Convert to tensors
        owner_tensor = torch.tensor(owner_list, dtype=torch.long, device=device)
        neighbour_tensor = torch.tensor(neighbour_list, dtype=torch.long, device=device)

        # Create mesh
        poly = PolyMesh(
            points=points,
            faces=faces_list,
            owner=owner_tensor,
            neighbour=neighbour_tensor,
            boundary=boundary_patches,
        )
        self._mesh = FvMesh.from_poly_mesh(poly)

        # Verify mesh quality
        min_vol = self._mesh.cell_volumes.min().item()
        max_vol = self._mesh.cell_volumes.max().item()
        logger.info(f"Mesh: {self._mesh.n_cells} cells, volumes: min={min_vol:.6e}, max={max_vol:.6e}")

        if min_vol < 1e-10:
            logger.warning("Mesh has near-zero cell volumes! Check face definitions.")

        # Initialize fields
        n_cells = nx * ny
        U = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        p = torch.zeros(n_cells, dtype=dtype, device=device)
        phi = torch.zeros(len(faces_list), dtype=dtype, device=device)

        # Set lid velocity on top wall cells
        for i in range(nx):
            idx = (ny - 1) * nx + i
            U[idx, 0] = self.U_lid

        # Create boundary condition tensor (NaN = no BC, value = fixed BC)
        U_bc = torch.full((n_cells, 3), float('nan'), dtype=dtype, device=device)

        # Bottom wall cells: U = (0, 0, 0)
        for i in range(nx):
            U_bc[i, 0] = 0.0
            U_bc[i, 1] = 0.0
            U_bc[i, 2] = 0.0

        # Left wall cells: U = (0, 0, 0)
        for j in range(ny):
            idx = j * nx
            U_bc[idx, 0] = 0.0
            U_bc[idx, 1] = 0.0
            U_bc[idx, 2] = 0.0

        # Right wall cells: U = (0, 0, 0)
        for j in range(ny):
            idx = j * nx + (nx - 1)
            U_bc[idx, 0] = 0.0
            U_bc[idx, 1] = 0.0
            U_bc[idx, 2] = 0.0

        # Top wall cells: U = (U_lid, 0, 0) - set LAST to override corners
        for i in range(nx):
            idx = (ny - 1) * nx + i
            U_bc[idx, 0] = self.U_lid
            U_bc[idx, 1] = 0.0
            U_bc[idx, 2] = 0.0

        self._U_init = U.clone()
        self._p_init = p.clone()
        self._phi_init = phi.clone()
        self._U_bc = U_bc
        self._nx = nx
        self._ny = ny

        logger.info(f"Lid-driven cavity setup: {nx}x{ny} mesh, Re={self.Re:.0f}, nu={self.nu:.6e}")

    def run(self) -> dict[str, Any]:
        """Run the SIMPLE solver for lid-driven cavity."""
        mesh = self._mesh
        dtype = torch.float64
        device = torch.device("cpu")

        config = SIMPLEConfig(
            relaxation_factor_U=0.7,
            relaxation_factor_p=0.02,
            nu=self.nu,
            consistent=False,
            p_tolerance=1e-6,
            p_max_iter=50,
        )
        solver = SIMPLESolver(mesh, config)

        U, p, phi, convergence = solver.solve(
            self._U_init.clone(),
            self._p_init.clone(),
            self._phi_init.clone(),
            U_bc=self._U_bc,
            max_outer_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

        self._U_computed = U
        self._p_computed = p

        self._solver_info = {
            "iterations": convergence.outer_iterations,
            "converged": convergence.converged,
            "final_residual": convergence.continuity_error,
        }

        return self._solver_info

    def get_reference(self) -> dict[str, torch.Tensor]:
        """Compute reference solution from Ghia et al. (1982) data."""
        ny = self._ny
        nx = self._nx
        dtype = torch.float64
        n_cells = nx * ny

        U_ref = torch.zeros(n_cells, 3, dtype=dtype)

        ghia_y = [d[0] for d in GHIA_RE100_U_VCENTRELINE]
        ghia_u = [d[1] for d in GHIA_RE100_U_VCENTRELINE]

        for j in range(ny):
            y = (j + 0.5) / ny
            u_interp = self._interpolate(y, ghia_y, ghia_u) * self.U_lid
            for i in range(nx):
                idx = j * nx + i
                U_ref[idx, 0] = u_interp

        ghia_x = [d[0] for d in GHIA_RE100_V_HCENTRELINE]
        ghia_v = [d[1] for d in GHIA_RE100_V_HCENTRELINE]

        j_mid = ny // 2
        for i in range(nx):
            x = (i + 0.5) / nx
            v_interp = self._interpolate(x, ghia_x, ghia_v) * self.U_lid
            idx = j_mid * nx + i
            U_ref[idx, 1] = v_interp

        return {"U": U_ref}

    def get_computed(self) -> dict[str, torch.Tensor]:
        return {"U": self._U_computed.clone()}

    def get_tolerances(self) -> dict[str, float]:
        return {"l2_tol": 0.15, "max_tol": 0.15}

    @staticmethod
    def _interpolate(x, x_data, y_data):
        if x <= x_data[0]:
            return y_data[0]
        if x >= x_data[-1]:
            return y_data[-1]
        for i in range(len(x_data) - 1):
            if x_data[i] <= x <= x_data[i + 1]:
                t = (x - x_data[i]) / (x_data[i + 1] - x_data[i])
                return y_data[i] + t * (y_data[i + 1] - y_data[i])
        return y_data[-1]
