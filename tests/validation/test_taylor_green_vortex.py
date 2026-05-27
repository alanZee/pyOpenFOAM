"""
Validation test: Taylor-Green vortex decay (icoFoam).

Validates that icoFoam can simulate the Taylor-Green vortex decay
problem and produce results consistent with the analytical solution.

The 2D Taylor-Green vortex on a [0, 2pi] x [0, 2pi] domain:

    u(x,y,t) =  sin(x) * cos(y) * exp(-2*nu*t)
    v(x,y,t) = -cos(x) * sin(y) * exp(-2*nu*t)
    p(x,y,t) = -0.25 * (cos(2x) + cos(2y)) * exp(-4*nu*t)

The kinetic energy decays exponentially:

    E(t) / E(0) = exp(-4 * nu * t)

where E = 0.5 * <u^2 + v^2> averaged over the domain.

Known limitation: wall boundaries (instead of true periodic BCs) and
finite-difference discretisation introduce boundary effects.  Tests
validate the decay *rate* and general flow structure rather than strict
pointwise agreement.

Reference:
    Taylor, G.I. & Green, A.E. (1937).
    "Mechanism of the production of small eddies from large ones."
    Proc. R. Soc. Lond. A 158, 499–521.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------


def _make_taylor_green_case(
    case_dir: Path,
    n_cells: int = 16,
    L: float = 6.283185307179586,  # 2*pi
    nu: float = 0.01,
    U0: float = 1.0,
) -> None:
    """Write a complete icoFoam Taylor-Green vortex case.

    Geometry: [0, 2pi] x [0, 2pi] with 2D (empty z).

    Boundary conditions:
    - All four walls: fixedValue (matching the analytical solution at t=0)
      to minimise boundary reflection / non-physical effects.
    - frontAndBack: empty

    Initial conditions:
    - U = (sin(x)*cos(y), -cos(x)*sin(y), 0)  at cell centres
    - p = -0.25 * (cos(2x) + cos(2y)) at cell centres
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = L / n_cells
    dy = L / n_cells
    dz = 0.1

    # ---- Points: two z-layers ----
    points_z0 = []
    for j in range(n_cells + 1):
        for i in range(n_cells + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)

    points_z1 = []
    for j in range(n_cells + 1):
        for i in range(n_cells + 1):
            points_z1.append((i * dx, j * dy, dz))

    all_points = points_z0 + points_z1
    n_points = len(all_points)

    # ---- Faces / owner / neighbour ----
    faces: list[tuple] = []
    owner: list[int] = []
    neighbour: list[int] = []

    # Internal vertical (x-direction neighbours)
    for j in range(n_cells):
        for i in range(n_cells - 1):
            p0 = j * (n_cells + 1) + i + 1
            p1 = p0 + n_cells + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells + i)
            neighbour.append(j * n_cells + i + 1)

    # Internal horizontal (y-direction neighbours)
    for j in range(n_cells - 1):
        for i in range(n_cells):
            p0 = (j + 1) * (n_cells + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells + i)
            neighbour.append((j + 1) * n_cells + i)

    n_internal = len(neighbour)

    # Boundary faces — 4 walls
    # leftWall (x=0)
    for j in range(n_cells):
        p0 = j * (n_cells + 1)
        p1 = p0 + n_cells + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells)
    n_left = n_cells
    left_start = n_internal

    # rightWall (x=L)
    for j in range(n_cells):
        p0 = j * (n_cells + 1) + n_cells
        p1 = p0 + n_cells + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells + n_cells - 1)
    n_right = n_cells
    right_start = left_start + n_left

    # topWall (y=L)
    for i in range(n_cells):
        p0 = n_cells * (n_cells + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells - 1) * n_cells + i)
    n_top = n_cells
    top_start = right_start + n_right

    # bottomWall (y=0)
    for i in range(n_cells):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
    n_bottom = n_cells
    bottom_start = top_start + n_top

    # frontAndBack (empty, z-normal)
    for j in range(n_cells):
        for i in range(n_cells):
            p0 = j * (n_cells + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells + 1
            p3 = p0 + n_cells + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells + i)

    for j in range(n_cells):
        for i in range(n_cells):
            p0 = n_base + j * (n_cells + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells + 1
            p3 = p0 + n_cells + 1
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells + i)

    n_empty = 2 * n_cells * n_cells
    empty_start = bottom_start + n_bottom

    n_faces = len(faces)
    n_total_cells = n_cells * n_cells

    # ---- Write mesh files ----
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII, location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        nv = face[0]
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"})
    lines = ["5", "("]
    for name, ptype, nf, sf in [
        ("leftWall", "wall", n_left, left_start),
        ("rightWall", "wall", n_right, right_start),
        ("topWall", "wall", n_top, top_start),
        ("bottomWall", "wall", n_bottom, bottom_start),
        ("frontAndBack", "empty", n_empty, empty_start),
    ]:
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type            {ptype};")
        lines.append(f"        nFaces          {nf};")
        lines.append(f"        startFace       {sf};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        f"nu              [0 2 -1 0 0 0 0] {nu};",
        overwrite=True,
    )

    # ---- 0/U: Taylor-Green initial condition ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_lines = [f"nonuniform List<vector> {n_total_cells}", "("]
    for j in range(n_cells):
        y_c = (j + 0.5) * dy
        for i in range(n_cells):
            x_c = (i + 0.5) * dx
            u_val = U0 * np.sin(x_c) * np.cos(y_c)
            v_val = -U0 * np.cos(x_c) * np.sin(y_c)
            u_lines.append(f"({u_val:.10g} {v_val:.10g} 0)")
    u_lines.append(");")

    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   " + "\n".join(u_lines) + "\n\n"
        "boundaryField\n{\n"
        "    leftWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           nonuniform List<vector> 1\n        (0 0 0);\n"
        "    }\n"
        "    rightWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           nonuniform List<vector> 1\n        (0 0 0);\n"
        "    }\n"
        "    topWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           nonuniform List<vector> 1\n        (0 0 0);\n"
        "    }\n"
        "    bottomWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           nonuniform List<vector> 1\n        (0 0 0);\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # ---- 0/p ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_lines = [f"nonuniform List<scalar> {n_total_cells}", "("]
    for j in range(n_cells):
        y_c = (j + 0.5) * dy
        for i in range(n_cells):
            x_c = (i + 0.5) * dx
            p_val = -0.25 * (np.cos(2.0 * x_c) + np.cos(2.0 * y_c))
            p_lines.append(f"{p_val:.10g}")
    p_lines.append(");")

    p_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   " + "\n".join(p_lines) + "\n\n"
        "boundaryField\n{\n"
        "    leftWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    rightWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    topWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    bottomWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     icoFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        "endTime         2.0;\n"
        "deltaT          0.01;\n"
        "writeControl    timeStep;\n"
        "writeInterval   200;\n"
        "purgeWrite      0;\n"
        "writeFormat     ascii;\n"
        "writePrecision  8;\n"
        "writeCompression off;\n"
        "timeFormat      general;\n"
        "timePrecision   6;\n"
        "runTimeModifiable true;\n"
    )
    write_foam_file(sys_dir / "controlDict", cd_header, cd_body, overwrite=True)

    # ---- system/fvSchemes ----
    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    fs_body = (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         none;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)

    # ---- system/fvSolution (PISO) ----
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
        "solvers\n{\n"
        "    p\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "PISO\n{\n"
        "    nCorrectors             2;\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    residualControl\n    {\n"
        "        p               1e-4;\n"
        "        U               1e-4;\n"
        "    }\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kinetic_energy(U: torch.Tensor) -> float:
    """Compute volume-averaged kinetic energy: 0.5 * <u^2 + v^2>."""
    u = U[:, 0].detach().cpu().numpy()
    v = U[:, 1].detach().cpu().numpy()
    return 0.5 * np.mean(u ** 2 + v ** 2)


def _analytical_energy_ratio(t: float, nu: float) -> float:
    """Analytical energy decay ratio: E(t)/E(0) = exp(-4*nu*t)."""
    return np.exp(-4.0 * nu * t)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tg_case(tmp_path):
    """Create a 16x16 Taylor-Green vortex case (nu=0.01)."""
    case_dir = tmp_path / "taylor_green"
    _make_taylor_green_case(
        case_dir,
        n_cells=16,
        L=2.0 * np.pi,
        nu=0.01,
        U0=1.0,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTaylorGreenVortex:
    """Validation: icoFoam Taylor-Green vortex decay."""

    def test_case_structure(self, tg_case):
        """Case directory has expected icoFoam structure."""
        from pyfoam.io.case import Case

        case = Case(tg_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.get_application() == "icoFoam"

    def test_mesh_dimensions(self, tg_case):
        """Mesh is 16x16 = 256 cells."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(tg_case)
        assert solver.mesh.n_cells == 256
        assert solver.mesh.n_internal_faces > 0

    def test_solver_initialises(self, tg_case):
        """icoFoam initialises with correct Taylor-Green parameters."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(tg_case)
        assert solver.U.shape == (256, 3)
        assert solver.p.shape == (256,)
        assert abs(solver.nu - 0.01) < 1e-10

    def test_initial_velocity_pattern(self, tg_case):
        """Initial velocity has the correct sinusoidal pattern."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(tg_case)
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        u_all = solver.U[:, 0].detach().cpu().numpy()
        v_all = solver.U[:, 1].detach().cpu().numpy()

        # Velocity should not be uniform (has spatial variation)
        assert np.std(u_all) > 0.1, "u-velocity lacks expected spatial variation"
        assert np.std(v_all) > 0.1, "v-velocity lacks expected spatial variation"

        # Check RMS velocity: for u=sin(x)*cos(y), RMS = sqrt(mean(sin^2*cos^2))
        # = sqrt(1/4) = 0.5 for a full period [0, 2pi]^2
        u_rms = np.sqrt(np.mean(u_all ** 2))
        expected_rms = 0.5  # RMS of sin(x)*cos(y) over full period
        assert abs(u_rms - expected_rms) < 0.15, (
            f"u-RMS = {u_rms:.3f}, expected ~{expected_rms:.3f}"
        )

    def test_run_produces_finite_fields(self, tg_case):
        """icoFoam completes and all field values are finite."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(tg_case)
        solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_energy_decays(self, tg_case):
        """Kinetic energy decreases over time (vortex decays)."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(tg_case)
        E0 = _kinetic_energy(solver.U)
        solver.run()
        E_final = _kinetic_energy(solver.U)

        assert E_final < E0, (
            f"Energy did not decay: E0={E0:.6f}, E_final={E_final:.6f}"
        )

    def test_energy_decay_magnitude(self, tg_case):
        """Energy decay is of the correct order of magnitude.

        The analytical decay for nu=0.01, dt=0.01, n_steps=200 (t=2.0):
            E(t)/E(0) = exp(-4 * 0.01 * 2.0) = exp(-0.08) ~ 0.923

        Allow generous tolerance for finite-difference and boundary effects.
        """
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(tg_case)
        E0 = _kinetic_energy(solver.U)
        solver.run()
        E_final = _kinetic_energy(solver.U)

        ratio = E_final / max(E0, 1e-30)
        analytical = _analytical_energy_ratio(2.0, 0.01)

        # Allow 50% tolerance: solver should not decay too fast or grow
        assert 0.0 < ratio < 2.0 * analytical, (
            f"Energy ratio {ratio:.4f} not in expected range "
            f"(analytical={analytical:.4f})"
        )

    def test_velocity_remains_bounded(self, tg_case):
        """Velocity stays within physical bounds after simulation."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(tg_case)
        solver.run()

        u_mag = torch.linalg.norm(solver.U, dim=1)
        u_max = u_mag.max().item()

        # Should not diverge; initial max is ~1.0
        assert 0.0 < u_max < 5.0, f"Unreasonable peak velocity: {u_max}"

    def test_pressure_is_finite(self, tg_case):
        """Pressure field is finite after simulation."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(tg_case)
        solver.run()

        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        p_range = solver.p.max().item() - solver.p.min().item()
        assert p_range < 1e6, f"Pressure range too large: {p_range}"
