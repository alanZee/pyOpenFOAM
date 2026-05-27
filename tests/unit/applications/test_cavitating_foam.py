"""
Test cavitatingFoam solver.

Uses a dam break case with cavitation two-phase properties.
Verifies:
- Case loading and field initialisation
- Cavitation model initialisation
- PISO inner loop execution
- Run produces finite, bounded fields
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------

def _make_cavitation_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    dx: float = 1.0,
    dy: float = 1.0,
    delta_t: float = 0.001,
    end_time: float = 0.005,
    n_outer: int = 2,
    n_correctors: int = 2,
) -> None:
    """Write a complete cavitatingFoam dam break case (no T field)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    dz = 0.1
    Lx = n_cells_x * dx
    Ly = n_cells_y * dy

    # ---- Points ----
    points_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)
    points_z1 = [
        (i * dx, j * dy, dz)
        for j in range(n_cells_y + 1)
        for i in range(n_cells_x + 1)
    ]
    all_points = points_z0 + points_z1
    n_points = len(all_points)

    # ---- Faces ----
    faces = []
    owner = []
    neighbour = []

    # Internal vertical (x-direction)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal (y-direction)
    for j in range(n_cells_y - 1):
        for i in range(n_cells_x):
            p0 = (j + 1) * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append((j + 1) * n_cells_x + i)

    n_internal = len(neighbour)

    # Boundary: top
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)
    n_top = n_cells_x
    top_start = n_internal

    # Bottom
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
    n_bottom = n_cells_x
    bottom_start = top_start + n_top

    # Left
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)
    n_left = n_cells_y
    left_start = bottom_start + n_bottom

    # Right
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)
    n_right = n_cells_y
    right_start = left_start + n_left

    # Front / back (empty)
    for _ in range(2):
        for j in range(n_cells_y):
            for i in range(n_cells_x):
                p0 = j * (n_cells_x + 1) + i
                p1 = p0 + 1
                p2 = p1 + n_cells_x + 1
                p3 = p0 + n_cells_x + 1
                faces.append((4, p0, p1, p2, p3))
                owner.append(j * n_cells_x + i)
    n_empty = 2 * n_cells_x * n_cells_y
    empty_start = right_start + n_right

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # ---- Write mesh ----
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location="constant/polyMesh",
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
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{face[0]}({verts})")
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
    for name, n_f, start in [
        ("topWall", n_top, top_start),
        ("bottomWall", n_bottom, bottom_start),
        ("leftWall", n_left, left_start),
        ("rightWall", n_right, right_start),
        ("frontAndBack", n_empty, empty_start),
    ]:
        lines.append(f"    {name}")
        lines.append("    {")
        if name == "frontAndBack":
            lines.append("        type            empty;")
        else:
            lines.append("        type            wall;")
        lines.append(f"        nFaces          {n_f};")
        lines.append(f"        startFace       {start};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- constant/transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    tp_body = (
        "nu1             [0 2 -1 0 0 0 0] 1e-6;\n"
        "nu2             [0 2 -1 0 0 0 0] 1.47e-5;\n"
        "rho1            [1 -3 0 0 0 0 0] 1000;\n"
        "rho2            [1 -3 0 0 0 0 0] 1.225;\n"
        "sigma           [1 0 -2 0 0 0 0] 0.07;\n"
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header, tp_body,
        overwrite=True,
    )

    # ---- constant/g ----
    g_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="uniformVectorField", location="constant", object="g",
    )
    write_foam_file(
        case_dir / "constant" / "g", g_header,
        "uniform (0 -9.81 0);\n", overwrite=True,
    )

    # ---- 0/U ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    topWall\n    {\n        type            slip;\n    }\n"
        "    bottomWall\n    {\n        type            noSlip;\n    }\n"
        "    leftWall\n    {\n        type            noSlip;\n    }\n"
        "    rightWall\n    {\n        type            noSlip;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # ---- 0/p_rgh ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p_rgh",
    )
    p_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    topWall\n    {\n        type            zeroGradient;\n    }\n"
        "    bottomWall\n    {\n        type            zeroGradient;\n    }\n"
        "    leftWall\n    {\n        type            zeroGradient;\n    }\n"
        "    rightWall\n    {\n        type            zeroGradient;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p_rgh", p_header, p_body, overwrite=True)

    # ---- 0/alpha.vapor ----
    alpha_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="alpha.vapor",
    )
    dam_x = Lx / 4.0
    dam_y = Ly / 2.0
    alpha_lines = [
        "dimensions      [0 0 0 0 0 0 0];",
        "",
        f"internalField   nonuniform {n_cells}",
        "(",
    ]
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            xc = (i + 0.5) * dx
            yc = (j + 0.5) * dy
            alpha_lines.append("1" if (xc < dam_x and yc < dam_y) else "0")
    alpha_lines += [
        ")",
        "",
        "boundaryField\n{",
    ]
    for bname in ["topWall", "bottomWall", "leftWall", "rightWall"]:
        alpha_lines += [f"    {bname}", "    {", "        type            zeroGradient;", "    }"]
    alpha_lines += ["    frontAndBack", "    {", "        type            empty;", "    }", "}"]
    write_foam_file(
        zero_dir / "alpha.vapor", alpha_header,
        "\n".join(alpha_lines) + "\n", overwrite=True,
    )

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     cavitatingFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n"
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
        "divSchemes\n{\n    default         none;\n"
        "    div(phi,alpha)  Gauss vanLeer;\n"
        "    div(phi,U)      Gauss upwind;\n"
        "}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)

    # ---- system/fvSolution ----
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
        "        maxIter         100;\n"
        "    }\n"
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "        maxIter         100;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        f"    nOuterCorrectors    {n_outer};\n"
        f"    nCorrectors         {n_correctors};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dam_break_case(tmp_path):
    """Create a cavitating dam break case (4x4 mesh)."""
    case_dir = tmp_path / "cavitationDamBreak"
    _make_cavitation_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        delta_t=0.001,
        end_time=0.003,
        n_outer=2,
        n_correctors=2,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCavitatingFoam:
    """Tests for the cavitatingFoam solver."""

    def test_case_loads(self, dam_break_case):
        """Solver loads the case and reads all required fields."""
        from pyfoam.applications.cavitating_foam import CavitatingFoam

        solver = CavitatingFoam(dam_break_case)

        assert solver.mesh.n_cells == 16
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)
        assert solver.alpha.shape == (16,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

    def test_cavitation_model_init(self, dam_break_case):
        """Cavitation model is initialised with correct parameters."""
        from pyfoam.applications.cavitating_foam import CavitatingFoam

        solver = CavitatingFoam(dam_break_case)

        assert solver.cavitation_model is not None
        assert solver.cavitation_model.p_v == 2300.0
        assert solver.cavitation_model.n_b == 1e13

    def test_run_produces_valid_fields(self, dam_break_case):
        """After running, all fields are finite and alpha is bounded."""
        from pyfoam.applications.cavitating_foam import CavitatingFoam

        solver = CavitatingFoam(dam_break_case)
        conv = solver.run()

        # Shapes preserved
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)
        assert solver.alpha.shape == (16,)

        # No NaN/Inf
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.alpha).all(), "alpha contains NaN/Inf"

        # Alpha bounded
        assert solver.alpha.min() >= -1e-6, f"alpha min = {solver.alpha.min()}"
        assert solver.alpha.max() <= 1.0 + 1e-6, f"alpha max = {solver.alpha.max()}"

    def test_piso_loop_executes(self, dam_break_case):
        """PISO inner loop runs and produces valid fields."""
        from pyfoam.applications.cavitating_foam import CavitatingFoam

        solver = CavitatingFoam(dam_break_case)

        # Give non-zero initial velocity so PISO has a driving force
        solver.U = torch.ones_like(solver.U) * 0.1

        # Run one time step
        U, p, alpha, phi, conv = solver._pimple_cavitation_iteration()

        # At least one outer iteration should have run
        assert conv.outer_iterations >= 1

        # All fields should be finite after PISO corrections
        assert torch.isfinite(U).all(), "U contains NaN/Inf after PISO"
        assert torch.isfinite(p).all(), "p contains NaN/Inf after PISO"
        assert torch.isfinite(phi).all(), "phi contains NaN/Inf after PISO"
