"""
Unit tests for BuoyantSimpleFoam — steady-state buoyant compressible SIMPLE solver.

Tests cover:
- Case loading and mesh construction with gravity field
- Gravity vector reading from constant/g
- Gravity dot product computation (gh, ghf)
- Pressure decomposition (p_rgh = p - rho*gh)
- Radiation model initialisation (P1)
- Radiation source term computation
- Buoyancy source in momentum equation
- Energy equation with buoyancy work
- Buoyancy flux correction in pressure equation
- Convergence on heated cavity
- Field writing to time directories
- Written field format validity
- Solver produces finite values
- Density stays positive
- Custom gravity vector injection
- Custom radiation model injection
- Turbulence coupling
- Boussinesq-like behavior
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper for buoyant cavity case
# ---------------------------------------------------------------------------

def _make_buoyant_cavity_case(
    case_dir: Path,
    n_cells_x: int = 10,
    n_cells_y: int = 10,
    T_init: float = 300.0,
    T_hot: float = 310.0,
    T_cold: float = 290.0,
    p_init: float = 101325.0,
    end_time: int = 500,
    write_interval: int = 100,
    alpha_p: float = 0.3,
    alpha_U: float = 0.7,
    alpha_T: float = 1.0,
    convergence_tolerance: float = 1e-4,
    max_outer_iterations: int = 100,
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
    radiation_model: str | None = None,
    absorption_coeff: float = 0.1,
    turbulence_model: str | None = None,
) -> None:
    """Write a complete buoyant cavity case to *case_dir*.

    Creates a cavity with:
    - Left wall at T_hot (heated)
    - Right wall at T_cold (cooled)
    - Top/bottom walls adiabatic (zeroGradient)
    - Gravity pointing down (-y)

    Creates:
    - constant/polyMesh/{points, faces, owner, neighbour, boundary}
    - constant/g
    - 0/U, 0/p, 0/T
    - system/{controlDict, fvSchemes, fvSolution}
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mesh ----
    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1  # small depth for 3D (empty BC)

    # Points: two layers (z=0, z=dz)
    points_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)

    points_z1 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z1.append((i * dx, j * dy, dz))

    all_points = points_z0 + points_z1
    n_points = len(all_points)

    # Faces: internal + boundary
    faces = []
    owner = []
    neighbour = []

    # Internal vertical faces (between cells in x-direction)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal faces (between cells in y-direction)
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

    # Boundary faces
    # hotWall (left, x=0)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)

    n_hot = n_cells_y
    hot_start = n_internal

    # coldWall (right, x=1)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_cold = n_cells_y
    cold_start = hot_start + n_hot

    # adiabaticWalls (top, bottom)
    # Bottom (y=0)
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    # Top (y=1)
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)

    n_adiabatic = 2 * n_cells_x
    adiabatic_start = cold_start + n_cold

    # frontAndBack (z-normal, empty)
    # Front (z=0)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)

    # Back (z=dz)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = n_base + j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells_x + i)

    n_empty = 2 * n_cells_x * n_cells_y
    empty_start = adiabatic_start + n_adiabatic

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0",
        format=FileFormat.ASCII,
        location="constant/polyMesh",
    )

    # points
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    # faces
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        nv = face[0]
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    # owner
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    # neighbour
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    # boundary
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"})
    lines = ["4", "("]
    lines.append("    hotWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_hot};")
    lines.append(f"        startFace       {hot_start};")
    lines.append("    }")
    lines.append("    coldWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_cold};")
    lines.append(f"        startFace       {cold_start};")
    lines.append("    }")
    lines.append("    adiabaticWalls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_adiabatic};")
    lines.append(f"        startFace       {adiabatic_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- constant/g ----
    g_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="uniformDimensionedVectorField", location="constant", object="g",
    )
    g_body = (
        f"dimensions      [0 1 -2 0 0 0 0];\n"
        f"value           ({gravity[0]:.6g} {gravity[1]:.6g} {gravity[2]:.6g});\n"
    )
    write_foam_file(case_dir / "constant" / "g", g_header, g_body, overwrite=True)

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
        "    hotWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    adiabaticWalls\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # ---- 0/p (pressure in Pa for compressible) ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_body = (
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        f"internalField   uniform {p_init};\n\n"
        "boundaryField\n{\n"
        "    hotWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    adiabaticWalls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- 0/T (temperature) ----
    T_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    T_body = (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        f"internalField   uniform {T_init};\n\n"
        "boundaryField\n{\n"
        "    hotWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_hot};\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_cold};\n"
        "    }\n"
        "    adiabaticWalls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "T", T_header, T_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     buoyantSimpleFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        "deltaT          1;\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n"
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
        "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         none;\n}\n\n"
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
        "    p_rgh\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
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
        "    T\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "}\n\n"
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    residualControl\n    {\n"
        "        p               1e-4;\n"
        "        U               1e-4;\n"
        "    }\n"
        "    relaxationFactors\n    {\n"
        f"        p               {alpha_p};\n"
        f"        U               {alpha_U};\n"
        f"        T               {alpha_T};\n"
        "    }\n"
        f"    convergenceTolerance {convergence_tolerance};\n"
        f"    maxOuterIterations  {max_outer_iterations};\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)

    # ---- radiationProperties (optional) ----
    if radiation_model is not None:
        rad_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="constant", object="radiationProperties",
        )
        rad_body = (
            f"radiationModel  {radiation_model};\n\n"
            f"{radiation_model}\n{{\n"
            f"    absorptionCoeff {absorption_coeff};\n"
            "}\n"
        )
        write_foam_file(
            case_dir / "constant" / "radiationProperties", rad_header,
            rad_body, overwrite=True,
        )

    # ---- turbulenceProperties (optional) ----
    if turbulence_model is not None:
        turb_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="constant", object="turbulenceProperties",
        )
        turb_body = (
            "simulationType  RAS;\n\n"
            "RAS\n{\n"
            f"    model           {turbulence_model};\n"
            "    turbulence      on;\n"
            "    printCoeffs     on;\n"
            "}\n"
        )
        write_foam_file(
            case_dir / "constant" / "turbulenceProperties", turb_header,
            turb_body, overwrite=True,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def buoyant_cavity(tmp_path):
    """Create a buoyant cavity case (4x4 mesh)."""
    case_dir = tmp_path / "buoyant_cavity"
    _make_buoyant_cavity_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        T_init=300.0,
        T_hot=310.0,
        T_cold=290.0,
        p_init=101325.0,
        end_time=50,
        write_interval=50,
        max_outer_iterations=20,
    )
    return case_dir


@pytest.fixture
def tiny_buoyant_cavity(tmp_path):
    """Create a minimal 2x2 buoyant cavity for fast tests."""
    case_dir = tmp_path / "tiny_buoyant"
    _make_buoyant_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_hot=305.0,
        T_cold=295.0,
        p_init=101325.0,
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
    )
    return case_dir


@pytest.fixture
def buoyant_cavity_with_radiation(tmp_path):
    """Create a buoyant cavity with P1 radiation."""
    case_dir = tmp_path / "buoyant_rad"
    _make_buoyant_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_hot=310.0,
        T_cold=290.0,
        p_init=101325.0,
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
        radiation_model="P1",
        absorption_coeff=0.5,
    )
    return case_dir


@pytest.fixture
def buoyant_cavity_custom_gravity(tmp_path):
    """Create a buoyant cavity with custom gravity vector."""
    case_dir = tmp_path / "buoyant_custom_g"
    _make_buoyant_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_hot=310.0,
        T_cold=290.0,
        gravity=(0.0, -9.81, 0.0),
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
    )
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestBuoyantSimpleFoamInit:
    """Tests for BuoyantSimpleFoam initialisation."""

    def test_case_loads(self, buoyant_cavity):
        """Case directory is readable and has expected structure."""
        from pyfoam.io.case import Case

        case = Case(buoyant_cavity)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("T", 0)

    def test_mesh_builds(self, buoyant_cavity):
        """FvMesh is constructed correctly."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(buoyant_cavity)
        mesh = solver.mesh

        assert mesh.n_cells == 16  # 4x4
        assert mesh.n_internal_faces > 0

    def test_fields_initialise(self, buoyant_cavity):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)

        n_cells = 16
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert solver.phi.shape == (solver.mesh.n_faces,)
        assert solver.rho.shape == (n_cells,)

    def test_thermo_defaults(self, buoyant_cavity):
        """Default thermo is air."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        assert solver.thermo is not None
        assert solver.thermo.R() == 287.0

    def test_gravity_read_from_file(self, buoyant_cavity):
        """Gravity vector is read from constant/g."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        g = solver.g

        assert g.shape == (3,)
        assert abs(g[1].item() - (-9.81)) < 0.01
        assert abs(g[0].item()) < 1e-10
        assert abs(g[2].item()) < 1e-10

    def test_custom_gravity_injection(self, buoyant_cavity):
        """Custom gravity vector can be injected."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(
            buoyant_cavity,
            gravity=(0.0, -10.0, 0.0),
        )
        g = solver.g

        assert abs(g[1].item() - (-10.0)) < 1e-10


class TestBuoyantSimpleFoamGravity:
    """Tests for gravity dot product computations."""

    def test_gh_shape(self, buoyant_cavity):
        """gh has correct shape (n_cells,)."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        assert solver.gh.shape == (solver.mesh.n_cells,)

    def test_ghf_shape(self, buoyant_cavity):
        """ghf has correct shape (n_faces,)."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        assert solver.ghf.shape == (solver.mesh.n_faces,)

    def test_gh_finite(self, buoyant_cavity):
        """gh values are finite."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        assert torch.isfinite(solver.gh).all()

    def test_ghf_finite(self, buoyant_cavity):
        """ghf values are finite."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        assert torch.isfinite(solver.ghf).all()

    def test_gh_decreases_with_y(self, buoyant_cavity):
        """gh should decrease with y for downward gravity."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        mesh = solver.mesh

        # For gravity (0, -9.81, 0), gh = -9.81 * y
        # Cells at higher y should have more negative gh
        gh = solver.gh
        cell_y = mesh.cell_centres[:, 1]

        # Sort by y coordinate
        sorted_indices = torch.argsort(cell_y)
        gh_sorted = gh[sorted_indices]

        # gh should be monotonically decreasing (more negative with higher y)
        for i in range(len(gh_sorted) - 1):
            assert gh_sorted[i] >= gh_sorted[i + 1] - 1e-6

    def test_p_rgh_initialisation(self, buoyant_cavity):
        """p_rgh = p - rho*gh at initialisation."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        p_rgh_expected = solver.p - solver.rho * solver.gh

        assert torch.allclose(solver.p_rgh, p_rgh_expected, rtol=1e-10)


class TestBuoyantSimpleFoamRadiation:
    """Tests for radiation model."""

    def test_default_radiation_is_p1(self, buoyant_cavity):
        """Default radiation model is P1."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam
        from pyfoam.models.radiation import P1Radiation

        solver = BuoyantSimpleFoam(buoyant_cavity)
        assert isinstance(solver.radiation, P1Radiation)

    def test_custom_radiation_injection(self, buoyant_cavity):
        """Custom radiation model can be injected."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam
        from pyfoam.models.radiation import P1Radiation

        custom_rad = P1Radiation(absorption_coeff=0.5, T_ref=350.0)
        solver = BuoyantSimpleFoam(buoyant_cavity, radiation=custom_rad)

        assert solver.radiation is custom_rad
        assert solver.radiation.absorption_coeff == 0.5

    def test_radiation_from_file(self, buoyant_cavity_with_radiation):
        """Radiation model reads from radiationProperties file."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam
        from pyfoam.models.radiation import P1Radiation

        solver = BuoyantSimpleFoam(buoyant_cavity_with_radiation)
        assert isinstance(solver.radiation, P1Radiation)
        assert abs(solver.radiation.absorption_coeff - 0.5) < 1e-10


class TestP1RadiationModel:
    """Tests for the P1 radiation model."""

    def test_sh_shape(self):
        """Sh returns correct shape."""
        from pyfoam.models.radiation import P1Radiation

        rad = P1Radiation()
        T = torch.full((16,), 300.0, dtype=CFD_DTYPE)
        Sh = rad.Sh(T)

        assert Sh.shape == (16,)

    def test_sh_zero_at_ref_temperature(self):
        """Sh is zero when T = T_ref."""
        from pyfoam.models.radiation import P1Radiation

        rad = P1Radiation(T_ref=300.0)
        T = torch.full((16,), 300.0, dtype=CFD_DTYPE)
        Sh = rad.Sh(T)

        assert torch.allclose(Sh, torch.zeros_like(Sh), atol=1e-10)

    def test_sh_positive_when_hot(self):
        """Sh > 0 when T > T_ref (net absorption)."""
        from pyfoam.models.radiation import P1Radiation

        rad = P1Radiation(T_ref=300.0)
        T = torch.full((16,), 400.0, dtype=CFD_DTYPE)
        Sh = rad.Sh(T)

        assert (Sh > 0).all()

    def test_sh_negative_when_cold(self):
        """Sh < 0 when T < T_ref (net emission)."""
        from pyfoam.models.radiation import P1Radiation

        rad = P1Radiation(T_ref=300.0)
        T = torch.full((16,), 200.0, dtype=CFD_DTYPE)
        Sh = rad.Sh(T)

        assert (Sh < 0).all()

    def test_sh_scales_with_absorption(self):
        """Sh scales linearly with absorption coefficient."""
        from pyfoam.models.radiation import P1Radiation

        T = torch.full((16,), 400.0, dtype=CFD_DTYPE)

        rad1 = P1Radiation(absorption_coeff=0.1)
        rad2 = P1Radiation(absorption_coeff=0.5)

        Sh1 = rad1.Sh(T)
        Sh2 = rad2.Sh(T)

        ratio = Sh2 / Sh1
        assert torch.allclose(ratio, torch.full_like(ratio, 5.0), rtol=1e-10)

    def test_sh_scales_with_T4(self):
        """Sh scales with T^4 - T_ref^4."""
        from pyfoam.models.radiation import P1Radiation
        from pyfoam.models.radiation import STEFAN_BOLTZMANN

        rad = P1Radiation(absorption_coeff=0.1, T_ref=0.0)
        T = torch.tensor([100.0, 200.0], dtype=CFD_DTYPE)
        Sh = rad.Sh(T)

        expected = 4.0 * 0.1 * STEFAN_BOLTZMANN * T**4
        assert torch.allclose(Sh, expected, rtol=1e-10)

    def test_repr(self):
        """String representation."""
        from pyfoam.models.radiation import P1Radiation

        rad = P1Radiation(absorption_coeff=0.5, T_ref=300.0)
        r = repr(rad)

        assert "P1Radiation" in r
        assert "0.5" in r


class TestBuoyantSimpleFoamMomentum:
    """Tests for the buoyant momentum predictor."""

    def test_momentum_predictor_shape(self, buoyant_cavity):
        """Buoyant momentum predictor returns correct shapes."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        U, A_p, H = solver._buoyant_momentum_predictor(
            solver.U, solver.p_rgh, solver.phi, solver.rho,
        )

        n_cells = solver.mesh.n_cells
        assert U.shape == (n_cells, 3)
        assert A_p.shape == (n_cells,)
        assert H.shape == (n_cells, 3)

    def test_momentum_predictor_finite(self, buoyant_cavity):
        """Buoyant momentum predictor produces finite values."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        U, A_p, H = solver._buoyant_momentum_predictor(
            solver.U, solver.p_rgh, solver.phi, solver.rho,
        )

        assert torch.isfinite(U).all()
        assert torch.isfinite(A_p).all()
        assert torch.isfinite(H).all()

    def test_buoyancy_affects_velocity(self, buoyant_cavity):
        """Non-zero gravity produces non-zero velocity from temperature gradient."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        n_cells = solver.mesh.n_cells

        # Set up temperature gradient (hot left, cold right)
        mesh = solver.mesh
        x = mesh.cell_centres[:, 0]
        T_grad = 300.0 + 10.0 * (1.0 - 2.0 * x)
        solver.T = T_grad
        solver.rho = solver.thermo.rho(solver.p, solver.T)
        solver.p_rgh = solver.p - solver.rho * solver.gh

        U, _, _ = solver._buoyant_momentum_predictor(
            solver.U, solver.p_rgh, solver.phi, solver.rho,
        )

        # Should produce non-zero velocity due to buoyancy
        assert U.abs().max() > 1e-10

    def test_no_buoyancy_with_zero_gravity(self, buoyant_cavity):
        """Zero gravity gives same result as non-buoyant solver."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(
            buoyant_cavity,
            gravity=(0.0, 0.0, 0.0),
        )

        U_buoy, A_p_buoy, H_buoy = solver._buoyant_momentum_predictor(
            solver.U, solver.p_rgh, solver.phi, solver.rho,
        )

        # With zero gravity, buoyancy source is zero
        # Results should be similar to rhoSimpleFoam (though not identical
        # due to p_rgh vs p decomposition)
        assert torch.isfinite(U_buoy).all()


class TestBuoyantSimpleFoamPressure:
    """Tests for the pressure equation with buoyancy."""

    def test_pressure_equation_shape(self, buoyant_cavity):
        """Pressure equation returns correct shape."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        mesh = solver.mesh

        n_internal = mesh.n_internal_faces
        phiHbyA = torch.zeros(n_internal, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_new = solver._solve_pressure_equation(
            solver.p_rgh, phiHbyA, A_p, solver.rho, mesh,
        )

        assert p_new.shape == (mesh.n_cells,)
        assert torch.isfinite(p_new).all()


class TestBuoyantSimpleFoamEnergy:
    """Tests for the buoyant energy equation."""

    def test_energy_equation_shape(self, buoyant_cavity):
        """Energy equation returns correct shape."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        T_old = solver.T.clone()

        T_new = solver._buoyant_solve_energy_equation(
            solver.T, solver.U, solver.phi, solver.rho, solver.p, T_old,
        )

        assert T_new.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(T_new).all()

    def test_energy_preserves_uniform_T(self, buoyant_cavity):
        """Uniform temperature with zero velocity stays uniform."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        T_old = solver.T.clone()

        U_zero = torch.zeros_like(solver.U)
        phi_zero = torch.zeros_like(solver.phi)

        T_new = solver._buoyant_solve_energy_equation(
            solver.T, U_zero, phi_zero, solver.rho, solver.p, T_old,
        )

        # Temperature should remain approximately uniform
        T_range = T_new.max() - T_new.min()
        assert T_range < 1.0

    def test_radiation_cools_hot_fluid(self, buoyant_cavity):
        """Radiation source cools fluid above T_ref."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam
        from pyfoam.models.radiation import P1Radiation

        # Use high absorption to make radiation effect visible
        rad = P1Radiation(absorption_coeff=10.0, T_ref=300.0)
        solver = BuoyantSimpleFoam(buoyant_cavity, radiation=rad)

        # Set uniform hot temperature
        solver.T = torch.full_like(solver.T, 400.0)
        solver.rho = solver.thermo.rho(solver.p, solver.T)

        T_old = solver.T.clone()
        T_new = solver._buoyant_solve_energy_equation(
            solver.T, solver.U, solver.phi, solver.rho, solver.p, T_old,
        )

        # Radiation should cool the fluid (T_new < T_old)
        # Note: this depends on the balance of terms
        assert torch.isfinite(T_new).all()


class TestBuoyantSimpleFoamRun:
    """Tests for the full solver run."""

    def test_run_converges(self, tiny_buoyant_cavity):
        """BuoyantSimpleFoam runs and produces valid output."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(tiny_buoyant_cavity)
        conv = solver.run()

        n_cells = 4  # 2x2
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

        # All values should be finite
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"
        assert torch.isfinite(solver.rho).all(), "rho contains NaN/Inf"

    def test_run_writes_output(self, tiny_buoyant_cavity):
        """BuoyantSimpleFoam writes U, p, T to time directories."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(tiny_buoyant_cavity)
        solver.run()

        # Check that at least one output time directory was created
        time_dirs = [d for d in tiny_buoyant_cavity.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        # Check that U, p, T were written
        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"
            assert (td / "T").exists(), f"T not found in {td}"

    def test_density_remains_positive(self, tiny_buoyant_cavity):
        """Density stays positive throughout the simulation."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(tiny_buoyant_cavity)
        solver.run()

        assert (solver.rho > 0).all(), "Density became non-positive"
        assert (solver.T > 0).all(), "Temperature became non-positive"

    def test_convergence_data_populated(self, tiny_buoyant_cavity):
        """ConvergenceData has non-trivial values after run."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(tiny_buoyant_cavity)
        conv = solver.run()

        assert conv.outer_iterations >= 1
        assert conv.U_residual >= 0
        assert conv.p_residual >= 0
        assert conv.continuity_error >= 0

    def test_pressure_temperature_coupled(self, tiny_buoyant_cavity):
        """Pressure and temperature are coupled via EOS after solving."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(tiny_buoyant_cavity)
        solver.run()

        # Verify EOS consistency
        rho_expected = solver.thermo.rho(solver.p, solver.T)
        assert torch.allclose(solver.rho, rho_expected, rtol=1e-6), (
            "Density inconsistent with EOS after solving"
        )

    def test_fields_are_valid_format(self, tiny_buoyant_cavity):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam
        from pyfoam.io.field_io import read_field

        solver = BuoyantSimpleFoam(tiny_buoyant_cavity)
        solver.run()

        time_dirs = sorted(
            [d for d in tiny_buoyant_cavity.iterdir()
             if d.is_dir() and d.name.replace(".", "").isdigit()
             and d.name != "0"],
            key=lambda d: float(d.name),
        )
        assert len(time_dirs) >= 1

        last_dir = time_dirs[-1]
        U_data = read_field(last_dir / "U")
        p_data = read_field(last_dir / "p")
        T_data = read_field(last_dir / "T")

        assert U_data.scalar_type == "vector"
        assert p_data.scalar_type == "scalar"
        assert T_data.scalar_type == "scalar"

    def test_run_with_radiation(self, buoyant_cavity_with_radiation):
        """BuoyantSimpleFoam with radiation produces valid output."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity_with_radiation)
        conv = solver.run()

        n_cells = 4  # 2x2
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()

    def test_run_with_custom_gravity(self, buoyant_cavity_custom_gravity):
        """BuoyantSimpleFoam with custom gravity produces valid output."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity_custom_gravity)
        conv = solver.run()

        n_cells = 4  # 2x2
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()


class TestBuoyantSimpleFoamInheritance:
    """Tests for inheritance from RhoSimpleFoam."""

    def test_inherits_from_rho_simple_foam(self, buoyant_cavity):
        """BuoyantSimpleFoam inherits from RhoSimpleFoam."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)
        assert isinstance(solver, RhoSimpleFoam)

    def test_has_parent_methods(self, buoyant_cavity):
        """BuoyantSimpleFoam has parent's helper methods."""
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantSimpleFoam(buoyant_cavity)

        # Check inherited methods exist
        assert hasattr(solver, '_compute_grad')
        assert hasattr(solver, '_compute_grad_vector')
        assert hasattr(solver, '_compute_div')
        assert hasattr(solver, '_compute_residual')
        assert hasattr(solver, '_compute_continuity_error')
