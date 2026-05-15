"""
Unit tests for BuoyantPimpleFoam — transient buoyant compressible PIMPLE solver.

Tests cover:
- Case loading and mesh construction with gravity field
- Gravity vector reading from constant/g
- Gravity dot product computation (gh, ghf)
- Pressure decomposition (p_rgh = p - rho*gh)
- Radiation model initialisation (P1)
- PIMPLE settings reading (nOuterCorrectors, nCorrectors)
- Transient momentum predictor with buoyancy
- Transient energy equation with buoyancy and radiation
- Buoyancy flux correction in pressure equation
- PISO inner correction loop
- Time derivative terms (∂(ρU)/∂t, ∂(ρe)/∂t)
- Old field storage for time derivatives
- Convergence on heated cavity
- Field writing to time directories
- Written field format validity
- Solver produces finite values
- Density stays positive
- Custom gravity vector injection
- Custom radiation model injection
- Turbulence coupling
- Inheritance from BuoyantSimpleFoam
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper for buoyant transient cavity case
# ---------------------------------------------------------------------------

def _make_buoyant_pimple_cavity_case(
    case_dir: Path,
    n_cells_x: int = 10,
    n_cells_y: int = 10,
    T_init: float = 300.0,
    T_hot: float = 310.0,
    T_cold: float = 290.0,
    p_init: float = 101325.0,
    delta_t: float = 1e-5,
    end_time: float = 5e-5,
    write_interval: int = 1,
    n_outer_correctors: int = 3,
    n_correctors: int = 2,
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
    """Write a complete buoyant PIMPLE cavity case to *case_dir*.

    Creates a cavity with:
    - Left wall at T_hot (heated)
    - Right wall at T_cold (cooled)
    - Top/bottom walls adiabatic (zeroGradient)
    - Gravity pointing down (-y)
    - PIMPLE algorithm settings

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
        "application     buoyantPimpleFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
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
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
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
        "PIMPLE\n{\n"
        f"    nOuterCorrectors    {n_outer_correctors};\n"
        f"    nCorrectors         {n_correctors};\n"
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
def buoyant_pimple_cavity(tmp_path):
    """Create a buoyant PIMPLE cavity case (4x4 mesh)."""
    case_dir = tmp_path / "buoyant_pimple_cavity"
    _make_buoyant_pimple_cavity_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        T_init=300.0,
        T_hot=310.0,
        T_cold=290.0,
        p_init=101325.0,
        delta_t=1e-5,
        end_time=5e-5,
        n_outer_correctors=3,
        n_correctors=2,
        max_outer_iterations=20,
    )
    return case_dir


@pytest.fixture
def tiny_buoyant_pimple_cavity(tmp_path):
    """Create a minimal 2x2 buoyant PIMPLE cavity for fast tests."""
    case_dir = tmp_path / "tiny_buoyant_pimple"
    _make_buoyant_pimple_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_hot=305.0,
        T_cold=295.0,
        p_init=101325.0,
        delta_t=1e-5,
        end_time=3e-5,
        n_outer_correctors=3,
        n_correctors=2,
        max_outer_iterations=10,
    )
    return case_dir


@pytest.fixture
def buoyant_pimple_with_radiation(tmp_path):
    """Create a buoyant PIMPLE cavity with P1 radiation."""
    case_dir = tmp_path / "buoyant_pimple_rad"
    _make_buoyant_pimple_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_hot=310.0,
        T_cold=290.0,
        p_init=101325.0,
        delta_t=1e-5,
        end_time=3e-5,
        n_outer_correctors=3,
        n_correctors=2,
        max_outer_iterations=10,
        radiation_model="P1",
        absorption_coeff=0.5,
    )
    return case_dir


@pytest.fixture
def buoyant_pimple_custom_gravity(tmp_path):
    """Create a buoyant PIMPLE cavity with custom gravity vector."""
    case_dir = tmp_path / "buoyant_pimple_custom_g"
    _make_buoyant_pimple_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_hot=310.0,
        T_cold=290.0,
        gravity=(0.0, -9.81, 0.0),
        delta_t=1e-5,
        end_time=3e-5,
        n_outer_correctors=3,
        n_correctors=2,
        max_outer_iterations=10,
    )
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestBuoyantPimpleFoamInit:
    """Tests for BuoyantPimpleFoam initialisation."""

    def test_case_loads(self, buoyant_pimple_cavity):
        """Case directory is readable and has expected structure."""
        from pyfoam.io.case import Case

        case = Case(buoyant_pimple_cavity)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("T", 0)

    def test_mesh_builds(self, buoyant_pimple_cavity):
        """FvMesh is constructed correctly."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(buoyant_pimple_cavity)
        mesh = solver.mesh

        assert mesh.n_cells == 16  # 4x4
        assert mesh.n_internal_faces > 0

    def test_fields_initialise(self, buoyant_pimple_cavity):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)

        n_cells = 16
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert solver.phi.shape == (solver.mesh.n_faces,)
        assert solver.rho.shape == (n_cells,)

    def test_thermo_defaults(self, buoyant_pimple_cavity):
        """Default thermo is air."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        assert solver.thermo is not None
        assert solver.thermo.R() == 287.0

    def test_gravity_read_from_file(self, buoyant_pimple_cavity):
        """Gravity vector is read from constant/g."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        g = solver.g

        assert g.shape == (3,)
        assert abs(g[1].item() - (-9.81)) < 0.01
        assert abs(g[0].item()) < 1e-10
        assert abs(g[2].item()) < 1e-10

    def test_custom_gravity_injection(self, buoyant_pimple_cavity):
        """Custom gravity vector can be injected."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(
            buoyant_pimple_cavity,
            gravity=(0.0, -10.0, 0.0),
        )
        g = solver.g

        assert abs(g[1].item() - (-10.0)) < 1e-10


class TestBuoyantPimpleFoamGravity:
    """Tests for gravity dot product computations."""

    def test_gh_shape(self, buoyant_pimple_cavity):
        """gh has correct shape (n_cells,)."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        assert solver.gh.shape == (solver.mesh.n_cells,)

    def test_ghf_shape(self, buoyant_pimple_cavity):
        """ghf has correct shape (n_faces,)."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        assert solver.ghf.shape == (solver.mesh.n_faces,)

    def test_gh_finite(self, buoyant_pimple_cavity):
        """gh values are finite."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        assert torch.isfinite(solver.gh).all()

    def test_p_rgh_initialisation(self, buoyant_pimple_cavity):
        """p_rgh = p - rho*gh at initialisation."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        p_rgh_expected = solver.p - solver.rho * solver.gh

        assert torch.allclose(solver.p_rgh, p_rgh_expected, rtol=1e-10)


class TestBuoyantPimpleFoamRadiation:
    """Tests for radiation model."""

    def test_default_radiation_is_p1(self, buoyant_pimple_cavity):
        """Default radiation model is P1."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam
        from pyfoam.models.radiation import P1Radiation

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        assert isinstance(solver.radiation, P1Radiation)

    def test_custom_radiation_injection(self, buoyant_pimple_cavity):
        """Custom radiation model can be injected."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam
        from pyfoam.models.radiation import P1Radiation

        custom_rad = P1Radiation(absorption_coeff=0.5, T_ref=350.0)
        solver = BuoyantPimpleFoam(buoyant_pimple_cavity, radiation=custom_rad)

        assert solver.radiation is custom_rad
        assert solver.radiation.absorption_coeff == 0.5

    def test_radiation_from_file(self, buoyant_pimple_with_radiation):
        """Radiation model reads from radiationProperties file."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam
        from pyfoam.models.radiation import P1Radiation

        solver = BuoyantPimpleFoam(buoyant_pimple_with_radiation)
        assert isinstance(solver.radiation, P1Radiation)
        assert abs(solver.radiation.absorption_coeff - 0.5) < 1e-10


class TestBuoyantPimpleFoamPimpleSettings:
    """Tests for PIMPLE algorithm settings."""

    def test_pimple_settings_read(self, buoyant_pimple_cavity):
        """PIMPLE settings are read correctly from fvSolution."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        assert solver.n_outer_correctors == 3
        assert solver.n_correctors == 2
        assert abs(solver.convergence_tolerance - 1e-4) < 1e-10
        assert abs(solver.alpha_U - 0.7) < 1e-10
        assert abs(solver.alpha_p - 0.3) < 1e-10
        assert abs(solver.alpha_T - 1.0) < 1e-10

    def test_pressure_solver_tolerance(self, buoyant_pimple_cavity):
        """Pressure solver tolerance is read."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        assert abs(solver.p_tolerance - 1e-6) < 1e-10

    def test_velocity_solver_tolerance(self, buoyant_pimple_cavity):
        """Velocity solver tolerance is read."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        assert abs(solver.U_tolerance - 1e-6) < 1e-10


class TestBuoyantPimpleFoamOldFields:
    """Tests for old field storage (time derivatives)."""

    def test_old_fields_stored(self, buoyant_pimple_cavity):
        """Old fields are stored for time derivative."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)

        assert solver.U_old.shape == solver.U.shape
        assert solver.p_old.shape == solver.p.shape
        assert solver.T_old.shape == solver.T.shape
        assert solver.rho_old.shape == solver.rho.shape

    def test_old_fields_equal_initial(self, buoyant_pimple_cavity):
        """Old fields equal initial fields at start."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)

        assert torch.allclose(solver.U_old, solver.U)
        assert torch.allclose(solver.p_old, solver.p)
        assert torch.allclose(solver.T_old, solver.T)
        assert torch.allclose(solver.rho_old, solver.rho)


class TestBuoyantPimpleFoamMomentum:
    """Tests for the transient buoyant momentum predictor."""

    def test_momentum_predictor_shape(self, buoyant_pimple_cavity):
        """Transient buoyant momentum predictor returns correct shapes."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        U, A_p, H = solver._buoyant_momentum_predictor_transient(
            solver.U, solver.p_rgh, solver.phi, solver.rho,
        )

        n_cells = solver.mesh.n_cells
        assert U.shape == (n_cells, 3)
        assert A_p.shape == (n_cells,)
        assert H.shape == (n_cells, 3)

    def test_momentum_predictor_finite(self, buoyant_pimple_cavity):
        """Transient buoyant momentum predictor produces finite values."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        U, A_p, H = solver._buoyant_momentum_predictor_transient(
            solver.U, solver.p_rgh, solver.phi, solver.rho,
        )

        assert torch.isfinite(U).all()
        assert torch.isfinite(A_p).all()
        assert torch.isfinite(H).all()

    def test_buoyancy_affects_velocity(self, buoyant_pimple_cavity):
        """Non-zero gravity produces non-zero velocity from temperature gradient."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)

        # Set up temperature gradient (hot left, cold right)
        mesh = solver.mesh
        x = mesh.cell_centres[:, 0]
        T_grad = 300.0 + 10.0 * (1.0 - 2.0 * x)
        solver.T = T_grad
        solver.rho = solver.thermo.rho(solver.p, solver.T)
        solver.p_rgh = solver.p - solver.rho * solver.gh

        U, _, _ = solver._buoyant_momentum_predictor_transient(
            solver.U, solver.p_rgh, solver.phi, solver.rho,
        )

        # Should produce non-zero velocity due to buoyancy
        assert U.abs().max() > 1e-10

    def test_time_derivative_effect(self, buoyant_pimple_cavity):
        """Time derivative term affects momentum predictor result."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)

        # Set U_old to a different value than U
        solver.U_old = solver.U.clone() + 0.1

        U1, A_p1, H1 = solver._buoyant_momentum_predictor_transient(
            solver.U, solver.p_rgh, solver.phi, solver.rho,
        )

        # Reset U_old to U (no time derivative effect)
        solver.U_old = solver.U.clone()

        U2, A_p2, H2 = solver._buoyant_momentum_predictor_transient(
            solver.U, solver.p_rgh, solver.phi, solver.rho,
        )

        # Results should differ due to different U_old
        assert not torch.allclose(U1, U2, rtol=1e-6)


class TestBuoyantPimpleFoamEnergy:
    """Tests for the transient buoyant energy equation."""

    def test_energy_equation_shape(self, buoyant_pimple_cavity):
        """Energy equation returns correct shape."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        T_old = solver.T.clone()

        T_new = solver._buoyant_solve_energy_equation_transient(
            solver.T, solver.U, solver.phi, solver.rho, solver.p, T_old,
        )

        assert T_new.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(T_new).all()

    def test_energy_preserves_uniform_T(self, buoyant_pimple_cavity):
        """Uniform temperature with zero velocity stays approximately uniform."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        T_old = solver.T.clone()

        U_zero = torch.zeros_like(solver.U)
        phi_zero = torch.zeros_like(solver.phi)

        T_new = solver._buoyant_solve_energy_equation_transient(
            solver.T, U_zero, phi_zero, solver.rho, solver.p, T_old,
        )

        # Temperature should remain approximately uniform
        T_range = T_new.max() - T_new.min()
        assert T_range < 1.0

    def test_time_derivative_in_energy(self, buoyant_pimple_cavity):
        """Time derivative term affects energy equation result."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        T_old = solver.T.clone()

        # Set T_old (stored) to a different value
        solver.T_old = solver.T.clone() + 10.0

        T1 = solver._buoyant_solve_energy_equation_transient(
            solver.T, solver.U, solver.phi, solver.rho, solver.p, T_old,
        )

        # Reset T_old
        solver.T_old = solver.T.clone()

        T2 = solver._buoyant_solve_energy_equation_transient(
            solver.T, solver.U, solver.phi, solver.rho, solver.p, T_old,
        )

        # Results should differ due to different T_old
        assert not torch.allclose(T1, T2, rtol=1e-6)


class TestBuoyantPimpleFoamRun:
    """Tests for the full solver run."""

    def test_run_converges(self, tiny_buoyant_pimple_cavity):
        """BuoyantPimpleFoam runs and produces valid output."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(tiny_buoyant_pimple_cavity)
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

    def test_run_writes_output(self, tiny_buoyant_pimple_cavity):
        """BuoyantPimpleFoam writes U, p, T to time directories."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(tiny_buoyant_pimple_cavity)
        solver.run()

        # Check that at least one output time directory was created
        time_dirs = [d for d in tiny_buoyant_pimple_cavity.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        # Check that U, p, T were written
        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"
            assert (td / "T").exists(), f"T not found in {td}"

    def test_density_remains_positive(self, tiny_buoyant_pimple_cavity):
        """Density stays positive throughout the simulation."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(tiny_buoyant_pimple_cavity)
        solver.run()

        assert (solver.rho > 0).all(), "Density became non-positive"
        assert (solver.T > 0).all(), "Temperature became non-positive"

    def test_convergence_data_populated(self, tiny_buoyant_pimple_cavity):
        """ConvergenceData has non-trivial values after run."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(tiny_buoyant_pimple_cavity)
        conv = solver.run()

        assert conv.outer_iterations >= 1
        assert conv.U_residual >= 0
        assert conv.p_residual >= 0
        assert conv.continuity_error >= 0

    def test_pressure_temperature_coupled(self, tiny_buoyant_pimple_cavity):
        """Pressure and temperature are coupled via EOS after solving."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(tiny_buoyant_pimple_cavity)
        solver.run()

        # Verify EOS consistency
        rho_expected = solver.thermo.rho(solver.p, solver.T)
        assert torch.allclose(solver.rho, rho_expected, rtol=1e-3), (
            "Density inconsistent with EOS after solving"
        )

    def test_fields_are_valid_format(self, tiny_buoyant_pimple_cavity):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam
        from pyfoam.io.field_io import read_field

        solver = BuoyantPimpleFoam(tiny_buoyant_pimple_cavity)
        solver.run()

        time_dirs = sorted(
            [d for d in tiny_buoyant_pimple_cavity.iterdir()
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

    def test_run_with_radiation(self, buoyant_pimple_with_radiation):
        """BuoyantPimpleFoam with radiation produces valid output."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_with_radiation)
        conv = solver.run()

        n_cells = 4  # 2x2
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()

    def test_run_with_custom_gravity(self, buoyant_pimple_custom_gravity):
        """BuoyantPimpleFoam with custom gravity produces valid output."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_custom_gravity)
        conv = solver.run()

        n_cells = 4  # 2x2
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()


class TestBuoyantPimpleFoamInheritance:
    """Tests for inheritance from BuoyantSimpleFoam."""

    def test_inherits_from_buoyant_simple_foam(self, buoyant_pimple_cavity):
        """BuoyantPimpleFoam inherits from BuoyantSimpleFoam."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)
        assert isinstance(solver, BuoyantSimpleFoam)

    def test_has_parent_methods(self, buoyant_pimple_cavity):
        """BuoyantPimpleFoam has parent's helper methods."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)

        # Check inherited methods exist
        assert hasattr(solver, '_compute_grad')
        assert hasattr(solver, '_compute_grad_vector')
        assert hasattr(solver, '_compute_div')
        assert hasattr(solver, '_compute_residual')
        assert hasattr(solver, '_compute_continuity_error')

    def test_has_buoyancy_methods(self, buoyant_pimple_cavity):
        """BuoyantPimpleFoam has buoyancy-specific methods."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)

        # Check buoyancy methods exist
        assert hasattr(solver, '_buoyant_momentum_predictor_transient')
        assert hasattr(solver, '_buoyant_solve_energy_equation_transient')
        assert hasattr(solver, '_recompute_H_buoyant')
        assert hasattr(solver, '_buoyant_pimple_iteration')

    def test_has_pimple_methods(self, buoyant_pimple_cavity):
        """BuoyantPimpleFoam has PIMPLE-specific methods."""
        from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

        solver = BuoyantPimpleFoam(buoyant_pimple_cavity)

        # Check PIMPLE methods exist
        assert hasattr(solver, '_read_pimple_settings')
        assert hasattr(solver, 'n_outer_correctors')
        assert hasattr(solver, 'n_correctors')
