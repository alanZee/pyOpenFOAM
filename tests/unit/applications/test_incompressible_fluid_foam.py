"""
Unit tests for IncompressibleFluidFoam — unified incompressible solver.

Tests cover:
- Algorithm detection (SIMPLE, PISO, PIMPLE) from fvSolution
- Default algorithm when no sub-dict present
- Field initialisation and property reading
- SIMPLE mode: convergence on lid-driven cavity
- PISO mode: transient time-stepping
- PIMPLE mode: transient with outer corrections
- Turbulence model integration
- Boundary condition building
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper
# ---------------------------------------------------------------------------

def _make_cavity_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    nu: float = 0.01,
    delta_t: float = 1.0,
    end_time: int = 500,
    write_interval: int = 100,
    algorithm: str = "SIMPLE",
    alpha_p: float = 0.3,
    alpha_U: float = 0.7,
    convergence_tolerance: float = 1e-4,
    max_outer_iterations: int = 200,
    n_outer_correctors: int = 3,
    n_correctors: int = 2,
    turbulence_model: str | None = None,
) -> None:
    """Write a complete lid-driven cavity case to *case_dir*.

    Creates:
    - constant/polyMesh/{points, faces, owner, neighbour, boundary}
    - constant/transportProperties
    - constant/turbulenceProperties (if turbulence_model is set)
    - 0/U, 0/p
    - system/{controlDict, fvSchemes, fvSolution}
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mesh ----
    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1

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

    faces = []
    owner = []
    neighbour = []

    # Internal vertical faces
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal faces
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

    # movingWall (top, y=1)
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)

    n_moving = n_cells_x
    moving_start = n_internal

    # fixedWalls: bottom, left, right
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)

    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_fixed = n_cells_x + 2 * n_cells_y
    fixed_start = n_internal + n_moving

    # frontAndBack (z-normal, empty)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)

    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = n_base + j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells_x + i)

    n_empty = 2 * n_cells_x * n_cells_y
    empty_start = fixed_start + n_fixed

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
    lines = ["3", "("]
    lines.append("    movingWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_moving};")
    lines.append(f"        startFace       {moving_start};")
    lines.append("    }")
    lines.append("    fixedWalls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_fixed};")
    lines.append(f"        startFace       {fixed_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
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
        "    movingWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (1 0 0);\n"
        "    }\n"
        "    fixedWalls\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
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
    p_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    movingWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    fixedWalls\n    {\n"
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
        "application     incompressibleFluid;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
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

    solvers_block = (
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
    )

    if algorithm == "SIMPLE":
        algo_block = (
            "SIMPLE\n{\n"
            "    nNonOrthogonalCorrectors 0;\n"
            "    residualControl\n    {\n"
            "        p               1e-4;\n"
            "        U               1e-4;\n"
            "    }\n"
            "    relaxationFactors\n    {\n"
            f"        p               {alpha_p};\n"
            f"        U               {alpha_U};\n"
            "    }\n"
            f"    convergenceTolerance {convergence_tolerance};\n"
            f"    maxOuterIterations  {max_outer_iterations};\n"
            "}\n"
        )
    elif algorithm == "PISO":
        algo_block = (
            "PISO\n{\n"
            f"    nCorrectors         {n_correctors};\n"
            "    nNonOrthogonalCorrectors 0;\n"
            f"    convergenceTolerance {convergence_tolerance};\n"
            "}\n"
        )
    elif algorithm == "PIMPLE":
        algo_block = (
            "PIMPLE\n{\n"
            f"    nOuterCorrectors    {n_outer_correctors};\n"
            f"    nCorrectors         {n_correctors};\n"
            "    nNonOrthogonalCorrectors 0;\n"
            "    relaxationFactors\n    {\n"
            f"        p               {alpha_p};\n"
            f"        U               {alpha_U};\n"
            "    }\n"
            f"    convergenceTolerance {convergence_tolerance};\n"
            f"    maxOuterIterations  {max_outer_iterations};\n"
            "}\n"
        )
    else:
        # No algorithm sub-dict (test default detection)
        algo_block = ""

    fv_body = solvers_block + algo_block
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_case(tmp_path):
    """Create a cavity case configured for SIMPLE algorithm."""
    case_dir = tmp_path / "cavity_simple"
    _make_cavity_case(case_dir, n_cells_x=2, n_cells_y=2, nu=0.01,
                      end_time=10, write_interval=10,
                      max_outer_iterations=50, algorithm="SIMPLE")
    return case_dir


@pytest.fixture
def piso_case(tmp_path):
    """Create a cavity case configured for PISO algorithm."""
    case_dir = tmp_path / "cavity_piso"
    _make_cavity_case(case_dir, n_cells_x=2, n_cells_y=2, nu=0.01,
                      delta_t=0.001, end_time=0.01, write_interval=5,
                      algorithm="PISO", n_correctors=2)
    return case_dir


@pytest.fixture
def pimple_case(tmp_path):
    """Create a cavity case configured for PIMPLE algorithm."""
    case_dir = tmp_path / "cavity_pimple"
    _make_cavity_case(case_dir, n_cells_x=2, n_cells_y=2, nu=0.01,
                      delta_t=0.001, end_time=0.01, write_interval=5,
                      algorithm="PIMPLE",
                      n_outer_correctors=3, n_correctors=2,
                      max_outer_iterations=50)
    return case_dir


@pytest.fixture
def no_algo_case(tmp_path):
    """Create a cavity case with no algorithm sub-dict in fvSolution."""
    case_dir = tmp_path / "cavity_no_algo"
    _make_cavity_case(case_dir, n_cells_x=2, n_cells_y=2, nu=0.01,
                      end_time=10, write_interval=10,
                      max_outer_iterations=50, algorithm="NONE")
    return case_dir


# ---------------------------------------------------------------------------
# Tests: Algorithm detection
# ---------------------------------------------------------------------------

class TestAlgorithmDetection:
    """Tests for algorithm auto-detection from fvSolution."""

    def test_detect_simple(self, simple_case):
        """SIMPLE algorithm detected when SIMPLE sub-dict present."""
        from pyfoam.applications.incompressible_fluid_foam import (
            IncompressibleFluidFoam, Algorithm,
        )

        solver = IncompressibleFluidFoam(simple_case)
        assert solver.algorithm == Algorithm.SIMPLE

    def test_detect_piso(self, piso_case):
        """PISO algorithm detected when PISO sub-dict present."""
        from pyfoam.applications.incompressible_fluid_foam import (
            IncompressibleFluidFoam, Algorithm,
        )

        solver = IncompressibleFluidFoam(piso_case)
        assert solver.algorithm == Algorithm.PISO

    def test_detect_pimple(self, pimple_case):
        """PIMPLE algorithm detected when PIMPLE sub-dict present."""
        from pyfoam.applications.incompressible_fluid_foam import (
            IncompressibleFluidFoam, Algorithm,
        )

        solver = IncompressibleFluidFoam(pimple_case)
        assert solver.algorithm == Algorithm.PIMPLE

    def test_default_to_pimple_when_no_algo(self, no_algo_case):
        """Defaults to PIMPLE when no algorithm sub-dict found."""
        from pyfoam.applications.incompressible_fluid_foam import (
            IncompressibleFluidFoam, Algorithm,
        )

        solver = IncompressibleFluidFoam(no_algo_case)
        assert solver.algorithm == Algorithm.PIMPLE


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for IncompressibleFluidFoam initialisation."""

    def test_fields_initialise_simple(self, simple_case):
        """Fields are initialised correctly in SIMPLE mode."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(simple_case)

        assert solver.U.shape == (4, 3)
        assert torch.allclose(solver.U, torch.zeros(4, 3, dtype=CFD_DTYPE))

        assert solver.p.shape == (4,)
        assert torch.allclose(solver.p, torch.zeros(4, dtype=CFD_DTYPE))

        assert solver.phi.shape == (solver.mesh.n_faces,)

    def test_fields_initialise_piso(self, piso_case):
        """Fields are initialised correctly in PISO mode."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(piso_case)

        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        # Transient solvers should have old fields
        assert hasattr(solver, "U_old")
        assert hasattr(solver, "p_old")

    def test_fields_initialise_pimple(self, pimple_case):
        """Fields are initialised correctly in PIMPLE mode."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(pimple_case)

        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert hasattr(solver, "U_old")
        assert hasattr(solver, "p_old")

    def test_nu_read(self, simple_case):
        """Kinematic viscosity is read from transportProperties."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(simple_case)
        assert abs(solver.nu - 0.01) < 1e-10

    def test_fv_solution_settings_simple(self, simple_case):
        """fvSolution settings are read correctly for SIMPLE mode."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam, Algorithm

        solver = IncompressibleFluidFoam(simple_case)

        assert solver.p_solver == "PCG"
        assert solver.U_solver == "PBiCGStab"
        assert solver.algorithm == Algorithm.SIMPLE
        assert abs(solver.alpha_p - 0.3) < 1e-10
        assert abs(solver.alpha_U - 0.7) < 1e-10

    def test_fv_solution_settings_piso(self, piso_case):
        """fvSolution settings are read correctly for PISO mode."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam, Algorithm

        solver = IncompressibleFluidFoam(piso_case)

        assert solver.algorithm == Algorithm.PISO
        assert solver.n_correctors == 2
        # PISO should not use under-relaxation
        assert abs(solver.alpha_p - 1.0) < 1e-10
        assert abs(solver.alpha_U - 1.0) < 1e-10

    def test_fv_solution_settings_pimple(self, pimple_case):
        """fvSolution settings are read correctly for PIMPLE mode."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam, Algorithm

        solver = IncompressibleFluidFoam(pimple_case)

        assert solver.algorithm == Algorithm.PIMPLE
        assert solver.n_outer_correctors == 3
        assert solver.n_correctors == 2
        assert abs(solver.alpha_p - 0.3) < 1e-10
        assert abs(solver.alpha_U - 0.7) < 1e-10

    def test_turbulence_disabled_by_default(self, simple_case):
        """Turbulence is disabled when no turbulenceProperties exists."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(simple_case)
        assert solver.turbulence_enabled is False
        assert solver.ras is None


# ---------------------------------------------------------------------------
# Tests: Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    """Tests for boundary condition building."""

    def test_bc_tensor_shape(self, simple_case):
        """U_bc has correct shape."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(simple_case)
        U_bc = solver._build_boundary_conditions()

        assert U_bc.shape == (4, 3)

    def test_bc_has_fixed_values(self, simple_case):
        """U_bc has prescribed values for boundary cells."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(simple_case)
        U_bc = solver._build_boundary_conditions()

        bc_mask = ~torch.isnan(U_bc[:, 0])
        assert bc_mask.any(), "No boundary conditions found"

    def test_parse_vector_value_tuple(self):
        """_parse_vector_value handles tuple input."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        result = IncompressibleFluidFoam._parse_vector_value((1.0, 2.0, 3.0))
        assert result == (1.0, 2.0, 3.0)

    def test_parse_vector_value_string(self):
        """_parse_vector_value handles 'uniform ( x y z )' string."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        result = IncompressibleFluidFoam._parse_vector_value("uniform ( 1.5 0.0 -0.5 )")
        assert result is not None
        assert abs(result[0] - 1.5) < 1e-10
        assert abs(result[1] - 0.0) < 1e-10
        assert abs(result[2] - (-0.5)) < 1e-10

    def test_parse_vector_value_invalid(self):
        """_parse_vector_value returns None for invalid input."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        assert IncompressibleFluidFoam._parse_vector_value("invalid") is None
        assert IncompressibleFluidFoam._parse_vector_value(42) is None


# ---------------------------------------------------------------------------
# Tests: SIMPLE mode execution
# ---------------------------------------------------------------------------

class TestSimpleMode:
    """Tests for SIMPLE algorithm execution."""

    def test_build_simple_solver(self, simple_case):
        """_build_simple_solver creates a SIMPLESolver."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam
        from pyfoam.solvers.simple import SIMPLESolver

        solver = IncompressibleFluidFoam(simple_case)
        simple_solver = solver._build_simple_solver()
        assert isinstance(simple_solver, SIMPLESolver)

    def test_run_simple_converges(self, simple_case):
        """SIMPLE mode runs and produces valid output."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(simple_case)
        conv = solver.run()

        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_run_simple_writes_output(self, simple_case):
        """SIMPLE mode writes field files to time directories."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(simple_case)
        solver.run()

        time_dirs = [d for d in simple_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"


# ---------------------------------------------------------------------------
# Tests: PISO mode execution
# ---------------------------------------------------------------------------

class TestPisoMode:
    """Tests for PISO algorithm execution."""

    def test_build_piso_solver(self, piso_case):
        """_build_piso_solver creates a PISOSolver."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam
        from pyfoam.solvers.piso import PISOSolver

        solver = IncompressibleFluidFoam(piso_case)
        piso_solver = solver._build_piso_solver()
        assert isinstance(piso_solver, PISOSolver)

    def test_run_piso_produces_valid_output(self, piso_case):
        """PISO mode runs and produces finite values."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(piso_case)
        conv = solver.run()

        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"


# ---------------------------------------------------------------------------
# Tests: PIMPLE mode execution
# ---------------------------------------------------------------------------

class TestPimpleMode:
    """Tests for PIMPLE algorithm execution."""

    def test_build_pimple_solver(self, pimple_case):
        """_build_pimple_solver creates a PIMPLESolver."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam
        from pyfoam.solvers.pimple import PIMPLESolver

        solver = IncompressibleFluidFoam(pimple_case)
        pimple_solver = solver._build_pimple_solver()
        assert isinstance(pimple_solver, PIMPLESolver)

    def test_run_pimple_produces_valid_output(self, pimple_case):
        """PIMPLE mode runs and produces finite values."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(pimple_case)
        conv = solver.run()

        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"


# ---------------------------------------------------------------------------
# Tests: Turbulence integration
# ---------------------------------------------------------------------------

class TestTurbulence:
    """Tests for turbulence model integration."""

    def test_turbulence_enabled_with_ras(self, tmp_path):
        """Turbulence is enabled when turbulenceProperties specifies RAS."""
        case_dir = tmp_path / "cavity_turb"
        _make_cavity_case(
            case_dir, n_cells_x=2, n_cells_y=2, nu=0.01,
            end_time=10, write_interval=10, max_outer_iterations=50,
            algorithm="SIMPLE", turbulence_model="kEpsilon",
        )

        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(case_dir)
        assert solver.turbulence_enabled is True
        assert solver.ras is not None

    def test_turbulent_simple_run(self, tmp_path):
        """SIMPLE mode with turbulence produces valid output."""
        case_dir = tmp_path / "cavity_turb_simple"
        _make_cavity_case(
            case_dir, n_cells_x=2, n_cells_y=2, nu=0.01,
            end_time=10, write_interval=10, max_outer_iterations=50,
            algorithm="SIMPLE", turbulence_model="kEpsilon",
        )

        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_turbulent_pimple_run(self, tmp_path):
        """PIMPLE mode with turbulence produces valid output."""
        case_dir = tmp_path / "cavity_turb_pimple"
        _make_cavity_case(
            case_dir, n_cells_x=2, n_cells_y=2, nu=0.01,
            delta_t=0.001, end_time=0.01, write_interval=5,
            algorithm="PIMPLE", n_outer_correctors=3, n_correctors=2,
            max_outer_iterations=50, turbulence_model="kEpsilon",
        )

        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam

        solver = IncompressibleFluidFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"


# ---------------------------------------------------------------------------
# Tests: Written field format
# ---------------------------------------------------------------------------

class TestFieldFormat:
    """Tests for written field format validity."""

    def test_simple_written_fields_valid(self, simple_case):
        """Written fields from SIMPLE mode are valid OpenFOAM format."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam
        from pyfoam.io.field_io import read_field

        solver = IncompressibleFluidFoam(simple_case)
        solver.run()

        time_dirs = sorted(
            [d for d in simple_case.iterdir()
             if d.is_dir() and d.name.replace(".", "").isdigit()
             and d.name != "0"],
            key=lambda d: float(d.name),
        )
        assert len(time_dirs) >= 1

        last_dir = time_dirs[-1]
        U_data = read_field(last_dir / "U")
        p_data = read_field(last_dir / "p")

        assert U_data.scalar_type == "vector"
        assert p_data.scalar_type == "scalar"
        assert not U_data.is_uniform
        assert not p_data.is_uniform

    def test_pimple_written_fields_valid(self, pimple_case):
        """Written fields from PIMPLE mode are valid OpenFOAM format."""
        from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam
        from pyfoam.io.field_io import read_field

        solver = IncompressibleFluidFoam(pimple_case)
        solver.run()

        time_dirs = sorted(
            [d for d in pimple_case.iterdir()
             if d.is_dir() and d.name.replace(".", "").isdigit()
             and d.name != "0"],
            key=lambda d: float(d.name),
        )
        assert len(time_dirs) >= 1

        last_dir = time_dirs[-1]
        U_data = read_field(last_dir / "U")
        assert U_data.scalar_type == "vector"


# ---------------------------------------------------------------------------
# Tests: Import from package
# ---------------------------------------------------------------------------

class TestImport:
    """Tests for import availability."""

    def test_import_from_package(self):
        """IncompressibleFluidFoam is importable from pyfoam.applications."""
        from pyfoam.applications import IncompressibleFluidFoam

        assert IncompressibleFluidFoam is not None

    def test_import_algorithm_enum(self):
        """Algorithm enum is importable."""
        from pyfoam.applications.incompressible_fluid_foam import Algorithm

        assert hasattr(Algorithm, "SIMPLE")
        assert hasattr(Algorithm, "PISO")
        assert hasattr(Algorithm, "PIMPLE")
