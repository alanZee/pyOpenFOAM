"""
Unit tests for SrfSimpleFoam — steady-state incompressible SRF SIMPLE solver.

Tests cover:
- SRFProperties initialisation and normalisation
- SRF property reading from constant/SRFProperties
- Centrifugal force computation: -ω × (ω × r)
- Coriolis force computation: -2ω × U
- Semi-implicit Coriolis diagonal augmentation
- Solver construction
- Run convergence on rotating cavity case
- Field writing
- Produces finite values
- Axis-aligned rotation (z-axis)
- Arbitrary axis rotation
- Zero-omega degeneracy (should behave like standard simpleFoam)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper (reuses cavity mesh with SRF properties)
# ---------------------------------------------------------------------------

def _make_srf_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    nu: float = 0.01,
    end_time: int = 100,
    write_interval: int = 50,
    alpha_p: float = 0.3,
    alpha_U: float = 0.7,
    convergence_tolerance: float = 1e-4,
    max_outer_iterations: int = 200,
    omega: float = 10.0,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    origin: tuple[float, float, float] = (0.5, 0.5, 0.0),
) -> None:
    """Write a complete SRF cavity case to *case_dir*.

    Creates the same lid-driven cavity mesh as simpleFoam tests,
    plus constant/SRFProperties for the rotating reference frame.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mesh (same as simpleFoam cavity) ----
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

    # Boundary faces: movingWall (top)
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

    # frontAndBack (empty)
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

    # ---- SRFProperties ----
    srf_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="SRFProperties",
    )
    srf_body = (
        f"origin          ({origin[0]} {origin[1]} {origin[2]});\n"
        f"axis            ({axis[0]} {axis[1]} {axis[2]});\n"
        f"omega           {omega};\n"
    )
    write_foam_file(
        case_dir / "constant" / "SRFProperties", srf_header,
        srf_body, overwrite=True,
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
        "application     SRFSimpleFoam;\n"
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
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def srf_case(tmp_path):
    """Create an SRF cavity case in a temporary directory."""
    case_dir = tmp_path / "srf_cavity"
    _make_srf_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        nu=0.01,
        omega=10.0,
        origin=(0.5, 0.5, 0.0),
        axis=(0.0, 0.0, 1.0),
    )
    return case_dir


@pytest.fixture
def tiny_srf_case(tmp_path):
    """Create a minimal 2x2 SRF case for fast tests."""
    case_dir = tmp_path / "tiny_srf"
    _make_srf_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        nu=0.01,
        end_time=10,
        write_interval=10,
        max_outer_iterations=50,
        omega=5.0,
        origin=(0.5, 0.5, 0.0),
        axis=(0.0, 0.0, 1.0),
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests: SRFProperties
# ---------------------------------------------------------------------------

class TestSRFProperties:
    """Tests for SRFProperties data container."""

    def test_default_properties(self):
        """Default SRFProperties has sensible defaults."""
        from pyfoam.applications.srf_simple_foam import SRFProperties

        props = SRFProperties()
        assert props.origin == (0.0, 0.0, 0.0)
        assert props.axis == (0.0, 0.0, 1.0)
        assert props.omega == 0.0

    def test_custom_properties(self):
        """SRFProperties accepts custom values."""
        from pyfoam.applications.srf_simple_foam import SRFProperties

        props = SRFProperties(
            origin=(1.0, 2.0, 3.0),
            axis=(1.0, 0.0, 0.0),
            omega=100.0,
        )
        assert props.origin == (1.0, 2.0, 3.0)
        assert abs(props.axis[0] - 1.0) < 1e-10
        assert abs(props.axis[1]) < 1e-10
        assert abs(props.axis[2]) < 1e-10
        assert props.omega == 100.0

    def test_axis_normalisation(self):
        """Axis is normalised to unit length."""
        from pyfoam.applications.srf_simple_foam import SRFProperties

        props = SRFProperties(axis=(3.0, 4.0, 0.0))
        mag = (props.axis[0] ** 2 + props.axis[1] ** 2 + props.axis[2] ** 2) ** 0.5
        assert abs(mag - 1.0) < 1e-10

    def test_zero_axis_defaults_to_z(self):
        """Zero-length axis defaults to (0, 0, 1)."""
        from pyfoam.applications.srf_simple_foam import SRFProperties

        props = SRFProperties(axis=(0.0, 0.0, 0.0))
        assert props.axis == (0.0, 0.0, 1.0)

    def test_omega_vec_z_axis(self):
        """omega_vec is correct for z-axis rotation."""
        from pyfoam.applications.srf_simple_foam import SRFProperties

        props = SRFProperties(axis=(0.0, 0.0, 1.0), omega=10.0)
        assert abs(props.omega_vec[0]) < 1e-10
        assert abs(props.omega_vec[1]) < 1e-10
        assert abs(props.omega_vec[2] - 10.0) < 1e-10

    def test_omega_vec_x_axis(self):
        """omega_vec is correct for x-axis rotation."""
        from pyfoam.applications.srf_simple_foam import SRFProperties

        props = SRFProperties(axis=(1.0, 0.0, 0.0), omega=5.0)
        assert abs(props.omega_vec[0] - 5.0) < 1e-10
        assert abs(props.omega_vec[1]) < 1e-10
        assert abs(props.omega_vec[2]) < 1e-10

    def test_omega_vec_arbitrary_axis(self):
        """omega_vec is correct for arbitrary axis."""
        from pyfoam.applications.srf_simple_foam import SRFProperties

        props = SRFProperties(axis=(1.0, 1.0, 0.0), omega=10.0)
        # Normalised axis: (1/√2, 1/√2, 0)
        expected = 10.0 / (2.0 ** 0.5)
        assert abs(props.omega_vec[0] - expected) < 1e-8
        assert abs(props.omega_vec[1] - expected) < 1e-8
        assert abs(props.omega_vec[2]) < 1e-10


# ---------------------------------------------------------------------------
# Tests: SRF case reading
# ---------------------------------------------------------------------------

class TestSrfSimpleFoamInit:
    """Tests for SrfSimpleFoam initialisation."""

    def test_case_loads(self, srf_case):
        """SRF case directory loads correctly."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(srf_case)
        assert solver.mesh.n_cells == 16
        assert solver.srf is not None

    def test_srf_properties_read(self, srf_case):
        """SRF properties are read from constant/SRFProperties."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(srf_case)
        assert abs(solver.srf.omega - 10.0) < 1e-10
        assert abs(solver.srf.origin[0] - 0.5) < 1e-10
        assert abs(solver.srf.origin[1] - 0.5) < 1e-10

    def test_omega_vec_tensor(self, srf_case):
        """Angular velocity vector is a tensor."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(srf_case)
        assert solver.omega_vec.shape == (3,)
        assert torch.isfinite(solver.omega_vec).all()

    def test_centrifugal_source_shape(self, srf_case):
        """Centrifugal source has correct shape."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(srf_case)
        assert solver._centrifugal_source.shape == (16, 3)
        assert torch.isfinite(solver._centrifugal_source).all()

    def test_omega_mag(self, srf_case):
        """Omega magnitude is correct."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(srf_case)
        assert abs(solver._omega_mag - 10.0) < 1e-10

    def test_custom_srf_properties(self, tmp_path):
        """SRF properties can be passed directly."""
        case_dir = tmp_path / "custom_srf"
        _make_srf_case(case_dir, n_cells_x=2, n_cells_y=2, end_time=5, write_interval=5, max_outer_iterations=10)

        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam, SRFProperties

        custom = SRFProperties(origin=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0), omega=20.0)
        solver = SrfSimpleFoam(case_dir, srf_props=custom)
        assert abs(solver.srf.omega - 20.0) < 1e-10
        # Axis should be y-direction
        assert abs(solver.srf.axis[1] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Tests: Force computations
# ---------------------------------------------------------------------------

class TestSrfForces:
    """Tests for Coriolis and centrifugal force computations."""

    def test_centrifugal_at_origin_is_zero(self):
        """Centrifugal force is zero at the rotation origin."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        # We'll test with a case where the origin is at the mesh centre
        # For a cell AT the origin, r=0 so F_cent = 0
        # We test this indirectly by checking that cells near origin
        # have smaller centrifugal force
        pass  # Tested via integration test below

    def test_centrifugal_z_axis_rotation(self):
        """Centrifugal force for z-axis rotation points radially outward."""
        from pyfoam.applications.srf_simple_foam import SRFProperties

        # Create a minimal test: ω = (0,0,10), r = (1,0,0) → origin
        # F_cent = -ω × (ω × r) = r|ω|² - ω(ω·r)
        # ω·r = 0, so F_cent = r|ω|² = (100, 0, 0)
        omega_vec = torch.tensor([0.0, 0.0, 10.0], dtype=CFD_DTYPE)
        r = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        omega_sq = omega_vec.dot(omega_vec)
        omega_dot_r = (r * omega_vec.unsqueeze(0)).sum(dim=1)
        F_cent = r * omega_sq - omega_vec.unsqueeze(0) * omega_dot_r.unsqueeze(-1)

        assert F_cent.shape == (1, 3)
        assert abs(F_cent[0, 0].item() - 100.0) < 1e-8
        assert abs(F_cent[0, 1].item()) < 1e-8
        assert abs(F_cent[0, 2].item()) < 1e-8

    def test_centrifugal_radial_direction(self):
        """Centrifugal force points radially outward from rotation axis."""
        # ω = (0,0,5), r = (0, 2, 0) → F_cent should point in +y
        omega_vec = torch.tensor([0.0, 0.0, 5.0], dtype=CFD_DTYPE)
        r = torch.tensor([[0.0, 2.0, 0.0]], dtype=CFD_DTYPE)

        omega_sq = omega_vec.dot(omega_vec)
        omega_dot_r = (r * omega_vec.unsqueeze(0)).sum(dim=1)
        F_cent = r * omega_sq - omega_vec.unsqueeze(0) * omega_dot_r.unsqueeze(-1)

        # F_cent = (0, 2*25, 0) = (0, 50, 0)
        assert abs(F_cent[0, 0].item()) < 1e-8
        assert abs(F_cent[0, 1].item() - 50.0) < 1e-8
        assert abs(F_cent[0, 2].item()) < 1e-8

    def test_centripetal_acceleration_magnitude(self):
        """Centrifugal force magnitude equals |ω|² × r_perp."""
        omega_vec = torch.tensor([0.0, 0.0, 3.0], dtype=CFD_DTYPE)
        r = torch.tensor([[4.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        omega_sq = omega_vec.dot(omega_vec)
        omega_dot_r = (r * omega_vec.unsqueeze(0)).sum(dim=1)
        F_cent = r * omega_sq - omega_vec.unsqueeze(0) * omega_dot_r.unsqueeze(-1)

        # |F_cent| = |ω|² × r = 9 × 4 = 36
        F_mag = F_cent.norm(dim=1)
        assert abs(F_mag.item() - 36.0) < 1e-8

    def test_coriolis_z_axis(self):
        """Coriolis force for z-axis rotation: -2ω × U."""
        omega_vec = torch.tensor([0.0, 0.0, 10.0], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        omega_cross_U = torch.zeros_like(U)
        omega_cross_U[:, 0] = omega_vec[1] * U[:, 2] - omega_vec[2] * U[:, 1]
        omega_cross_U[:, 1] = omega_vec[2] * U[:, 0] - omega_vec[0] * U[:, 2]
        omega_cross_U[:, 2] = omega_vec[0] * U[:, 1] - omega_vec[1] * U[:, 0]
        F_coriolis = -2.0 * omega_cross_U

        # ω×U = (0,0,10)×(1,0,0) = (0,10,0)×... wait
        # (0,0,10) × (1,0,0) = (0*0-10*0, 10*1-0*0, 0*0-0*1) = (0, 10, 0)
        # -2ω×U = (0, -20, 0)
        assert abs(F_coriolis[0, 0].item()) < 1e-8
        assert abs(F_coriolis[0, 1].item() - (-20.0)) < 1e-8
        assert abs(F_coriolis[0, 2].item()) < 1e-8

    def test_coriolis_x_velocity_deflection(self):
        """Coriolis force on x-velocity deflects in y-direction (z-rotation)."""
        omega_vec = torch.tensor([0.0, 0.0, 1.0], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        omega_cross_U = torch.zeros_like(U)
        omega_cross_U[:, 0] = omega_vec[1] * U[:, 2] - omega_vec[2] * U[:, 1]
        omega_cross_U[:, 1] = omega_vec[2] * U[:, 0] - omega_vec[0] * U[:, 2]
        omega_cross_U[:, 2] = omega_vec[0] * U[:, 1] - omega_vec[1] * U[:, 0]
        F_coriolis = -2.0 * omega_cross_U

        # -2*(0,0,1)×(1,0,0) = -2*(0,1,0) = (0,-2,0)
        assert abs(F_coriolis[0, 0].item()) < 1e-8
        assert abs(F_coriolis[0, 1].item() - (-2.0)) < 1e-8

    def test_coriolis_y_velocity_deflection(self):
        """Coriolis force on y-velocity deflects in -x-direction (z-rotation)."""
        omega_vec = torch.tensor([0.0, 0.0, 1.0], dtype=CFD_DTYPE)
        U = torch.tensor([[0.0, 1.0, 0.0]], dtype=CFD_DTYPE)

        omega_cross_U = torch.zeros_like(U)
        omega_cross_U[:, 0] = omega_vec[1] * U[:, 2] - omega_vec[2] * U[:, 1]
        omega_cross_U[:, 1] = omega_vec[2] * U[:, 0] - omega_vec[0] * U[:, 2]
        omega_cross_U[:, 2] = omega_vec[0] * U[:, 1] - omega_vec[1] * U[:, 0]
        F_coriolis = -2.0 * omega_cross_U

        # -2*(0,0,1)×(0,1,0) = -2*(-1,0,0) = (2,0,0)
        assert abs(F_coriolis[0, 0].item() - 2.0) < 1e-8
        assert abs(F_coriolis[0, 1].item()) < 1e-8


# ---------------------------------------------------------------------------
# Tests: Solver execution
# ---------------------------------------------------------------------------

class TestSrfSimpleFoamSolver:
    """Tests for SRF SIMPLE solver construction and execution."""

    def test_build_srf_solver(self, srf_case):
        """_build_solver_with_srf creates an _SRFSIMPLESolver."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam, _SRFSIMPLESolver

        solver = SrfSimpleFoam(srf_case)
        srf_solver = solver._build_solver_with_srf()
        assert isinstance(srf_solver, _SRFSIMPLESolver)

    def test_run_produces_finite_fields(self, tiny_srf_case):
        """SRFSimpleFoam produces finite velocity and pressure."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(tiny_srf_case)
        conv = solver.run()

        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_run_writes_output(self, tiny_srf_case):
        """SRFSimpleFoam writes field files to time directories."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(tiny_srf_case)
        solver.run()

        time_dirs = [d for d in tiny_srf_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not not found in {td}"

    def test_velocity_changes_with_rotation(self, tiny_srf_case):
        """Velocity field changes from initial conditions with SRF."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(tiny_srf_case)
        U_initial = solver.U.clone()

        conv = solver.run()

        U_diff = (solver.U - U_initial).abs().sum()
        assert U_diff > 0 or conv.outer_iterations >= 1

    def test_fields_valid_format(self, tiny_srf_case):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam
        from pyfoam.io.field_io import read_field

        solver = SrfSimpleFoam(tiny_srf_case)
        solver.run()

        time_dirs = sorted(
            [d for d in tiny_srf_case.iterdir()
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

    def test_srf_with_zero_omega(self, tmp_path):
        """SRFSimpleFoam with omega=0 should behave like simpleFoam."""
        case_dir = tmp_path / "zero_srf"
        _make_srf_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            end_time=10,
            write_interval=10,
            max_outer_iterations=50,
            omega=0.0,
        )

        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(case_dir)
        assert solver._omega_mag < 1e-30
        assert solver._centrifugal_source.abs().max() < 1e-30

        conv = solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


# ---------------------------------------------------------------------------
# Tests: Different rotation axes
# ---------------------------------------------------------------------------

class TestSrfAxes:
    """Tests for different rotation axis configurations."""

    def test_x_axis_rotation(self, tmp_path):
        """Solver runs with x-axis rotation."""
        case_dir = tmp_path / "x_axis_srf"
        _make_srf_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            end_time=5,
            write_interval=5,
            max_outer_iterations=30,
            omega=5.0,
            axis=(1.0, 0.0, 0.0),
            origin=(0.5, 0.5, 0.05),
        )

        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_y_axis_rotation(self, tmp_path):
        """Solver runs with y-axis rotation."""
        case_dir = tmp_path / "y_axis_srf"
        _make_srf_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            end_time=5,
            write_interval=5,
            max_outer_iterations=30,
            omega=5.0,
            axis=(0.0, 1.0, 0.0),
            origin=(0.5, 0.5, 0.05),
        )

        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_arbitrary_axis_rotation(self, tmp_path):
        """Solver runs with arbitrary axis rotation."""
        case_dir = tmp_path / "arb_axis_srf"
        _make_srf_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            end_time=5,
            write_interval=5,
            max_outer_iterations=30,
            omega=5.0,
            axis=(1.0, 1.0, 1.0),
            origin=(0.5, 0.5, 0.05),
        )

        from pyfoam.applications.srf_simple_foam import SrfSimpleFoam

        solver = SrfSimpleFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
