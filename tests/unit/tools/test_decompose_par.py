"""
Unit tests for decompose_par -- domain decomposition tool.

Tests cover:
- Basic decomposition with simple method
- Processor directory creation
- Mesh file writing
- Field file writing
- DecomposeParDict writing
- Error handling
- Imbalance metrics
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file
from pyfoam.mesh.fv_mesh import FvMesh


# ---------------------------------------------------------------------------
# Mesh helper
# ---------------------------------------------------------------------------

def _make_simple_case(case_dir: Path, nx: int = 4, ny: int = 2) -> None:
    """Create a minimal 2D case for decomposition testing."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / nx
    dy = 1.0 / ny
    dz = 0.1

    # Points
    points_z0 = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)

    points_z1 = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            points_z1.append((i * dx, j * dy, dz))

    all_points = points_z0 + points_z1
    n_points = len(all_points)

    faces = []
    owner = []
    neighbour = []

    # Internal vertical faces
    for j in range(ny):
        for i in range(nx - 1):
            p0 = j * (nx + 1) + i + 1
            p1 = p0 + nx + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * nx + i)
            neighbour.append(j * nx + i + 1)

    # Internal horizontal faces
    for j in range(ny - 1):
        for i in range(nx):
            p0 = (j + 1) * (nx + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * nx + i)
            neighbour.append((j + 1) * nx + i)

    n_internal = len(neighbour)

    # Boundary faces
    for j in range(ny):
        faces.append((4, j * (nx + 1), j * (nx + 1) + nx + 1,
                       j * (nx + 1) + nx + 1 + n_base, j * (nx + 1) + n_base))
        owner.append(j * nx)
    for j in range(ny):
        faces.append((4, j * (nx + 1) + nx, j * (nx + 1) + nx + nx + 1,
                       j * (nx + 1) + nx + nx + 1 + n_base, j * (nx + 1) + nx + n_base))
        owner.append(j * nx + nx - 1)

    for i in range(nx):
        faces.append((4, i, i + 1, i + 1 + n_base, i + n_base))
        owner.append(i)
    for i in range(nx):
        faces.append((4, ny * (nx + 1) + i, ny * (nx + 1) + i + 1,
                       ny * (nx + 1) + i + 1 + n_base, ny * (nx + 1) + i + n_base))
        owner.append((ny - 1) * nx + i)

    # Empty faces
    for _ in range(2 * nx * ny):
        faces.append((4, 0, 1, 2, 3))
        owner.append(0)

    n_faces = len(faces)

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for p in all_points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
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
    n_patches = 4
    bnd_start = n_internal
    lines = [f"{n_patches}", "("]
    for name, nf in [("inlet", ny), ("outlet", ny), ("walls", 2 * nx), ("frontAndBack", 2 * nx * ny)]:
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type            patch;")
        lines.append(f"        nFaces          {nf};")
        lines.append(f"        startFace       {bnd_start};")
        lines.append("    }")
        bnd_start += nf
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # 0/U
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)
    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    write_foam_file(zero_dir / "U", u_header, (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (1 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type fixedValue;\n        value uniform (1 0 0);\n    }\n"
        "    outlet\n    {\n        type zeroGradient;\n    }\n"
        "    walls\n    {\n        type noSlip;\n    }\n"
        "    frontAndBack\n    {\n        type empty;\n    }\n"
        "}\n"
    ), overwrite=True)

    # 0/p
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    write_foam_file(zero_dir / "p", p_header, (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type zeroGradient;\n    }\n"
        "    outlet\n    {\n        type fixedValue;\n        value uniform 0;\n    }\n"
        "    walls\n    {\n        type zeroGradient;\n    }\n"
        "    frontAndBack\n    {\n        type empty;\n    }\n"
        "}\n"
    ), overwrite=True)

    # system/
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)
    for fname, content in [
        ("controlDict", "application icoFoam;\nstartTime 0;\nendTime 1;\ndeltaT 0.001;\n"),
        ("fvSchemes", "gradSchemes\n{\n    default Gauss linear;\n}\n"),
        ("fvSolution", "solvers\n{\n    p { solver PCG; tolerance 1e-6; }\n    U { solver PBiCGStab; tolerance 1e-6; }\n}\n"),
    ]:
        h = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="system", object=fname,
        )
        write_foam_file(sys_dir / fname, h, content, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_case(tmp_path):
    case_dir = tmp_path / "simple"
    _make_simple_case(case_dir, nx=4, ny=2)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestDecomposeParBasic:
    """Basic decomposition tests."""

    def test_returns_result(self, simple_case):
        """decompose_par returns a DecomposeParResult."""
        from pyfoam.tools.decompose_par import decompose_par
        result = decompose_par(simple_case, n_proc=2, method="simple")
        assert result is not None
        assert result.n_proc == 2

    def test_creates_processor_dirs(self, simple_case):
        """Processor directories are created."""
        from pyfoam.tools.decompose_par import decompose_par
        decompose_par(simple_case, n_proc=2, method="simple")
        assert (simple_case / "processor0").is_dir()
        assert (simple_case / "processor1").is_dir()

    def test_cells_per_proc(self, simple_case):
        """Total cells across processors equals global cell count."""
        from pyfoam.tools.decompose_par import decompose_par
        result = decompose_par(simple_case, n_proc=2, method="simple")
        total = sum(result.cells_per_proc)
        assert total == 8  # 4x2 mesh

    def test_all_cells_assigned(self, simple_case):
        """All cells are assigned to some processor."""
        from pyfoam.tools.decompose_par import decompose_par
        result = decompose_par(simple_case, n_proc=4, method="simple")
        total = sum(result.cells_per_proc)
        assert total == 8

    def test_imbalance_ratio(self, simple_case):
        """Imbalance ratio is reasonable."""
        from pyfoam.tools.decompose_par import decompose_par
        result = decompose_par(simple_case, n_proc=2, method="simple")
        assert result.imbalance_ratio >= 1.0
        assert result.imbalance_ratio < 2.0  # Should be balanced


class TestDecomposeParMeshFiles:
    """Test that processor mesh files are written."""

    def test_points_written(self, simple_case):
        """Points file exists in each processor."""
        from pyfoam.tools.decompose_par import decompose_par
        decompose_par(simple_case, n_proc=2, method="simple")
        for p in range(2):
            pts = simple_case / f"processor{p}" / "constant" / "polyMesh" / "points"
            assert pts.exists()

    def test_faces_written(self, simple_case):
        """Faces file exists in each processor."""
        from pyfoam.tools.decompose_par import decompose_par
        decompose_par(simple_case, n_proc=2, method="simple")
        for p in range(2):
            fc = simple_case / f"processor{p}" / "constant" / "polyMesh" / "faces"
            assert fc.exists()

    def test_owner_written(self, simple_case):
        """Owner file exists in each processor."""
        from pyfoam.tools.decompose_par import decompose_par
        decompose_par(simple_case, n_proc=2, method="simple")
        for p in range(2):
            ow = simple_case / f"processor{p}" / "constant" / "polyMesh" / "owner"
            assert ow.exists()

    def test_boundary_written(self, simple_case):
        """Boundary file exists in each processor."""
        from pyfoam.tools.decompose_par import decompose_par
        decompose_par(simple_case, n_proc=2, method="simple")
        for p in range(2):
            bd = simple_case / f"processor{p}" / "constant" / "polyMesh" / "boundary"
            assert bd.exists()


class TestDecomposeParFields:
    """Test that processor fields are written."""

    def test_u_written(self, simple_case):
        """U field exists in each processor 0/ directory."""
        from pyfoam.tools.decompose_par import decompose_par
        decompose_par(simple_case, n_proc=2, method="simple")
        for p in range(2):
            u = simple_case / f"processor{p}" / "0" / "U"
            assert u.exists()

    def test_p_written(self, simple_case):
        """p field exists in each processor 0/ directory."""
        from pyfoam.tools.decompose_par import decompose_par
        decompose_par(simple_case, n_proc=2, method="simple")
        for p in range(2):
            pc = simple_case / f"processor{p}" / "0" / "p"
            assert pc.exists()


class TestDecomposeParSystem:
    """Test that system files are written."""

    def test_decomposepardict_written(self, simple_case):
        """decomposeParDict exists in each processor."""
        from pyfoam.tools.decompose_par import decompose_par
        decompose_par(simple_case, n_proc=2, method="simple")
        for p in range(2):
            dpd = simple_case / f"processor{p}" / "system" / "decomposeParDict"
            assert dpd.exists()

    def test_controldict_copied(self, simple_case):
        """controlDict is copied to each processor."""
        from pyfoam.tools.decompose_par import decompose_par
        decompose_par(simple_case, n_proc=2, method="simple")
        for p in range(2):
            cd = simple_case / f"processor{p}" / "system" / "controlDict"
            assert cd.exists()


class TestDecomposeParErrors:
    """Error handling tests."""

    def test_n_proc_zero_raises(self, simple_case):
        """Zero processors raises ValueError."""
        from pyfoam.tools.decompose_par import decompose_par
        with pytest.raises(ValueError, match="n_proc must be >= 1"):
            decompose_par(simple_case, n_proc=0)

    def test_n_proc_exceeds_cells_raises(self, simple_case):
        """More processors than cells raises ValueError."""
        from pyfoam.tools.decompose_par import decompose_par
        with pytest.raises(ValueError, match="n_proc.*exceeds n_cells"):
            decompose_par(simple_case, n_proc=100)
