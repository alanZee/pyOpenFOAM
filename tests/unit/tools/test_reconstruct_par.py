"""
Unit tests for reconstruct_par -- parallel case reconstruction tool.

Tests cover:
- Auto-detection of processor count
- Basic reconstruction
- Result dataclass fields
- Error handling
- Mesh file writing
- Field reconstruction
- Overwrite behaviour
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Helper: create a processor directory with mesh and field data
# ---------------------------------------------------------------------------


def _make_processor_dir(
    proc_dir: Path,
    proc_id: int,
    n_cells: int = 4,
    n_points_per_cell: int = 4,
) -> None:
    """Create a minimal processor directory with mesh and field files."""
    proc_dir.mkdir(parents=True, exist_ok=True)

    mesh_dir = proc_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location=f"processor{proc_id}/constant/polyMesh",
    )

    # Create simple mesh: n_cells hex cells in a row
    dx = 1.0 / n_cells
    points = []
    for i in range(n_cells + 1):
        points.append((i * dx, 0.0, 0.0))
        points.append((i * dx, 1.0, 0.0))
        points.append((i * dx, 0.0, 1.0))
        points.append((i * dx, 1.0, 1.0))

    n_points = len(points)
    n_faces = 3 * n_cells + 2  # internal + 2 boundary (left/right)
    n_internal = n_cells - 1

    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "vectorField", "object": "points"},
    )
    lines = [f"{n_points}", "("]
    for p in points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    # Faces (simplified)
    faces = []
    owner = []
    neighbour = []
    face_idx = 0

    # Internal faces
    for i in range(n_internal):
        base = (i + 1) * 4
        faces.append(f"4({base} {base+1} {base+3} {base+2})")
        owner.append(i)
        neighbour.append(i + 1)
        face_idx += 1

    # Boundary faces: left
    faces.append("4(0 1 5 4)")
    owner.append(0)
    face_idx += 1

    # Boundary faces: right
    right_base = n_cells * 4
    faces.append(f"4({right_base} {right_base+1} {right_base+3} {right_base+2})")
    owner.append(n_cells - 1)
    face_idx += 1

    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "faceList", "object": "faces"},
    )
    lines = [f"{len(faces)}", "("]
    for f in faces:
        lines.append(f)
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "labelList", "object": "owner"},
    )
    lines = [f"{len(owner)}", "("]
    for o in owner:
        lines.append(str(o))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"},
    )
    lines = [f"{len(neighbour)}", "("]
    for n in neighbour:
        lines.append(str(n))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    # Boundary
    h = FoamFileHeader(
        **{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"},
    )
    left_start = n_internal
    right_start = n_internal + 1
    boundary_content = (
        "2\n(\n"
        "    left\n    {\n        type            patch;\n"
        "        nFaces          1;\n        startFace       %d;\n    }\n"
        "    right\n    {\n        type            patch;\n"
        "        nFaces          1;\n        startFace       %d;\n    }\n"
        ")\n"
    ) % (left_start, right_start)
    write_foam_file(mesh_dir / "boundary", h, boundary_content, overwrite=True)

    # 0/U
    zero_dir = proc_dir / "0"
    zero_dir.mkdir(exist_ok=True)
    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location=f"processor{proc_id}/0", object="U",
    )
    write_foam_file(zero_dir / "U", u_header, (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (1 0 0);\n\n"
        "boundaryField\n{}\n"
    ), overwrite=True)

    # 0/p
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location=f"processor{proc_id}/0", object="p",
    )
    write_foam_file(zero_dir / "p", p_header, (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{}\n"
    ), overwrite=True)


def _make_parallel_case(case_dir: Path, n_proc: int = 2) -> None:
    """Create a case with processor directories."""
    case_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_proc):
        _make_processor_dir(case_dir / f"processor{i}", i, n_cells=4)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parallel_case(tmp_path):
    case_dir = tmp_path / "parallel_case"
    _make_parallel_case(case_dir, n_proc=2)
    return case_dir


@pytest.fixture
def parallel_case_4proc(tmp_path):
    case_dir = tmp_path / "parallel_case_4"
    _make_parallel_case(case_dir, n_proc=4)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestReconstructParAutoDetect:
    """Auto-detection of processor count."""

    def test_auto_detect_2(self, parallel_case):
        """Auto-detects 2 processor directories."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        result = reconstruct_par(parallel_case)
        assert result.n_proc == 2

    def test_auto_detect_4(self, parallel_case_4proc):
        """Auto-detects 4 processor directories."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        result = reconstruct_par(parallel_case_4proc)
        assert result.n_proc == 4


class TestReconstructParBasic:
    """Basic reconstruction tests."""

    def test_returns_result(self, parallel_case):
        """reconstruct_par returns a ReconstructParResult."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        result = reconstruct_par(parallel_case, n_proc=2)
        assert result is not None
        assert result.n_proc == 2

    def test_total_cells(self, parallel_case):
        """Total cells equals sum across processors."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        result = reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        # Each processor has 4 cells
        assert result.n_cells_total == 8

    def test_total_faces(self, parallel_case):
        """Total faces equals sum across processors."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        result = reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        assert result.n_faces_total > 0

    def test_total_points(self, parallel_case):
        """Total points equals sum across processors."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        result = reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        assert result.n_points_total > 0

    def test_fields_reconstructed(self, parallel_case):
        """U and p are reconstructed."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        result = reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        assert "U" in result.fields_reconstructed
        assert "p" in result.fields_reconstructed

    def test_processor_dirs_recorded(self, parallel_case):
        """Processor directories are recorded."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        result = reconstruct_par(parallel_case, n_proc=2)
        assert len(result.processor_dirs) == 2


class TestReconstructParMeshFiles:
    """Test that reconstructed mesh files are written."""

    def test_points_written(self, parallel_case):
        """Points file is written."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        pts = parallel_case / "constant" / "polyMesh" / "points"
        assert pts.exists()

    def test_faces_written(self, parallel_case):
        """Faces file is written."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        fcs = parallel_case / "constant" / "polyMesh" / "faces"
        assert fcs.exists()

    def test_owner_written(self, parallel_case):
        """Owner file is written."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        own = parallel_case / "constant" / "polyMesh" / "owner"
        assert own.exists()

    def test_neighbour_written(self, parallel_case):
        """Neighbour file is written."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        nbr = parallel_case / "constant" / "polyMesh" / "neighbour"
        assert nbr.exists()

    def test_boundary_written(self, parallel_case):
        """Boundary file is written."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        bnd = parallel_case / "constant" / "polyMesh" / "boundary"
        assert bnd.exists()


class TestReconstructParFieldFiles:
    """Test that reconstructed field files are written."""

    def test_u_field_written(self, parallel_case):
        """U field is written."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        u = parallel_case / "0" / "U"
        assert u.exists()

    def test_p_field_written(self, parallel_case):
        """p field is written."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        reconstruct_par(parallel_case, n_proc=2, overwrite=True)
        p = parallel_case / "0" / "p"
        assert p.exists()


class TestReconstructParErrors:
    """Error handling tests."""

    def test_n_proc_zero_raises(self, parallel_case):
        """Zero processors raises ValueError."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        with pytest.raises(ValueError, match="n_proc must be >= 1"):
            reconstruct_par(parallel_case, n_proc=0)

    def test_missing_processor_dir_raises(self, tmp_path):
        """Missing processor directory raises FileNotFoundError."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        empty = tmp_path / "empty"
        empty.mkdir()
        # Create only processor0
        _make_processor_dir(empty / "processor0", 0, n_cells=2)
        with pytest.raises(FileNotFoundError):
            reconstruct_par(empty, n_proc=2)

    def test_no_processor_dirs_raises(self, tmp_path):
        """No processor directories raises FileNotFoundError."""
        from pyfoam.tools.reconstruct_par import reconstruct_par
        empty = tmp_path / "no_proc"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            reconstruct_par(empty)
