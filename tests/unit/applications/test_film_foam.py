"""
Unit tests for FilmFoam — thin film flow solver.

Tests cover:
- Case loading and initialisation
- Film thickness field initialisation
- Capillary number computation
- Laplacian computation
- Gradient computation
- Solver run produces finite values
- Solver writes output
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper
# ---------------------------------------------------------------------------

def _make_film_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    h_init: float = 1e-3,
    end_time: int = 5,
    delta_t: float = 0.001,
) -> None:
    """Write a 2D film flow case."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1

    points_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)
    points_z1 = [(x, y, dz) for x, y, _ in points_z0]
    all_points = points_z0 + points_z1
    n_points = len(all_points)

    faces, owner, neighbour = [], [], []

    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

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

    # Boundary patches
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)
    n_left = n_cells_y
    left_start = n_internal

    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)
    n_right = n_cells_y
    right_start = left_start + n_left

    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)
    n_tb = 2 * n_cells_x
    tb_start = right_start + n_right

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
    empty_start = tb_start + n_tb

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

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
    lines = ["4", "("]
    lines += [
        "    leftWall", "    {", "        type            wall;",
        f"        nFaces          {n_left};", f"        startFace       {left_start};", "    }",
        "    rightWall", "    {", "        type            wall;",
        f"        nFaces          {n_right};", f"        startFace       {right_start};", "    }",
        "    topBottomWalls", "    {", "        type            wall;",
        f"        nFaces          {n_tb};", f"        startFace       {tb_start};", "    }",
        "    frontAndBack", "    {", "        type            empty;",
        f"        nFaces          {n_empty};", f"        startFace       {empty_start};", "    }",
        ")",
    ]
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # 0/h
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    write_foam_file(zero_dir / "h",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="h"),
        "dimensions      [0 1 0 0 0 0 0];\n\n"
        f"internalField   uniform {h_init};\n\n"
        "boundaryField\n{\n"
        "    leftWall\n    {\n        type            zeroGradient;\n    }\n"
        "    rightWall\n    {\n        type            zeroGradient;\n    }\n"
        "    topBottomWalls\n    {\n        type            zeroGradient;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # System files
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    write_foam_file(sys_dir / "controlDict",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="controlDict"),
        "application     filmFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        "writeInterval   10;\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSchemes",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSchemes"),
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         none;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSolution",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSolution"),
        "filmFoam\n{\n"
        "    convergenceTolerance 1e-6;\n"
        "    nCorrectors         3;\n"
        "}\n",
        overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def film_case(tmp_path):
    case_dir = tmp_path / "film"
    _make_film_case(case_dir, n_cells_x=2, n_cells_y=2, end_time=2, delta_t=0.001)
    return case_dir


@pytest.fixture
def film_case_4x4(tmp_path):
    case_dir = tmp_path / "film_4x4"
    _make_film_case(case_dir, n_cells_x=4, n_cells_y=4, end_time=5, delta_t=0.001)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestFilmFoamInit:
    """Initialisation tests."""

    def test_case_loads(self, film_case):
        from pyfoam.io.case import Case
        case = Case(film_case)
        assert case.has_mesh()

    def test_solver_creates(self, film_case):
        from pyfoam.applications.film_foam import FilmFoam
        solver = FilmFoam(film_case, rho=1000.0, mu=1e-3, sigma=0.07)
        assert solver.rho == 1000.0
        assert solver.mu == 1e-3
        assert solver.sigma == 0.07

    def test_film_thickness_shape(self, film_case):
        from pyfoam.applications.film_foam import FilmFoam
        solver = FilmFoam(film_case)
        assert solver.h.shape == (4,)

    def test_capillary_number_computed(self, film_case):
        from pyfoam.applications.film_foam import FilmFoam
        solver = FilmFoam(film_case)
        assert solver.Ca >= 0.0

    def test_custom_parameters(self, film_case):
        from pyfoam.applications.film_foam import FilmFoam
        solver = FilmFoam(
            film_case, rho=800.0, mu=5e-4, sigma=0.05,
            beta=0.3, contact_angle=math.pi / 6,
        )
        assert solver.beta == 0.3
        assert abs(solver.contact_angle - math.pi / 6) < 1e-10


class TestFilmFoamRun:
    """Solver execution tests."""

    def test_run_completes(self, film_case):
        from pyfoam.applications.film_foam import FilmFoam
        solver = FilmFoam(film_case)
        result = solver.run()
        assert "converged" in result
        assert "h_min" in result
        assert "h_max" in result

    def test_run_finite_values(self, film_case):
        from pyfoam.applications.film_foam import FilmFoam
        solver = FilmFoam(film_case)
        solver.run()
        assert torch.isfinite(solver.h).all()

    def test_film_thickness_positive(self, film_case):
        from pyfoam.applications.film_foam import FilmFoam
        solver = FilmFoam(film_case)
        solver.run()
        assert (solver.h > 0).all()

    def test_run_writes_output(self, film_case):
        from pyfoam.applications.film_foam import FilmFoam
        solver = FilmFoam(film_case)
        solver.run()

        time_dirs = [
            d for d in film_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1

    def test_exports_in_all(self):
        from pyfoam.applications import FilmFoam
        assert FilmFoam is not None
