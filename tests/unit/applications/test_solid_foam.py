"""
Unit tests for SolidFoam — solid mechanics solver with thermal stress.

Tests cover:
- Case loading and mesh construction
- Mechanical property reading
- Lamé parameter computation
- Field initialisation (displacement, temperature)
- Thermal strain computation
- Von Mises stress computation
- Solver run
- Output writing
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

def _make_solid_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    T_init: float = 300.0,
    E: float = 200e9,
    nu: float = 0.3,
    end_time: int = 5,
    delta_t: float = 1.0,
) -> None:
    """Write a complete solid mechanics case."""
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

    # Front/back (empty)
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

    # 0/ directory
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    write_foam_file(zero_dir / "D",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volVectorField",
                       location="0", object="D"),
        "dimensions      [0 1 0 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    leftWall\n    {\n        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n    }\n"
        "    rightWall\n    {\n        type            zeroGradient;\n    }\n"
        "    topBottomWalls\n    {\n        type            zeroGradient;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    write_foam_file(zero_dir / "T",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="T"),
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        f"internalField   uniform {T_init};\n\n"
        "boundaryField\n{\n"
        "    leftWall\n    {\n        type            fixedValue;\n"
        f"        value           uniform {T_init};\n    }}\n"
        "    rightWall\n    {\n        type            zeroGradient;\n    }\n"
        "    topBottomWalls\n    {\n        type            zeroGradient;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    write_foam_file(zero_dir / "U",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volVectorField",
                       location="0", object="U"),
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    leftWall\n    {\n        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n    }\n"
        "    rightWall\n    {\n        type            zeroGradient;\n    }\n"
        "    topBottomWalls\n    {\n        type            zeroGradient;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    write_foam_file(zero_dir / "p",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="p"),
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        "internalField   uniform 101325;\n\n"
        "boundaryField\n{\n"
        "    leftWall\n    {\n        type            zeroGradient;\n    }\n"
        "    rightWall\n    {\n        type            zeroGradient;\n    }\n"
        "    topBottomWalls\n    {\n        type            zeroGradient;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    write_foam_file(sys_dir / "controlDict",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="controlDict"),
        "application     solidFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        "writeInterval   10;\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSchemes",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSchemes"),
        "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         none;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSolution",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSolution"),
        "solvers\n{\n"
        "    D\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "}\n\n"
        "solidMechanics\n{\n"
        "    convergenceTolerance 1e-5;\n"
        "    nCorrectors         1;\n"
        "}\n",
        overwrite=True)

    # Mechanical properties
    write_foam_file(
        case_dir / "constant" / "mechanicalProperties",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="constant", object="mechanicalProperties"),
        f"E           {E};\n"
        f"nu          {nu};\n"
        f"alpha_th    12e-6;\n"
        "rho_s       7800;\n",
        overwrite=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solid_case(tmp_path):
    case_dir = tmp_path / "solid"
    _make_solid_case(case_dir, n_cells_x=2, n_cells_y=2, T_init=300.0, end_time=2, delta_t=1.0)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestSolidFoamInit:
    """Initialisation tests."""

    def test_case_loads(self, solid_case):
        from pyfoam.io.case import Case
        case = Case(solid_case)
        assert case.has_mesh()

    def test_solver_creates(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case, E=200e9, nu=0.3)
        assert solver.E == 200e9
        assert solver.nu == 0.3

    def test_lame_parameters(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case, E=200e9, nu=0.3)
        E, nu = 200e9, 0.3
        lam_expected = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu_expected = E / (2 * (1 + nu))
        assert abs(solver.lam - lam_expected) / lam_expected < 1e-10
        assert abs(solver.mu - mu_expected) / mu_expected < 1e-10

    def test_alpha_th_injection(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case, alpha_th=1e-5)
        assert solver.alpha_th == 1e-5

    def test_displacement_shape(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case)
        assert solver.D.shape == (4, 3)

    def test_temperature_shape(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case)
        assert solver.T.shape == (4,)

    def test_stress_shape(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case)
        assert solver.sigma.shape == (4, 6)


class TestSolidFoamThermalStrain:
    """Tests for thermal strain computation."""

    def test_zero_thermal_strain_at_ref_temp(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case, T_ref=300.0)
        # When T = T_ref, thermal strain should be zero
        solver.T = torch.full_like(solver.T, 300.0)
        solver.epsilon_th = solver._compute_thermal_strain()
        assert torch.allclose(solver.epsilon_th, torch.zeros_like(solver.epsilon_th))

    def test_positive_thermal_strain_when_hot(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case, T_ref=300.0, alpha_th=12e-6)
        solver.T = torch.full_like(solver.T, 600.0)
        solver.epsilon_th = solver._compute_thermal_strain()
        # Normal strains should be positive
        assert (solver.epsilon_th[:, 0] > 0).all()
        assert (solver.epsilon_th[:, 1] > 0).all()
        assert (solver.epsilon_th[:, 2] > 0).all()


class TestSolidFoamRun:
    """Solver execution tests."""

    def test_run_completes(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case, E=200e9, nu=0.3, alpha_th=12e-6)
        result = solver.run()
        assert "converged" in result
        assert "von_mises_max" in result

    def test_run_finite_values(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case)
        solver.run()
        assert torch.isfinite(solver.D).all()
        assert torch.isfinite(solver.sigma).all()

    def test_von_mises_positive(self, solid_case):
        from pyfoam.applications.solid_foam import SolidFoam
        solver = SolidFoam(solid_case)
        solver.run()
        von_mises = solver._compute_von_mises_stress()
        assert (von_mises >= 0).all()

    def test_exports_in_all(self):
        from pyfoam.applications import SolidFoam
        assert SolidFoam is not None
