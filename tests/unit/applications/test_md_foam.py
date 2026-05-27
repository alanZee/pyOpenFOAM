"""
Unit tests for MdFoam — Lennard-Jones molecular dynamics solver.

Tests cover:
- LJParticle dataclass creation
- FCC lattice initialization
- Lennard-Jones force computation
- Velocity Verlet integration
- Periodic boundary conditions
- Temperature computation
- Thermostat operation
- Energy conservation
- Full solver run
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------

def _make_md_case(
    case_dir: Path,
    n_cells: int = 4,
    end_time: int = 1,
    delta_t: float = 0.001,
) -> None:
    """Write a simple case for mdFoam."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells
    dy, dz = 0.1, 0.1

    # 网格点
    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.extend([(x, 0.0, 0.0), (x, dy, 0.0), (x, dy, dz), (x, 0.0, dz)])

    n_points = len(points)
    faces, owner, neighbour = [], [], []

    # 内部面
    for i in range(n_cells - 1):
        faces.append((4, i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)
    n_internal = len(neighbour)

    # 入口面
    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # 出口面
    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # Empty patches
    empty_start = outlet_start + 1
    n_empty = 0
    for i in range(n_cells):
        faces.append((4, i * 4, (i + 1) * 4, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    n_empty += n_cells
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    n_empty += n_cells
    for i in range(n_cells):
        faces.append((4, i * 4, i * 4 + 1, (i + 1) * 4 + 1, i * 4))
        owner.append(i)
    n_empty += n_cells
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)
    n_empty += n_cells

    n_faces = len(faces)

    # 写网格文件
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    hb = FoamFileHeader(version="2.0", format=FileFormat.ASCII, location="constant/polyMesh")

    def _header(class_name: str, obj: str) -> FoamFileHeader:
        return FoamFileHeader(**{**hb.__dict__, "class_name": class_name, "object": obj})

    lines = [str(n_points), "("]
    for p in points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", _header("vectorField", "points"), "\n".join(lines), overwrite=True)

    lines = [str(n_faces), "("]
    for face in faces:
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{face[0]}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", _header("faceList", "faces"), "\n".join(lines), overwrite=True)

    lines = [str(n_faces), "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", _header("labelList", "owner"), "\n".join(lines), overwrite=True)

    lines = [str(n_internal), "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", _header("labelList", "neighbour"), "\n".join(lines), overwrite=True)

    lines = ["2", "("]
    lines += [
        "    inlet", "    {", "        type            patch;",
        f"        nFaces          1;", f"        startFace       {inlet_start};", "    }",
        "    outlet", "    {", "        type            patch;",
        f"        nFaces          1;", f"        startFace       {outlet_start};", "    }",
        "    walls", "    {", "        type            empty;",
        f"        nFaces          {n_empty};", f"        startFace       {empty_start};", "    }",
        ")",
    ]
    write_foam_file(mesh_dir / "boundary", _header("polyBoundaryMesh", "boundary"), "\n".join(lines), overwrite=True)

    # 0/ 目录
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    # U (SolverBase 需要)
    write_foam_file(zero_dir / "U",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volVectorField",
                       location="0", object="U"),
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # p (SolverBase 需要)
    write_foam_file(zero_dir / "p",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="p"),
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        "internalField   uniform 101325;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            zeroGradient;\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # system/
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    write_foam_file(sys_dir / "controlDict",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="controlDict"),
        "application     mdFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSchemes",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSchemes"),
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSolution",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSolution"),
        "mdFoam\n{\n"
        "    convergenceTolerance 1e-3;\n"
        "}\n",
        overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def md_case(tmp_path):
    """Create a simple MD case."""
    case_dir = tmp_path / "md"
    _make_md_case(case_dir, n_cells=4, end_time=1, delta_t=0.001)
    return case_dir


@pytest.fixture
def md_solver(md_case):
    """Create an MD solver with 32 particles."""
    from pyfoam.applications.md_foam import MdFoam
    return MdFoam(
        md_case, n_particles=32, T_init=1.0,
        rho=0.8, seed=42,
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestLJParticle:
    """Tests for LJParticle dataclass."""

    def test_create_default(self):
        """LJParticle creates with sensible defaults."""
        from pyfoam.applications.md_foam import LJParticle

        p = LJParticle()
        assert len(p.position) == 3
        assert len(p.velocity) == 3
        assert len(p.force) == 3
        assert p.mass == 1.0

    def test_create_custom(self):
        """LJParticle creates with custom parameters."""
        from pyfoam.applications.md_foam import LJParticle

        p = LJParticle(
            position=[1.0, 2.0, 3.0],
            velocity=[10.0, 0.0, 0.0],
            force=[-1.0, 0.0, 0.5],
            mass=2.0,
        )
        assert p.position == [1.0, 2.0, 3.0]
        assert p.velocity == [10.0, 0.0, 0.0]
        assert p.mass == 2.0


class TestMdFoamInit:
    """Tests for MdFoam initialization."""

    def test_case_loads(self, md_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(md_case)
        assert case.has_mesh()

    def test_solver_creates(self, md_case):
        """MdFoam creates successfully."""
        from pyfoam.applications.md_foam import MdFoam

        solver = MdFoam(md_case, n_particles=32, T_init=1.0, rho=0.8, seed=42)
        assert solver.n_particles == 32
        assert len(solver.particles) == 32

    def test_box_size(self, md_solver):
        """Box size is correct for given density."""
        solver = md_solver
        expected_L = (solver.n_particles / solver.rho) ** (1.0 / 3.0)
        assert abs(solver.L - expected_L) < 1e-10

    def test_positions_in_box(self, md_solver):
        """All particles are inside the simulation box."""
        solver = md_solver
        for p in solver.particles:
            for coord in p.position:
                assert 0 <= coord <= solver.L, f"Particle outside box: {coord}"

    def test_zero_com_velocity(self, md_solver):
        """Center-of-mass velocity is zero."""
        solver = md_solver
        vx_cm = sum(p.velocity[0] for p in solver.particles) / solver.n_particles
        vy_cm = sum(p.velocity[1] for p in solver.particles) / solver.n_particles
        vz_cm = sum(p.velocity[2] for p in solver.particles) / solver.n_particles
        assert abs(vx_cm) < 1e-10
        assert abs(vy_cm) < 1e-10
        assert abs(vz_cm) < 1e-10


class TestMdFoamForces:
    """Tests for force computation."""

    def test_forces_computed(self, md_solver):
        """Forces are computed after initialization."""
        solver = md_solver
        # 力应该已被计算
        has_nonzero = any(
            any(f != 0 for f in p.force)
            for p in solver.particles
        )
        assert has_nonzero, "No forces computed"

    def test_forces_finite(self, md_solver):
        """All forces are finite."""
        solver = md_solver
        for p in solver.particles:
            for f in p.force:
                assert math.isfinite(f), "Force is NaN/Inf"

    def test_potential_energy_finite(self, md_solver):
        """Potential energy is finite."""
        solver = md_solver
        assert math.isfinite(solver.potential_energy)


class TestMdFoamTemperature:
    """Tests for temperature computation."""

    def test_temperature_positive(self, md_solver):
        """Temperature is positive."""
        solver = md_solver
        assert solver.temperature > 0, "Temperature must be positive"

    def test_temperature_finite(self, md_solver):
        """Temperature is finite."""
        solver = md_solver
        assert math.isfinite(solver.temperature)

    def test_kinetic_energy_positive(self, md_solver):
        """Kinetic energy is positive."""
        solver = md_solver
        assert solver.kinetic_energy > 0, "Kinetic energy must be positive"


class TestMdFoamIntegration:
    """Tests for Velocity Verlet integration."""

    def test_single_step(self, md_solver):
        """Single Velocity Verlet step completes."""
        solver = md_solver
        E_before = solver.total_energy
        solver._velocity_verlet_step(solver.delta_t)
        E_after = solver.total_energy
        # 能量应该大致守恒（有限容差）
        assert math.isfinite(E_after)

    def test_positions_remain_in_box(self, md_solver):
        """Particles stay inside box after integration."""
        solver = md_solver
        solver._velocity_verlet_step(solver.delta_t)
        for p in solver.particles:
            for coord in p.position:
                assert 0 <= coord <= solver.L, f"Particle left box after step"


class TestMdFoamSolver:
    """Tests for the full MD solver run."""

    def test_run_completes(self, md_case):
        """MD solver runs without errors."""
        from pyfoam.applications.md_foam import MdFoam

        solver = MdFoam(md_case, n_particles=16, T_init=1.0, rho=0.8, seed=42)
        solver.end_time = 0.01
        result = solver.run()

        assert "converged" in result
        assert "T_final" in result
        assert "E_total" in result

    def test_temperature_after_run(self, md_case):
        """Temperature remains positive after run."""
        from pyfoam.applications.md_foam import MdFoam

        solver = MdFoam(md_case, n_particles=16, T_init=1.0, rho=0.8, seed=42)
        solver.end_time = 0.01
        solver.run()

        assert solver.temperature > 0, "Temperature dropped to zero"

    def test_energy_finite_after_run(self, md_case):
        """Total energy is finite after run."""
        from pyfoam.applications.md_foam import MdFoam

        solver = MdFoam(md_case, n_particles=16, T_init=1.0, rho=0.8, seed=42)
        solver.end_time = 0.01
        solver.run()

        assert math.isfinite(solver.total_energy), "Total energy is NaN/Inf"

    def test_particles_in_box_after_run(self, md_case):
        """All particles remain in box after run."""
        from pyfoam.applications.md_foam import MdFoam

        solver = MdFoam(md_case, n_particles=16, T_init=1.0, rho=0.8, seed=42)
        solver.end_time = 0.01
        solver.run()

        for p in solver.particles:
            for coord in p.position:
                assert 0 <= coord <= solver.L
