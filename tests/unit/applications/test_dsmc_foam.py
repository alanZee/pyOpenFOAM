"""
Unit tests for DsmcFoam — Direct Simulation Monte Carlo solver.

Tests cover:
- DSMC particle creation and properties
- VHS collision model
- Particle initialisation and cell assignment
- Macroscopic field sampling
- Particle movement and re-indexing
- Collision mechanics
- Solver run to completion
- Field output
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

def _make_dsmc_case(
    case_dir: Path,
    n_cells: int = 4,
    L: float = 1.0,
    end_time: int = 2,
    delta_t: float = 0.01,
    T_init: float = 300.0,
) -> None:
    """Write a 1D DSMC case."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = L / n_cells
    dy = 0.1
    dz = 0.1

    # Points
    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.append((x, 0.0, 0.0))
        points.append((x, dy, 0.0))
        points.append((x, dy, dz))
        points.append((x, 0.0, dz))

    n_points = len(points)

    # Faces
    faces = []
    owner = []
    neighbour = []

    for i in range(n_cells - 1):
        faces.append((4, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)

    n_internal = len(neighbour)

    # Inlet
    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # Outlet
    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4 + 0, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # Empty patches (sides)
    empty_start = outlet_start + 1
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, (i + 1) * 4 + 0, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, i * 4 + 1, (i + 1) * 4 + 1, (i + 1) * 4 + 0))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)

    n_faces = len(faces)

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII, location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for p in points:
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
    n_empty = 4 * n_cells
    lines = ["2", "("]
    lines.append("    inlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {inlet_start};")
    lines.append("    }")
    lines.append("    outlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {outlet_start};")
    lines.append("    }")
    lines.append("    walls")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # 0/T
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    T_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    T_body = (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        f"internalField   uniform {T_init};\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_init};\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "T", T_header, T_body, overwrite=True)

    # 0/U
    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (1 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (1 0 0);\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # system/controlDict
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     dsmcFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n"
    )
    write_foam_file(sys_dir / "controlDict", cd_header, cd_body, overwrite=True)

    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
        "dsmc\n{\n"
        "    convergenceTolerance 1e-3;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)

    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    fs_body = (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dsmc_case(tmp_path):
    """Create a simple DSMC case."""
    case_dir = tmp_path / "dsmc"
    _make_dsmc_case(case_dir, n_cells=3, end_time=1, delta_t=0.01)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestDSMCParticle:
    """Tests for DSMCParticle dataclass."""

    def test_create_default(self):
        """DSMCParticle creates with sensible defaults."""
        from pyfoam.applications.dsmc_foam import DSMCParticle

        p = DSMCParticle()
        assert len(p.position) == 3
        assert len(p.velocity) == 3
        assert p.alive is True
        assert p.cell_id == -1

    def test_create_custom(self):
        """DSMCParticle creates with custom parameters."""
        from pyfoam.applications.dsmc_foam import DSMCParticle

        p = DSMCParticle(
            position=[1.0, 2.0, 3.0],
            velocity=[10.0, 0.0, 0.0],
            cell_id=5,
            mass=1e-26,
            n_real=1e10,
        )
        assert p.position == [1.0, 2.0, 3.0]
        assert p.cell_id == 5
        assert p.mass == 1e-26
        assert p.n_real == 1e10


class TestVHSModel:
    """Tests for VHS collision model."""

    def test_create_default(self):
        """VHSModel creates with N2 defaults."""
        from pyfoam.applications.dsmc_foam import VHSModel

        vhs = VHSModel()
        assert vhs.d_ref > 0
        assert vhs.T_ref > 0
        assert 0 <= vhs.omega <= 2.0

    def test_custom_parameters(self):
        """VHSModel accepts custom parameters."""
        from pyfoam.applications.dsmc_foam import VHSModel

        vhs = VHSModel(d_ref=5e-10, T_ref=273.0, omega=0.5)
        assert vhs.d_ref == 5e-10
        assert vhs.omega == 0.5


class TestDsmcFoamInit:
    """Tests for DsmcFoam initialization."""

    def test_case_loads(self, dsmc_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(dsmc_case)
        assert case.has_mesh()

    def test_solver_creates(self, dsmc_case):
        """DsmcFoam creates successfully."""
        from pyfoam.applications.dsmc_foam import DsmcFoam

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=5, seed=42)
        assert solver.n_cells == 3
        assert solver.n_particles_per_cell == 5

    def test_particles_initialized(self, dsmc_case):
        """Particles are placed in all cells."""
        from pyfoam.applications.dsmc_foam import DsmcFoam

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=8, seed=42)
        assert solver.total_particles == 3 * 8

    def test_particle_positions_finite(self, dsmc_case):
        """All particle positions are finite after initialization."""
        from pyfoam.applications.dsmc_foam import DsmcFoam

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=5, seed=42)
        for cell_id in range(solver.n_cells):
            for p in solver.particles[cell_id]:
                for coord in p.position:
                    assert math.isfinite(coord)
                for vel in p.velocity:
                    assert math.isfinite(vel)


class TestDSMCSampling:
    """Tests for macroscopic sampling from particles."""

    def test_sample_shape(self, dsmc_case):
        """Sampled fields have correct shape."""
        from pyfoam.applications.dsmc_foam import DsmcFoam

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=10, seed=42)
        solver._sample_macroscopic()

        assert solver.T.shape == (3,)
        assert solver.U.shape == (3, 3)
        assert solver.rho.shape == (3,)

    def test_sample_temperature_positive(self, dsmc_case):
        """Sampled temperature is positive."""
        from pyfoam.applications.dsmc_foam import DsmcFoam

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=10, seed=42)
        solver._sample_macroscopic()

        assert (solver.T > 0).all(), "Temperature must be positive"

    def test_sample_fields_finite(self, dsmc_case):
        """All sampled fields are finite."""
        from pyfoam.applications.dsmc_foam import DsmcFoam

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=10, seed=42)
        solver._sample_macroscopic()

        assert torch.isfinite(solver.T).all()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.rho).all()


class TestDSMCCollision:
    """Tests for DSMC collision mechanics."""

    def test_vhs_collision_conserves_momentum(self, dsmc_case):
        """VHS collision conserves momentum."""
        from pyfoam.applications.dsmc_foam import DsmcFoam, DSMCParticle

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=2, seed=42)

        import random
        rng = random.Random(123)

        p1 = DSMCParticle(velocity=[100.0, 0.0, 0.0], mass=1e-26)
        p2 = DSMCParticle(velocity=[-50.0, 0.0, 0.0], mass=1e-26)

        # 记录碰撞前动量
        px_before = p1.mass * p1.velocity[0] + p2.mass * p2.velocity[0]
        py_before = p1.mass * p1.velocity[1] + p2.mass * p2.velocity[1]
        pz_before = p1.mass * p1.velocity[2] + p2.mass * p2.velocity[2]

        g_rel = [
            p1.velocity[0] - p2.velocity[0],
            p1.velocity[1] - p2.velocity[1],
            p1.velocity[2] - p2.velocity[2],
        ]
        g_mag = math.sqrt(sum(g ** 2 for g in g_rel))

        solver._vhs_collision(p1, p2, g_rel, g_mag, rng)

        # 检查碰撞后动量守恒
        px_after = p1.mass * p1.velocity[0] + p2.mass * p2.velocity[0]
        py_after = p1.mass * p1.velocity[1] + p2.mass * p2.velocity[1]
        pz_after = p1.mass * p1.velocity[2] + p2.mass * p2.velocity[2]

        assert abs(px_after - px_before) < 1e-10, "X-momentum not conserved"
        assert abs(py_after - py_before) < 1e-10, "Y-momentum not conserved"
        assert abs(pz_after - pz_before) < 1e-10, "Z-momentum not conserved"

    def test_vhs_collision_conserves_energy(self, dsmc_case):
        """VHS collision conserves kinetic energy."""
        from pyfoam.applications.dsmc_foam import DsmcFoam, DSMCParticle

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=2, seed=42)

        import random
        rng = random.Random(456)

        p1 = DSMCParticle(velocity=[200.0, 50.0, -30.0], mass=1e-26)
        p2 = DSMCParticle(velocity=[-100.0, 20.0, 10.0], mass=1e-26)

        def kinetic_energy():
            return 0.5 * p1.mass * sum(v ** 2 for v in p1.velocity) + \
                   0.5 * p2.mass * sum(v ** 2 for v in p2.velocity)

        ke_before = kinetic_energy()

        g_rel = [
            p1.velocity[0] - p2.velocity[0],
            p1.velocity[1] - p2.velocity[1],
            p1.velocity[2] - p2.velocity[2],
        ]
        g_mag = math.sqrt(sum(g ** 2 for g in g_rel))

        solver._vhs_collision(p1, p2, g_rel, g_mag, rng)

        ke_after = kinetic_energy()
        assert abs(ke_after - ke_before) / max(ke_before, 1e-30) < 1e-8, \
            "Kinetic energy not conserved"


class TestDsmcFoamRun:
    """Tests for the full DSMC solver run."""

    def test_run_completes(self, dsmc_case):
        """DSMC solver runs without errors."""
        from pyfoam.applications.dsmc_foam import DsmcFoam

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=5, seed=42)
        result = solver.run()

        assert "converged" in result
        assert "n_particles_total" in result

    def test_particles_survive(self, dsmc_case):
        """Particles survive after a short run."""
        from pyfoam.applications.dsmc_foam import DsmcFoam

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=5, seed=42)
        solver.run()

        assert solver.total_particles > 0, "All particles died"

    def test_temperature_stays_positive(self, dsmc_case):
        """Temperature remains positive after run."""
        from pyfoam.applications.dsmc_foam import DsmcFoam

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=8, seed=42)
        solver.run()

        assert (solver.T > 0).all(), "Temperature dropped to zero or negative"

    def test_fields_finite(self, dsmc_case):
        """All macroscopic fields are finite after run."""
        from pyfoam.applications.dsmc_foam import DsmcFoam

        solver = DsmcFoam(dsmc_case, n_particles_per_cell=8, seed=42)
        solver.run()

        assert torch.isfinite(solver.T).all(), "T has NaN/Inf"
        assert torch.isfinite(solver.U).all(), "U has NaN/Inf"
