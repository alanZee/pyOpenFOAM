"""
Tutorial validation: incompressible fluid cases.

Runs pyOpenFOAM on standard OpenFOAM tutorial configurations and
validates physical correctness (no NaN, correct trends).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.io.mesh_io import read_mesh
from tests.tutorials.helpers import (
    make_structured_mesh,
    write_control_dict,
    write_fv_schemes,
    write_fv_solution,
    write_pressure_field,
    write_transport_properties,
    write_velocity_field,
)


def _load_mesh(mesh_dir: Path) -> FvMesh:
    """Load an FvMesh from a polyMesh directory."""
    md = read_mesh(mesh_dir)
    faces_t = [torch.tensor(f, dtype=INDEX_DTYPE) for f in md.faces]
    mesh = FvMesh(
        points=md.points, faces=faces_t,
        owner=md.owner, neighbour=md.neighbour,
        boundary=md.boundary,
    )
    mesh.compute_geometry()
    return mesh


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def cavity_case(tmp_path: Path) -> Path:
    case = tmp_path / "cavity"
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=10, ny=10)
    write_transport_properties(case, nu=0.01)
    write_control_dict(case, delta_t=0.005, end_time=0.5, write_interval=100)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_velocity_field(
        case,
        patches={"movingWall": (1, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


@pytest.fixture
def couette_case(tmp_path: Path) -> Path:
    """Plane Couette: long domain with inlet/outlet for PISO."""
    case = tmp_path / "couette"
    mesh_dir = case / "constant" / "polyMesh"
    # Use long domain (100:1 aspect) to minimize inlet/outlet effects
    make_structured_mesh(mesh_dir, nx=20, ny=5, x_range=(0.0, 10.0), y_range=(0.0, 1.0))
    write_transport_properties(case, nu=0.01)
    write_control_dict(case, delta_t=0.001, end_time=0.1, write_interval=100)
    write_fv_schemes(case)
    write_fv_solution(case, algorithm="PISO")
    # Inlet at left, outlet at right — need to add these patches
    # For now, use zeroGradient on left/right (acts as open boundaries)
    write_velocity_field(
        case,
        patches={"movingWall": (1, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


# ── Tests ────────────────────────────────────────────────────────────────

class TestLidDrivenCavity:
    """Lid-driven cavity at Re=100 (SIMPLE algorithm)."""

    def test_simple_completes(self, cavity_case: Path):
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(cavity_case)
        solver.run()
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_velocity_bounded(self, cavity_case: Path):
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(cavity_case)
        solver.run()
        u_max = solver.U[:, 0].max().item()
        assert u_max <= 1.5, f"u_max={u_max:.3f} exceeds physical bound"

    def test_cavity_recirculation(self, cavity_case: Path):
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(cavity_case)
        solver.run()
        u = solver.U[:, 0]
        assert (u < 0).any(), "No recirculation detected"


class TestPlaneCouette:
    """Plane Couette flow — requires inlet/outlet patches (covered in tests/validation/)."""

    @pytest.mark.xfail(reason="Couette needs inlet/outlet patches; covered in validation/test_couette_flow.py")
    def test_piso_completes(self, couette_case: Path):
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(couette_case)
        solver.run()
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    @pytest.mark.xfail(reason="Couette needs inlet/outlet patches; covered in validation/test_couette_flow.py")
    def test_linear_velocity_profile(self, couette_case: Path):
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(couette_case)
        solver.run()
        y = solver.mesh.cell_centres[:, 1]
        u = solver.U[:, 0]
        sorted_idx = torch.argsort(y)
        u_sorted = u[sorted_idx]
        u_diff = u_sorted[1:] - u_sorted[:-1]
        n_positive = (u_diff > -0.01).sum().item()
        assert n_positive > len(u_diff) * 0.8, "Velocity profile not approximately monotonic"


class TestTutorialMeshGeneration:
    """Verify the structured mesh helper produces valid meshes."""

    def test_mesh_connectivity(self, tmp_path: Path):
        mesh_dir = tmp_path / "mesh"
        make_structured_mesh(mesh_dir, nx=5, ny=5)
        mesh = _load_mesh(mesh_dir)
        assert mesh.n_cells == 25
        assert mesh.n_internal_faces > 0
        assert mesh.n_faces > mesh.n_internal_faces, "No boundary faces"

    def test_cell_volumes_positive(self, tmp_path: Path):
        mesh_dir = tmp_path / "mesh"
        make_structured_mesh(mesh_dir, nx=5, ny=5)
        mesh = _load_mesh(mesh_dir)
        assert (mesh.cell_volumes > 0).all(), "Non-positive cell volumes detected"
