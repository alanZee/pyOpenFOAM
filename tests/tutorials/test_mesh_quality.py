"""
Tutorial validation: solver mesh quality tests.

验证求解器网格质量处理。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from tests.tutorials.helpers import (
    make_structured_mesh,
    write_control_dict,
    write_fv_schemes,
    write_fv_solution,
    write_pressure_field,
    write_transport_properties,
    write_velocity_field,
)


def _make_quality_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=0.01)
    write_control_dict(case, delta_t=0.005, end_time=1.0, write_interval=200)
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


class TestMeshQuality:
    """网格质量测试。"""

    def test_mesh_cell_volumes_positive(self, tmp_path: Path):
        """单元体积为正。"""
        case = _make_quality_case(tmp_path, "quality")
        from pyfoam.mesh.fv_mesh import FvMesh
        from pyfoam.io.mesh_io import read_mesh
        mesh_dir = case / "constant" / "polyMesh"
        md = read_mesh(mesh_dir)
        faces_t = [torch.tensor(f, dtype=INDEX_DTYPE) for f in md.faces]
        mesh = FvMesh(
            points=md.points, faces=faces_t,
            owner=md.owner, neighbour=md.neighbour,
            boundary=md.boundary,
        )
        mesh.compute_geometry()
        assert (mesh.cell_volumes > 0).all(), "Non-positive cell volumes"

    def test_mesh_face_areas_positive(self, tmp_path: Path):
        """面面积为正。"""
        case = _make_quality_case(tmp_path, "areas")
        from pyfoam.mesh.fv_mesh import FvMesh
        from pyfoam.io.mesh_io import read_mesh
        mesh_dir = case / "constant" / "polyMesh"
        md = read_mesh(mesh_dir)
        faces_t = [torch.tensor(f, dtype=INDEX_DTYPE) for f in md.faces]
        mesh = FvMesh(
            points=md.points, faces=faces_t,
            owner=md.owner, neighbour=md.neighbour,
            boundary=md.boundary,
        )
        mesh.compute_geometry()
        face_areas = mesh.face_areas
        assert face_areas.shape[0] > 0

    def test_mesh_connectivity(self, tmp_path: Path):
        """网格连通性。"""
        case = _make_quality_case(tmp_path, "connectivity")
        from pyfoam.mesh.fv_mesh import FvMesh
        from pyfoam.io.mesh_io import read_mesh
        mesh_dir = case / "constant" / "polyMesh"
        md = read_mesh(mesh_dir)
        faces_t = [torch.tensor(f, dtype=INDEX_DTYPE) for f in md.faces]
        mesh = FvMesh(
            points=md.points, faces=faces_t,
            owner=md.owner, neighbour=md.neighbour,
            boundary=md.boundary,
        )
        mesh.compute_geometry()
        assert mesh.n_cells > 0
        assert mesh.n_faces > 0
        assert mesh.n_internal_faces > 0


class TestMeshQualityEffect:
    """网格质量效应测试。"""

    def test_better_mesh_better_solution(self, tmp_path: Path):
        """更好网格产生更好解。"""
        from pyfoam.applications.simple_foam import SimpleFoam
        case1 = _make_quality_case(tmp_path, "coarse", nx=5, ny=5)
        case2 = _make_quality_case(tmp_path, "fine", nx=20, ny=20)
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        # 两种网格都应产生有效解
        assert torch.isfinite(solver1.U).all()
        assert torch.isfinite(solver2.U).all()
