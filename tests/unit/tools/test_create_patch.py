"""Tests for create_patch — create a new boundary patch from selected faces."""
from __future__ import annotations
import pytest, torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_patch import create_patch
from tests.unit.mesh.conftest import make_fv_mesh


class TestBasic:
    def test_returns_fv_mesh(self):
        m = make_fv_mesh()
        r = create_patch(m, [1], "new_patch")
        assert isinstance(r, FvMesh)

    def test_new_patch_in_boundary(self):
        m = make_fv_mesh()
        r = create_patch(m, [1], "new_patch")
        names = [p["name"] for p in r.boundary]
        assert "new_patch" in names

    def test_boundary_face_moved(self):
        """移动一个 boundary face 到新 patch。"""
        m = make_fv_mesh()
        # face 1 是 boundary face (bottom patch)
        r = create_patch(m, [1], "moved")
        moved_patch = next(p for p in r.boundary if p["name"] == "moved")
        assert moved_patch["nFaces"] == 1

    def test_internal_face_creates_old_internal(self):
        """将 internal face 移到新 patch → 产生 oldInternal。"""
        m = make_fv_mesh()
        r = create_patch(m, [0], "new_internal")
        names = [p["name"] for p in r.boundary]
        assert "new_internal" in names
        assert "oldInternal" in names
        # internal face 移走后应该减少
        assert r.n_internal_faces < m.n_internal_faces


class TestNaming:
    def test_duplicate_name_raises(self):
        m = make_fv_mesh()
        with pytest.raises(ValueError, match="already exists"):
            create_patch(m, [1], "bottom")  # 'bottom' 已存在

    def test_custom_patch_type(self):
        m = make_fv_mesh()
        r = create_patch(m, [1], "inlet", patch_type="patch")
        inlet = next(p for p in r.boundary if p["name"] == "inlet")
        assert inlet["type"] == "patch"


class TestTopology:
    def test_owner_lt_neighbour(self):
        m = make_fv_mesh()
        r = create_patch(m, [1], "new")
        for i in range(r.n_internal_faces):
            assert r.owner[i].item() < r.neighbour[i].item()

    def test_indices_valid(self):
        m = make_fv_mesh()
        r = create_patch(m, [1], "new")
        np_ = r.points.shape[0]
        for fi in range(r.n_faces):
            assert r.faces[fi].min().item() >= 0
            assert r.faces[fi].max().item() < np_


class TestVolume:
    def test_preserves_volume(self):
        m = make_fv_mesh()
        v0 = m.total_volume.item()
        r = create_patch(m, [1], "new")
        r.compute_geometry()
        assert abs(r.total_volume.item() - v0) < 1e-8


class TestEdgeCases:
    def test_does_not_modify_original(self):
        m = make_fv_mesh()
        nc = m.n_cells
        create_patch(m, [1], "new")
        assert m.n_cells == nc
