"""Tests for create_baffles — create baffle faces from internal faces."""
from __future__ import annotations
import pytest, torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_baffles import create_baffles
from tests.unit.mesh.conftest import make_fv_mesh


class TestBasic:
    def test_returns_fv_mesh(self):
        m = make_fv_mesh()
        r = create_baffles(m, [0])
        assert isinstance(r, FvMesh)

    def test_single_baffle(self):
        """一个 internal face → 两个 baffle boundary faces。"""
        m = make_fv_mesh()
        n_int_before = m.n_internal_faces
        r = create_baffles(m, [0])
        # 1 个 internal face 消失，变成 2 个 boundary faces
        assert r.n_internal_faces == n_int_before - 1
        assert r.n_faces == m.n_faces + 1  # -1 internal + 2 baffle = +1

    def test_baffle_patch_name(self):
        m = make_fv_mesh()
        r = create_baffles(m, [0], patch_name="my_baffle")
        names = [p["name"] for p in r.boundary]
        assert "my_baffle" in names

    def test_baffle_patch_type(self):
        m = make_fv_mesh()
        r = create_baffles(m, [0], patch_type="patch")
        for p in r.boundary:
            if p["name"] == "baffle":
                assert p["type"] == "patch"


class TestTopology:
    def test_owner_lt_neighbour(self):
        """所有新 internal faces 满足 owner < neighbour。"""
        m = make_fv_mesh()
        r = create_baffles(m, [0])
        for i in range(r.n_internal_faces):
            assert r.owner[i].item() < r.neighbour[i].item()

    def test_indices_valid(self):
        m = make_fv_mesh()
        r = create_baffles(m, [0])
        np_ = r.points.shape[0]
        for fi in range(r.n_faces):
            assert r.faces[fi].min().item() >= 0
            assert r.faces[fi].max().item() < np_
        assert r.owner.min().item() >= 0
        assert r.owner.max().item() < r.n_cells


class TestVolume:
    def test_preserves_volume(self):
        """baffle 操作不改变体积。"""
        m = make_fv_mesh()
        v0 = m.total_volume.item()
        r = create_baffles(m, [0])
        r.compute_geometry()
        assert abs(r.total_volume.item() - v0) < 1e-8


class TestEdgeCases:
    def test_invalid_face_index(self):
        m = make_fv_mesh()
        with pytest.raises(ValueError, match="not an internal face"):
            create_baffles(m, [m.n_faces - 1])

    def test_empty_selection(self):
        m = make_fv_mesh()
        r = create_baffles(m, [])
        assert r.n_faces == m.n_faces
        assert r.n_internal_faces == m.n_internal_faces

    def test_does_not_modify_original(self):
        m = make_fv_mesh()
        nc = m.n_internal_faces
        create_baffles(m, [0])
        assert m.n_internal_faces == nc
