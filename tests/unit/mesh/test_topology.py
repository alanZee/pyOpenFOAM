"""Tests for pyfoam.mesh.topology — face-cell connectivity utilities."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.topology import (
    boundary_face_mask,
    build_cell_to_faces,
    build_face_to_cells,
    cell_neighbours,
    count_internal_faces,
    internal_face_mask,
    validate_owner_neighbour,
)

# ---------------------------------------------------------------------------
# 2-cell hex mesh data (matches conftest)
# ---------------------------------------------------------------------------

_N_CELLS = 2
_N_FACES = 11
_N_INTERNAL = 1

_OWNER = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=INDEX_DTYPE)
_NEIGHBOUR = torch.tensor([1], dtype=INDEX_DTYPE)


# ===================================================================
# validate_owner_neighbour
# ===================================================================


class TestValidateOwnerNeighbour:
    """validate_owner_neighbour 校验 owner/neighbour 数组。"""

    def test_valid_mesh(self):
        """合法 mesh 不抛异常。"""
        validate_owner_neighbour(_OWNER, _NEIGHBOUR, _N_CELLS, _N_INTERNAL)

    def test_owner_length_mismatch(self):
        """owner 长度 < n_internal_faces 应报错。"""
        short_owner = torch.tensor([0], dtype=INDEX_DTYPE)
        with pytest.raises(ValueError, match="Total faces.*n_internal_faces"):
            validate_owner_neighbour(short_owner, _NEIGHBOUR, _N_CELLS, 5)

    def test_neighbour_length_mismatch(self):
        """neighbour 长度 != n_internal_faces 应报错。"""
        bad_nbr = torch.tensor([1, 0], dtype=INDEX_DTYPE)
        with pytest.raises(ValueError, match="neighbour length"):
            validate_owner_neighbour(_OWNER, bad_nbr, _N_CELLS, _N_INTERNAL)

    def test_owner_index_negative(self):
        """owner 含负索引应报错。"""
        bad = _OWNER.clone()
        bad[0] = -1
        with pytest.raises(ValueError, match="owner indices out of range"):
            validate_owner_neighbour(bad, _NEIGHBOUR, _N_CELLS, _N_INTERNAL)

    def test_owner_index_too_large(self):
        """owner 索引 >= n_cells 应报错。"""
        bad = _OWNER.clone()
        bad[0] = _N_CELLS
        with pytest.raises(ValueError, match="owner indices out of range"):
            validate_owner_neighbour(bad, _NEIGHBOUR, _N_CELLS, _N_INTERNAL)

    def test_neighbour_index_negative(self):
        """neighbour 含负索引应报错。"""
        bad = torch.tensor([-1], dtype=INDEX_DTYPE)
        with pytest.raises(ValueError, match="neighbour indices out of range"):
            validate_owner_neighbour(_OWNER, bad, _N_CELLS, _N_INTERNAL)

    def test_neighbour_index_too_large(self):
        """neighbour 索引 >= n_cells 应报错。"""
        bad = torch.tensor([99], dtype=INDEX_DTYPE)
        with pytest.raises(ValueError, match="neighbour indices out of range"):
            validate_owner_neighbour(_OWNER, bad, _N_CELLS, _N_INTERNAL)

    def test_owner_ge_neighbour_violation(self):
        """owner >= neighbour 违反 OpenFOAM 约定。"""
        owner = torch.tensor([1, 0], dtype=INDEX_DTYPE)
        nbr = torch.tensor([0], dtype=INDEX_DTYPE)
        with pytest.raises(ValueError, match="owner < neighbour"):
            validate_owner_neighbour(owner, nbr, _N_CELLS, 1)

    def test_zero_internal_faces(self):
        """n_internal_faces == 0 时 neighbour 为空也应通过。"""
        owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        nbr = torch.tensor([], dtype=INDEX_DTYPE)
        validate_owner_neighbour(owner, nbr, _N_CELLS, 0)


# ===================================================================
# internal_face_mask / boundary_face_mask
# ===================================================================


class TestFaceMasks:
    """internal_face_mask 和 boundary_face_mask 布尔掩码。"""

    def test_internal_mask_shape_and_values(self):
        mask = internal_face_mask(_N_FACES, _N_INTERNAL)
        assert mask.shape == (_N_FACES,)
        assert mask.dtype == torch.bool
        assert mask[:_N_INTERNAL].all()
        assert not mask[_N_INTERNAL:].any()

    def test_boundary_mask_shape_and_values(self):
        mask = boundary_face_mask(_N_FACES, _N_INTERNAL)
        assert mask.shape == (_N_FACES,)
        assert mask.dtype == torch.bool
        assert not mask[:_N_INTERNAL].any()
        assert mask[_N_INTERNAL:].all()

    def test_masks_are_complementary(self):
        """内部面 + 边界面 = 全部面。"""
        i_mask = internal_face_mask(_N_FACES, _N_INTERNAL)
        b_mask = boundary_face_mask(_N_FACES, _N_INTERNAL)
        assert (i_mask | b_mask).all()
        assert not (i_mask & b_mask).any()

    def test_zero_internal_faces(self):
        mask = internal_face_mask(5, 0)
        assert not mask.any()
        mask = boundary_face_mask(5, 0)
        assert mask.all()

    def test_all_internal_faces(self):
        mask = internal_face_mask(3, 3)
        assert mask.all()
        mask = boundary_face_mask(3, 3)
        assert not mask.any()


# ===================================================================
# count_internal_faces
# ===================================================================


class TestCountInternalFaces:
    """count_internal_faces 直接返回 neighbour 长度。"""

    def test_returns_neighbour_length(self):
        assert count_internal_faces(_OWNER, _NEIGHBOUR.shape[0]) == 1

    def test_zero(self):
        assert count_internal_faces(_OWNER, 0) == 0


# ===================================================================
# build_face_to_cells
# ===================================================================


class TestBuildFaceToCells:
    """build_face_to_cells 构建 (n_faces, 2) 面-单元映射。"""

    def test_shape_and_dtype(self):
        f2c = build_face_to_cells(_OWNER, _NEIGHBOUR, _N_INTERNAL)
        assert f2c.shape == (_N_FACES, 2)
        assert f2c.dtype == INDEX_DTYPE

    def test_owner_column(self):
        f2c = build_face_to_cells(_OWNER, _NEIGHBOUR, _N_INTERNAL)
        assert torch.equal(f2c[:, 0], _OWNER)

    def test_internal_faces_have_neighbour(self):
        f2c = build_face_to_cells(_OWNER, _NEIGHBOUR, _N_INTERNAL)
        assert torch.equal(f2c[:_N_INTERNAL, 1], _NEIGHBOUR)

    def test_boundary_faces_neighbour_is_neg1(self):
        f2c = build_face_to_cells(_OWNER, _NEIGHBOUR, _N_INTERNAL)
        assert (f2c[_N_INTERNAL:, 1] == -1).all()

    def test_zero_internal_faces(self):
        owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        nbr = torch.tensor([], dtype=INDEX_DTYPE)
        f2c = build_face_to_cells(owner, nbr, 0)
        assert f2c.shape == (2, 2)
        assert (f2c[:, 1] == -1).all()


# ===================================================================
# build_cell_to_faces
# ===================================================================


class TestBuildCellToFaces:
    """build_cell_to_faces 构建 cell->faces 列表。"""

    def test_length(self):
        c2f = build_cell_to_faces(_OWNER, _NEIGHBOUR, _N_CELLS, _N_INTERNAL)
        assert len(c2f) == _N_CELLS

    def test_cell0_faces(self):
        """Cell 0 owns faces 0-5 + 是 face 0 的 neighbour -> 6 个面。"""
        c2f = build_cell_to_faces(_OWNER, _NEIGHBOUR, _N_CELLS, _N_INTERNAL)
        cell0 = set(c2f[0].tolist())
        # owner of faces 0..5, neighbour of face 0
        assert cell0 == {0, 1, 2, 3, 4, 5}

    def test_cell1_faces(self):
        """Cell 1 owns faces 6-10 + 是 face 0 的 neighbour -> 6 个面。"""
        c2f = build_cell_to_faces(_OWNER, _NEIGHBOUR, _N_CELLS, _N_INTERNAL)
        cell1 = set(c2f[1].tolist())
        assert cell1 == {0, 6, 7, 8, 9, 10}

    def test_face_coverage(self):
        """所有面至少属于一个 cell。"""
        c2f = build_cell_to_faces(_OWNER, _NEIGHBOUR, _N_CELLS, _N_INTERNAL)
        all_faces = set()
        for faces in c2f:
            all_faces.update(faces.tolist())
        assert all_faces == set(range(_N_FACES))

    def test_boundary_only_mesh(self):
        """无内部面时，cell 只通过 owner 关联面。"""
        owner = torch.tensor([0, 0, 1], dtype=INDEX_DTYPE)
        nbr = torch.tensor([], dtype=INDEX_DTYPE)
        c2f = build_cell_to_faces(owner, nbr, 2, 0)
        assert set(c2f[0].tolist()) == {0, 1}
        assert set(c2f[1].tolist()) == {2}


# ===================================================================
# cell_neighbours
# ===================================================================


class TestCellNeighbours:
    """cell_neighbours 返回给定 cell 的邻居单元。"""

    def test_cell0_neighbour(self):
        """Cell 0 的邻居应为 [1]（通过内部面 0）。"""
        nbrs = cell_neighbours(0, _OWNER, _NEIGHBOUR, _N_INTERNAL)
        assert nbrs.tolist() == [1]

    def test_cell1_neighbour(self):
        """Cell 1 的邻居应为 [0]。"""
        nbrs = cell_neighbours(1, _OWNER, _NEIGHBOUR, _N_INTERNAL)
        assert nbrs.tolist() == [0]

    def test_unique_sorted(self):
        """结果应已去重排序。"""
        nbrs = cell_neighbours(0, _OWNER, _NEIGHBOUR, _N_INTERNAL)
        assert nbrs.shape == torch.unique(nbrs).shape

    def test_no_internal_faces(self):
        """无内部面时邻居为空。"""
        owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        nbr = torch.tensor([], dtype=INDEX_DTYPE)
        nbrs = cell_neighbours(0, owner, nbr, 0)
        assert nbrs.numel() == 0
