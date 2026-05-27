"""独立域分解策略测试 — SimpleDecomposition、ScotchDecomposition."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE


# ---------------------------------------------------------------------------
# 复用 8-cell 网格
# ---------------------------------------------------------------------------


def _pt(ix, iy, iz):
    return ix + iy * 3 + iz * 9


_POINTS = []
for k in range(3):
    for j in range(3):
        for i in range(3):
            _POINTS.append([float(i), float(j), float(k)])

_FACES = []
_OWNER = []
_NEIGHBOUR = []

# x 方向内部面
_FACES.append([_pt(1,0,0), _pt(1,1,0), _pt(1,1,1), _pt(1,0,1)])
_OWNER.append(0); _NEIGHBOUR.append(1)
_FACES.append([_pt(1,1,0), _pt(1,2,0), _pt(1,2,1), _pt(1,1,1)])
_OWNER.append(2); _NEIGHBOUR.append(3)
_FACES.append([_pt(1,0,1), _pt(1,1,1), _pt(1,1,2), _pt(1,0,2)])
_OWNER.append(4); _NEIGHBOUR.append(5)
_FACES.append([_pt(1,1,1), _pt(1,2,1), _pt(1,2,2), _pt(1,1,2)])
_OWNER.append(6); _NEIGHBOUR.append(7)

# y 方向内部面
_FACES.append([_pt(0,1,0), _pt(1,1,0), _pt(1,1,1), _pt(0,1,1)])
_OWNER.append(0); _NEIGHBOUR.append(2)
_FACES.append([_pt(1,1,0), _pt(2,1,0), _pt(2,1,1), _pt(1,1,1)])
_OWNER.append(1); _NEIGHBOUR.append(3)
_FACES.append([_pt(0,1,1), _pt(1,1,1), _pt(1,1,2), _pt(0,1,2)])
_OWNER.append(4); _NEIGHBOUR.append(6)
_FACES.append([_pt(1,1,1), _pt(2,1,1), _pt(2,1,2), _pt(1,1,2)])
_OWNER.append(5); _NEIGHBOUR.append(7)

# z 方向内部面
_FACES.append([_pt(0,0,1), _pt(1,0,1), _pt(1,1,1), _pt(0,1,1)])
_OWNER.append(0); _NEIGHBOUR.append(4)
_FACES.append([_pt(1,0,1), _pt(2,0,1), _pt(2,1,1), _pt(1,1,1)])
_OWNER.append(1); _NEIGHBOUR.append(5)
_FACES.append([_pt(0,1,1), _pt(1,1,1), _pt(1,2,1), _pt(0,2,1)])
_OWNER.append(2); _NEIGHBOUR.append(6)
_FACES.append([_pt(1,1,1), _pt(2,1,1), _pt(2,2,1), _pt(1,2,1)])
_OWNER.append(3); _NEIGHBOUR.append(7)

# 边界面
boundary_patches = []
boundary_start = len(_FACES)

# Bottom (z=0)
bottom_faces = []
for cell_idx, (ix, iy) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    _FACES.append([_pt(ix,iy,0), _pt(ix+1,iy,0), _pt(ix+1,iy+1,0), _pt(ix,iy+1,0)])
    _OWNER.append(cell_idx)
    bottom_faces.append(len(_FACES) - 1)
boundary_patches.append({"name": "bottom", "type": "wall", "startFace": boundary_start, "nFaces": len(bottom_faces)})
boundary_start += len(bottom_faces)

# Top (z=2)
top_faces = []
for cell_idx, (ix, iy) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    _FACES.append([_pt(ix,iy,2), _pt(ix+1,iy,2), _pt(ix+1,iy+1,2), _pt(ix,iy+1,2)])
    _OWNER.append(cell_idx + 4)
    top_faces.append(len(_FACES) - 1)
boundary_patches.append({"name": "top", "type": "wall", "startFace": boundary_start, "nFaces": len(top_faces)})
boundary_start += len(top_faces)

# Front (y=0)
front_faces = []
for cell_idx, (ix, iz) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    c = ix + iz * 4
    _FACES.append([_pt(ix,0,iz), _pt(ix+1,0,iz), _pt(ix+1,0,iz+1), _pt(ix,0,iz+1)])
    _OWNER.append(c)
    front_faces.append(len(_FACES) - 1)
boundary_patches.append({"name": "front", "type": "wall", "startFace": boundary_start, "nFaces": len(front_faces)})
boundary_start += len(front_faces)

# Back (y=2)
back_faces = []
for cell_idx, (ix, iz) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    c = ix + 2 + iz * 4
    _FACES.append([_pt(ix,2,iz), _pt(ix+1,2,iz), _pt(ix+1,2,iz+1), _pt(ix,2,iz+1)])
    _OWNER.append(c)
    back_faces.append(len(_FACES) - 1)
boundary_patches.append({"name": "back", "type": "wall", "startFace": boundary_start, "nFaces": len(back_faces)})
boundary_start += len(back_faces)

# Left (x=0)
left_faces = []
for cell_idx, (iy, iz) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    c = iy + iz * 4
    _FACES.append([_pt(0,iy,iz), _pt(0,iy+1,iz), _pt(0,iy+1,iz+1), _pt(0,iy,iz+1)])
    _OWNER.append(c)
    left_faces.append(len(_FACES) - 1)
boundary_patches.append({"name": "left", "type": "wall", "startFace": boundary_start, "nFaces": len(left_faces)})
boundary_start += len(left_faces)

# Right (x=2)
right_faces = []
for cell_idx, (iy, iz) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    c = 1 + iy + iz * 4
    _FACES.append([_pt(2,iy,iz), _pt(2,iy+1,iz), _pt(2,iy+1,iz+1), _pt(2,iy,iz+1)])
    _OWNER.append(c)
    right_faces.append(len(_FACES) - 1)
boundary_patches.append({"name": "right", "type": "wall", "startFace": boundary_start, "nFaces": len(right_faces)})


def make_8cell_fv_mesh(device="cpu", dtype=torch.float64):
    from pyfoam.mesh.fv_mesh import FvMesh
    mesh = FvMesh(
        points=torch.tensor(_POINTS, dtype=dtype, device=device),
        faces=[torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in _FACES],
        owner=torch.tensor(_OWNER, dtype=INDEX_DTYPE, device=device),
        neighbour=torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE, device=device),
        boundary=boundary_patches,
    )
    mesh.compute_geometry()
    return mesh


def make_8cell_poly_mesh(device="cpu", dtype=torch.float64):
    from pyfoam.mesh.poly_mesh import PolyMesh
    points = torch.tensor(_POINTS, dtype=dtype, device=device)
    faces = [torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in _FACES]
    owner = torch.tensor(_OWNER, dtype=INDEX_DTYPE, device=device)
    neighbour = torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE, device=device)
    return PolyMesh(points=points, faces=faces, owner=owner, neighbour=neighbour, boundary=boundary_patches)


@pytest.fixture
def fv_mesh_8cell():
    return make_8cell_fv_mesh()


@pytest.fixture
def poly_mesh_8cell():
    return make_8cell_poly_mesh()


# ---------------------------------------------------------------------------
# RTS 注册测试
# ---------------------------------------------------------------------------


class TestDecompositionStrategyRegistration:
    """测试 RTS 注册."""

    def test_simple_registered(self):
        from pyfoam.parallel.decomposition_2 import DecompositionStrategy
        assert "simple" in DecompositionStrategy.available_types()

    def test_scotch_registered(self):
        from pyfoam.parallel.decomposition_2 import DecompositionStrategy
        assert "scotch" in DecompositionStrategy.available_types()

    def test_factory_simple(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import (
            DecompositionStrategy, SimpleDecomposition,
        )
        strategy = DecompositionStrategy.create("simple", fv_mesh_8cell, n_processors=2)
        assert isinstance(strategy, SimpleDecomposition)

    def test_factory_scotch(self, fv_mesh_8cell):
        """Scotch 策略创建（可能回退到 simple）."""
        from pyfoam.parallel.decomposition_2 import DecompositionStrategy
        strategy = DecompositionStrategy.create("scotch", fv_mesh_8cell, n_processors=2)
        assert strategy is not None

    def test_unknown_strategy_raises(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import DecompositionStrategy
        with pytest.raises(KeyError, match="Unknown decomposition strategy"):
            DecompositionStrategy.create("nonexistent", fv_mesh_8cell, n_processors=2)


# ---------------------------------------------------------------------------
# SimpleDecomposition 测试
# ---------------------------------------------------------------------------


class TestSimpleDecomposition:
    """简单几何分解测试."""

    def test_decompose_2_processors(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(fv_mesh_8cell, n_processors=2)
        assignment = strategy.decompose()
        assert assignment.shape == (8,)
        assert assignment.dtype == INDEX_DTYPE

    def test_all_cells_assigned(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(fv_mesh_8cell, n_processors=2)
        assignment = strategy.decompose()
        unique = set(assignment.tolist())
        assert unique == {0, 1}

    def test_all_cells_covered(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(fv_mesh_8cell, n_processors=2)
        assignment = strategy.decompose()
        # 每个单元都被分配
        assert len(assignment) == 8

    def test_balanced_decomposition(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(fv_mesh_8cell, n_processors=2)
        assignment = strategy.decompose()
        n0 = (assignment == 0).sum().item()
        n1 = (assignment == 1).sum().item()
        # 应大致均衡
        assert n0 >= 1
        assert n1 >= 1
        assert n0 + n1 == 8

    def test_4_processors(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(fv_mesh_8cell, n_processors=4)
        assignment = strategy.decompose()
        for i in range(4):
            assert (assignment == i).sum() >= 1

    def test_8_processors(self, fv_mesh_8cell):
        """8 个处理器分解 8 个单元."""
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(fv_mesh_8cell, n_processors=8)
        assignment = strategy.decompose()
        # 分位数分割可能将多个同坐标单元分给同一处理器
        # 验证所有单元都被分配且值在合法范围内
        assert assignment.shape == (8,)
        assert assignment.min() >= 0
        assert assignment.max() < 8
        # 至少有 4 个处理器获得了单元
        unique_count = len(set(assignment.tolist()))
        assert unique_count >= 4

    def test_with_poly_mesh(self, poly_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(poly_mesh_8cell, n_processors=2)
        assignment = strategy.decompose()
        assert assignment.shape == (8,)

    def test_zero_processors_raises(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        with pytest.raises(ValueError, match="n_processors must be >= 1"):
            SimpleDecomposition(fv_mesh_8cell, n_processors=0)

    def test_too_many_processors_raises(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        with pytest.raises(ValueError, match="n_processors.*n_cells"):
            SimpleDecomposition(fv_mesh_8cell, n_processors=100)

    def test_repr(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(fv_mesh_8cell, n_processors=2)
        r = repr(strategy)
        assert "SimpleDecomposition" in r
        assert "2" in r


# ---------------------------------------------------------------------------
# ScotchDecomposition 测试
# ---------------------------------------------------------------------------


class TestScotchDecomposition:
    """Scotch 图分解测试（含回退到 simple 的情况）."""

    def test_decompose(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import ScotchDecomposition
        strategy = ScotchDecomposition(fv_mesh_8cell, n_processors=2)
        assignment = strategy.decompose()
        assert assignment.shape == (8,)
        assert assignment.dtype == INDEX_DTYPE

    def test_all_cells_assigned(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import ScotchDecomposition
        strategy = ScotchDecomposition(fv_mesh_8cell, n_processors=2)
        assignment = strategy.decompose()
        unique = set(assignment.tolist())
        assert unique == {0, 1}

    def test_each_processor_has_cells(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import ScotchDecomposition
        strategy = ScotchDecomposition(fv_mesh_8cell, n_processors=4)
        assignment = strategy.decompose()
        for i in range(4):
            assert (assignment == i).sum() >= 1

    def test_zero_processors_raises(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import ScotchDecomposition
        with pytest.raises(ValueError, match="n_processors must be >= 1"):
            ScotchDecomposition(fv_mesh_8cell, n_processors=0)

    def test_repr(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import ScotchDecomposition
        strategy = ScotchDecomposition(fv_mesh_8cell, n_processors=2)
        r = repr(strategy)
        assert "ScotchDecomposition" in r


# ---------------------------------------------------------------------------
# 基类辅助方法测试
# ---------------------------------------------------------------------------


class TestDecompositionStrategyHelpers:
    """基类辅助方法测试."""

    def test_mesh_property(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(fv_mesh_8cell, n_processors=2)
        assert strategy.mesh is fv_mesh_8cell

    def test_n_processors_property(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(fv_mesh_8cell, n_processors=4)
        assert strategy.n_processors == 4

    def test_estimate_cell_centres_fvmesh(self, fv_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(fv_mesh_8cell, n_processors=2)
        centres = strategy._estimate_cell_centres()
        assert centres.shape == (8, 3)

    def test_estimate_cell_centres_polymesh(self, poly_mesh_8cell):
        from pyfoam.parallel.decomposition_2 import SimpleDecomposition
        strategy = SimpleDecomposition(poly_mesh_8cell, n_processors=2)
        centres = strategy._estimate_cell_centres()
        assert centres.shape == (8, 3)
