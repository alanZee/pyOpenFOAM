"""
Tests for poly_topo_change module.
"""
import pytest
import torch

from pyfoam.poly_topo_change import PolyTopoChange, TopoSet, BoxToCell, CylinderToCell


class TestPolyTopoChange:
    """拓扑修改测试。"""

    def test_add_cell(self):
        topo = PolyTopoChange(n_cells=10)
        idx = topo.add_cell(points=[0, 1, 2, 3, 4, 5, 6, 7])
        assert idx == 10
        assert topo.n_cells == 11
        assert topo.n_pending == 1

    def test_remove_cell(self):
        topo = PolyTopoChange(n_cells=10)
        topo.remove_cell(5)
        assert topo.n_pending == 1

    def test_add_face(self):
        topo = PolyTopoChange(n_faces=20)
        idx = topo.add_face(vertices=[0, 1, 2], owner=0)
        assert idx == 20
        assert topo.n_faces == 21

    def test_modify_face(self):
        topo = PolyTopoChange(n_faces=20)
        topo.modify_face(5, new_owner=1)
        assert topo.n_pending == 1

    def test_get_changes(self):
        topo = PolyTopoChange(n_cells=10)
        topo.add_cell([0, 1, 2, 3])
        topo.remove_cell(5)
        changes = topo.get_changes()
        assert len(changes) == 2

    def test_clear(self):
        topo = PolyTopoChange(n_cells=10)
        topo.add_cell([0, 1, 2, 3])
        topo.clear()
        assert topo.n_pending == 0


class TestTopoSet:
    """拓扑集合测试。"""

    def test_add_remove(self):
        s = TopoSet("test")
        s.add(5)
        s.add(10)
        assert len(s) == 2
        assert 5 in s
        s.remove(5)
        assert len(s) == 1

    def test_add_range(self):
        s = TopoSet("test")
        s.add_range(0, 10)
        assert len(s) == 10

    def test_invert(self):
        s = TopoSet("test")
        s.add(0)
        s.add(1)
        s.invert(5)
        assert len(s) == 3
        assert 2 in s
        assert 3 in s
        assert 4 in s

    def test_to_tensor(self):
        s = TopoSet("test")
        s.add(3)
        s.add(1)
        s.add(2)
        t = s.to_tensor()
        assert t.tolist() == [1, 2, 3]


class TestBoxToCell:
    """盒形选择源测试。"""

    def test_select_from_centres(self):
        box = BoxToCell(min_point=(0, 0, 0), max_point=(0.5, 0.5, 0.5))
        centres = torch.tensor([
            [0.25, 0.25, 0.25],  # inside
            [0.75, 0.75, 0.75],  # outside
            [0.1, 0.1, 0.1],    # inside
        ], dtype=torch.float64)
        s = TopoSet("test")
        box.select_from_centres(s, centres)
        assert len(s) == 2
        assert 0 in s
        assert 2 in s


class TestCylinderToCell:
    """圆柱形选择源测试。"""

    def test_select_from_centres(self):
        cyl = CylinderToCell(
            point1=(0, 0, 0),
            point2=(1, 0, 0),
            radius=0.3,
        )
        centres = torch.tensor([
            [0.5, 0.1, 0.0],   # inside
            [0.5, 0.5, 0.0],   # outside (r > 0.3)
            [1.5, 0.0, 0.0],   # outside (beyond end)
        ], dtype=torch.float64)
        s = TopoSet("test")
        cyl.select_from_centres(s, centres)
        assert len(s) == 1
        assert 0 in s
