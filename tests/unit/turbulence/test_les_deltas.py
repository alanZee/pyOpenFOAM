"""Tests for LES delta models (CubeRootVolDelta, MaxDeltaXYZ, VanDriestDelta, SmoothDelta)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


class FakeMesh:
    """Minimal mesh stub for delta model tests."""

    def __init__(self, cell_volumes, cell_deltas=None, y_plus=None):
        self.cell_volumes = cell_volumes
        self.n_cells = len(cell_volumes)
        if cell_deltas is not None:
            self.cell_deltas = cell_deltas
        if y_plus is not None:
            self.y_plus = y_plus


class TestCubeRootVolDelta:
    """Tests for CubeRootVolDelta."""

    def test_registration(self):
        from pyfoam.turbulence.les_deltas import LESDelta

        assert "cubeRootVol" in LESDelta.available_types()

    def test_factory_creation(self):
        from pyfoam.turbulence.les_deltas import LESDelta, CubeRootVolDelta

        delta = LESDelta.create("cubeRootVol")
        assert isinstance(delta, CubeRootVolDelta)

    def test_uniform_cells(self):
        from pyfoam.turbulence.les_deltas import CubeRootVolDelta

        # V = 8 for each cell => delta = 2.0
        volumes = torch.full((5,), 8.0, dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes)

        delta_fn = CubeRootVolDelta()
        result = delta_fn(mesh)

        expected = torch.full((5,), 2.0, dtype=CFD_DTYPE)
        assert torch.allclose(result, expected)

    def test_single_cell(self):
        from pyfoam.turbulence.les_deltas import CubeRootVolDelta

        volumes = torch.tensor([27.0], dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes)

        delta_fn = CubeRootVolDelta()
        result = delta_fn(mesh)

        assert torch.allclose(result, torch.tensor([3.0], dtype=CFD_DTYPE))

    def test_zero_volume_clamped(self):
        """Zero-volume cells should not produce NaN."""
        from pyfoam.turbulence.les_deltas import CubeRootVolDelta

        volumes = torch.tensor([0.0, 1.0, 8.0], dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes)

        delta_fn = CubeRootVolDelta()
        result = delta_fn(mesh)

        assert torch.isfinite(result).all()
        assert result[0] > 0  # clamped away from zero

    def test_various_volumes(self):
        from pyfoam.turbulence.les_deltas import CubeRootVolDelta

        volumes = torch.tensor([1.0, 8.0, 64.0, 125.0], dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes)

        delta_fn = CubeRootVolDelta()
        result = delta_fn(mesh)

        expected = torch.tensor([1.0, 2.0, 4.0, 5.0], dtype=CFD_DTYPE)
        assert torch.allclose(result, expected)


class TestMaxDeltaXYZ:
    """Tests for MaxDeltaXYZ."""

    def test_registration(self):
        from pyfoam.turbulence.les_deltas import LESDelta

        assert "maxDeltaxyz" in LESDelta.available_types()

    def test_factory_creation(self):
        from pyfoam.turbulence.les_deltas import LESDelta, MaxDeltaXYZ

        delta = LESDelta.create("maxDeltaxyz")
        assert isinstance(delta, MaxDeltaXYZ)

    def test_with_cell_deltas(self):
        """When mesh has cell_deltas, max of axis values is returned."""
        from pyfoam.turbulence.les_deltas import MaxDeltaXYZ

        volumes = torch.full((3,), 1.0, dtype=CFD_DTYPE)
        cell_deltas = torch.tensor([
            [1.0, 2.0, 3.0],
            [5.0, 1.0, 2.0],
            [0.5, 0.5, 0.5],
        ], dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes, cell_deltas=cell_deltas)

        delta_fn = MaxDeltaXYZ()
        result = delta_fn(mesh)

        expected = torch.tensor([3.0, 5.0, 0.5], dtype=CFD_DTYPE)
        assert torch.allclose(result, expected)

    def test_fallback_to_cuberoot(self):
        """Without cell_deltas, falls back to V^(1/3)."""
        from pyfoam.turbulence.les_deltas import MaxDeltaXYZ

        volumes = torch.tensor([8.0, 27.0, 64.0], dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes)

        delta_fn = MaxDeltaXYZ()
        result = delta_fn(mesh)

        expected = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)
        assert torch.allclose(result, expected)


class TestVanDriestDelta:
    """Tests for VanDriestDelta."""

    def test_registration(self):
        from pyfoam.turbulence.les_deltas import LESDelta

        assert "vanDriest" in LESDelta.available_types()

    def test_factory_creation(self):
        from pyfoam.turbulence.les_deltas import LESDelta, VanDriestDelta

        delta = LESDelta.create("vanDriest", A_plus=25.0)
        assert isinstance(delta, VanDriestDelta)

    def test_custom_A_plus(self):
        from pyfoam.turbulence.les_deltas import VanDriestDelta

        delta_fn = VanDriestDelta(A_plus=50.0)
        assert delta_fn.A_plus == pytest.approx(50.0)

    def test_no_y_plus_fallback(self):
        """Without y_plus on mesh, returns plain cube-root volume."""
        from pyfoam.turbulence.les_deltas import VanDriestDelta

        volumes = torch.tensor([8.0, 27.0], dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes)

        delta_fn = VanDriestDelta()
        result = delta_fn(mesh)

        expected = torch.tensor([2.0, 3.0], dtype=CFD_DTYPE)
        assert torch.allclose(result, expected)

    def test_with_y_plus_wall_cells(self):
        """Near wall (small y+), delta should be damped toward zero."""
        from pyfoam.turbulence.les_deltas import VanDriestDelta

        volumes = torch.full((3,), 8.0, dtype=CFD_DTYPE)  # V^(1/3) = 2.0
        y_plus = torch.tensor([0.0, 1.0, 1000.0], dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes, y_plus=y_plus)

        delta_fn = VanDriestDelta(A_plus=25.0)
        result = delta_fn(mesh)

        # At y+=0, damping=0 => delta=0
        assert result[0] < 1e-10
        # At y+=1, damping is small
        assert result[1] < 2.0
        # At y+=1000, damping ~1 => delta ~ 2.0
        assert result[2] > 1.99

    def test_damping_increases_with_y_plus(self):
        """Delta should monotonically increase with y+."""
        from pyfoam.turbulence.les_deltas import VanDriestDelta

        volumes = torch.full((5,), 27.0, dtype=CFD_DTYPE)  # V^(1/3) = 3.0
        y_plus = torch.tensor([0.0, 5.0, 10.0, 50.0, 200.0], dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes, y_plus=y_plus)

        delta_fn = VanDriestDelta(A_plus=25.0)
        result = delta_fn(mesh)

        # Each successive value should be larger
        for i in range(4):
            assert result[i + 1] > result[i], (
                f"delta at y+={y_plus[i+1]} should be > delta at y+={y_plus[i]}"
            )

    def test_far_field_equals_base(self):
        """At large y+, Van Driest delta approaches the base delta."""
        from pyfoam.turbulence.les_deltas import VanDriestDelta, CubeRootVolDelta

        volumes = torch.tensor([8.0, 64.0, 125.0], dtype=CFD_DTYPE)
        y_plus = torch.tensor([1e6, 1e6, 1e6], dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes, y_plus=y_plus)

        vd = VanDriestDelta()
        cr = CubeRootVolDelta()

        result_vd = vd(mesh)
        result_cr = cr(mesh)

        assert torch.allclose(result_vd, result_cr, atol=1e-6)

    def test_finite_output(self):
        from pyfoam.turbulence.les_deltas import VanDriestDelta

        volumes = torch.rand(20, dtype=CFD_DTYPE).clamp(0.01, 100.0)
        y_plus = torch.rand(20, dtype=CFD_DTYPE) * 100
        mesh = FakeMesh(volumes, y_plus=y_plus)

        delta_fn = VanDriestDelta()
        result = delta_fn(mesh)

        assert torch.isfinite(result).all()
        assert (result >= 0).all()


class FakeMeshWithFaces(FakeMesh):
    """Mesh stub with owner/neighbour for smooth delta tests."""

    def __init__(self, cell_volumes, owner, neighbour, n_internal_faces=None):
        super().__init__(cell_volumes)
        self.owner = owner
        self.neighbour = neighbour
        self.n_internal_faces = n_internal_faces if n_internal_faces is not None else len(neighbour)


class TestSmoothDelta:
    """Tests for SmoothDelta."""

    def test_registration(self):
        from pyfoam.turbulence.les_deltas import LESDelta
        assert "smoothDelta" in LESDelta.available_types()

    def test_factory_creation(self):
        from pyfoam.turbulence.les_deltas import LESDelta, SmoothDelta
        delta = LESDelta.create("smoothDelta", n_passes=2)
        assert isinstance(delta, SmoothDelta)
        assert delta.n_passes == 2

    def test_default_n_passes(self):
        from pyfoam.turbulence.les_deltas import SmoothDelta
        delta_fn = SmoothDelta()
        assert delta_fn.n_passes == 1

    def test_n_passes_min_one(self):
        from pyfoam.turbulence.les_deltas import SmoothDelta
        delta_fn = SmoothDelta(n_passes=0)
        assert delta_fn.n_passes == 1

    def test_uniform_cells_unchanged(self):
        """均匀网格的 smooth delta 应接近基础 delta。"""
        from pyfoam.turbulence.les_deltas import SmoothDelta

        volumes = torch.full((4,), 8.0, dtype=CFD_DTYPE)  # V^(1/3) = 2.0
        owner = torch.tensor([0, 0, 1], dtype=torch.long)
        neighbour = torch.tensor([1, 2, 3], dtype=torch.long)
        mesh = FakeMeshWithFaces(volumes, owner, neighbour)

        delta_fn = SmoothDelta()
        result = delta_fn(mesh)

        expected = torch.full((4,), 2.0, dtype=CFD_DTYPE)
        assert torch.allclose(result, expected, atol=1e-10)

    def test_reduces_variation(self):
        """平滑应减小空间变化。"""
        from pyfoam.turbulence.les_deltas import SmoothDelta

        # 2 个单元，体积差别大
        volumes = torch.tensor([1.0, 1000.0], dtype=CFD_DTYPE)
        owner = torch.tensor([0], dtype=torch.long)
        neighbour = torch.tensor([1], dtype=torch.long)
        mesh = FakeMeshWithFaces(volumes, owner, neighbour)

        delta_fn = SmoothDelta()
        result = delta_fn(mesh)

        # 原始 delta: [1.0, 10.0]
        # 平滑后两个值应更接近
        base = volumes.pow(1.0 / 3.0)
        spread_orig = (base[1] - base[0]).item()
        spread_smooth = (result[1] - result[0]).item()
        assert spread_smooth < spread_orig

    def test_multiple_passes_more_smoothing(self):
        """多次迭代平滑更强。"""
        from pyfoam.turbulence.les_deltas import SmoothDelta

        volumes = torch.tensor([1.0, 1000.0], dtype=CFD_DTYPE)
        owner = torch.tensor([0], dtype=torch.long)
        neighbour = torch.tensor([1], dtype=torch.long)
        mesh = FakeMeshWithFaces(volumes, owner, neighbour)

        f1 = SmoothDelta(n_passes=1)
        f5 = SmoothDelta(n_passes=5)

        result1 = f1(mesh)
        result5 = f5(mesh)

        spread1 = (result1[1] - result1[0]).abs().item()
        spread5 = (result5[1] - result5[0]).abs().item()
        assert spread5 <= spread1 + 1e-10

    def test_no_mesh_faces_fallback(self):
        """无 owner/neighbour 时返回基础 delta。"""
        from pyfoam.turbulence.les_deltas import SmoothDelta, CubeRootVolDelta

        volumes = torch.tensor([8.0, 27.0, 64.0], dtype=CFD_DTYPE)
        mesh = FakeMesh(volumes)

        smooth_fn = SmoothDelta()
        cuberoot_fn = CubeRootVolDelta()

        result_smooth = smooth_fn(mesh)
        result_cuberoot = cuberoot_fn(mesh)

        assert torch.allclose(result_smooth, result_cuberoot)

    def test_shape(self):
        """输出形状正确。"""
        from pyfoam.turbulence.les_deltas import SmoothDelta

        volumes = torch.rand(10, dtype=CFD_DTYPE).clamp(0.01, 100.0)
        owner = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
        neighbour = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
        mesh = FakeMeshWithFaces(volumes, owner, neighbour)

        delta_fn = SmoothDelta()
        result = delta_fn(mesh)
        assert result.shape == (10,)

    def test_finite_output(self):
        """输出不含 NaN/Inf。"""
        from pyfoam.turbulence.les_deltas import SmoothDelta

        volumes = torch.rand(20, dtype=CFD_DTYPE).clamp(0.01, 100.0)
        owner = torch.arange(19, dtype=torch.long)
        neighbour = torch.arange(1, 20, dtype=torch.long)
        mesh = FakeMeshWithFaces(volumes, owner, neighbour)

        delta_fn = SmoothDelta()
        result = delta_fn(mesh)

        assert torch.isfinite(result).all()
        assert (result >= 0).all()
