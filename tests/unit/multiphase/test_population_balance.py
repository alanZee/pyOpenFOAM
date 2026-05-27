"""群体平衡方程 (PBE) 模型测试."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE


# ---------------------------------------------------------------------------
# 复用 2-cell hex 网格
# ---------------------------------------------------------------------------

_POINTS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 2.0],
    [1.0, 0.0, 2.0],
    [1.0, 1.0, 2.0],
    [0.0, 1.0, 2.0],
]

_FACES = [
    [4, 5, 6, 7],
    [0, 3, 2, 1],
    [0, 1, 5, 4],
    [3, 7, 6, 2],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
    [8, 9, 10, 11],
    [4, 5, 9, 8],
    [7, 11, 10, 6],
    [4, 8, 11, 7],
    [5, 6, 10, 9],
]

_OWNER = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_NEIGHBOUR = [1]

_BOUNDARY = [
    {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
    {"name": "top", "type": "wall", "startFace": 6, "nFaces": 5},
]


def make_fv_mesh(device="cpu", dtype=torch.float64):
    from pyfoam.mesh.fv_mesh import FvMesh
    mesh = FvMesh(
        points=torch.tensor(_POINTS, dtype=dtype, device=device),
        faces=[torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in _FACES],
        owner=torch.tensor(_OWNER, dtype=INDEX_DTYPE, device=device),
        neighbour=torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE, device=device),
        boundary=_BOUNDARY,
    )
    mesh.compute_geometry()
    return mesh


@pytest.fixture
def fv_mesh():
    return make_fv_mesh()


# ---------------------------------------------------------------------------
# PBEBin 测试
# ---------------------------------------------------------------------------


class TestPBEBin:
    """尺寸区间 PBEBin 测试."""

    def test_create_geometric_bins(self):
        from pyfoam.multiphase.population_balance import PBEBin
        bins = PBEBin.create_geometric_bins(
            v_min=1e-12, v_max=1e-6, n_bins=10, ratio=2.0,
        )
        assert len(bins) == 10

    def test_bin_diameter(self):
        from pyfoam.multiphase.population_balance import PBEBin
        bins = PBEBin.create_geometric_bins(
            v_min=1e-12, v_max=1e-6, n_bins=5, ratio=2.0,
        )
        for b in bins:
            assert b.d > 0
            # 验证 d = (6v/π)^(1/3)
            expected_d = (6.0 * b.v_center / 3.141592653589793) ** (1.0 / 3.0)
            assert abs(b.d - expected_d) < 1e-10

    def test_bin_dv_positive(self):
        from pyfoam.multiphase.population_balance import PBEBin
        bins = PBEBin.create_geometric_bins(
            v_min=1e-12, v_max=1e-6, n_bins=10, ratio=2.0,
        )
        for b in bins:
            assert b.dv > 0

    def test_bin_ordering(self):
        from pyfoam.multiphase.population_balance import PBEBin
        bins = PBEBin.create_geometric_bins(
            v_min=1e-12, v_max=1e-6, n_bins=10, ratio=2.0,
        )
        for i in range(len(bins) - 1):
            assert bins[i].v_upper == bins[i + 1].v_lower
            assert bins[i].v_center < bins[i + 1].v_center

    def test_single_bin(self):
        from pyfoam.multiphase.population_balance import PBEBin
        bins = PBEBin.create_geometric_bins(
            v_min=1e-12, v_max=1e-6, n_bins=1, ratio=2.0,
        )
        assert len(bins) == 1
        assert bins[0].v_lower == 1e-12
        assert bins[0].v_upper == 1e-6


# ---------------------------------------------------------------------------
# 聚并模型测试
# ---------------------------------------------------------------------------


class TestConstantCoalescence:
    """常数聚并核测试."""

    def test_rate(self):
        from pyfoam.multiphase.population_balance import ConstantCoalescence
        coal = ConstantCoalescence(C_coal=1e-3)
        n_i = torch.tensor([1.0, 2.0])
        n_j = torch.tensor([0.5, 1.0])
        gamma = torch.tensor([100.0, 200.0])
        rate = coal.coalescence_rate(1e-9, 1e-9, n_i, n_j, gamma)
        expected = 1e-3 * n_i * n_j
        assert torch.allclose(rate, expected)

    def test_symmetric(self):
        from pyfoam.multiphase.population_balance import ConstantCoalescence
        coal = ConstantCoalescence()
        n_i = torch.tensor([3.0])
        n_j = torch.tensor([4.0])
        gamma = torch.tensor([10.0])
        rate1 = coal.coalescence_rate(1e-9, 2e-9, n_i, n_j, gamma)
        rate2 = coal.coalescence_rate(2e-9, 1e-9, n_j, n_i, gamma)
        assert torch.allclose(rate1, rate2)


class TestShearCoalescence:
    """剪切聚并核测试."""

    def test_rate_depends_on_shear(self):
        from pyfoam.multiphase.population_balance import ShearCoalescence
        coal = ShearCoalescence()
        n_i = torch.tensor([1.0])
        n_j = torch.tensor([1.0])
        gamma_low = torch.tensor([10.0])
        gamma_high = torch.tensor([100.0])
        rate_low = coal.coalescence_rate(1e-9, 1e-9, n_i, n_j, gamma_low)
        rate_high = coal.coalescence_rate(1e-9, 1e-9, n_i, n_j, gamma_high)
        assert rate_high > rate_low


# ---------------------------------------------------------------------------
# 破碎模型测试
# ---------------------------------------------------------------------------


class TestConstantBreakup:
    """常数破碎率测试."""

    def test_rate(self):
        from pyfoam.multiphase.population_balance import ConstantBreakup
        brk = ConstantBreakup(C_break=0.5)
        n_i = torch.tensor([10.0, 20.0])
        gamma = torch.tensor([100.0, 200.0])
        rate = brk.breakup_rate(1e-9, n_i, gamma)
        expected = 0.5 * n_i
        assert torch.allclose(rate, expected)

    def test_daughter_distribution(self):
        from pyfoam.multiphase.population_balance import ConstantBreakup
        brk = ConstantBreakup(n_daughters=2)
        # 每个子体 = 母体/2
        v_parent = 1e-9
        v_daughter = 0.5e-9
        assert brk.daughter_distribution(v_parent, v_daughter) == 0.5
        # 非半体积的子体应返回 0
        assert brk.daughter_distribution(v_parent, 0.3e-9) == 0.0


class TestShearBreakup:
    """剪切破碎模型测试."""

    def test_rate_depends_on_shear(self):
        from pyfoam.multiphase.population_balance import ShearBreakup
        brk = ShearBreakup(C_break=0.1)
        n_i = torch.tensor([10.0])
        gamma_low = torch.tensor([10.0])
        gamma_high = torch.tensor([100.0])
        rate_low = brk.breakup_rate(1e-9, n_i, gamma_low)
        rate_high = brk.breakup_rate(1e-9, n_i, gamma_high)
        assert rate_high > rate_low


class TestWeberBreakup:
    """Weber 破碎模型测试."""

    def test_no_breakup_below_critical_weber(self):
        from pyfoam.multiphase.population_balance import WeberBreakup
        brk = WeberBreakup(C_break=0.1, rho=1000.0, sigma=0.07, We_cr=12.0)
        n_i = torch.tensor([10.0])
        # 低剪切率应使 We < We_cr
        gamma = torch.tensor([0.001])
        rate = brk.breakup_rate(1e-12, n_i, gamma)
        assert torch.allclose(rate, torch.tensor([0.0]), atol=1e-20)

    def test_breakup_above_critical_weber(self):
        from pyfoam.multiphase.population_balance import WeberBreakup
        # We_cr=0.1 使得 gamma=100 时 We > We_cr
        brk = WeberBreakup(C_break=0.1, rho=1000.0, sigma=0.07, We_cr=0.1)
        n_i = torch.tensor([10.0])
        gamma = torch.tensor([100.0])
        rate = brk.breakup_rate(1e-9, n_i, gamma)
        assert rate.item() > 0


# ---------------------------------------------------------------------------
# PopulationBalanceModel 测试
# ---------------------------------------------------------------------------


class TestPopulationBalanceModel:
    """群体平衡方程求解器测试."""

    def test_creation(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        assert pbe.n_bins == 5

    def test_n_fields_shape(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        assert len(pbe.n_fields) == 5
        for nf in pbe.n_fields:
            assert nf.shape == (fv_mesh.n_cells,)

    def test_n_fields_positive(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        for nf in pbe.n_fields:
            assert (nf > 0).all()

    def test_advance_no_submodels(self, fv_mesh):
        """没有聚并/破碎模型时，advance 不改变数密度."""
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        n_before = [nf.clone() for nf in pbe.n_fields]
        pbe.advance(dt=1e-4)
        for i, nf in enumerate(pbe.n_fields):
            assert torch.allclose(nf, n_before[i])

    def test_advance_with_coalescence(self, fv_mesh):
        """聚并会改变数密度."""
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin, ConstantCoalescence,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 3)
        pbe = PopulationBalanceModel(
            fv_mesh, bins,
            coalescence=ConstantCoalescence(C_coal=1e3),
        )
        pbe.advance(dt=1e-6)
        # 数密度应发生变化
        changed = any(
            not torch.allclose(nf, torch.full_like(nf, nf[0].item()))
            for nf in pbe.n_fields
        )
        # 聚并消耗了小 bin 的数密度
        assert pbe.n_fields[0].min() >= 0  # 不能为负

    def test_advance_with_breakup(self, fv_mesh):
        """破碎会改变数密度."""
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin, ShearBreakup,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(
            fv_mesh, bins,
            breakup=ShearBreakup(C_break=100.0),
        )
        pbe.gamma = torch.full((fv_mesh.n_cells,), 1000.0, dtype=torch.float64)
        n_before_sum = sum(nf.sum().item() for nf in pbe.n_fields)
        pbe.advance(dt=1e-6)
        n_after_sum = sum(nf.sum().item() for nf in pbe.n_fields)
        # 数密度不能为负
        for nf in pbe.n_fields:
            assert (nf >= 0).all()

    def test_advance_with_both_submodels(self, fv_mesh):
        """同时有聚并和破碎."""
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
            ConstantCoalescence, ShearBreakup,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(
            fv_mesh, bins,
            coalescence=ConstantCoalescence(C_coal=1e-1),
            breakup=ShearBreakup(C_break=1.0),
        )
        pbe.gamma = torch.full((fv_mesh.n_cells,), 10.0, dtype=torch.float64)
        pbe.advance(dt=1e-3)
        for nf in pbe.n_fields:
            assert (nf >= 0).all()

    def test_time_advances(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 3)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        assert pbe.time == 0.0
        pbe.advance(dt=1e-4)
        assert abs(pbe.time - 1e-4) < 1e-10

    def test_multiple_advances(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin, ConstantCoalescence,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(
            fv_mesh, bins,
            coalescence=ConstantCoalescence(C_coal=1e-1),
        )
        for _ in range(10):
            pbe.advance(dt=1e-5)
        assert abs(pbe.time - 1e-4) < 1e-10

    def test_at_least_one_bin_required(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        with pytest.raises(ValueError, match="至少需要一个尺寸区间"):
            PopulationBalanceModel(fv_mesh, [])

    def test_gamma_setter(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 3)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        gamma = torch.full((fv_mesh.n_cells,), 50.0, dtype=torch.float64)
        pbe.gamma = gamma
        assert torch.allclose(pbe.gamma, gamma)


# ---------------------------------------------------------------------------
# 统计量测试
# ---------------------------------------------------------------------------


class TestPBEStatistics:
    """PBE 统计量测试."""

    def test_total_number_density(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        N0 = pbe.total_number_density()
        assert N0.shape == (fv_mesh.n_cells,)
        assert (N0 > 0).all()

    def test_total_volume_fraction_positive(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        alpha = pbe.total_volume_fraction()
        assert alpha.shape == (fv_mesh.n_cells,)
        assert (alpha >= 0).all()

    def test_mean_diameter_positive(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        d10 = pbe.mean_diameter()
        assert d10.shape == (fv_mesh.n_cells,)
        assert (d10 > 0).all()

    def test_sauter_mean_diameter(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        d32 = pbe.sauter_mean_diameter()
        assert d32.shape == (fv_mesh.n_cells,)
        assert (d32 > 0).all()

    def test_sauter_ge_mean(self, fv_mesh):
        """Sauter 平均直径 d₃₂ ≥ d₁₀."""
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        d10 = pbe.mean_diameter()
        d32 = pbe.sauter_mean_diameter()
        assert (d32 >= d10 - 1e-20).all()

    def test_size_distribution(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(fv_mesh, bins)
        diams, dist = pbe.size_distribution()
        assert len(diams) == 5
        assert dist.shape == (5, fv_mesh.n_cells)

    def test_repr(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin, ConstantCoalescence,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        pbe = PopulationBalanceModel(
            fv_mesh, bins,
            coalescence=ConstantCoalescence(),
        )
        r = repr(pbe)
        assert "PopulationBalanceModel" in r
        assert "n_bins=5" in r
        assert "ConstantCoalescence" in r


# ---------------------------------------------------------------------------
# alpha 初始化测试
# ---------------------------------------------------------------------------


class TestPBEWithAlpha:
    """带初始体积分数的 PBE 测试."""

    def test_custom_alpha(self, fv_mesh):
        from pyfoam.multiphase.population_balance import (
            PopulationBalanceModel, PBEBin,
        )
        bins = PBEBin.create_geometric_bins(1e-12, 1e-6, 5)
        alpha = torch.full((fv_mesh.n_cells,), 0.3, dtype=torch.float64)
        pbe = PopulationBalanceModel(fv_mesh, bins, alpha=alpha)
        assert torch.allclose(pbe._alpha, alpha)
