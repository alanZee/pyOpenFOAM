"""
Tests for chemistry models: ChemistryModel, ODEChemistrySolver, SRMChemistrySolver.

Test cases:
1. RTS registry
2. ODE chemistry solver source terms
3. SRM chemistry solver source terms
4. Edge cases and validation
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.thermophysical.chemistry import (
    ChemistryModel,
    ODEChemistrySolver,
    SRMChemistrySolver,
)


# ===================================================================
# ChemistryModel RTS 注册表
# ===================================================================


class TestChemistryRTS:
    """ChemistryModel 运行时选择注册表测试。"""

    def test_registry_contains_all_models(self):
        types = ChemistryModel.available_types()
        assert "ODE" in types
        assert "SRM" in types

    def test_factory_create_ode(self):
        chem = ChemistryModel.create("ODE", species=["A", "B"])
        assert isinstance(chem, ODEChemistrySolver)

    def test_factory_create_srm(self):
        chem = ChemistryModel.create("SRM", species=["fuel", "products"])
        assert isinstance(chem, SRMChemistrySolver)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown chemistry model"):
            ChemistryModel.create("nonexistent")

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ChemistryModel()


# ===================================================================
# ODEChemistrySolver
# ===================================================================


class TestODEChemistrySolver:
    """ODE 化学求解器测试。"""

    def test_species_property(self):
        """species 属性应返回组分名列表。"""
        chem = ODEChemistrySolver(species=["O2", "N2", "CH4"])
        assert chem.species == ["O2", "N2", "CH4"]
        assert chem.n_species == 3

    def test_n_reactions(self):
        """n_reactions 应返回反应数量。"""
        chem = ODEChemistrySolver(
            species=["A", "B", "C"],
            reactions=[
                {"A": 1e10, "Ea": 8e4, "reactants": {"A": 1}, "products": {"B": 1}},
                {"A": 1e8, "Ea": 5e4, "reactants": {"B": 1}, "products": {"C": 1}},
            ],
        )
        assert chem.n_reactions == 2

    def test_source_shape(self):
        """source 应返回正确形状的张量。"""
        chem = ODEChemistrySolver(
            species=["A", "B"],
            reactions=[{
                "A": 1e10, "Ea": 0.0, "b": 0.0,
                "reactants": {"A": 1},
                "products": {"B": 1},
                "Q": 0.0,
            }],
        )
        n_cells = 5
        Y = torch.zeros(n_cells, 2)
        Y[:, 0] = 1.0  # 全是 A
        T = torch.full((n_cells,), 1000.0)
        p = 101325.0
        rho = 1.0
        dt = 1e-4

        dYdt, dTdt = chem.source(Y, T, p, rho, dt)

        assert dYdt.shape == (n_cells, 2)
        assert dTdt.shape == (n_cells,)
        assert torch.isfinite(dYdt).all()
        assert torch.isfinite(dTdt).all()

    def test_species_consumption(self):
        """反应物应被消耗。"""
        chem = ODEChemistrySolver(
            species=["A", "B"],
            reactions=[{
                "A": 1e12, "Ea": 0.0, "b": 0.0,
                "reactants": {"A": 1},
                "products": {"B": 1},
                "Q": 0.0,
            }],
            dt_max=1e-6,
        )
        Y = torch.tensor([[0.9, 0.1]])
        T = torch.tensor([1000.0])
        dt = 1e-3

        dYdt, _ = chem.source(Y, T, 101325.0, 1.0, dt)

        # A 应被消耗（dY/dt < 0），B 应被生成（dY/dt > 0）
        assert float(dYdt[0, 0].item()) < 0.0, "A should be consumed"
        assert float(dYdt[0, 1].item()) > 0.0, "B should be produced"

    def test_heat_release(self):
        """放热反应应使温度上升。"""
        chem = ODEChemistrySolver(
            species=["A", "B"],
            reactions=[{
                "A": 1e12, "Ea": 0.0, "b": 0.0,
                "reactants": {"A": 1},
                "products": {"B": 1},
                "Q": 1e6,  # 放热
            }],
            dt_max=1e-6,
            Cp=1005.0,
        )
        Y = torch.tensor([[0.9, 0.1]])
        T = torch.tensor([1000.0])
        dt = 1e-3

        _, dTdt = chem.source(Y, T, 101325.0, 1.0, dt)

        # 放热反应 → 温度上升 → dT/dt > 0
        assert float(dTdt[0].item()) > 0.0, "Temperature should increase"

    def test_scalar_input(self):
        """应支持标量 T、p、rho 输入。"""
        chem = ODEChemistrySolver(species=["A", "B"])
        Y = torch.tensor([[0.5, 0.5]])
        dYdt, dTdt = chem.source(Y, 300.0, 101325.0, 1.225, 1e-4)
        assert dYdt.shape == (1, 2)
        assert dTdt.shape == (1,)

    def test_default_species(self):
        """默认组分应为 ["A", "B"]。"""
        chem = ODEChemistrySolver()
        assert chem.species == ["A", "B"]

    def test_repr(self):
        """repr 应包含类名和关键参数。"""
        chem = ODEChemistrySolver(species=["A", "B", "C"])
        r = repr(chem)
        assert "ODEChemistrySolver" in r
        assert "3" in r  # n_species


# ===================================================================
# SRMChemistrySolver
# ===================================================================


class TestSRMChemistrySolver:
    """SRM 化学求解器测试。"""

    def test_species_property(self):
        """species 属性应返回组分名列表。"""
        chem = SRMChemistrySolver(species=["fuel", "O2", "products"])
        assert chem.species == ["fuel", "O2", "products"]

    def test_tau_chem_property(self):
        """tau_chem 属性应返回化学时间尺度。"""
        chem = SRMChemistrySolver(tau_chem=5e-4)
        assert chem.tau_chem == 5e-4

    def test_table_resolution_property(self):
        """table_resolution 属性应返回表分辨率。"""
        chem = SRMChemistrySolver(table_resolution=200)
        assert chem.table_resolution == 200

    def test_source_shape(self):
        """source 应返回正确形状的张量。"""
        chem = SRMChemistrySolver(species=["fuel", "products"])
        n_cells = 5
        Y = torch.zeros(n_cells, 2)
        Y[:, 0] = 0.5
        Y[:, 1] = 0.5
        T = torch.full((n_cells,), 1200.0)

        dYdt, dTdt = chem.source(Y, T, 101325.0, 1.0, 1e-3)

        assert dYdt.shape == (n_cells, 2)
        assert dTdt.shape == (n_cells,)
        assert torch.isfinite(dYdt).all()
        assert torch.isfinite(dTdt).all()

    def test_unburned_state(self):
        """未燃烧状态（T = T_unburned）时源项应趋近于零。"""
        chem = SRMChemistrySolver(
            species=["fuel", "products"],
            T_unburned=300.0,
            T_burned=2000.0,
            tau_chem=1e-3,
        )
        Y = torch.tensor([[1.0, 0.0]])
        T = torch.tensor([300.0])

        dYdt, dTdt = chem.source(Y, T, 101325.0, 1.0, 1e-3)

        # 进度变量 c = 0 → dc/dt = 0 → 源项应为零
        assert abs(float(dYdt[0, 0].item())) < 1e-10
        assert abs(float(dTdt[0].item())) < 1e-10

    def test_burned_state(self):
        """完全燃烧状态（T = T_burned）时源项应趋近于零。"""
        chem = SRMChemistrySolver(
            species=["fuel", "products"],
            T_unburned=300.0,
            T_burned=2000.0,
            tau_chem=1e-3,
        )
        Y = torch.tensor([[0.0, 1.0]])
        T = torch.tensor([2000.0])

        dYdt, dTdt = chem.source(Y, T, 101325.0, 1.0, 1e-3)

        # 进度变量 c = 1 → dc/dt = 0 → 源项应为零
        assert abs(float(dYdt[0, 0].item())) < 1e-10
        assert abs(float(dTdt[0].item())) < 1e-10

    def test_mid_flame_state(self):
        """中间温度时应有非零源项。"""
        chem = SRMChemistrySolver(
            species=["fuel", "products"],
            T_unburned=300.0,
            T_burned=2000.0,
            tau_chem=1e-3,
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([1150.0])  # 中间温度

        dYdt, dTdt = chem.source(Y, T, 101325.0, 1.0, 1e-3)

        # c = 0.5 → dc/dt = 0.5*0.5/tau = 0.25/tau > 0
        assert abs(float(dYdt[0, 0].item())) > 0.0  # fuel consumed
        assert float(dYdt[0, 1].item()) > 0.0  # products produced
        assert float(dTdt[0].item()) > 0.0  # temperature increases

    def test_fuel_consumed_products_produced(self):
        """燃料消耗、产物生成。"""
        chem = SRMChemistrySolver(
            species=["fuel", "products"],
            T_unburned=300.0,
            T_burned=2000.0,
            tau_chem=1e-3,
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([1000.0])

        dYdt, _ = chem.source(Y, T, 101325.0, 1.0, 1e-3)

        # fuel 应被消耗
        assert float(dYdt[0, 0].item()) < 0.0
        # products 应被生成
        assert float(dYdt[0, 1].item()) > 0.0

    def test_source_increases_with_tau_chem_decrease(self):
        """更小的 tau_chem 应产生更大的源项。"""
        chem_fast = SRMChemistrySolver(
            species=["fuel", "products"],
            tau_chem=1e-4,
        )
        chem_slow = SRMChemistrySolver(
            species=["fuel", "products"],
            tau_chem=1e-2,
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([1000.0])

        dYdt_fast, _ = chem_fast.source(Y, T, 101325.0, 1.0, 1e-3)
        dYdt_slow, _ = chem_slow.source(Y, T, 101325.0, 1.0, 1e-3)

        assert abs(float(dYdt_fast[0, 0].item())) > abs(float(dYdt_slow[0, 0].item()))

    def test_default_species(self):
        """默认组分应为 ["fuel", "products"]。"""
        chem = SRMChemistrySolver()
        assert chem.species == ["fuel", "products"]

    def test_repr(self):
        """repr 应包含类名和关键参数。"""
        chem = SRMChemistrySolver(tau_chem=1e-3, table_resolution=50)
        r = repr(chem)
        assert "SRMChemistrySolver" in r
        assert "0.001" in r

    def test_scalar_temperature_input(self):
        """应支持标量温度输入。"""
        chem = SRMChemistrySolver(species=["fuel", "products"])
        Y = torch.tensor([[0.5, 0.5]])
        dYdt, dTdt = chem.source(Y, 1000.0, 101325.0, 1.0, 1e-3)
        assert dYdt.shape == (1, 2)
        assert dTdt.shape == (1,)

    def test_high_table_resolution(self):
        """高分辨率查找表应工作正常。"""
        chem = SRMChemistrySolver(
            species=["fuel", "products"],
            table_resolution=500,
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([1000.0])
        dYdt, dTdt = chem.source(Y, T, 101325.0, 1.0, 1e-3)
        assert torch.isfinite(dYdt).all()
        assert torch.isfinite(dTdt).all()


# ===================================================================
# 跨模型集成测试
# ===================================================================


class TestChemistryIntegration:
    """集成测试：组合使用多个化学模型。"""

    def test_ode_different_conditions(self):
        """ODE 模型在不同工况下应产生合理结果。"""
        chem = ODEChemistrySolver(
            species=["fuel", "O2", "products"],
            reactions=[{
                "A": 1e10, "Ea": 8e4, "b": 0.0,
                "reactants": {"fuel": 1, "O2": 2},
                "products": {"products": 3},
                "Q": 5e5,
            }],
            dt_max=1e-5,
        )

        conditions = [
            (torch.tensor([[0.1, 0.2, 0.0]]), 500.0),   # 低温
            (torch.tensor([[0.1, 0.2, 0.0]]), 1500.0),  # 高温
            (torch.tensor([[0.05, 0.3, 0.0]]), 1000.0), # 稀混合
        ]

        for Y, T in conditions:
            dYdt, dTdt = chem.source(Y, T, 101325.0, 1.0, 1e-3)
            assert torch.isfinite(dYdt).all()
            assert torch.isfinite(dTdt).all()

    def test_srm_different_conditions(self):
        """SRM 模型在不同温度下应产生合理结果。"""
        chem = SRMChemistrySolver(
            species=["fuel", "products"],
            T_unburned=300.0,
            T_burned=2000.0,
        )

        temperatures = [300.0, 500.0, 1000.0, 1500.0, 2000.0]
        Y = torch.tensor([[0.5, 0.5]])

        for T in temperatures:
            dYdt, dTdt = chem.source(Y, T, 101325.0, 1.0, 1e-3)
            assert torch.isfinite(dYdt).all()
            assert torch.isfinite(dTdt).all()

    def test_ode_and_srm_produce_different_results(self):
        """ODE 和 SRM 模型应产生不同的数值。"""
        Y = torch.tensor([[0.5, 0.5]])
        T = 1000.0

        ode_chem = ODEChemistrySolver(
            species=["fuel", "products"],
            reactions=[{
                "A": 1e10, "Ea": 8e4,
                "reactants": {"fuel": 1},
                "products": {"products": 1},
            }],
        )
        srm_chem = SRMChemistrySolver(species=["fuel", "products"])

        dYdt_ode, _ = ode_chem.source(Y, T, 101325.0, 1.0, 1e-3)
        dYdt_srm, _ = srm_chem.source(Y, T, 101325.0, 1.0, 1e-3)

        # 两种方法的数值应不同
        assert not torch.allclose(dYdt_ode, dYdt_srm, atol=1e-6)
