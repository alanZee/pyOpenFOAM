"""
Tests for reaction kinetics and combustion models.

覆盖：
- ReactionRateModel RTS 注册表
- ArrheniusReaction 速率计算
- ThirdBodyReaction 第三体增强
- FallOffReaction Lindemann 降压
- CombustionModel RTS 注册表
- PaSRModel 燃烧源项
- EDCModel 燃烧源项
- InfinitelyFastChemistry 燃烧源项
"""

import pytest
import torch

from pyfoam.thermophysical.reaction import (
    R_UNIVERSAL,
    ReactionRateModel,
    ArrheniusReaction,
    ThirdBodyReaction,
    FallOffReaction,
    CombustionModel,
    PaSRModel,
    EDCModel,
    InfinitelyFastChemistry,
)


# ===================================================================
# ReactionRateModel RTS 注册表
# ===================================================================


class TestReactionRateRTS:
    """ReactionRateModel 运行时选择注册表测试。"""

    def test_registry_contains_all_models(self):
        types = ReactionRateModel.available_types()
        assert "Arrhenius" in types
        assert "thirdBody" in types
        assert "fallOff" in types

    def test_factory_create_arrhenius(self):
        rxn = ReactionRateModel.create("Arrhenius", A=1e10, b=0.5, Ea=8e4)
        assert isinstance(rxn, ArrheniusReaction)
        assert rxn.A == 1e10
        assert rxn.b == 0.5
        assert rxn.Ea == 8e4

    def test_factory_create_third_body(self):
        rxn = ReactionRateModel.create("thirdBody")
        assert isinstance(rxn, ThirdBodyReaction)

    def test_factory_create_fall_off(self):
        rxn = ReactionRateModel.create("fallOff")
        assert isinstance(rxn, FallOffReaction)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown reaction rate model"):
            ReactionRateModel.create("nonexistent")

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ReactionRateModel()


# ===================================================================
# ArrheniusReaction
# ===================================================================


class TestArrheniusReaction:
    """Arrhenius 反应速率测试。"""

    def test_basic_rate(self):
        """k = A * T^b * exp(-Ea/(RT)) 的基本计算。"""
        rxn = ArrheniusReaction(A=1e10, b=0.0, Ea=8e4)
        T = 1000.0
        expected = 1e10 * torch.exp(torch.tensor(-8e4 / (R_UNIVERSAL * 1000.0)))
        k = rxn.rate(T=T)
        # 使用相对容差，避免 float32/float64 精度差异
        assert abs(float(k.item()) - float(expected.item())) / max(abs(float(expected.item())), 1e-30) < 1e-5

    def test_temperature_exponent(self):
        """非零 b 值时 T^b 项应正确参与计算。"""
        rxn = ArrheniusReaction(A=1.0, b=2.0, Ea=0.0)
        T = 500.0
        # k = 1.0 * 500^2 * exp(0) = 250000
        k = rxn.rate(T=T)
        assert abs(float(k.item()) - 250000.0) < 1.0

    def test_zero_activation_energy(self):
        """Ea=0 时 exp(0)=1，速率仅由 A*T^b 决定。"""
        rxn = ArrheniusReaction(A=100.0, b=0.0, Ea=0.0)
        k = rxn.rate(T=500.0)
        assert abs(float(k.item()) - 100.0) < 1e-6

    def test_zero_rate_at_low_temperature(self):
        """低温时 Arrhenius 速率应趋近于零（指数衰减）。"""
        rxn = ArrheniusReaction(A=1e10, b=0.0, Ea=8e4)
        k_low = rxn.rate(T=100.0)   # 低温
        k_high = rxn.rate(T=2000.0) # 高温
        # 低温速率应比高温速率小很多个数量级
        assert float(k_low.item()) < 1e-30
        assert float(k_high.item()) > 0.0

    def test_rate_increases_with_temperature(self):
        """速率应随温度单调递增。"""
        rxn = ArrheniusReaction(A=1e10, b=0.0, Ea=8e4)
        temps = [500.0, 1000.0, 1500.0, 2000.0]
        rates = [float(rxn.rate(T=t).item()) for t in temps]
        for i in range(len(rates) - 1):
            assert rates[i] < rates[i + 1]

    def test_tensor_input(self):
        """应支持张量输入。"""
        rxn = ArrheniusReaction(A=1e10, b=0.0, Ea=8e4)
        T = torch.tensor([500.0, 1000.0, 1500.0])
        k = rxn.rate(T=T)
        assert k.shape == (3,)
        assert torch.isfinite(k).all()
        assert (k > 0).all()

    def test_tensor_output_finite(self):
        """输出应始终有限。"""
        rxn = ArrheniusReaction(A=1e20, b=1.0, Ea=1e5)
        T = torch.tensor([100.0, 500.0, 1000.0, 5000.0, 10000.0])
        k = rxn.rate(T=T)
        assert torch.isfinite(k).all()

    def test_repr(self):
        rxn = ArrheniusReaction(A=1e10, b=0.5, Ea=8e4)
        r = repr(rxn)
        assert "ArrheniusReaction" in r
        assert "1e+10" in r or "10000000000" in r

    def test_default_parameters(self):
        """默认参数: A=1, b=0, Ea=0 → k=1 对所有 T。"""
        rxn = ArrheniusReaction()
        k = rxn.rate(T=500.0)
        assert abs(float(k.item()) - 1.0) < 1e-6


# ===================================================================
# ThirdBodyReaction
# ===================================================================


class TestThirdBodyReaction:
    """第三体反应速率测试。"""

    def test_uniform_efficiency(self):
        """无效率表时所有组分效率为 1.0，k_eff = k_base * sum(C_i)。"""
        base = ArrheniusReaction(A=100.0, Ea=0.0)  # k=100
        rxn = ThirdBodyReaction(base_rate=base)
        concs = {"N2": 0.5, "O2": 0.3}
        k = rxn.rate(T=300.0, concentrations=concs)
        # [M] = 0.5*1.0 + 0.3*1.0 = 0.8, k_eff = 100 * 0.8 = 80
        assert abs(float(k.item()) - 80.0) < 1e-3

    def test_custom_efficiencies(self):
        """自定义效率应正确加权。"""
        base = ArrheniusReaction(A=100.0, Ea=0.0)
        rxn = ThirdBodyReaction(base_rate=base, efficiencies={"H2O": 12.0, "N2": 0.5})
        concs = {"H2O": 0.1, "N2": 0.5}
        k = rxn.rate(T=300.0, concentrations=concs)
        # [M] = 0.1*12.0 + 0.5*0.5 = 1.2 + 0.25 = 1.45, k_eff = 100 * 1.45 = 145
        assert abs(float(k.item()) - 145.0) < 1e-3

    def test_no_concentrations_defaults_to_one(self):
        """无浓度信息时 [M] = 1.0。"""
        base = ArrheniusReaction(A=50.0, Ea=0.0)
        rxn = ThirdBodyReaction(base_rate=base)
        k = rxn.rate(T=300.0)
        assert abs(float(k.item()) - 50.0) < 1e-6

    def test_tensor_concentrations(self):
        """应支持张量浓度输入。"""
        base = ArrheniusReaction(A=1.0, Ea=0.0)
        rxn = ThirdBodyReaction(base_rate=base)
        concs = {"A": torch.tensor([0.1, 0.2, 0.3])}
        k = rxn.rate(T=torch.tensor([300.0, 400.0, 500.0]), concentrations=concs)
        assert k.shape == (3,)

    def test_rts_create(self):
        """RTS 工厂应能创建 ThirdBodyReaction。"""
        rxn = ReactionRateModel.create(
            "thirdBody",
            base_rate=ArrheniusReaction(A=100.0, Ea=0.0),
            efficiencies={"O2": 2.0},
        )
        assert isinstance(rxn, ThirdBodyReaction)


# ===================================================================
# FallOffReaction
# ===================================================================


class TestFallOffReaction:
    """Lindemann 降压反应速率测试。"""

    def test_low_pressure_limit(self):
        """低压极限: [M] → 0 时 k → 0。"""
        k0 = ArrheniusReaction(A=1e10, Ea=0.0)
        k_inf = ArrheniusReaction(A=1e6, Ea=0.0)
        rxn = FallOffReaction(k0=k0, k_inf=k_inf)
        concs = {"M": 1e-10}
        k = rxn.rate(T=300.0, concentrations=concs)
        # Pr = k0 * [M] / k_inf = 1e10 * 1e-10 / 1e6 = 1e-6
        # k = k_inf * Pr / (1+Pr) ≈ k_inf * Pr = 1e6 * 1e-6 = 1.0
        assert float(k.item()) < 2.0

    def test_high_pressure_limit(self):
        """高压极限: [M] → inf 时 k → k_inf。"""
        k0 = ArrheniusReaction(A=1e10, Ea=0.0)
        k_inf = ArrheniusReaction(A=1e6, Ea=0.0)
        rxn = FallOffReaction(k0=k0, k_inf=k_inf)
        concs = {"M": 1e10}
        k = rxn.rate(T=300.0, concentrations=concs)
        # Pr = 1e10 * 1e10 / 1e6 = 1e14, k = 1e6 * 1e14/(1+1e14) ≈ 1e6
        assert abs(float(k.item()) - 1e6) / 1e6 < 0.01

    def test_transition_regime(self):
        """过渡区域: Pr ~ 1 时 k 介于 k0*[M] 和 k_inf 之间。"""
        k0 = ArrheniusReaction(A=1e10, Ea=0.0)
        k_inf = ArrheniusReaction(A=1e8, Ea=0.0)
        rxn = FallOffReaction(k0=k0, k_inf=k_inf)
        # 设 Pr ≈ 1: [M] = k_inf / k0 = 1e8/1e10 = 0.01
        concs = {"M": 0.01}
        k = rxn.rate(T=300.0, concentrations=concs)
        # k = k_inf * 0.5 = 5e7
        assert abs(float(k.item()) - 5e7) / 5e7 < 0.01

    def test_no_concentrations(self):
        """无浓度信息时 [M] = 1.0。"""
        k0 = ArrheniusReaction(A=1e10, Ea=0.0)
        k_inf = ArrheniusReaction(A=1e6, Ea=0.0)
        rxn = FallOffReaction(k0=k0, k_inf=k_inf)
        k = rxn.rate(T=300.0)
        # Pr = 1e10 * 1 / 1e6 = 1e4, k = 1e6 * 1e4/(1+1e4) ≈ 1e6
        assert abs(float(k.item()) - 1e6) / 1e6 < 0.01

    def test_tensor_input(self):
        """应支持张量输入。"""
        k0 = ArrheniusReaction(A=1e10, Ea=0.0)
        k_inf = ArrheniusReaction(A=1e6, Ea=0.0)
        rxn = FallOffReaction(k0=k0, k_inf=k_inf)
        T = torch.tensor([300.0, 600.0, 1000.0])
        concs = {"M": torch.tensor([1.0, 1.0, 1.0])}
        k = rxn.rate(T=T, concentrations=concs)
        assert k.shape == (3,)
        assert torch.isfinite(k).all()
        assert (k > 0).all()

    def test_rts_create(self):
        """RTS 工厂应能创建 FallOffReaction。"""
        rxn = ReactionRateModel.create("fallOff")
        assert isinstance(rxn, FallOffReaction)


# ===================================================================
# CombustionModel RTS 注册表
# ===================================================================


class TestCombustionRTS:
    """CombustionModel 运行时选择注册表测试。"""

    def test_registry_contains_all_models(self):
        types = CombustionModel.available_types()
        assert "PaSR" in types
        assert "EDC" in types
        assert "infinitelyFast" in types

    def test_factory_create_pasr(self):
        model = CombustionModel.create("PaSR", A=1e10, Ea=8e4, C_mix=0.1)
        assert isinstance(model, PaSRModel)

    def test_factory_create_edc(self):
        model = CombustionModel.create("EDC")
        assert isinstance(model, EDCModel)

    def test_factory_create_infinitely_fast(self):
        model = CombustionModel.create("infinitelyFast")
        assert isinstance(model, InfinitelyFastChemistry)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown combustion model"):
            CombustionModel.create("nonexistent")

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            CombustionModel()


# ===================================================================
# PaSRModel
# ===================================================================


class TestPaSRModel:
    """PaSR 燃烧模型测试。"""

    def test_finite_source_terms(self):
        """源项应对合理输入产生有限值。"""
        model = PaSRModel(A=1e10, Ea=8e4, C_mix=0.1)
        Su, Sp = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
        assert torch.isfinite(Su).all()
        assert torch.isfinite(Sp).all()

    def test_source_positive(self):
        """正的燃料和氧化剂浓度应产生正的燃烧源项。"""
        model = PaSRModel(A=1e10, Ea=8e4, C_mix=0.1)
        Su, Sp = model.source(Y_fuel=0.05, Y_ox=0.23, T=1500.0, rho=1.2)
        assert float(Su.item()) > 0.0

    def test_zero_fuel_gives_zero_source(self):
        """无燃料时燃烧源项为零。"""
        model = PaSRModel(A=1e10, Ea=8e4, C_mix=0.1)
        Su, Sp = model.source(Y_fuel=0.0, Y_ox=0.23, T=1500.0, rho=1.0)
        assert abs(float(Su.item())) < 1e-30

    def test_zero_oxidizer_gives_zero_source(self):
        """无氧化剂时燃烧源项为零。"""
        model = PaSRModel(A=1e10, Ea=8e4, C_mix=0.1)
        Su, Sp = model.source(Y_fuel=0.05, Y_ox=0.0, T=1500.0, rho=1.0)
        assert abs(float(Su.item())) < 1e-30

    def test_source_increases_with_temperature(self):
        """燃烧源项应随温度增大。"""
        model = PaSRModel(A=1e10, Ea=8e4, C_mix=0.1)
        Su_low, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=500.0, rho=1.0)
        Su_high, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=2000.0, rho=1.0)
        assert float(Su_high.item()) > float(Su_low.item())

    def test_sp_positive(self):
        """Sp（dSu/dT）应为正值（燃烧随温度增强）。"""
        model = PaSRModel(A=1e10, Ea=8e4, C_mix=0.1)
        _, Sp = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
        assert float(Sp.item()) > 0.0

    def test_tensor_input(self):
        """应支持张量输入。"""
        model = PaSRModel(A=1e10, Ea=8e4, C_mix=0.1)
        n = 5
        Yf = torch.full((n,), 0.05)
        Yo = torch.full((n,), 0.23)
        T = torch.full((n,), 1000.0)
        rho = torch.full((n,), 1.0)
        Su, Sp = model.source(Yf, Yo, T, rho)
        assert Su.shape == (n,)
        assert Sp.shape == (n,)
        assert torch.isfinite(Su).all()

    def test_stoich_ratio_limits_fuel(self):
        """化学计量比应限制燃料消耗量。"""
        model = PaSRModel(A=1e10, Ea=0.0, C_mix=1.0, stoich_ratio=2.0)
        # Y_fuel=0.1, Y_ox=0.1, s=2 → Y_limit = min(0.1, 0.1/2) = 0.05
        Su1, _ = model.source(Y_fuel=0.1, Y_ox=0.1, T=300.0, rho=1.0)
        model_no_limit = PaSRModel(A=1e10, Ea=0.0, C_mix=1.0, stoich_ratio=1.0)
        Su2, _ = model_no_limit.source(Y_fuel=0.1, Y_ox=0.1, T=300.0, rho=1.0)
        assert float(Su1.item()) < float(Su2.item())

    def test_repr(self):
        model = PaSRModel(A=1e10, Ea=8e4)
        r = repr(model)
        assert "PaSRModel" in r


# ===================================================================
# EDCModel
# ===================================================================


class TestEDCModel:
    """EDC 燃烧模型测试。"""

    def test_finite_source_terms(self):
        """源项应对合理输入产生有限值。"""
        model = EDCModel(C_tau=0.4, tau_chem=1e-3)
        Su, Sp = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
        assert torch.isfinite(Su).all()
        assert torch.isfinite(Sp).all()

    def test_source_positive(self):
        """正的燃料和氧化剂浓度应产生正的燃烧源项。"""
        model = EDCModel(C_tau=0.4, tau_chem=1e-3)
        Su, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
        assert float(Su.item()) > 0.0

    def test_zero_fuel_gives_zero_source(self):
        """无燃料时源项为零。"""
        model = EDCModel(C_tau=0.4, tau_chem=1e-3)
        Su, _ = model.source(Y_fuel=0.0, Y_ox=0.23, T=1000.0, rho=1.0)
        assert abs(float(Su.item())) < 1e-30

    def test_gamma_limits_to_one(self):
        """当 tau_star >= tau_chem 时 gamma 应被截断到 1。"""
        # C_tau >= tau_chem 时 gamma >= 1，应被 clamp 到 1
        model = EDCModel(C_tau=1.0, tau_chem=0.5)
        # gamma = sqrt(1.0 / 0.5) = sqrt(2) ≈ 1.414 → clamped to 1.0
        # gamma^2/(1-gamma^2) 当 gamma→1 时发散，应被 clamp(min=1e-10) 限制
        Su, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
        assert torch.isfinite(Su).all()

    def test_small_gamma_gives_small_source(self):
        """小的 gamma（小 C_tau 或大 tau_chem）应产生小的源项。"""
        model = EDCModel(C_tau=1e-6, tau_chem=1.0)
        Su, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
        # gamma = sqrt(1e-6/1.0) = 0.001, gamma^2/(1-gamma^2) ≈ 1e-6
        assert float(Su.item()) < 1e-4

    def test_source_scales_with_density(self):
        """源项应与密度成正比。"""
        model = EDCModel(C_tau=0.4, tau_chem=1e-3)
        Su1, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
        Su2, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=2.0)
        assert abs(float(Su2.item()) / float(Su1.item()) - 2.0) < 1e-6

    def test_stoich_ratio(self):
        """化学计量比应限制氧化剂效率。"""
        model = EDCModel(C_tau=0.4, tau_chem=1e-3, stoich_ratio=2.0)
        # Y_fuel=0.1, Y_ox=0.1, s=2 → Y_limit = min(0.1, 0.05) = 0.05
        Su, _ = model.source(Y_fuel=0.1, Y_ox=0.1, T=1000.0, rho=1.0)
        # Y_fuel=0.1, Y_ox=0.5, s=2 → Y_limit = min(0.1, 0.25) = 0.1
        Su2, _ = model.source(Y_fuel=0.1, Y_ox=0.5, T=1000.0, rho=1.0)
        assert float(Su2.item()) > float(Su.item())

    def test_tensor_input(self):
        """应支持张量输入。"""
        model = EDCModel()
        n = 5
        Yf = torch.full((n,), 0.05)
        Yo = torch.full((n,), 0.23)
        T = torch.full((n,), 1000.0)
        rho = torch.full((n,), 1.0)
        Su, Sp = model.source(Yf, Yo, T, rho)
        assert Su.shape == (n,)
        assert Sp.shape == (n,)

    def test_repr(self):
        model = EDCModel(C_tau=0.4, tau_chem=1e-3)
        r = repr(model)
        assert "EDCModel" in r


# ===================================================================
# InfinitelyFastChemistry
# ===================================================================


class TestInfinitelyFastChemistry:
    """无限快化学燃烧模型测试。"""

    def test_basic_source(self):
        """Su = rho * min(Y_fuel, Y_ox/s) / dt。"""
        model = InfinitelyFastChemistry(dt=1e-3, stoich_ratio=1.0)
        Su, Sp = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
        # Y_limit = min(0.05, 0.23) = 0.05
        expected = 1.0 * 0.05 / 1e-3
        assert abs(float(Su.item()) - expected) < 1e-6

    def test_fuel_limited(self):
        """燃料受限时 Y_fuel < Y_ox/s。"""
        model = InfinitelyFastChemistry(dt=1e-3, stoich_ratio=1.0)
        Su, _ = model.source(Y_fuel=0.01, Y_ox=0.23, T=1000.0, rho=1.0)
        expected = 1.0 * 0.01 / 1e-3
        assert abs(float(Su.item()) - expected) < 1e-6

    def test_oxidizer_limited(self):
        """氧化剂受限时 Y_ox/s < Y_fuel。"""
        model = InfinitelyFastChemistry(dt=1e-3, stoich_ratio=4.0)
        Su, _ = model.source(Y_fuel=0.1, Y_ox=0.2, T=1000.0, rho=1.0)
        # Y_limit = min(0.1, 0.2/4) = min(0.1, 0.05) = 0.05
        expected = 1.0 * 0.05 / 1e-3
        assert abs(float(Su.item()) - expected) < 1e-6

    def test_independent_of_temperature(self):
        """源项应与温度无关。"""
        model = InfinitelyFastChemistry(dt=1e-3, stoich_ratio=1.0)
        Su1, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=300.0, rho=1.0)
        Su2, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=3000.0, rho=1.0)
        assert abs(float(Su1.item()) - float(Su2.item())) < 1e-10

    def test_sp_zero(self):
        """无限快化学的 Sp 应为零（与温度无关）。"""
        model = InfinitelyFastChemistry()
        _, Sp = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
        assert float(Sp.item()) == 0.0

    def test_scales_with_density(self):
        """源项应与密度成正比。"""
        model = InfinitelyFastChemistry(dt=1e-3)
        Su1, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
        Su2, _ = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=3.0)
        assert abs(float(Su2.item()) / float(Su1.item()) - 3.0) < 1e-6

    def test_zero_reactants(self):
        """无反应物时源项为零。"""
        model = InfinitelyFastChemistry()
        Su, _ = model.source(Y_fuel=0.0, Y_ox=0.0, T=1000.0, rho=1.0)
        assert abs(float(Su.item())) < 1e-30

    def test_tensor_input(self):
        """应支持张量输入。"""
        model = InfinitelyFastChemistry(dt=1e-3)
        n = 5
        Yf = torch.full((n,), 0.05)
        Yo = torch.full((n,), 0.23)
        T = torch.full((n,), 1000.0)
        rho = torch.full((n,), 1.0)
        Su, Sp = model.source(Yf, Yo, T, rho)
        assert Su.shape == (n,)
        assert Sp.shape == (n,)
        assert torch.isfinite(Su).all()

    def test_repr(self):
        model = InfinitelyFastChemistry(dt=1e-3, stoich_ratio=2.0)
        r = repr(model)
        assert "InfinitelyFastChemistry" in r
        assert "0.001" in r


# ===================================================================
# 跨模型集成测试
# ===================================================================


class TestIntegration:
    """集成测试：组合使用多个模型。"""

    def test_arrhenius_in_third_body(self):
        """Arrhenius + 第三体组合。"""
        base = ArrheniusReaction(A=1e10, b=0.0, Ea=8e4)
        rxn = ThirdBodyReaction(base_rate=base, efficiencies={"H2O": 12.0})
        concs = {"H2O": 0.05, "N2": 0.78}
        k = rxn.rate(T=1000.0, concentrations=concs)
        assert float(k.item()) > 0.0
        assert torch.isfinite(k).all()

    def test_arrhenius_in_fall_off(self):
        """Arrhenius + Lindemann 降压组合。"""
        k0 = ArrheniusReaction(A=1e14, b=-0.5, Ea=8e4)
        k_inf = ArrheniusReaction(A=1e11, b=0.0, Ea=8e4)
        rxn = FallOffReaction(k0=k0, k_inf=k_inf)
        concs = {"N2": 0.78, "O2": 0.22}
        k = rxn.rate(T=1000.0, concentrations=concs)
        assert float(k.item()) > 0.0
        assert float(k.item()) < float(k_inf.rate(T=1000.0).item())

    def test_pasr_at_different_conditions(self):
        """PaSR 在不同工况下应产生合理结果。"""
        model = PaSRModel(A=1e10, Ea=8e4, C_mix=0.1)
        conditions = [
            (0.05, 0.23, 300.0, 1.2),    # 低温
            (0.05, 0.23, 1500.0, 1.0),   # 高温
            (0.01, 0.23, 1000.0, 0.5),   # 稀混合
            (0.10, 0.23, 1000.0, 2.0),   # 浓混合
        ]
        for Yf, Yo, T, rho in conditions:
            Su, Sp = model.source(Yf, Yo, T, rho)
            assert torch.isfinite(Su).all()
            assert torch.isfinite(Sp).all()

    def test_edc_at_different_conditions(self):
        """EDC 在不同工况下应产生合理结果。"""
        model = EDCModel(C_tau=0.4, tau_chem=1e-3)
        conditions = [
            (0.05, 0.23, 300.0, 1.2),
            (0.05, 0.23, 1500.0, 1.0),
            (0.01, 0.23, 1000.0, 0.5),
            (0.10, 0.23, 1000.0, 2.0),
        ]
        for Yf, Yo, T, rho in conditions:
            Su, Sp = model.source(Yf, Yo, T, rho)
            assert torch.isfinite(Su).all()

    def test_models_produce_different_results(self):
        """不同燃烧模型应产生不同结果。"""
        Yf, Yo, T, rho = 0.05, 0.23, 1000.0, 1.0
        pasr = PaSRModel(A=1e10, Ea=8e4, C_mix=0.1)
        edc = EDCModel(C_tau=0.4, tau_chem=1e-3)
        inf_model = InfinitelyFastChemistry(dt=1e-3)

        Su_pasr, _ = pasr.source(Yf, Yo, T, rho)
        Su_edc, _ = edc.source(Yf, Yo, T, rho)
        Su_inf, _ = inf_model.source(Yf, Yo, T, rho)

        # 三个模型应产生不同的数值
        vals = [float(Su_pasr.item()), float(Su_edc.item()), float(Su_inf.item())]
        assert len(set(f"{v:.6e}" for v in vals)) > 1
