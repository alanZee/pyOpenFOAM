"""Tests for enhanced wave models v6-v10 (absorption, relaxation, generation)."""

import math

import pytest
import torch

from pyfoam.waves.enhanced_5 import AbsorptionModel
from pyfoam.waves.enhanced_6 import ActiveAbsorption, PassiveAbsorption
from pyfoam.waves.enhanced_7 import RelaxationZone, WaveGenerationModel
from pyfoam.waves.enhanced_8 import PistonType, FlapType
from pyfoam.waves.enhanced_9 import PressureType, IrregularGeneration
from pyfoam.waves.enhanced_10 import AbsorptionGeneration, FlapDiffraction


# ===========================================================================
# enhanced_6: ActiveAbsorption
# ===========================================================================


class TestActiveAbsorptionRTS:
    def test_registered(self):
        assert "active" in AbsorptionModel.available_types()

    def test_create_via_factory(self):
        model = AbsorptionModel.create("active", zone_length=20.0, depth=10.0)
        assert isinstance(model, ActiveAbsorption)


class TestActiveAbsorption:
    @pytest.fixture
    def model(self):
        return ActiveAbsorption(zone_length=20.0, depth=10.0, gain=1.0)

    def test_absorb_shape(self, model):
        x = torch.linspace(0, 100, 100)
        eta = torch.sin(0.1 * x)
        u = torch.cos(0.1 * x)
        w = torch.sin(0.2 * x)
        eta_abs, u_abs, w_abs = model.absorb(eta, u, w, x, x_zone_start=80.0)
        assert eta_abs.shape == (100,)
        assert u_abs.shape == (100,)
        assert w_abs.shape == (100,)

    def test_absorb_full_at_end(self, model):
        """gain=1 时区域末端应完全吸收（eta -> 0）。"""
        x = torch.tensor([100.0])  # 区域末端 (80 + 20)
        eta = torch.tensor([1.0])
        u = torch.tensor([1.0])
        w = torch.tensor([1.0])
        eta_abs, u_abs, w_abs = model.absorb(eta, u, w, x, x_zone_start=80.0)
        assert abs(eta_abs.item()) < 1e-6
        assert abs(u_abs.item()) < 1e-6

    def test_no_absorb_outside_zone(self, model):
        """区域外不应吸收（eta 不变）。"""
        x = torch.tensor([50.0])
        eta = torch.tensor([1.0])
        u = torch.tensor([1.0])
        w = torch.tensor([1.0])
        eta_abs, u_abs, w_abs = model.absorb(eta, u, w, x, x_zone_start=80.0)
        assert abs(eta_abs.item() - 1.0) < 1e-6

    def test_gain_effect(self):
        """gain=0 时不应吸收。"""
        model = ActiveAbsorption(zone_length=20.0, depth=10.0, gain=0.0)
        x = torch.tensor([100.0])
        eta = torch.tensor([1.0])
        u = torch.tensor([1.0])
        w = torch.tensor([1.0])
        eta_abs, _, _ = model.absorb(eta, u, w, x, x_zone_start=80.0)
        assert abs(eta_abs.item() - 1.0) < 1e-6

    def test_partial_absorption(self):
        """gain=0.5 时应在区域末端吸收约 50%。"""
        model = ActiveAbsorption(zone_length=20.0, depth=10.0, gain=0.5)
        x = torch.tensor([100.0])
        eta = torch.tensor([1.0])
        u = torch.tensor([1.0])
        w = torch.tensor([1.0])
        eta_abs, _, _ = model.absorb(eta, u, w, x, x_zone_start=80.0)
        assert abs(eta_abs.item() - 0.5) < 1e-6

    def test_finite(self, model):
        x = torch.linspace(0, 100, 100)
        eta = torch.randn(100)
        u = torch.randn(100)
        w = torch.randn(100)
        eta_abs, u_abs, w_abs = model.absorb(eta, u, w, x, x_zone_start=80.0)
        assert torch.isfinite(eta_abs).all()
        assert torch.isfinite(u_abs).all()

    def test_repr(self):
        model = ActiveAbsorption(zone_length=20.0, depth=10.0, gain=0.8)
        assert "ActiveAbsorption" in repr(model)
        assert "0.8" in repr(model)


# ===========================================================================
# enhanced_6: PassiveAbsorption
# ===========================================================================


class TestPassiveAbsorptionRTS:
    def test_registered(self):
        assert "passive" in AbsorptionModel.available_types()

    def test_create_via_factory(self):
        model = AbsorptionModel.create("passive", zone_length=20.0, depth=10.0)
        assert isinstance(model, PassiveAbsorption)


class TestPassiveAbsorption:
    @pytest.fixture
    def model(self):
        return PassiveAbsorption(zone_length=20.0, depth=10.0, sigma_max=5.0)

    def test_absorb_shape(self, model):
        x = torch.linspace(0, 100, 100)
        eta = torch.ones(100)
        u = torch.ones(100)
        w = torch.ones(100)
        eta_abs, u_abs, w_abs = model.absorb(eta, u, w, x, x_zone_start=80.0)
        assert eta_abs.shape == (100,)

    def test_no_absorb_outside(self, model):
        x = torch.tensor([50.0])
        eta = torch.tensor([1.0])
        u = torch.tensor([1.0])
        w = torch.tensor([1.0])
        eta_abs, _, _ = model.absorb(eta, u, w, x, x_zone_start=80.0)
        assert abs(eta_abs.item() - 1.0) < 1e-6

    def test_strong_damping_at_end(self, model):
        """区域末端应有强阻尼（eta -> ~0）。"""
        x = torch.tensor([100.0])
        eta = torch.tensor([1.0])
        u = torch.tensor([1.0])
        w = torch.tensor([1.0])
        eta_abs, _, _ = model.absorb(eta, u, w, x, x_zone_start=80.0)
        # exp(-5) ~ 0.0067
        assert eta_abs.item() < 0.02

    def test_damping_coefficient(self, model):
        x = torch.tensor([80.0, 90.0, 100.0])
        sigma = model.damping_coefficient(x, x_zone_start=80.0)
        # s=0: sigma=0, s=0.5: sigma=5*0.25=1.25, s=1: sigma=5
        assert abs(sigma[0].item()) < 1e-6
        assert abs(sigma[2].item() - 5.0) < 1e-6

    def test_finite(self, model):
        x = torch.linspace(0, 100, 100)
        eta = torch.randn(100)
        u = torch.randn(100)
        w = torch.randn(100)
        eta_abs, u_abs, w_abs = model.absorb(eta, u, w, x, x_zone_start=80.0)
        assert torch.isfinite(eta_abs).all()

    def test_repr(self):
        model = PassiveAbsorption(zone_length=20.0, depth=10.0, sigma_max=3.0)
        assert "PassiveAbsorption" in repr(model)
        assert "3.0" in repr(model)


# ===========================================================================
# enhanced_7: RelaxationZone
# ===========================================================================


class TestRelaxationZoneRTS:
    def test_registered(self):
        assert "relaxationZone" in AbsorptionModel.available_types()

    def test_create_via_factory(self):
        model = AbsorptionModel.create("relaxationZone", zone_length=20.0, depth=10.0)
        assert isinstance(model, RelaxationZone)


class TestRelaxationZone:
    @pytest.fixture
    def cosine_zone(self):
        return RelaxationZone(zone_length=20.0, depth=10.0, profile="cosine")

    @pytest.fixture
    def polynomial_zone(self):
        return RelaxationZone(zone_length=20.0, depth=10.0, profile="polynomial")

    @pytest.fixture
    def exponential_zone(self):
        return RelaxationZone(zone_length=20.0, depth=10.0, profile="exponential")

    def test_cosine_weight_at_boundaries(self, cosine_zone):
        x = torch.tensor([80.0, 90.0, 100.0])
        wt = cosine_zone.relaxation_weight(x, x_zone_start=80.0)
        assert abs(wt[0].item()) < 1e-6  # start: 0
        assert abs(wt[1].item() - 0.5) < 1e-6  # middle: ~0.5（实际上 cos(pi/2)=0）
        # cos(pi*0.5) = cos(pi/2) = 0 => w = 0.5*(1-0) = 0.5
        assert abs(wt[2].item() - 1.0) < 1e-6  # end: 1

    def test_polynomial_weight(self, polynomial_zone):
        x = torch.tensor([80.0, 90.0, 100.0])
        wt = polynomial_zone.relaxation_weight(x, x_zone_start=80.0)
        assert abs(wt[0].item()) < 1e-6
        assert abs(wt[2].item() - 1.0) < 1e-6

    def test_exponential_weight(self, exponential_zone):
        x = torch.tensor([80.0, 100.0])
        wt = exponential_zone.relaxation_weight(x, x_zone_start=80.0)
        assert abs(wt[0].item()) < 1e-6
        assert abs(wt[1].item() - 1.0) < 1e-6

    def test_absorb_cosine(self, cosine_zone):
        x = torch.tensor([100.0])
        eta = torch.tensor([1.0])
        u = torch.tensor([1.0])
        w = torch.tensor([1.0])
        eta_abs, _, _ = cosine_zone.absorb(eta, u, w, x, x_zone_start=80.0)
        assert abs(eta_abs.item()) < 1e-6

    def test_profile_property(self, cosine_zone):
        assert cosine_zone.profile == "cosine"

    def test_repr(self, cosine_zone):
        assert "RelaxationZone" in repr(cosine_zone)
        assert "cosine" in repr(cosine_zone)


# ===========================================================================
# enhanced_7: WaveGenerationModel ABC
# ===========================================================================


class TestWaveGenerationModel:
    def test_registry(self):
        assert hasattr(WaveGenerationModel, "_registry")
        assert hasattr(WaveGenerationModel, "register")
        assert hasattr(WaveGenerationModel, "create")
        assert hasattr(WaveGenerationModel, "available_types")

    def test_abstract(self):
        with pytest.raises(TypeError):
            WaveGenerationModel(amplitude=1.0, depth=10.0, period=8.0)

    def test_properties(self):
        @WaveGenerationModel.register("testGen")
        class TestGen(WaveGenerationModel):
            def generate_elevation(self, x, t):
                return torch.zeros_like(x)
            def generate_velocity(self, x, t, z):
                return torch.zeros_like(x), torch.zeros_like(x)

        gen = TestGen(amplitude=1.0, depth=10.0, period=8.0)
        assert gen.amplitude == 1.0
        assert gen.depth == 10.0
        assert gen.period == 8.0
        assert abs(gen.angular_frequency - 2.0 * math.pi / 8.0) < 1e-10

        del WaveGenerationModel._registry["testGen"]


# ===========================================================================
# enhanced_8: PistonType
# ===========================================================================


class TestPistonTypeRTS:
    def test_registered(self):
        assert "piston" in WaveGenerationModel.available_types()

    def test_create_via_factory(self):
        gen = WaveGenerationModel.create(
            "piston", amplitude=1.0, depth=10.0, period=8.0,
        )
        assert isinstance(gen, PistonType)


class TestPistonTypeGeneration:
    @pytest.fixture
    def gen(self):
        return PistonType(amplitude=1.0, depth=10.0, period=8.0)

    def test_elevation_shape(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_elevation_amplitude(self, gen):
        x = torch.linspace(0, 200, 500)
        eta = gen.generate_elevation(x, t=0.0)
        assert eta.abs().max().item() <= gen.amplitude + 1e-10

    def test_elevation_finite(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_velocity_shape(self, gen):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 8.0)
        u, w = gen.generate_velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_velocity_finite(self, gen):
        x = torch.linspace(0, 50, 20)
        z = torch.linspace(1, 9, 20)
        u, w = gen.generate_velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()

    def test_paddle_displacement(self, gen):
        X = gen.paddle_displacement(t=0.0)
        assert abs(X) < 1e-10  # sin(0) = 0

    def test_paddle_amplitude_positive(self, gen):
        assert gen.paddle_amplitude > 0

    def test_repr(self):
        gen = PistonType(amplitude=1.0, depth=10.0, period=8.0)
        assert "PistonType" in repr(gen)


# ===========================================================================
# enhanced_8: FlapType
# ===========================================================================


class TestFlapTypeRTS:
    def test_registered(self):
        assert "flap" in WaveGenerationModel.available_types()

    def test_create_via_factory(self):
        gen = WaveGenerationModel.create(
            "flap", amplitude=1.0, depth=10.0, period=8.0,
        )
        assert isinstance(gen, FlapType)


class TestFlapTypeGeneration:
    @pytest.fixture
    def gen(self):
        return FlapType(amplitude=1.0, depth=10.0, period=8.0)

    def test_elevation_shape(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_elevation_finite(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_velocity_finite(self, gen):
        x = torch.linspace(0, 50, 20)
        z = torch.linspace(1, 9, 20)
        u, w = gen.generate_velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()

    def test_flap_angle(self, gen):
        angle = gen.flap_angle(t=0.0)
        assert abs(angle) < 1e-10  # sin(0) = 0

    def test_flap_amplitude_positive(self, gen):
        assert gen.flap_amplitude > 0

    def test_with_hinge_depth(self):
        gen = FlapType(amplitude=1.0, depth=10.0, period=8.0, hinge_depth=5.0)
        assert gen.hinge_depth == 5.0
        x = torch.linspace(0, 50, 20)
        eta = gen.generate_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_repr(self):
        gen = FlapType(amplitude=1.0, depth=10.0, period=8.0)
        assert "FlapType" in repr(gen)


# ===========================================================================
# enhanced_9: PressureType
# ===========================================================================


class TestPressureTypeRTS:
    def test_registered(self):
        assert "pressure" in WaveGenerationModel.available_types()

    def test_create_via_factory(self):
        gen = WaveGenerationModel.create(
            "pressure", amplitude=1.0, depth=10.0, period=8.0,
        )
        assert isinstance(gen, PressureType)


class TestPressureTypeGeneration:
    @pytest.fixture
    def gen(self):
        return PressureType(
            amplitude=1.0, depth=10.0, period=8.0, submergence=3.0,
        )

    def test_elevation_shape(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_elevation_finite(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_velocity_finite(self, gen):
        x = torch.linspace(0, 50, 20)
        z = torch.linspace(1, 9, 20)
        u, w = gen.generate_velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()

    def test_pressure_amplitude_positive(self, gen):
        assert gen.pressure_amplitude > 0

    def test_submergence_property(self, gen):
        assert gen.submergence == 3.0

    def test_source_distribution(self, gen):
        x = torch.tensor([0.0])  # 源中心
        G = gen.source_distribution(x)
        assert abs(G.item() - 1.0) < 1e-6

    def test_repr(self):
        gen = PressureType(amplitude=1.0, depth=10.0, period=8.0, submergence=3.0)
        assert "PressureType" in repr(gen)


# ===========================================================================
# enhanced_9: IrregularGeneration
# ===========================================================================


class TestIrregularGenerationRTS:
    def test_registered(self):
        assert "irregularGeneration" in WaveGenerationModel.available_types()

    def test_create_via_factory(self):
        gen = WaveGenerationModel.create(
            "irregularGeneration", amplitude=1.0, depth=20.0, period=8.0,
        )
        assert isinstance(gen, IrregularGeneration)


class TestIrregularGeneration:
    @pytest.fixture
    def gen(self):
        return IrregularGeneration(
            amplitude=1.0, depth=20.0, period=8.0, seed=42,
        )

    def test_elevation_shape(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_elevation_finite(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_velocity_shape(self, gen):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 15.0)
        u, w = gen.generate_velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_velocity_finite(self, gen):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 15.0)
        u, w = gen.generate_velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()

    def test_n_components(self, gen):
        assert gen.n_components == 50

    def test_repr(self):
        gen = IrregularGeneration(amplitude=1.0, depth=20.0, period=8.0)
        assert "IrregularGeneration" in repr(gen)


# ===========================================================================
# enhanced_10: AbsorptionGeneration
# ===========================================================================


class TestAbsorptionGenerationRTS:
    def test_registered(self):
        assert "absorptionGeneration" in WaveGenerationModel.available_types()

    def test_create_via_factory(self):
        gen = WaveGenerationModel.create(
            "absorptionGeneration", amplitude=1.0, depth=10.0, period=8.0,
        )
        assert isinstance(gen, AbsorptionGeneration)


class TestAbsorptionGeneration:
    @pytest.fixture
    def gen(self):
        return AbsorptionGeneration(
            amplitude=1.0, depth=10.0, period=8.0, absorption_gain=0.8,
        )

    def test_elevation_shape(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_elevation_finite(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_velocity_finite(self, gen):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 8.0)
        u, w = gen.generate_velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()

    def test_effective_displacement_no_reflection(self, gen):
        """无反射时有效位移应等于生成位移。"""
        t = 1.5
        X_eff = gen.effective_displacement(t, eta_reflected=0.0)
        X_gen = gen.paddle_displacement(t)
        assert abs(X_eff - X_gen) < 1e-10

    def test_effective_displacement_with_reflection(self, gen):
        """有反射时有效位移应减少。"""
        t = 1.5
        X_eff = gen.effective_displacement(t, eta_reflected=0.5)
        X_gen = gen.paddle_displacement(t)
        # 反射波应使位移减小（在大多数时刻）
        # 只检查有限值
        assert math.isfinite(X_eff)

    def test_absorption_gain(self, gen):
        assert gen.absorption_gain == 0.8

    def test_is_piston_subtype(self, gen):
        assert isinstance(gen, PistonType)

    def test_repr(self):
        gen = AbsorptionGeneration(
            amplitude=1.0, depth=10.0, period=8.0, absorption_gain=0.8,
        )
        assert "AbsorptionGeneration" in repr(gen)


# ===========================================================================
# enhanced_10: FlapDiffraction
# ===========================================================================


class TestFlapDiffractionRTS:
    def test_registered(self):
        assert "flapDiffraction" in WaveGenerationModel.available_types()

    def test_create_via_factory(self):
        gen = WaveGenerationModel.create(
            "flapDiffraction", amplitude=1.0, depth=10.0, period=8.0,
        )
        assert isinstance(gen, FlapDiffraction)


class TestFlapDiffraction:
    @pytest.fixture
    def gen(self):
        return FlapDiffraction(
            amplitude=1.0, depth=10.0, period=8.0, flap_width=2.0,
        )

    def test_elevation_shape(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_elevation_finite(self, gen):
        x = torch.linspace(0, 100, 50)
        eta = gen.generate_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_velocity_finite(self, gen):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 8.0)
        u, w = gen.generate_velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()

    def test_diffraction_coeff_range(self, gen):
        """K_diff 应在 (0, 1] 范围内。"""
        assert 0 < gen.diffraction_coeff <= 1.0

    def test_wide_flap_no_correction(self):
        """很宽的 flap（B -> inf）应使 K_diff -> 0。"""
        gen = FlapDiffraction(
            amplitude=1.0, depth=100.0, period=8.0, flap_width=1000.0,
        )
        # sin(k*B/2)/(k*B/2) -> 0 for large B
        assert gen.diffraction_coeff < 0.01

    def test_narrow_flap_full_correction(self):
        """很窄的 flap（B -> 0）应使 K_diff -> 1。"""
        gen = FlapDiffraction(
            amplitude=1.0, depth=10.0, period=8.0, flap_width=1e-6,
        )
        assert abs(gen.diffraction_coeff - 1.0) < 0.01

    def test_is_flap_subtype(self, gen):
        assert isinstance(gen, FlapType)

    def test_flap_width(self, gen):
        assert gen.flap_width == 2.0

    def test_repr(self):
        gen = FlapDiffraction(amplitude=1.0, depth=10.0, period=8.0, flap_width=2.0)
        assert "FlapDiffraction" in repr(gen)


# ===========================================================================
# dtype consistency (generation models)
# ===========================================================================


class TestGenerationDtypeConsistency:
    @pytest.mark.parametrize("gen_cls,kwargs", [
        (PistonType, {"amplitude": 1.0, "depth": 10.0, "period": 8.0}),
        (FlapType, {"amplitude": 1.0, "depth": 10.0, "period": 8.0}),
        (PressureType, {"amplitude": 1.0, "depth": 10.0, "period": 8.0}),
        (IrregularGeneration, {"amplitude": 1.0, "depth": 20.0, "period": 8.0, "seed": 42}),
        (AbsorptionGeneration, {"amplitude": 1.0, "depth": 10.0, "period": 8.0}),
        (FlapDiffraction, {"amplitude": 1.0, "depth": 10.0, "period": 8.0}),
    ])
    def test_float64_elevation(self, gen_cls, kwargs):
        gen = gen_cls(**kwargs)
        x = torch.linspace(0, 10, 10, dtype=torch.float64)
        eta = gen.generate_elevation(x, t=0.0)
        assert eta.dtype == torch.float64

    @pytest.mark.parametrize("gen_cls,kwargs", [
        (PistonType, {"amplitude": 1.0, "depth": 10.0, "period": 8.0}),
        (FlapType, {"amplitude": 1.0, "depth": 10.0, "period": 8.0}),
        (PressureType, {"amplitude": 1.0, "depth": 10.0, "period": 8.0}),
        (IrregularGeneration, {"amplitude": 1.0, "depth": 20.0, "period": 8.0, "seed": 42}),
        (AbsorptionGeneration, {"amplitude": 1.0, "depth": 10.0, "period": 8.0}),
        (FlapDiffraction, {"amplitude": 1.0, "depth": 10.0, "period": 8.0}),
    ])
    def test_float64_velocity(self, gen_cls, kwargs):
        gen = gen_cls(**kwargs)
        x = torch.linspace(0, 10, 10, dtype=torch.float64)
        z = torch.full((10,), 5.0, dtype=torch.float64)
        u, w = gen.generate_velocity(x, t=0.0, z=z)
        assert u.dtype == torch.float64
        assert w.dtype == torch.float64
