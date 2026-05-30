"""
Unit tests for enhanced Lagrangian models v2.
"""

from __future__ import annotations

import math
import pytest


class TestInjectionEnhanced2:
    def test_cloud_injector_count(self):
        from pyfoam.lagrangian.injection import PointInjector
        from pyfoam.lagrangian.injection_enhanced_2 import CloudInjector

        base = PointInjector(origin=[0, 0, 0], velocity=[1, 0, 0], n_particles=3)
        source = base.inject()
        inj = CloudInjector(source_particles=source, n_particles=5, seed=42)
        assert len(inj.inject()) == 5

    def test_cloud_injector_properties(self):
        from pyfoam.lagrangian.injection import PointInjector
        from pyfoam.lagrangian.injection_enhanced_2 import CloudInjector

        base = PointInjector(origin=[0, 0, 0], velocity=[1, 0, 0], diameter=2e-4, density=2000.0)
        source = base.inject()
        inj = CloudInjector(source_particles=source, n_particles=2, seed=42)
        for p in inj.inject():
            assert p.diameter == 2e-4
            assert p.density == 2000.0

    def test_cloud_injector_empty_raises(self):
        from pyfoam.lagrangian.injection_enhanced_2 import CloudInjector
        with pytest.raises(ValueError, match="non-empty"):
            CloudInjector(source_particles=[], n_particles=1)

    def test_field_injector_count(self):
        from pyfoam.lagrangian.injection_enhanced_2 import FieldInjector
        inj = FieldInjector(
            centre=[0, 0, 0], sigma=[0.1, 0.1, 0.1],
            mean_velocity=[1, 0, 0], n_particles=10, seed=42,
        )
        assert len(inj.inject()) == 10

    def test_field_injector_centre_bias(self):
        from pyfoam.lagrangian.injection_enhanced_2 import FieldInjector
        inj = FieldInjector(
            centre=[5.0, 5.0, 5.0], sigma=[0.01, 0.01, 0.01],
            mean_velocity=[0, 0, 0], n_particles=100, seed=42,
        )
        particles = inj.inject()
        avg_x = sum(p.position[0] for p in particles) / len(particles)
        assert abs(avg_x - 5.0) < 0.5


class TestForcesEnhanced2:
    def test_virtual_mass_zero_accel(self):
        from pyfoam.lagrangian.forces_enhanced_2 import VirtualMassForce
        f = VirtualMassForce()
        a = f.acceleration([0, 0, 0], 1e-4, 1000.0)
        assert a == [0.0, 0.0, 0.0]

    def test_saffman_mei_zero_vorticity(self):
        from pyfoam.lagrangian.forces_enhanced_2 import SaffmanMeiLift
        f = SaffmanMeiLift(vorticity=[0, 0, 0])
        a = f.acceleration([0, 0, 0], 1e-4, 1000.0, fluid_velocity=[1, 0, 0])
        assert a == [0.0, 0.0, 0.0]


class TestBreakupEnhanced2:
    def test_pilch_erdman_no_breakup(self):
        from pyfoam.lagrangian.breakup_enhanced_2 import PilchErdman
        m = PilchErdman()
        r = m.breakup(dt=1e-4, diameter=1e-3, relative_velocity=0.0)
        assert r["broken"] is False

    def test_pilch_erdman_low_we(self):
        from pyfoam.lagrangian.breakup_enhanced_2 import PilchErdman
        m = PilchErdman()
        r = m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=1.0)
        assert r["broken"] is False

    def test_reitz_khrt_no_breakup(self):
        from pyfoam.lagrangian.breakup_enhanced_2 import ReitzKHRT
        m = ReitzKHRT()
        r = m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=0.0)
        assert r["broken"] is False


class TestCollisionEnhanced2:
    def test_trajectory_model_pass_through(self):
        from pyfoam.lagrangian.collision_enhanced_2 import TrajectoryModel
        m = TrajectoryModel()
        v1, v2 = m.collide([0, 0, 0], [1, 0, 0], 1e-4, 1000.0,
                           [1, 0, 0], [-1, 0, 0], 1e-4, 1000.0)
        assert len(v1) == 3
        assert len(v2) == 3

    def test_orourke_no_collision_far(self):
        from pyfoam.lagrangian.collision_enhanced_2 import ORourkeCollision
        m = ORourkeCollision(seed=42)
        v1, v2 = m.collide([0, 0, 0], [1, 0, 0], 1e-4, 1000.0,
                           [10, 0, 0], [-1, 0, 0], 1e-4, 1000.0)
        assert v1 == [1, 0, 0]


class TestDispersionEnhanced2:
    def test_gradient_rng_no_turbulence(self):
        from pyfoam.lagrangian.dispersion_enhanced_2 import GradientDispersionRNG
        m = GradientDispersionRNG()
        dv = m.disperse(dt=1e-4, turbulent_kinetic_energy=0.0)
        assert dv == [0.0, 0.0, 0.0]

    def test_stochastic_rng_no_turbulence(self):
        from pyfoam.lagrangian.dispersion_enhanced_2 import StochasticDispersionRNG
        m = StochasticDispersionRNG()
        dv = m.disperse(dt=1e-4, turbulent_kinetic_energy=0.0)
        assert dv == [0.0, 0.0, 0.0]


class TestEvaporationEnhanced2:
    def test_standard_evap_cold(self):
        from pyfoam.lagrangian.evaporation_enhanced_2 import StandardEvaporation
        m = StandardEvaporation()
        dm = m.evaporate(dt=1e-4, diameter=1e-3, temperature=400.0, fluid_temperature=300.0)
        assert dm == 0.0

    def test_diffusion_evap_cold(self):
        from pyfoam.lagrangian.evaporation_enhanced_2 import DiffusionEvaporation
        m = DiffusionEvaporation()
        dm = m.evaporate(dt=1e-4, diameter=1e-3, temperature=400.0, fluid_temperature=300.0)
        assert dm == 0.0


class TestMPPICEnhanced2:
    def test_syamlal_rogers_below_threshold(self):
        from pyfoam.lagrangian.mppic_enhanced_2 import SyamlalRogersFriction
        m = SyamlalRogersFriction()
        tau = m.friction_stress(alpha=0.1, strain_rate=10.0)
        assert tau == 0.0

    def test_gidaspow_below_threshold(self):
        from pyfoam.lagrangian.mppic_enhanced_2 import GidaspowFriction
        m = GidaspowFriction()
        tau = m.friction_stress(alpha=0.01, strain_rate=10.0)
        assert tau == 0.0


class TestOxidationEnhanced2:
    def test_intrinsic_cold(self):
        from pyfoam.lagrangian.oxidation_enhanced_2 import IntrinsicOxidation
        m = IntrinsicOxidation()
        dm = m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0)
        assert dm < 1e-20

    def test_cka_cold(self):
        from pyfoam.lagrangian.oxidation_enhanced_2 import CKAOxidation
        m = CKAOxidation()
        dm = m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0)
        assert dm < 1e-20


class TestSprayEnhanced2:
    def test_khrt_atomize_no_velocity(self):
        from pyfoam.lagrangian.spray_enhanced_2 import KHRTAtomization
        m = KHRTAtomization()
        r = m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)
        assert r["atomized"] is False

    def test_lisa_atomize_no_velocity(self):
        from pyfoam.lagrangian.spray_enhanced_2 import LISAAtomization
        m = LISAAtomization()
        r = m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)
        assert r["atomized"] is False


class TestWallEnhanced2:
    def test_bai_gosman_far(self):
        from pyfoam.lagrangian.wall_enhanced_2 import BaiGosmanSplash
        m = BaiGosmanSplash()
        r = m.interact(velocity=[0.1, 0, 0], wall_normal=[0, 1, 0])
        assert r["stuck"] is False

    def test_kuhnke_far(self):
        from pyfoam.lagrangian.wall_enhanced_2 import KuhnkeSplash
        m = KuhnkeSplash()
        r = m.interact(velocity=[0.1, 0, 0], wall_normal=[0, 1, 0])
        assert r["stuck"] is False


class TestReactingEnhanced2:
    def test_composition_cold(self):
        from pyfoam.lagrangian.reacting_enhanced_2 import CompositionModel
        m = CompositionModel()
        r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
        assert r["mass_loss"] < 1e-20

    def test_phase_change_cold(self):
        from pyfoam.lagrangian.reacting_enhanced_2 import PhaseChangeModel
        m = PhaseChangeModel()
        r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
        assert r["mass_loss"] == 0.0


class TestInjectionEnhanced3:
    def test_manual_injection_rate(self):
        from pyfoam.lagrangian.injection_enhanced_3 import ManualInjectionRate
        inj = ManualInjectionRate(rate_table=[(0.0, 5), (0.1, 10)])
        ps1 = inj.inject()
        assert len(ps1) == 5
        ps2 = inj.inject()
        assert len(ps2) == 10
        ps3 = inj.inject()
        assert len(ps3) == 0

    def test_lagrangian_mapping(self):
        from pyfoam.lagrangian.injection_enhanced_3 import LagrangianMappingInjector
        inj = LagrangianMappingInjector(
            cell_centres=[[0, 0, 0], [1, 0, 0]],
            particles_per_cell=2,
        )
        ps = inj.inject()
        assert len(ps) == 4


class TestInjectionEnhanced4:
    def test_thermo_cloud_injector(self):
        from pyfoam.lagrangian.injection_enhanced_4 import ThermoCloudInjector
        inj = ThermoCloudInjector(
            origin=[0, 0, 0], direction=[1, 0, 0],
            n_particles=5, seed=42,
        )
        ps = inj.inject()
        assert len(ps) == 5
        for p in ps:
            assert hasattr(p, 'enthalpy')

    def test_reacting_cloud_injector(self):
        from pyfoam.lagrangian.injection_enhanced_4 import ReactingCloudInjector
        inj = ReactingCloudInjector(
            surface_points=[[0, 0, 0]],
            surface_normals=[[0, 0, 1]],
            n_particles=3,
            species={"C": 0.8, "H2O": 0.2},
        )
        ps = inj.inject()
        assert len(ps) == 3
        for p in ps:
            assert hasattr(p, 'species')


class TestInjectionEnhanced5:
    def test_kinematic_parcel(self):
        from pyfoam.lagrangian.injection_enhanced_5 import KinematicParcelInjector
        inj = KinematicParcelInjector(
            origin=[0, 0, 0], velocity=[1, 0, 0],
            n_particles=3, mass_flow_rate=0.03,
        )
        ps = inj.inject()
        assert len(ps) == 3
        for p in ps:
            assert hasattr(p, 'parcel_mass')

    def test_solid_particle(self):
        from pyfoam.lagrangian.injection_enhanced_5 import SolidParticleInjector
        inj = SolidParticleInjector(
            origin=[0, 0, 0], direction=[1, 0, 0],
            n_particles=3, seed=42,
        )
        ps = inj.inject()
        assert len(ps) == 3
        for p in ps:
            assert hasattr(p, 'is_solid')


class TestForcesEnhanced3:
    def test_thermophoretic_zero_grad(self):
        from pyfoam.lagrangian.forces_enhanced_3 import ThermophoreticForce
        f = ThermophoreticForce(temperature_gradient=[0, 0, 0])
        a = f.acceleration([0, 0, 0], 1e-4, 1000.0)
        assert a == [0.0, 0.0, 0.0]

    def test_brownian_large_particle(self):
        from pyfoam.lagrangian.forces_enhanced_3 import BrownianMotionForce
        f = BrownianMotionForce()
        a = f.acceleration([0, 0, 0], 1e-3, 1000.0)  # d > 10 micron
        assert a == [0.0, 0.0, 0.0]


class TestForcesEnhanced4:
    def test_pressure_gradient(self):
        from pyfoam.lagrangian.forces_enhanced_4 import PressureGradientForce
        f = PressureGradientForce(pressure_gradient=[1000, 0, 0])
        a = f.acceleration([0, 0, 0], 1e-4, 1000.0)
        assert a[0] == pytest.approx(-1.0)
        assert a[1] == pytest.approx(0.0)

    def test_buoyancy_heavy_particle(self):
        from pyfoam.lagrangian.forces_enhanced_4 import BuoyancyForce
        f = BuoyancyForce()
        a = f.acceleration([0, 0, 0], 1e-4, 1000.0, fluid_density=1.225)
        # (rho_f/rho_p - 1) * g_z = (0.001225 - 1) * (-9.81) = +9.798
        # 浮力使有效重力减小（向上方向）
        assert a[2] > 0


class TestForcesEnhanced5:
    def test_magnus_zero_rotation(self):
        from pyfoam.lagrangian.forces_enhanced_5 import MagnusForce
        f = MagnusForce(angular_velocity=[0, 0, 0])
        a = f.acceleration([0, 0, 0], 1e-4, 1000.0, fluid_velocity=[1, 0, 0])
        assert a == [0.0, 0.0, 0.0]

    def test_paramagnetic_zero_grad(self):
        from pyfoam.lagrangian.forces_enhanced_5 import ParamagneticForce
        f = ParamagneticForce()
        a = f.acceleration([0, 0, 0], 1e-4, 1000.0)
        assert a == [0.0, 0.0, 0.0]


class TestBreakupEnhanced3:
    def test_tab_breakup_low_we(self):
        from pyfoam.lagrangian.breakup_enhanced_3 import TABBreakup
        m = TABBreakup()
        r = m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=1.0)
        assert r["broken"] is False

    def test_shf_breakup_low_we(self):
        from pyfoam.lagrangian.breakup_enhanced_3 import SHFBreakup
        m = SHFBreakup()
        r = m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=1.0)
        assert r["broken"] is False


class TestBreakupEnhanced4:
    def test_etab_breakup_low_we(self):
        from pyfoam.lagrangian.breakup_enhanced_4 import ETABBreakup
        m = ETABBreakup()
        r = m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=1.0)
        assert r["broken"] is False

    def test_ssd_breakup_low_we(self):
        from pyfoam.lagrangian.breakup_enhanced_4 import SSDBreakup
        m = SSDBreakup(seed=42)
        r = m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=1.0)
        assert r["broken"] is False


class TestBreakupEnhanced5:
    def test_enhanced_tab_low_we(self):
        from pyfoam.lagrangian.breakup_enhanced_5 import EnhancedTaylorAnalogy
        m = EnhancedTaylorAnalogy()
        r = m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=1.0)
        assert r["broken"] is False

    def test_khrt_breakup_low_we(self):
        from pyfoam.lagrangian.breakup_enhanced_5 import KHRTBreakup
        m = KHRTBreakup()
        r = m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=1.0)
        assert r["broken"] is False


class TestCollisionEnhanced3:
    def test_stochastic_collision_far(self):
        from pyfoam.lagrangian.collision_enhanced_3 import StochasticCollision
        m = StochasticCollision(seed=42)
        v1, v2 = m.collide([0, 0, 0], [1, 0, 0], 1e-4, 1000.0,
                           [10, 0, 0], [-1, 0, 0], 1e-4, 1000.0)
        assert v1 == [1, 0, 0]

    def test_no_separation_far(self):
        from pyfoam.lagrangian.collision_enhanced_3 import NoSeparation
        m = NoSeparation()
        v1, v2 = m.collide([0, 0, 0], [1, 0, 0], 1e-4, 1000.0,
                           [10, 0, 0], [-1, 0, 0], 1e-4, 1000.0)
        assert v1 == [1, 0, 0]


class TestCollisionEnhanced4:
    def test_spring_dashpot_no_contact(self):
        from pyfoam.lagrangian.collision_enhanced_4 import SpringDashpot
        m = SpringDashpot()
        v1, v2 = m.collide([0, 0, 0], [1, 0, 0], 1e-4, 1000.0,
                           [1, 0, 0], [-1, 0, 0], 1e-4, 1000.0)
        assert v1 == [1, 0, 0]

    def test_pair_collision_wall_far(self):
        from pyfoam.lagrangian.collision_enhanced_4 import PairCollisionWall
        m = PairCollisionWall()
        v1, v2 = m.collide([0, 0, 0], [1, 0, 0], 1e-4, 1000.0,
                           [1, 0, 0], [0, 0, 0], 1e-4, 1000.0)
        assert v1 == [1, 0, 0]


class TestCollisionEnhanced5:
    def test_subcycled_no_contact(self):
        from pyfoam.lagrangian.collision_enhanced_5 import SubCycledCollision
        m = SubCycledCollision(n_subcycles=3)
        v1, v2 = m.collide([0, 0, 0], [1, 0, 0], 1e-4, 1000.0,
                           [1, 0, 0], [-1, 0, 0], 1e-4, 1000.0)
        assert v1 == [1, 0, 0]

    def test_coulaloglu_far(self):
        from pyfoam.lagrangian.collision_enhanced_5 import CoulalogluCollision
        m = CoulalogluCollision()
        v1, v2 = m.collide([0, 0, 0], [1, 0, 0], 1e-4, 1000.0,
                           [10, 0, 0], [-1, 0, 0], 1e-4, 1000.0)
        assert v1 == [1, 0, 0]


class TestDispersionEnhanced3:
    def test_turbulent_no_turb(self):
        from pyfoam.lagrangian.dispersion_enhanced_3 import TurbulentDispersion
        m = TurbulentDispersion()
        assert m.disperse(dt=1e-4) == [0.0, 0.0, 0.0]

    def test_gradient_b_no_turb(self):
        from pyfoam.lagrangian.dispersion_enhanced_3 import GradientDispersionB
        m = GradientDispersionB()
        assert m.disperse(dt=1e-4) == [0.0, 0.0, 0.0]


class TestDispersionEnhanced4:
    def test_brownian_no_turb(self):
        from pyfoam.lagrangian.dispersion_enhanced_4 import BrownianDispersion
        m = BrownianDispersion()
        # 即使无湍流，Brownian 也有贡献（但很小）
        dv = m.disperse(dt=1e-4, particle_diameter=1e-7)
        assert len(dv) == 3

    def test_ras_no_turb(self):
        from pyfoam.lagrangian.dispersion_enhanced_4 import DispersionModelRAS
        m = DispersionModelRAS()
        assert m.disperse(dt=1e-4) == [0.0, 0.0, 0.0]


class TestDispersionEnhanced5:
    def test_ke_model_no_turb(self):
        from pyfoam.lagrangian.dispersion_enhanced_5 import DispersionModelKE
        m = DispersionModelKE()
        assert m.disperse(dt=1e-4) == [0.0, 0.0, 0.0]

    def test_inverse_no_turb(self):
        from pyfoam.lagrangian.dispersion_enhanced_5 import InverseTimeScaleDispersion
        m = InverseTimeScaleDispersion()
        assert m.disperse(dt=1e-4) == [0.0, 0.0, 0.0]


class TestEvaporationEnhanced3:
    def test_liquid_evap_cold(self):
        from pyfoam.lagrangian.evaporation_enhanced_3 import LiquidEvaporation
        m = LiquidEvaporation()
        dm = m.evaporate(dt=1e-4, diameter=1e-3, temperature=400, fluid_temperature=300)
        assert dm == 0.0

    def test_multi_component_cold(self):
        from pyfoam.lagrangian.evaporation_enhanced_3 import MultiComponentEvaporation
        m = MultiComponentEvaporation()
        dm = m.evaporate(dt=1e-4, diameter=1e-3, temperature=400, fluid_temperature=300)
        assert dm == 0.0


class TestEvaporationEnhanced4:
    def test_blowing_cold(self):
        from pyfoam.lagrangian.evaporation_enhanced_4 import BlowingEvaporation
        m = BlowingEvaporation()
        dm = m.evaporate(dt=1e-4, diameter=1e-3, temperature=400, fluid_temperature=300)
        assert dm == 0.0

    def test_no_evap_two_phase(self):
        from pyfoam.lagrangian.evaporation_enhanced_4 import NoEvaporationTwoPhase
        m = NoEvaporationTwoPhase()
        dm = m.evaporate(dt=1e-4, diameter=1e-3, temperature=300, fluid_temperature=500)
        assert dm == 0.0


class TestEvaporationEnhanced5:
    def test_evap_di_cold(self):
        from pyfoam.lagrangian.evaporation_enhanced_5 import EvaporationDI
        m = EvaporationDI()
        dm = m.evaporate(dt=1e-4, diameter=1e-3, temperature=400, fluid_temperature=300)
        assert dm == 0.0

    def test_frossling_cold(self):
        from pyfoam.lagrangian.evaporation_enhanced_5 import FrosslingEvaporation
        m = FrosslingEvaporation()
        dm = m.evaporate(dt=1e-4, diameter=1e-3, temperature=400, fluid_temperature=300)
        assert dm == 0.0


class TestMPPICEnhanced3:
    def test_ergun_below_threshold(self):
        from pyfoam.lagrangian.mppic_enhanced_3 import ErgunFriction
        m = ErgunFriction()
        assert m.packing_stress(alpha=0.0) == 0.0

    def test_packing_limit_below_crit(self):
        from pyfoam.lagrangian.mppic_enhanced_3 import PackingLimitModel
        m = PackingLimitModel()
        assert m.packing_stress(alpha=0.1) == 0.0


class TestMPPICEnhanced4:
    def test_isotropic_damping_zero_alpha(self):
        from pyfoam.lagrangian.mppic_enhanced_4 import IsotropicDamping
        m = IsotropicDamping()
        assert m.packing_stress(alpha=0.0) == 0.0

    def test_velocity_limiter_within_bounds(self):
        from pyfoam.lagrangian.mppic_enhanced_4 import VelocityLimiter
        m = VelocityLimiter()
        v = m.limit_velocity([1.0, 0.0, 0.0], alpha=0.1)
        assert v == [1.0, 0.0, 0.0]


class TestMPPICEnhanced5:
    def test_min_mass_limiter_ok(self):
        from pyfoam.lagrangian.mppic_enhanced_5 import MinimumMassLimiter
        m = MinimumMassLimiter()
        assert m.should_remove(1e-10) is False
        assert m.should_remove(1e-20) is True

    def test_temp_limiter_ok(self):
        from pyfoam.lagrangian.mppic_enhanced_5 import MaximumTemperatureLimiter
        m = MaximumTemperatureLimiter()
        assert m.limit_temperature(1000.0, 0.1) == 1000.0
        assert m.limit_temperature(3000.0, 0.1) == 2000.0


class TestOxidationEnhanced3:
    def test_kinetic_diffusion_cold(self):
        from pyfoam.lagrangian.oxidation_enhanced_3 import KineticDiffusionOxidation
        m = KineticDiffusionOxidation()
        assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0) < 1e-20

    def test_diffusion_limited_no_o2(self):
        from pyfoam.lagrangian.oxidation_enhanced_3 import DiffusionLimitedOxidation
        m = DiffusionLimitedOxidation()
        assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=1500.0, oxygen_mass_fraction=0.0) == 0.0


class TestOxidationEnhanced4:
    def test_liquid_evap_ox_cold(self):
        from pyfoam.lagrangian.oxidation_enhanced_4 import LiquidEvaporationOxidation
        m = LiquidEvaporationOxidation()
        assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0) < 1e-20

    def test_surface_reaction_cold(self):
        from pyfoam.lagrangian.oxidation_enhanced_4 import SurfaceReaction
        m = SurfaceReaction()
        assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0) < 1e-20


class TestOxidationEnhanced5:
    def test_random_pore_cold(self):
        from pyfoam.lagrangian.oxidation_enhanced_5 import RandomPoreModel
        m = RandomPoreModel()
        assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0) < 1e-20

    def test_shrinking_core_cold(self):
        from pyfoam.lagrangian.oxidation_enhanced_5 import ShrinkingCoreModel
        m = ShrinkingCoreModel()
        assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0) < 1e-20


class TestSprayEnhanced3:
    def test_reitz_khrt_no_velocity(self):
        from pyfoam.lagrangian.spray_enhanced_3 import ReitzKHRTBreakup
        m = ReitzKHRTBreakup()
        r = m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)
        assert r["atomized"] is False

    def test_ssd_atomization_no_velocity(self):
        from pyfoam.lagrangian.spray_enhanced_3 import SSDAtomization
        m = SSDAtomization(seed=42)
        r = m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)
        assert r["atomized"] is False


class TestSprayEnhanced4:
    def test_blob_nozzle_no_velocity(self):
        from pyfoam.lagrangian.spray_enhanced_4 import BlobAtomizationNozzle
        m = BlobAtomizationNozzle()
        r = m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)
        assert r["atomized"] is False

    def test_spray_post_processing(self):
        from pyfoam.lagrangian.spray_enhanced_4 import SprayPostProcessing
        m = SprayPostProcessing()
        r = m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=10.0)
        assert r["atomized"] is False
        assert m.smd >= 0


class TestSprayEnhanced5:
    def test_nozzle_flow_no_velocity(self):
        from pyfoam.lagrangian.spray_enhanced_5 import NozzleFlowModel
        m = NozzleFlowModel()
        r = m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)
        assert r["atomized"] is False

    def test_sheet_atomization_no_velocity(self):
        from pyfoam.lagrangian.spray_enhanced_5 import SheetAtomization
        m = SheetAtomization()
        r = m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)
        assert r["atomized"] is False


class TestWallEnhanced3:
    def test_rebound_no_approach(self):
        from pyfoam.lagrangian.wall_enhanced_3 import ReboundModel
        m = ReboundModel()
        r = m.interact(velocity=[0.1, 0, 0], wall_normal=[0, 1, 0])
        assert r["stuck"] is False

    def test_stochastic_splash_approach(self):
        from pyfoam.lagrangian.wall_enhanced_3 import StochasticSplash
        m = StochasticSplash(seed=42)
        r = m.interact(velocity=[0, -0.01, 0], wall_normal=[0, 1, 0])
        # 低速时 We < we_splash，粒子反弹（不是粘附）
        assert r["stuck"] is False
        assert r["splashed"] is False


class TestWallEnhanced4:
    def test_thermal_wall(self):
        from pyfoam.lagrangian.wall_enhanced_4 import ThermalWallInteraction
        m = ThermalWallInteraction()
        r = m.interact(velocity=[0.1, 0, 0], wall_normal=[0, 1, 0])
        assert "heat_transferred" in r

    def test_wet_wall_approach(self):
        from pyfoam.lagrangian.wall_enhanced_4 import WetWallInteraction
        m = WetWallInteraction()
        r = m.interact(velocity=[0, -0.01, 0], wall_normal=[0, 1, 0])
        assert r.get("absorbed") is True


class TestWallEnhanced5:
    def test_bounce_distribution_no_approach(self):
        from pyfoam.lagrangian.wall_enhanced_5 import WallBounceDistribution
        m = WallBounceDistribution(seed=42)
        r = m.interact(velocity=[0.1, 0, 0], wall_normal=[0, 1, 0])
        assert r["stuck"] is False

    def test_critical_velocity_stick(self):
        from pyfoam.lagrangian.wall_enhanced_5 import CriticalVelocityModel
        m = CriticalVelocityModel(v_crit=1.0)
        r = m.interact(velocity=[0, -0.01, 0], wall_normal=[0, 1, 0])
        assert r["stuck"] is True

    def test_critical_velocity_bounce(self):
        from pyfoam.lagrangian.wall_enhanced_5 import CriticalVelocityModel
        m = CriticalVelocityModel(v_crit=0.001, restitution=0.8)
        r = m.interact(velocity=[0, -10, 0], wall_normal=[0, 1, 0])
        assert r["stuck"] is False


class TestReactingEnhanced3:
    def test_multiphase_cold(self):
        from pyfoam.lagrangian.reacting_enhanced_3 import ReactingMultiphaseCloud
        m = ReactingMultiphaseCloud()
        r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
        assert r["mass_loss"] < 1e-20

    def test_two_phase_cold(self):
        from pyfoam.lagrangian.reacting_enhanced_3 import TwoPhaseReacting
        m = TwoPhaseReacting()
        r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
        assert r["mass_loss"] < 1e-20


class TestReactingEnhanced4:
    def test_heterogeneous_cold(self):
        from pyfoam.lagrangian.reacting_enhanced_4 import HeterogeneousReacting
        m = HeterogeneousReacting()
        r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
        assert r["mass_loss"] < 1e-20

    def test_catalytic_cold(self):
        from pyfoam.lagrangian.reacting_enhanced_4 import CatalyticReacting
        m = CatalyticReacting()
        r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
        assert r["mass_loss"] < 1e-20


class TestReactingEnhanced5:
    def test_devolatilisation_cold(self):
        from pyfoam.lagrangian.reacting_enhanced_5 import DevolatilisationModel
        m = DevolatilisationModel()
        r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
        assert r["mass_loss"] < 1e-20

    def test_char_burnout_cold(self):
        from pyfoam.lagrangian.reacting_enhanced_5 import CharBurnoutModel
        m = CharBurnoutModel()
        r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
        assert r["mass_loss"] < 1e-20

    def test_char_burnout_hot(self):
        from pyfoam.lagrangian.reacting_enhanced_5 import CharBurnoutModel
        m = CharBurnoutModel()
        r = m.react(dt=1e-4, diameter=1e-3, temperature=1500.0, fluid_temperature=2000.0, species_mass_fraction=0.23)
        assert r["mass_loss"] > 0.0
        assert r["heat_release"] > 0.0
        assert r["diameter"] < 1e-3
