"""Unit tests for enhanced Lagrangian models v6-v10."""
from __future__ import annotations
import pytest

# Injection v6-v10
def test_injection_from_file():
    from pyfoam.lagrangian.injection_enhanced_6 import InjectionFromFile
    m = InjectionFromFile(data_table=[{'position': [0,0,0], 'velocity': [1,0,0]}])
    assert len(m.inject()) == 1

def test_resettable_injector():
    from pyfoam.lagrangian.injection_enhanced_6 import ResettableInjector
    from pyfoam.lagrangian.injection import PointInjector
    inner = PointInjector(origin=[0,0,0], velocity=[1,0,0], n_particles=3)
    m = ResettableInjector(inner=inner)
    assert len(m.inject()) == 3
    m.reset()
    assert m._injection_count == 0

def test_surface_flux_injector():
    from pyfoam.lagrangian.injection_enhanced_7 import SurfaceFluxInjector
    m = SurfaceFluxInjector(n_particles=5)
    assert len(m.inject()) == 5

def test_volume_source_injector():
    from pyfoam.lagrangian.injection_enhanced_7 import VolumeSourceInjector
    m = VolumeSourceInjector(n_particles=5, seed=42)
    assert len(m.inject()) == 5

def test_rate_controlled_injector():
    from pyfoam.lagrangian.injection_enhanced_8 import RateControlledInjector
    m = RateControlledInjector(n_particles=5)
    ps = m.inject()
    assert len(ps) == 5
    assert hasattr(ps[0], 'parcel_mass')

def test_distribution_injector():
    from pyfoam.lagrangian.injection_enhanced_8 import DistributionInjector
    m = DistributionInjector(n_particles=5, seed=42)
    assert len(m.inject()) == 5

def test_temporal_injector():
    from pyfoam.lagrangian.injection_enhanced_9 import TemporalInjector
    from pyfoam.lagrangian.injection import PointInjector
    inner = PointInjector(origin=[0,0,0], velocity=[0,0,0], n_particles=2)
    m = TemporalInjector(inner=inner)
    assert len(m.inject()) == 2

def test_probabilistic_injector():
    from pyfoam.lagrangian.injection_enhanced_9 import ProbabilisticInjector
    from pyfoam.lagrangian.injection import PointInjector
    inner = PointInjector(origin=[0,0,0], velocity=[0,0,0], n_particles=2)
    m = ProbabilisticInjector(inner=inner, activation_probability=1.0, seed=42)
    assert len(m.inject()) == 2

def test_multi_point_injector():
    from pyfoam.lagrangian.injection_enhanced_10 import MultiPointInjector
    m = MultiPointInjector(origins=[[0,0,0],[1,0,0]], n_per_point=3)
    assert len(m.inject()) == 6

def test_adaptive_injector():
    from pyfoam.lagrangian.injection_enhanced_10 import AdaptiveInjector
    m = AdaptiveInjector(n_particles=5, size_factor=2.0)
    ps = m.inject()
    assert len(ps) == 5
    assert ps[0].diameter == 2e-4

# Forces v6-v10
def test_charged_particle_force():
    from pyfoam.lagrangian.forces_enhanced_6 import ChargedParticleForce
    f = ChargedParticleForce()
    a = f.acceleration([0,0,0], 1e-4, 1000.0)
    assert a == [0.0, 0.0, 0.0]

def test_basset_force():
    from pyfoam.lagrangian.forces_enhanced_6 import BassetForce
    f = BassetForce()
    assert f.acceleration([0,0,0], 1e-4, 1000.0) == [0.0, 0.0, 0.0]

def test_faxen_force():
    from pyfoam.lagrangian.forces_enhanced_7 import FaxenForce
    f = FaxenForce()
    assert f.acceleration([0,0,0], 1e-4, 1000.0) == [0.0, 0.0, 0.0]

def test_history_force():
    from pyfoam.lagrangian.forces_enhanced_7 import HistoryForce
    f = HistoryForce()
    assert f.acceleration([0,0,0], 1e-4, 1000.0) == [0.0, 0.0, 0.0]

def test_oseen_drag_force():
    from pyfoam.lagrangian.forces_enhanced_8 import OseenDragForce
    f = OseenDragForce()
    assert f.acceleration([0,0,0], 1e-4, 1000.0) == [0.0, 0.0, 0.0]

def test_coriolis_force():
    from pyfoam.lagrangian.forces_enhanced_9 import CoriolisForce
    f = CoriolisForce()
    assert f.acceleration([0,0,0], 1e-4, 1000.0) == [0.0, 0.0, 0.0]

def test_electrostatic_force():
    from pyfoam.lagrangian.forces_enhanced_10 import ElectrostaticForce
    f = ElectrostaticForce()
    assert f.acceleration([0,0,0], 1e-4, 1000.0) == [0.0, 0.0, 0.0]

# Breakup v6-v10
def test_lisa_breakup():
    from pyfoam.lagrangian.breakup_enhanced_6 import LISABreakup
    m = LISABreakup()
    r = m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=0.0)
    assert r['broken'] is False

def test_wave_breakup():
    from pyfoam.lagrangian.breakup_enhanced_7 import WAVEBreakup
    m = WAVEBreakup()
    assert m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=0.0)['broken'] is False

def test_cascade_breakup():
    from pyfoam.lagrangian.breakup_enhanced_8 import CascadeBreakup
    m = CascadeBreakup()
    assert m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=0.0)['broken'] is False

def test_fractal_breakup():
    from pyfoam.lagrangian.breakup_enhanced_9 import FractalBreakup
    m = FractalBreakup()
    assert m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=0.0)['broken'] is False

def test_frag_breakup():
    from pyfoam.lagrangian.breakup_enhanced_10 import FragBreakup
    m = FragBreakup()
    assert m.breakup(dt=1e-4, diameter=1e-4, relative_velocity=0.0)['broken'] is False

# Collision v6-v10
def test_deterministic_collision():
    from pyfoam.lagrangian.collision_enhanced_6 import DeterministicCollision
    m = DeterministicCollision()
    v1, v2 = m.collide([0,0,0], [1,0,0], 1e-4, 1000.0, [10,0,0], [-1,0,0], 1e-4, 1000.0)
    assert v1 == [1,0,0]

def test_adaptive_collision():
    from pyfoam.lagrangian.collision_enhanced_7 import AdaptiveCollision
    m = AdaptiveCollision()
    v1, v2 = m.collide([0,0,0], [1,0,0], 1e-4, 1000.0, [10,0,0], [-1,0,0], 1e-4, 1000.0)
    assert v1 == [1,0,0]

def test_dem_collision():
    from pyfoam.lagrangian.collision_enhanced_10 import DEMCollision
    m = DEMCollision()
    v1, v2 = m.collide([0,0,0], [1,0,0], 1e-4, 1000.0, [10,0,0], [-1,0,0], 1e-4, 1000.0)
    assert v1 == [1,0,0]

# Dispersion v6-v10
def test_langevin_dispersion():
    from pyfoam.lagrangian.dispersion_enhanced_6 import LangevinDispersion
    m = LangevinDispersion()
    assert m.disperse(dt=1e-4) == [0.0, 0.0, 0.0]

def test_tensor_dispersion():
    from pyfoam.lagrangian.dispersion_enhanced_7 import TensorDispersion
    m = TensorDispersion()
    assert m.disperse(dt=1e-4) == [0.0, 0.0, 0.0]

def test_eddy_interaction_dispersion():
    from pyfoam.lagrangian.dispersion_enhanced_8 import EddyInteractionDispersion
    m = EddyInteractionDispersion()
    assert m.disperse(dt=1e-4) == [0.0, 0.0, 0.0]

def test_diffusion_dispersion():
    from pyfoam.lagrangian.dispersion_enhanced_9 import DiffusionDispersion
    m = DiffusionDispersion()
    assert m.disperse(dt=1e-4) == [0.0, 0.0, 0.0]

def test_filtered_dns_dispersion():
    from pyfoam.lagrangian.dispersion_enhanced_10 import FilteredDNSDispersion
    m = FilteredDNSDispersion()
    assert m.disperse(dt=1e-4) == [0.0, 0.0, 0.0]

# Evaporation v6-v10
def test_skin_evaporation():
    from pyfoam.lagrangian.evaporation_enhanced_6 import SkinEvaporation
    m = SkinEvaporation()
    assert m.evaporate(dt=1e-4, diameter=1e-3, temperature=400, fluid_temperature=300) == 0.0

def test_shell_evaporation():
    from pyfoam.lagrangian.evaporation_enhanced_7 import ShellEvaporation
    m = ShellEvaporation()
    assert m.evaporate(dt=1e-4, diameter=1e-3, temperature=400, fluid_temperature=300) == 0.0

def test_fuel_evaporation():
    from pyfoam.lagrangian.evaporation_enhanced_8 import FuelEvaporation
    m = FuelEvaporation()
    assert m.evaporate(dt=1e-4, diameter=1e-3, temperature=400, fluid_temperature=300) == 0.0

def test_flash_evaporation():
    from pyfoam.lagrangian.evaporation_enhanced_9 import FlashEvaporation
    m = FlashEvaporation()
    assert m.evaporate(dt=1e-4, diameter=1e-3, temperature=400, fluid_temperature=300) == 0.0

def test_equilibrium_evaporation():
    from pyfoam.lagrangian.evaporation_enhanced_10 import EquilibriumEvaporation
    m = EquilibriumEvaporation()
    assert m.evaporate(dt=1e-4, diameter=1e-3, temperature=400, fluid_temperature=300) == 0.0

# MPPIC v6-v10
def test_johnson_jackson_friction():
    from pyfoam.lagrangian.mppic_models_enhanced_6 import JohnsonJacksonFriction
    m = JohnsonJacksonFriction()
    assert m.packing_stress(alpha=0.0) == 0.0

def test_lun_savage_friction():
    from pyfoam.lagrangian.mppic_models_enhanced_7 import LunSavageFriction
    m = LunSavageFriction()
    assert m.packing_stress(alpha=0.0) == 0.0

def test_implicit_damping():
    from pyfoam.lagrangian.mppic_models_enhanced_8 import ImplicitDamping
    m = ImplicitDamping()
    assert m.packing_stress(alpha=0.0) == 0.0

def test_max_velocity_limiter():
    from pyfoam.lagrangian.mppic_models_enhanced_9 import MaximumVelocityLimiter
    m = MaximumVelocityLimiter()
    assert m.packing_stress(alpha=0.0) == 0.0

def test_packing_limiter():
    from pyfoam.lagrangian.mppic_models_enhanced_10 import PackingLimiter
    m = PackingLimiter()
    assert m.packing_stress(alpha=0.0) == 0.0

# Oxidation v6-v10
def test_langmuir_hinshelwood_oxidation():
    from pyfoam.lagrangian.oxidation_enhanced_6 import LangmuirHinshelwoodOxidation
    m = LangmuirHinshelwoodOxidation()
    assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0) < 1e-20

def test_power_law_oxidation():
    from pyfoam.lagrangian.oxidation_enhanced_7 import PowerLawOxidation
    m = PowerLawOxidation()
    assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0) < 1e-20

def test_bidisperse_oxidation():
    from pyfoam.lagrangian.oxidation_enhanced_8 import BidisperseOxidation
    m = BidisperseOxidation()
    assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0) < 1e-20

def test_grain_model():
    from pyfoam.lagrangian.oxidation_enhanced_9 import GrainModel
    m = GrainModel()
    assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0) < 1e-20

def test_kinetic_diffusion_v2():
    from pyfoam.lagrangian.oxidation_enhanced_10 import KineticDiffusionV2
    m = KineticDiffusionV2()
    assert m.oxidise(dt=1e-4, diameter=1e-3, temperature=100.0) < 1e-20

# Spray v6-v10
def test_wave_atomization():
    from pyfoam.lagrangian.spray_models_enhanced_6 import WaveAtomization
    m = WaveAtomization()
    assert m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)['atomized'] is False

def test_blobsheet_atomization():
    from pyfoam.lagrangian.spray_models_enhanced_7 import BlobsheetAtomization
    m = BlobsheetAtomization()
    assert m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)['atomized'] is False

def test_cascade_atomization():
    from pyfoam.lagrangian.spray_models_enhanced_8 import CascadeAtomization
    m = CascadeAtomization()
    assert m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)['atomized'] is False

def test_rt_atomization():
    from pyfoam.lagrangian.spray_models_enhanced_9 import RTAtomization
    m = RTAtomization()
    assert m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)['atomized'] is False

def test_hybrid_atomization():
    from pyfoam.lagrangian.spray_models_enhanced_10 import HybridAtomization
    m = HybridAtomization()
    assert m.atomize(dt=1e-5, diameter=1e-3, relative_velocity=0.0)['atomized'] is False

# Wall v6-v10
def test_momentum_transfer_wall():
    from pyfoam.lagrangian.wall_interaction_enhanced_6 import MomentumTransferWall
    m = MomentumTransferWall()
    r = m.interact(velocity=[0.1, 0, 0], wall_normal=[0, 1, 0])
    assert r['stuck'] is False

def test_bounce_friction_wall():
    from pyfoam.lagrangian.wall_interaction_enhanced_7 import BounceFrictionWall
    m = BounceFrictionWall()
    r = m.interact(velocity=[0.1, 0, 0], wall_normal=[0, 1, 0])
    assert r['stuck'] is False

def test_splash_fragment_wall():
    from pyfoam.lagrangian.wall_interaction_enhanced_8 import SplashFragmentWall
    m = SplashFragmentWall()
    r = m.interact(velocity=[0.1, 0, 0], wall_normal=[0, 1, 0])
    assert r['stuck'] is False

def test_temp_dependent_wall():
    from pyfoam.lagrangian.wall_interaction_enhanced_9 import TemperatureDependentWall
    m = TemperatureDependentWall()
    r = m.interact(velocity=[0.1, 0, 0], wall_normal=[0, 1, 0])
    assert r['stuck'] is False

def test_probabilistic_wall():
    from pyfoam.lagrangian.wall_interaction_enhanced_10 import ProbabilisticWall
    m = ProbabilisticWall()
    r = m.interact(velocity=[0.1, 0, 0], wall_normal=[0, 1, 0])
    assert r['stuck'] is False

# Reacting v6-v10
def test_kinetic_reacting():
    from pyfoam.lagrangian.reacting_models_enhanced_6 import KineticReacting
    m = KineticReacting()
    r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
    assert r['mass_loss'] < 1e-20

def test_arrhenius_reacting():
    from pyfoam.lagrangian.reacting_models_enhanced_7 import ArrheniusReacting
    m = ArrheniusReacting()
    r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
    assert r['mass_loss'] < 1e-20

def test_shrinking_core_reacting():
    from pyfoam.lagrangian.reacting_models_enhanced_8 import ShrinkingCoreReacting
    m = ShrinkingCoreReacting()
    r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
    assert r['mass_loss'] < 1e-20

def test_global_reaction_model():
    from pyfoam.lagrangian.reacting_models_enhanced_9 import GlobalReactionModel
    m = GlobalReactionModel()
    r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
    assert r['mass_loss'] < 1e-20

def test_char_gasification_model():
    from pyfoam.lagrangian.reacting_models_enhanced_10 import CharGasificationModel
    m = CharGasificationModel()
    r = m.react(dt=1e-4, diameter=1e-3, temperature=100.0, fluid_temperature=200.0)
    assert r['mass_loss'] < 1e-20

# Hot temperature tests for Arrhenius models
def test_oxidation_hot():
    from pyfoam.lagrangian.oxidation_enhanced_6 import LangmuirHinshelwoodOxidation
    m = LangmuirHinshelwoodOxidation()
    dm = m.oxidise(dt=1e-4, diameter=1e-3, temperature=1500.0, oxygen_mass_fraction=0.23)
    assert dm > 0

def test_reacting_hot():
    from pyfoam.lagrangian.reacting_models_enhanced_6 import KineticReacting
    m = KineticReacting()
    r = m.react(dt=1e-4, diameter=1e-3, temperature=1500.0, fluid_temperature=2000.0, species_mass_fraction=0.8)
    assert r['mass_loss'] > 0
    assert r['heat_release'] > 0
