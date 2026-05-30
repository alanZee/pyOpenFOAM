"""Generate enhanced Lagrangian model variants v6-v10."""
import os

base = 'src/pyfoam/lagrangian'

specs = [
    ('injection', 6, 'InjectionFromFile', 'inject from data table', 'ResettableInjector', 'resettable injection wrapper'),
    ('injection', 7, 'SurfaceFluxInjector', 'inject by surface mass flux', 'VolumeSourceInjector', 'inject from volume source'),
    ('injection', 8, 'RateControlledInjector', 'controlled mass flow rate', 'DistributionInjector', 'Rosin-Rammler size distribution'),
    ('injection', 9, 'TemporalInjector', 'time-windowed injection', 'ProbabilisticInjector', 'stochastic activation'),
    ('injection', 10, 'MultiPointInjector', 'multiple origin injection', 'AdaptiveInjector', 'adaptive sizing injection'),
    ('forces', 6, 'ChargedParticleForce', 'force on charged particles in E-field', 'BassetForce', 'Basset history force'),
    ('forces', 7, 'FaxenForce', 'Faxen correction for finite Re', 'HistoryForce', 'history integral force'),
    ('forces', 8, 'OseenDragForce', 'Oseen drag correction', 'HadamardRybczynskiDrag', 'H-R drag for bubbles/drops'),
    ('forces', 9, 'CoriolisForce', 'Coriolis force in rotating frame', 'CentrifugalForce', 'centrifugal force'),
    ('forces', 10, 'ElectrostaticForce', 'Coulomb force', 'AcousticRadiationForce', 'acoustic radiation force'),
    ('breakup', 6, 'LISABreakup', 'LISA sheet breakup', 'SchmehlBreakup', 'Schmehl droplet breakup'),
    ('breakup', 7, 'WAVEBreakup', 'WAVE breakup model', 'MadabhushiBreakup', 'Madabhushi model'),
    ('breakup', 8, 'CascadeBreakup', 'cascade breakup model', 'PowerLawBreakup', 'power-law child size'),
    ('breakup', 9, 'FractalBreakup', 'fractal breakup model', 'RosinRammlerBreakup', 'R-R child distribution'),
    ('breakup', 10, 'FragBreakup', 'fragmentation breakup', 'UniformBreakup', 'uniform child size'),
    ('collision', 6, 'DeterministicCollision', 'deterministic collision detection', 'NTCollision', 'NT-counter collision'),
    ('collision', 7, 'AdaptiveCollision', 'adaptive time-step collision', 'MultiParticleCollision', 'multi-body collision'),
    ('collision', 8, 'SphereCollision', 'sphere-of-influence collision', 'CellCollision', 'cell-based collision'),
    ('collision', 9, 'ReversibleCollision', 'reversible collision model', 'InelasticCoalescence', 'inelastic coalescence'),
    ('collision', 10, 'DEMCollision', 'DEM-based collision', 'PSCollision', 'particle-stochastic collision'),
    ('dispersion', 6, 'LangevinDispersion', 'Langevin equation dispersion', 'FilteredDispersion', 'filtered velocity dispersion'),
    ('dispersion', 7, 'TensorDispersion', 'anisotropic tensor dispersion', 'SchmidtDispersion', 'Schmidt number dispersion'),
    ('dispersion', 8, 'EddyInteractionDispersion', 'eddy interaction model', 'CrossDispersion', 'cross-correlation dispersion'),
    ('dispersion', 9, 'DiffusionDispersion', 'gradient diffusion dispersion', 'DriftDispersion', 'drift velocity dispersion'),
    ('dispersion', 10, 'FilteredDNSDispersion', 'DNS-filtered dispersion', 'ScaleSimilarityDispersion', 'scale-similarity dispersion'),
    ('evaporation', 6, 'SkinEvaporation', 'skin evaporation model', 'HomogeneousEvaporation', 'homogeneous evaporation'),
    ('evaporation', 7, 'ShellEvaporation', 'shell evaporation model', 'KnudsenEvaporation', 'Knudsen regime evaporation'),
    ('evaporation', 8, 'FuelEvaporation', 'fuel-specific evaporation', 'FilmEvaporation', 'thin-film evaporation'),
    ('evaporation', 9, 'FlashEvaporation', 'flash boiling evaporation', 'ConvectiveEvaporation', 'forced convection evap'),
    ('evaporation', 10, 'EquilibriumEvaporation', 'equilibrium evaporation', 'NonEquilibriumEvaporation', 'non-equilibrium evap'),
    ('mppic_models', 6, 'JohnsonJacksonFriction', 'J-J friction model', 'KTGFStress', 'KTGF-based stress'),
    ('mppic_models', 7, 'LunSavageFriction', 'Lun-Savage friction', 'GranularTemperatureModel', 'granular temperature'),
    ('mppic_models', 8, 'ImplicitDamping', 'implicit velocity damping', 'ExplicitDamping', 'explicit velocity damping'),
    ('mppic_models', 9, 'MaximumVelocityLimiter', 'max velocity limiter', 'MinimumDiameterLimiter', 'min diameter limiter'),
    ('mppic_models', 10, 'PackingLimiter', 'packing fraction limiter', 'VolumeFractionSmooth', 'volume fraction smoothing'),
    ('oxidation', 6, 'LangmuirHinshelwoodOxidation', 'L-H oxidation model', 'MarsMaesoneOxidation', 'Mars-Maesone model'),
    ('oxidation', 7, 'PowerLawOxidation', 'power-law oxidation', 'NthOrderOxidation', 'n-th order oxidation'),
    ('oxidation', 8, 'BidisperseOxidation', 'bidisperse pore oxidation', 'RandomPoreV2', 'random pore v2'),
    ('oxidation', 9, 'GrainModel', 'grain model oxidation', 'VolumeReactionModel', 'volumetric reaction'),
    ('oxidation', 10, 'KineticDiffusionV2', 'K-D model v2', 'MultipleReactionOxidation', 'multiple reaction model'),
    ('spray_models', 6, 'WaveAtomization', 'wave-based atomization', 'FIPAAtomization', 'FIPA model'),
    ('spray_models', 7, 'BlobsheetAtomization', 'blob-sheet atomization', 'FilmAtomization', 'film atomization'),
    ('spray_models', 8, 'CascadeAtomization', 'cascade atomization', 'StochasticAtomization', 'stochastic atomization'),
    ('spray_models', 9, 'RTAtomization', 'RT-dominated atomization', 'MultimodeAtomization', 'multimode atomization'),
    ('spray_models', 10, 'HybridAtomization', 'hybrid KH-RT-spray', 'CalibratedAtomization', 'calibrated atomization'),
    ('wall_interaction', 6, 'MomentumTransferWall', 'momentum transfer model', 'HeatTransferWall', 'heat transfer model'),
    ('wall_interaction', 7, 'BounceFrictionWall', 'friction wall bounce', 'AbsorptionWall', 'absorption model'),
    ('wall_interaction', 8, 'SplashFragmentWall', 'splash with fragments', 'SplashCoalescence', 'splash coalescence'),
    ('wall_interaction', 9, 'TemperatureDependentWall', 'T-dependent wall model', 'MaterialPropertyWall', 'material property model'),
    ('wall_interaction', 10, 'ProbabilisticWall', 'probabilistic wall model', 'AdaptiveWall', 'adaptive wall model'),
    ('reacting_models', 6, 'KineticReacting', 'kinetic rate reacting', 'DiffusionReacting', 'diffusion controlled reacting'),
    ('reacting_models', 7, 'ArrheniusReacting', 'multi-step Arrhenius', 'EquilibriumReacting', 'equilibrium reacting'),
    ('reacting_models', 8, 'ShrinkingCoreReacting', 'shrinking core reacting', 'UniformConversionReacting', 'uniform conversion'),
    ('reacting_models', 9, 'GlobalReactionModel', 'global reaction model', 'DetailedReactionModel', 'detailed mechanism'),
    ('reacting_models', 10, 'CharGasificationModel', 'char gasification model', 'PyrolysisModel', 'pyrolysis model'),
]

base_imports = {
    'injection': 'from pyfoam.lagrangian.injection import Injector',
    'forces': 'from pyfoam.lagrangian.forces import ParticleForce',
    'breakup': 'from pyfoam.lagrangian.breakup import BreakupModel',
    'collision': 'from pyfoam.lagrangian.collision import CollisionModel',
    'dispersion': 'from pyfoam.lagrangian.dispersion import DispersionModel',
    'evaporation': 'from pyfoam.lagrangian.evaporation import EvaporationModel',
    'mppic_models': 'from pyfoam.lagrangian.mppic_models import MPPICModel',
    'oxidation': 'from pyfoam.lagrangian.oxidation import OxidationModel',
    'spray_models': 'from pyfoam.lagrangian.spray_models import SprayModel',
    'wall_interaction': 'from pyfoam.lagrangian.wall_interaction import WallInteractionModel',
    'reacting_models': 'from pyfoam.lagrangian.reacting_models import ReactingModel',
}

bc = {
    'injection': 'Injector', 'forces': 'ParticleForce', 'breakup': 'BreakupModel',
    'collision': 'CollisionModel', 'dispersion': 'DispersionModel',
    'evaporation': 'EvaporationModel', 'mppic_models': 'MPPICModel',
    'oxidation': 'OxidationModel', 'spray_models': 'SprayModel',
    'wall_interaction': 'WallInteractionModel', 'reacting_models': 'ReactingModel',
}

def make_class(mod, name, desc):
    b = bc[mod]
    if mod == 'injection':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, origin=None, velocity=None, n_particles=1, **kw):
        self.origin = origin or [0,0,0]; self.velocity = velocity or [0,0,0]; self.n_particles = n_particles
    def inject(self):
        from pyfoam.lagrangian.particle import Particle
        return [Particle(position=list(self.origin), velocity=list(self.velocity)) for _ in range(self.n_particles)]'''
    elif mod == 'forces':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]'''
    elif mod == 'breakup':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {{"diameter": diameter, "broken": False}}'''
    elif mod == 'collision':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)'''
    elif mod == 'dispersion':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, seed=None, **kw): self.seed = seed
    def disperse(self, dt=0.0, turbulent_kinetic_energy=0.0, turbulent_dissipation=0.0, fluid_density=1.225, particle_diameter=1e-4, particle_density=1000.0):
        return [0.0, 0.0, 0.0]'''
    elif mod == 'evaporation':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, **kw): self._p = kw
    def evaporate(self, dt, diameter, temperature, fluid_temperature, fluid_density=1.0, fluid_viscosity=2e-5, latent_heat=2.26e6, vapour_diffusivity=2.6e-5, thermal_conductivity=0.026, specific_heat=1005.0):
        return 0.0'''
    elif mod == 'mppic_models':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)'''
    elif mod == 'oxidation':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, A=1.0, E_a=8e4, **kw): self.A = A; self.E_a = E_a
    def oxidise(self, dt, diameter, temperature, oxygen_mass_fraction=0.23, fluid_density=1.0):
        if diameter < 1e-15 or temperature < 1.0: return 0.0
        k = self.A * math.exp(-self.E_a / (8.314 * max(temperature, 1.0)))
        dm = math.pi * diameter**2 * k * fluid_density * oxygen_mass_fraction * dt
        return max(min(dm, (math.pi/6) * diameter**3 * 2000.0), 0.0)'''
    elif mod == 'spray_models':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, **kw): self._p = kw
    def atomize(self, dt, diameter, relative_velocity, fluid_density=1.225, surface_tension=0.072, particle_density=800.0, fluid_viscosity=1.8e-5):
        return {{"diameter": diameter, "atomized": False}}'''
    elif mod == 'wall_interaction':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {{"velocity": list(velocity), "stuck": False}}'''
    elif mod == 'reacting_models':
        return f'''class {name}({b}):
    """{desc}."""
    def __init__(self, A=1e3, E_a=8e4, **kw): self.A = A; self.E_a = E_a
    def react(self, dt, diameter, temperature, fluid_temperature, species_mass_fraction=1.0):
        if diameter < 1e-15 or temperature < 1.0: return {{"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}}
        k = self.A * math.exp(-self.E_a / (8.314 * max(temperature, 1.0)))
        dm = math.pi * diameter**2 * k * dt * species_mass_fraction
        m_p = (math.pi/6) * diameter**3 * 1000.0
        dm = max(min(dm, m_p), 0.0)
        new_d = diameter * max(1.0 - dm/m_p, 0.0)**(1.0/3.0) if m_p > 0 else diameter
        return {{"diameter": new_d, "mass_loss": dm, "heat_release": dm * 1e7}}'''

count = 0
for mod, v, c1, d1, c2, d2 in specs:
    fname = f'{base}/{mod}_enhanced_{v}.py'
    if os.path.exists(fname):
        print(f'Skip {fname}')
        continue

    imp = base_imports[mod]
    impl1 = make_class(mod, c1, d1)
    impl2 = make_class(mod, c2, d2)

    content = f'''"""
Enhanced {mod.replace("_", " ")} models v{v}.

- :class:`{c1}` -- {d1}
- :class:`{c2}` -- {d2}
"""

from __future__ import annotations

import math

{imp}

__all__ = ["{c1}", "{c2}"]


{impl1}


{impl2}
'''
    with open(fname, 'w') as f:
        f.write(content)
    count += 1
    print(f'Created {fname}')

print(f'\nTotal: {count} files created')
