"""
pyfoam.postprocessing — Post-processing tools and function objects.

Provides:

- :class:`FunctionObject` — base class for all function objects
- :class:`Forces` — force and moment calculation on patches
- :class:`ForceCoeffs` — force coefficient calculation (Cd, Cl, Cm)
- :class:`WallShearStress` — wall shear stress computation
- :class:`YPlus` — y+ calculation for wall-bounded flows
- :class:`FieldOperations` — grad, div, curl field operations
- :class:`Probes` — point probe sampling
- :class:`LineSample` — line sampling (sets)
- :class:`SurfaceSample` — surface sampling
- :class:`VTKWriter` — VTK file output
- :class:`FoamToVTK` — case-level VTK conversion
- :class:`Vorticity` — vorticity (∇ × U) computation
- :class:`QCriterion` — Q-criterion vortex identification
- :class:`Lambda2` — λ₂ criterion vortex identification
- :class:`Enstrophy` — enstrophy (0.5 * |ω|²) computation
- :class:`TurbulentKineticEnergy` — TKE computation (resolved/RANS)
- :class:`FieldAverageEnhanced` — enhanced time-weighted averaging with Reynolds decomposition
- :class:`ProbesEnhanced` — enhanced probes with time interpolation and spectral analysis

All function objects follow OpenFOAM's function object API and can be
configured via dictionary entries in ``system/controlDict``.
"""

from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry
from pyfoam.postprocessing.forces import Forces, ForceCoeffs
from pyfoam.postprocessing.wall_shear_stress import WallShearStress
from pyfoam.postprocessing.y_plus import YPlus
from pyfoam.postprocessing.field_operations import FieldOperations
from pyfoam.postprocessing.sampling import Probes, LineSample, SurfaceSample
from pyfoam.postprocessing.vtk_output import VTKWriter, FoamToVTK
from pyfoam.postprocessing.noise import Noise
from pyfoam.postprocessing.temporal_interpolate import TemporalInterpolate
from pyfoam.postprocessing.particle_tracks import ParticleTracks
from pyfoam.postprocessing.field_min_max import FieldMinMax, MinMaxResult
from pyfoam.postprocessing.field_average import FieldAverage
from pyfoam.postprocessing.y_plus_enhanced import YPlusEnhanced, WallTreatment, YPatchStats
from pyfoam.postprocessing.noise_enhanced import NoiseEnhanced, ThirdOctaveBand
from pyfoam.postprocessing.particle_tracks_enhanced import ParticleTracksEnhanced, TrackStatistics
from pyfoam.postprocessing.field_min_max_enhanced import FieldMinMaxEnhanced, EnhancedMinMaxResult
from pyfoam.postprocessing.vorticity import Vorticity
from pyfoam.postprocessing.q_criterion import QCriterion
from pyfoam.postprocessing.lambda2 import Lambda2
from pyfoam.postprocessing.enstrophy import Enstrophy
from pyfoam.postprocessing.turbulent_kinetic_energy import TurbulentKineticEnergy
from pyfoam.postprocessing.field_average_enhanced import FieldAverageEnhanced
from pyfoam.postprocessing.probes_enhanced import ProbesEnhanced, SpectrumResult

__all__ = [
    # Framework
    "FunctionObject",
    "FunctionObjectRegistry",
    # Forces
    "Forces",
    "ForceCoeffs",
    # Wall quantities
    "WallShearStress",
    "YPlus",
    # Field operations
    "FieldOperations",
    # Sampling
    "Probes",
    "LineSample",
    "SurfaceSample",
    # VTK output
    "VTKWriter",
    "FoamToVTK",
    # Acoustic analysis
    "Noise",
    # Temporal interpolation
    "TemporalInterpolate",
    # Particle tracking
    "ParticleTracks",
    # Field min/max
    "FieldMinMax",
    "MinMaxResult",
    # Field averaging
    "FieldAverage",
    # Enhanced y+
    "YPlusEnhanced",
    "WallTreatment",
    "YPatchStats",
    # Enhanced noise analysis
    "NoiseEnhanced",
    "ThirdOctaveBand",
    # Enhanced particle tracking
    "ParticleTracksEnhanced",
    "TrackStatistics",
    # Enhanced field min/max
    "FieldMinMaxEnhanced",
    "EnhancedMinMaxResult",
    # Vortex identification
    "Vorticity",
    "QCriterion",
    "Lambda2",
    # Turbulence quantities
    "Enstrophy",
    "TurbulentKineticEnergy",
    # Enhanced field averaging
    "FieldAverageEnhanced",
    # Enhanced probes
    "ProbesEnhanced",
    "SpectrumResult",
]
