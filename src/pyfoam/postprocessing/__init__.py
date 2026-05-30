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
- :class:`FieldMinMaxEnhanced2` — enhanced field min/max v2 with per-region stats
- :class:`ProbesEnhanced2` — enhanced probes v2 with multi-probe groups and cross-spectra
- :class:`ForcesEnhanced` — enhanced forces with full decomposition and fluctuation stats
- :class:`WallShearStressEnhanced` — enhanced wall shear stress with u_tau and Cf
- :class:`YPlusEnhanced2` — enhanced y+ v2 with wall laws and mesh quality
- :class:`FieldMinMaxEnhanced3` — enhanced field min/max v3 with percentiles and histograms
- :class:`ProbesEnhanced3` — enhanced probes v3 with per-probe spectra and coherence matrices
- :class:`ForcesEnhanced2` — enhanced forces v2 with projected forces and Cd/Cl
- :class:`WallShearStressEnhanced2` — enhanced wall shear stress v2 with non-orthogonal correction
- :class:`YPlusEnhanced3` — enhanced y+ v3 with adaptive wall law and regime classification
- :class:`FieldMinMaxEnhanced4` — enhanced field min/max v4 with per-region stats and time history
- :class:`ProbesEnhanced4` — enhanced probes v4 with multi-probe management and signal filtering
- :class:`ForcesEnhanced3` — enhanced forces v3 with moment coefficients and spectral analysis
- :class:`WallShearStressEnhanced3` — enhanced wall shear stress v3 with adaptive near-wall treatment
- :class:`YPlusEnhanced4` — enhanced y+ v4 with improved wall distance and dt suggestion
- :class:`FieldMinMaxEnhanced5` — enhanced field min/max v5 with anomaly detection and trend analysis
- :class:`ProbesEnhanced5` — enhanced probes v5 with wavelet analysis and signal quality
- :class:`ForcesEnhanced4` — enhanced forces v4 with aeroacoustic sources and unsteady stats
- :class:`WallShearStressEnhanced4` — enhanced wall shear stress v4 with quadrant analysis and roughness
- :class:`YPlusEnhanced5` — enhanced y+ v5 with AMR suggestions and budget analysis
- :class:`FieldMinMaxEnhanced6` — enhanced field min/max v6 with SPC and predictive monitoring
- :class:`ProbesEnhanced6` — enhanced probes v6 with POD and Lagrangian tracking
- :class:`ForcesEnhanced5` — enhanced forces v5 with FSI coupling and fatigue estimation
- :class:`WallShearStressEnhanced5` — enhanced wall shear stress v5 with anisotropy and coherent structures
- :class:`YPlusEnhanced6` — enhanced y+ v6 with wall heat transfer and adaptive wall function selection

- :class:`FieldMinMaxEnhanced7` — enhanced field min/max v7 with multivariate anomaly detection and adaptive thresholds
- :class:`ProbesEnhanced7` — enhanced probes v7 with compressed sensing recovery and ROM
- :class:`ForcesEnhanced6` — enhanced forces v6 with DMD mode decomposition and frequency tracking
- :class:`WallShearStressEnhanced6` — enhanced wall shear stress v6 with WMLES interface and pressure-strain coupling
- :class:`YPlusEnhanced7` — enhanced y+ v7 with uncertainty quantification and ensemble analysis

- :class:`FieldMinMaxEnhanced8` — enhanced field min/max v8 with temporal clustering and alert rules
- :class:`ProbesEnhanced8` — enhanced probes v8 with streaming data processing and health monitoring
- :class:`ForcesEnhanced7` — enhanced forces v7 with wavelet analysis and coefficient statistics
- :class:`WallShearStressEnhanced7` — enhanced wall shear stress v7 with drag decomposition and wall turbulence stats
- :class:`YPlusEnhanced8` — enhanced y+ v8 with spectral analysis and mesh adaptation criteria

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

# Phase 11: Enhanced postprocessing models
from pyfoam.postprocessing.field_min_max_enhanced_2 import (
    FieldMinMaxEnhanced2,
    RegionMinMaxResult,
)
from pyfoam.postprocessing.probes_enhanced_2 import (
    ProbesEnhanced2,
    CrossSpectrumResult,
)
from pyfoam.postprocessing.forces_enhanced import (
    ForcesEnhanced,
    ForceDecomposition,
    FluctuationStats,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced import (
    WallShearStressEnhanced,
    WSSPatchStats,
)
from pyfoam.postprocessing.y_plus_enhanced_2 import (
    YPlusEnhanced2,
    MeshQualityMetrics,
    WallLawType,
)

# Phase 12: Enhanced postprocessing models
from pyfoam.postprocessing.field_min_max_enhanced_3 import (
    FieldMinMaxEnhanced3,
    PercentileStats,
    HistogramData,
)
from pyfoam.postprocessing.probes_enhanced_3 import (
    ProbesEnhanced3,
    ProbeSpectrumResult,
    CoherenceMatrix,
)
from pyfoam.postprocessing.forces_enhanced_2 import (
    ForcesEnhanced2,
    ProjectedForces,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_2 import (
    WallShearStressEnhanced2,
    WSSDistribution,
)
from pyfoam.postprocessing.y_plus_enhanced_3 import (
    YPlusEnhanced3,
    RegimeClassification,
    YPlusEvolution,
)

# Phase 13: Enhanced postprocessing models
from pyfoam.postprocessing.field_min_max_enhanced_4 import (
    FieldMinMaxEnhanced4,
    RegionStats,
    TimeHistoryEntry,
)
from pyfoam.postprocessing.probes_enhanced_4 import (
    ProbesEnhanced4,
    ProbeGroupManager,
    FrequencyTracker,
)
from pyfoam.postprocessing.forces_enhanced_3 import (
    ForcesEnhanced3,
    MomentCoefficients,
    ForceSpectrum,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_3 import (
    WallShearStressEnhanced3,
    CfDistribution,
    WSSEvolution,
)
from pyfoam.postprocessing.y_plus_enhanced_4 import (
    YPlusEnhanced4,
    WallDistanceMetrics,
    TimeStepSuggestion,
)

# Phase 14: Enhanced postprocessing models
from pyfoam.postprocessing.field_min_max_enhanced_5 import (
    FieldMinMaxEnhanced5,
    AnomalyEvent,
    TrendAnalysis,
)
from pyfoam.postprocessing.probes_enhanced_5 import (
    ProbesEnhanced5,
    WaveletResult,
    SignalQuality,
    AutoPlacementResult,
)
from pyfoam.postprocessing.forces_enhanced_4 import (
    ForcesEnhanced4,
    UnsteadyForceStats,
    AeroacousticSource,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_4 import (
    WallShearStressEnhanced4,
    QuadrantEvent,
    SpatialCorrelation,
)
from pyfoam.postprocessing.y_plus_enhanced_5 import (
    YPlusEnhanced5,
    AMRSuggestion,
    YPlusBudget,
    WallModelConsistency,
)

# Phase 15: Enhanced postprocessing models
from pyfoam.postprocessing.field_min_max_enhanced_6 import (
    FieldMinMaxEnhanced6,
    FieldCorrelation,
    SPCControlChart,
    PredictiveAlert,
)
from pyfoam.postprocessing.probes_enhanced_6 import (
    ProbesEnhanced6,
    PODResult,
    LagrangianTrack,
)
from pyfoam.postprocessing.forces_enhanced_5 import (
    ForcesEnhanced5,
    FSIForceData,
    FatigueSpectrum,
    MomentPSD,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_5 import (
    WallShearStressEnhanced5,
    AnisotropyTensor,
    CoherentStructure,
)
from pyfoam.postprocessing.y_plus_enhanced_6 import (
    YPlusEnhanced6,
    WallHeatTransfer,
    AdaptiveWallFunction,
    YPlusPrediction,
)

# Phase 16: Enhanced postprocessing models
from pyfoam.postprocessing.field_min_max_enhanced_7 import (
    FieldMinMaxEnhanced7,
    MultivariateAnomaly,
    AdaptiveThreshold,
)
from pyfoam.postprocessing.probes_enhanced_7 import (
    ProbesEnhanced7,
    CompressedSensingResult,
    SensorPlacementResult,
)
from pyfoam.postprocessing.forces_enhanced_6 import (
    ForcesEnhanced6,
    DMDMode,
    FrequencyTracker,
    ReferenceFrameTransform,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_6 import (
    WallShearStressEnhanced6,
    WMLESInterface,
    PressureStrainCorrelation,
)
from pyfoam.postprocessing.y_plus_enhanced_7 import (
    YPlusEnhanced7,
    YPlusUncertainty,
    WallFunctionEnsemble,
)

# Phase 17: Enhanced postprocessing models
from pyfoam.postprocessing.field_min_max_enhanced_8 import (
    FieldMinMaxEnhanced8,
    TemporalCluster,
    CrossFieldCorrelation,
    AlertRule,
)
from pyfoam.postprocessing.probes_enhanced_8 import (
    ProbesEnhanced8,
    StreamingStats,
    ProbeHealth,
)
from pyfoam.postprocessing.forces_enhanced_7 import (
    ForcesEnhanced7,
    WaveletDecomposition,
    MultiBodyForce,
    CoefficientStats,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_7 import (
    WallShearStressEnhanced7,
    DragDecomposition,
    WallTurbulenceStats,
)
from pyfoam.postprocessing.y_plus_enhanced_8 import (
    YPlusEnhanced8,
    YPlusSpectrum,
    MeshAdaptationCriterion,
)

# Phase 18: Enhanced postprocessing models
from pyfoam.postprocessing.field_min_max_enhanced_9 import (
    FieldMinMaxEnhanced9,
    SpatialCluster,
    SPCLimit,
)
from pyfoam.postprocessing.probes_enhanced_9 import (
    ProbesEnhanced9,
    ReconstructedSignal,
    NetworkTopology,
)
from pyfoam.postprocessing.forces_enhanced_8 import (
    ForcesEnhanced8,
    PODMode,
    FrequencyDomainResult,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_8 import (
    WallShearStressEnhanced8,
    StreakSpacing,
    SkinFrictionTopology,
)
from pyfoam.postprocessing.y_plus_enhanced_9 import (
    YPlusEnhanced9,
    PatchComparison,
    TBLClassification,
    CellHeightSuggestion,
)

# Phase 19: Enhanced postprocessing models
from pyfoam.postprocessing.field_min_max_enhanced_10 import (
    FieldMinMaxEnhanced10,
    TopologicalExtreme,
    MultiFieldCorrelation,
)
from pyfoam.postprocessing.probes_enhanced_10 import (
    ProbesEnhanced10,
    ProbeCorrelation,
    SpectralEntropy,
)
from pyfoam.postprocessing.forces_enhanced_9 import (
    ForcesEnhanced9,
    LoadCycle,
    FatigueDamage,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_9 import (
    WallShearStressEnhanced9,
    AveragedTopology,
    StreakDynamics,
)
from pyfoam.postprocessing.y_plus_enhanced_10 import (
    YPlusEnhanced10,
    WallFunctionConsistency,
    MeshConvergenceIndicator,
)

# Phase 20: Enhanced postprocessing models
from pyfoam.postprocessing.field_min_max_enhanced_11 import (
    FieldMinMaxEnhanced11,
    PersistenceExtreme,
    GradientAtExtreme,
)
from pyfoam.postprocessing.probes_enhanced_11 import (
    ProbesEnhanced11,
    TemporalCoherence,
    ProbeCluster,
)
from pyfoam.postprocessing.forces_enhanced_10 import (
    ForcesEnhanced10,
    SpectralMoment,
    SteadyStateIndicator,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_10 import (
    WallShearStressEnhanced10,
    ReynoldsAnalogyResult,
    FrictionDecomposition,
)
from pyfoam.postprocessing.y_plus_enhanced_11 import (
    YPlusEnhanced11,
    WallHeatTransferCoeff,
    PatchRanking,
)

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
    # Phase 11: Enhanced postprocessing v2
    "FieldMinMaxEnhanced2",
    "RegionMinMaxResult",
    "ProbesEnhanced2",
    "CrossSpectrumResult",
    "ForcesEnhanced",
    "ForceDecomposition",
    "FluctuationStats",
    "WallShearStressEnhanced",
    "WSSPatchStats",
    "YPlusEnhanced2",
    "MeshQualityMetrics",
    "WallLawType",
    # Phase 12: Enhanced postprocessing v3
    "FieldMinMaxEnhanced3",
    "PercentileStats",
    "HistogramData",
    "ProbesEnhanced3",
    "ProbeSpectrumResult",
    "CoherenceMatrix",
    "ForcesEnhanced2",
    "ProjectedForces",
    "WallShearStressEnhanced2",
    "WSSDistribution",
    "YPlusEnhanced3",
    "RegimeClassification",
    "YPlusEvolution",
    # Phase 13: Enhanced postprocessing v4
    "FieldMinMaxEnhanced4",
    "RegionStats",
    "TimeHistoryEntry",
    "ProbesEnhanced4",
    "ProbeGroupManager",
    "FrequencyTracker",
    "ForcesEnhanced3",
    "MomentCoefficients",
    "ForceSpectrum",
    "WallShearStressEnhanced3",
    "CfDistribution",
    "WSSEvolution",
    "YPlusEnhanced4",
    "WallDistanceMetrics",
    "TimeStepSuggestion",
    # Phase 14: Enhanced postprocessing v5
    "FieldMinMaxEnhanced5",
    "AnomalyEvent",
    "TrendAnalysis",
    "ProbesEnhanced5",
    "WaveletResult",
    "SignalQuality",
    "AutoPlacementResult",
    "ForcesEnhanced4",
    "UnsteadyForceStats",
    "AeroacousticSource",
    "WallShearStressEnhanced4",
    "QuadrantEvent",
    "SpatialCorrelation",
    "YPlusEnhanced5",
    "AMRSuggestion",
    "YPlusBudget",
    "WallModelConsistency",
    # Phase 15: Enhanced postprocessing v6
    "FieldMinMaxEnhanced6",
    "FieldCorrelation",
    "SPCControlChart",
    "PredictiveAlert",
    "ProbesEnhanced6",
    "PODResult",
    "LagrangianTrack",
    "ForcesEnhanced5",
    "FSIForceData",
    "FatigueSpectrum",
    "MomentPSD",
    "WallShearStressEnhanced5",
    "AnisotropyTensor",
    "CoherentStructure",
    "YPlusEnhanced6",
    "WallHeatTransfer",
    "AdaptiveWallFunction",
    "YPlusPrediction",
    # Phase 16: Enhanced postprocessing v7
    "FieldMinMaxEnhanced7",
    "MultivariateAnomaly",
    "AdaptiveThreshold",
    "ProbesEnhanced7",
    "CompressedSensingResult",
    "SensorPlacementResult",
    "ForcesEnhanced6",
    "DMDMode",
    "FrequencyTracker",
    "ReferenceFrameTransform",
    "WallShearStressEnhanced6",
    "WMLESInterface",
    "PressureStrainCorrelation",
    "YPlusEnhanced7",
    "YPlusUncertainty",
    "WallFunctionEnsemble",
    # Phase 17: Enhanced postprocessing v8
    "FieldMinMaxEnhanced8",
    "TemporalCluster",
    "CrossFieldCorrelation",
    "AlertRule",
    "ProbesEnhanced8",
    "StreamingStats",
    "ProbeHealth",
    "ForcesEnhanced7",
    "WaveletDecomposition",
    "MultiBodyForce",
    "CoefficientStats",
    "WallShearStressEnhanced7",
    "DragDecomposition",
    "WallTurbulenceStats",
    "YPlusEnhanced8",
    "YPlusSpectrum",
    "MeshAdaptationCriterion",
    # Phase 18: Enhanced postprocessing v9
    "FieldMinMaxEnhanced9",
    "SpatialCluster",
    "SPCLimit",
    "ProbesEnhanced9",
    "ReconstructedSignal",
    "NetworkTopology",
    "ForcesEnhanced8",
    "PODMode",
    "FrequencyDomainResult",
    "WallShearStressEnhanced8",
    "StreakSpacing",
    "SkinFrictionTopology",
    "YPlusEnhanced9",
    "PatchComparison",
    "TBLClassification",
    "CellHeightSuggestion",
    # Phase 19: Enhanced postprocessing models
    "FieldMinMaxEnhanced10",
    "TopologicalExtreme",
    "MultiFieldCorrelation",
    "ProbesEnhanced10",
    "ProbeCorrelation",
    "SpectralEntropy",
    "ForcesEnhanced9",
    "LoadCycle",
    "FatigueDamage",
    "WallShearStressEnhanced9",
    "AveragedTopology",
    "StreakDynamics",
    "YPlusEnhanced10",
    "WallFunctionConsistency",
    "MeshConvergenceIndicator",
    # Phase 20: Enhanced postprocessing models
    "FieldMinMaxEnhanced11",
    "PersistenceExtreme",
    "GradientAtExtreme",
    "ProbesEnhanced11",
    "TemporalCoherence",
    "ProbeCluster",
    "ForcesEnhanced10",
    "SpectralMoment",
    "SteadyStateIndicator",
    "WallShearStressEnhanced10",
    "ReynoldsAnalogyResult",
    "FrictionDecomposition",
    "YPlusEnhanced11",
    "WallHeatTransferCoeff",
    "PatchRanking",
]
