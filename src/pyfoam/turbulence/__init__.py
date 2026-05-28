"""
pyfoam.turbulence — RANS/LES/DES turbulence models.

Provides:

**RANS Models:**
- :class:`KEpsilonModel` — standard k-ε model
- :class:`RealizableKEpsilonModel` — realizable k-ε model
- :class:`KOmegaSSTModel` — k-ω SST model (Menter 1994)
- :class:`KOmegaSSTLMModel` — k-ω SST Langtry-Menter transition model
- :class:`KOmegaModel` — standard k-ω model (Wilcox 2006)
- :class:`KOmega2006Model` — k-ω model with cross-diffusion and low-Re correction (Wilcox 2006)
- :class:`SpalartAllmarasModel` — S-A one-equation model
- :class:`LaunderSharmaKEModel` — low-Re k-ε model
- :class:`V2FModel` — v²-f model (Durbin 1995)
- :class:`RNGkEpsilonModel` — RNG k-ε model
- :class:`LRRModel` — LRR Reynolds stress model (Launder-Reece-Rodi 1975)
- :class:`SSGModel` — SSG Reynolds stress model (Speziale-Sarkar-Gatski 1991)

**LES Models:**
- :class:`SmagorinskyModel` — Smagorinsky SGS model
- :class:`WALEModel` — WALE SGS model
- :class:`DynamicSmagorinskyModel` — dynamic Smagorinsky model
- :class:`DynamicLagrangianModel` — Lagrangian dynamic model
- :class:`KEqnModel` — one-equation k SGS model
- :class:`DeardorffDiffStressModel` — Deardorff diffusion stress SGS model

**DES/SAS Models:**
- :class:`KOmegaSSTDESModel` — k-ω SST DES model
- :class:`KOmegaSSTSASModel` — k-ω SST SAS model
- :class:`SpalartAllmarasDDESModel` — SA DDES model
- :class:`SpalartAllmarasDESModel` — SA DES model
- :class:`SpalartAllmarasIDDESModel` — SA IDDES model

**Enhanced RANS Models:**
- :class:`KOmegaSST2003Model` — k-ω SST Menter 2003 with improved cross-diffusion
- :class:`RealizableKE2Model` — Enhanced realizable k-ε with production limiter

**Wall Functions:**
- :class:`NutkWallFunctionBC` — k-based wall function for ν_t
- :class:`NutLowReWallFunctionBC` — low-Re wall function (ν_t=0)
- :class:`EpsilonWallFunctionBC` — ε wall function
- :class:`OmegaWallFunctionBC` — ω wall function
- :class:`KqRWallFunctionBC` — k wall function

**LES Delta Models:**
- :class:`CubeRootVolDelta` — cube root of cell volume (standard)
- :class:`MaxDeltaXYZ` — maximum direction cell size
- :class:`VanDriestDelta` — cube root volume with Van Driest wall damping

**Viscoelastic Models:**
- :class:`MaxwellModel` — upper-convected Maxwell viscoelastic model
- :class:`GiesekusModel` — Giesekus viscoelastic model
- :class:`PTTModel` — Phan-Thien-Tanner viscoelastic model

**Scalar Transport Models:**
- :class:`ScalarTransportModel` — Abstract base for turbulent scalar flux models
- :class:`SGDH` — Simple Gradient Diffusion Hypothesis
- :class:`GGDH` — Generalized Gradient Diffusion Hypothesis

**Turbulence Time Scale Models:**
- :class:`TurbulenceTimeScale` — Abstract base for turbulence time scale models
- :class:`KolmogorovTimeScale` — Kolmogorov micro time scale: tau = sqrt(nu/epsilon)
- :class:`IntegralTimeScale` — Integral (large-eddy) time scale: tau = k/epsilon

All RANS/DES models register themselves via ``@TurbulenceModel.register(name)``
and can be instantiated at run-time via ``TurbulenceModel.create(name, ...)``.

Usage::

    from pyfoam.turbulence import TurbulenceModel, RASModel, RASConfig

    # Direct model creation
    model = TurbulenceModel.create("kEpsilon", mesh, U, phi)
    model.correct()
    nut = model.nut()

    # RAS wrapper (preferred for solver integration)
    config = RASConfig(model_name="kOmegaSST", nu=1.5e-5)
    ras = RASModel(mesh, U, phi, config)
    ras.correct()
    mu_eff = ras.mu_eff()
"""

# Base class
from pyfoam.turbulence.turbulence_model import TurbulenceModel

# RANS models (each import triggers @TurbulenceModel.register)
from pyfoam.turbulence.k_epsilon import KEpsilonModel, RealizableKEpsilonModel, KEpsilonConstants
from pyfoam.turbulence.k_omega_sst import KOmegaSSTModel, KOmegaSSTConstants
from pyfoam.turbulence.k_omega_sst_lm import KOmegaSSTLMModel, KOmegaSSTLMConstants
from pyfoam.turbulence.k_omega import KOmegaModel, KOmegaConstants
from pyfoam.turbulence.k_omega_2006 import KOmega2006Model, KOmega2006Constants
from pyfoam.turbulence.spalart_allmaras import SpalartAllmarasModel, SpalartAllmarasConstants
from pyfoam.turbulence.launder_sharma_ke import LaunderSharmaKEModel, LaunderSharmaKEConstants
from pyfoam.turbulence.v2f import V2FModel, V2FConstants
from pyfoam.turbulence.rng_k_epsilon import RNGkEpsilonModel, RNGkEpsilonConstants
from pyfoam.turbulence.lrr import LRRModel, LRRConstants
from pyfoam.turbulence.ssg import SSGModel, SSGConstants

# LES models
from pyfoam.turbulence.smagorinsky import SmagorinskyModel
from pyfoam.turbulence.wale import WALEModel
from pyfoam.turbulence.dynamic_smagorinsky import DynamicSmagorinskyModel
from pyfoam.turbulence.dynamic_lagrangian import DynamicLagrangianModel
from pyfoam.turbulence.k_eqn import KEqnModel, KEqnConstants
from pyfoam.turbulence.deardorff_diff_stress import DeardorffDiffStressModel, DeardorffDiffStressConstants

# DES models (each import triggers @TurbulenceModel.register)
from pyfoam.turbulence.k_omega_sst_des import KOmegaSSTDESModel, KOmegaSSTDESConstants
from pyfoam.turbulence.k_omega_sst_sas import KOmegaSSTSASModel, KOmegaSSTSASConstants
from pyfoam.turbulence.sa_ddes import SpalartAllmarasDDESModel, SpalartAllmarasDDESConstants
from pyfoam.turbulence.sa_des import SpalartAllmarasDESModel, SpalartAllmarasDESConstants
from pyfoam.turbulence.sa_iddes import SpalartAllmarasIDDESModel, SpalartAllmarasIDDESConstants

# Enhanced RANS models
from pyfoam.turbulence.turbulence_2 import (
    KOmegaSST2003Model,
    KOmegaSST2003Constants,
    RealizableKE2Model,
    RealizableKE2Constants,
)

# Wall functions
from pyfoam.turbulence.wall_functions import (
    compute_nut_wall,
    compute_nut_low_re_wall,
    compute_k_wall,
    compute_omega_wall,
    compute_epsilon_wall,
    compute_y_plus,
)

# RAS wrapper
from pyfoam.turbulence.ras_model import RASModel, RASConfig

# Laminar models
from pyfoam.turbulence.laminar_models import (
    StokesModel,
    GeneralizedNewtonianModel,
    PowerLawViscosity,
    BirdCarreauViscosity,
    CrossViscosity,
    CassonViscosity,
    HerschelBulkleyViscosity,
)

# Viscoelastic models
from pyfoam.turbulence.viscoelastic import (
    ViscoelasticModel,
    MaxwellModel,
    GiesekusModel,
    PTTModel,
    ViscoelasticConstants,
)

# LES delta models
from pyfoam.turbulence.les_deltas import (
    LESDelta,
    CubeRootVolDelta,
    MaxDeltaXYZ,
    VanDriestDelta,
)

# LES spatial filters
from pyfoam.turbulence.les_filters import (
    LESFilter,
    SimpleFilter,
    LaplaceFilter,
)

# Wall distance calculators
from pyfoam.turbulence.wall_distance import (
    WallDistanceCalculator,
    ExactWallDistance,
    ApproximateWallDistance,
)

# Compressibility corrections
from pyfoam.turbulence.compressibility_corrections import (
    CompressibilityCorrection,
    SarkarModel,
    ZemanModel,
)

# Scalar transport models
from pyfoam.turbulence.scalar_transport import (
    ScalarTransportModel,
    SGDH,
    GGDH,
)

# Turbulence time scale models
from pyfoam.turbulence.turbulence_time_scale import (
    TurbulenceTimeScale,
    KolmogorovTimeScale,
    IntegralTimeScale,
)

# Production limiters
from pyfoam.turbulence.production_limiter import (
    ProductionLimiter,
    StandardLimiter,
    KatoLimiter,
)

# Wall treatment models
from pyfoam.turbulence.wall_treatment import (
    WallTreatment,
    StandardWallTreatment,
    AutomaticWallTreatment,
)

__all__ = [
    # Base
    "TurbulenceModel",
    # RANS Models
    "KEpsilonModel",
    "RealizableKEpsilonModel",
    "KOmegaSSTModel",
    "KOmegaSSTLMModel",
    "KOmegaModel",
    "KOmega2006Model",
    "SpalartAllmarasModel",
    "LaunderSharmaKEModel",
    "V2FModel",
    "RNGkEpsilonModel",
    "LRRModel",
    "SSGModel",
    # LES Models
    "SmagorinskyModel",
    "WALEModel",
    "DynamicSmagorinskyModel",
    "DynamicLagrangianModel",
    "KEqnModel",
    "DeardorffDiffStressModel",
    # DES Models
    "KOmegaSSTDESModel",
    "KOmegaSSTSASModel",
    "SpalartAllmarasDDESModel",
    "SpalartAllmarasDESModel",
    "SpalartAllmarasIDDESModel",
    # Enhanced RANS Models
    "KOmegaSST2003Model",
    "KOmegaSST2003Constants",
    "RealizableKE2Model",
    "RealizableKE2Constants",
    # Constants
    "KEpsilonConstants",
    "KOmegaSSTConstants",
    "KOmegaSSTLMConstants",
    "KOmegaConstants",
    "KOmega2006Constants",
    "SpalartAllmarasConstants",
    "LaunderSharmaKEConstants",
    "V2FConstants",
    "RNGkEpsilonConstants",
    "LRRConstants",
    "SSGConstants",
    "KEqnConstants",
    "DeardorffDiffStressConstants",
    "KOmegaSSTDESConstants",
    "KOmegaSSTSASConstants",
    "SpalartAllmarasDDESConstants",
    "SpalartAllmarasDESConstants",
    "SpalartAllmarasIDDESConstants",
    # Wall functions
    "compute_nut_wall",
    "compute_nut_low_re_wall",
    "compute_k_wall",
    "compute_omega_wall",
    "compute_epsilon_wall",
    "compute_y_plus",
    # RAS wrapper
    "RASModel",
    "RASConfig",
    # Laminar models
    "StokesModel",
    "GeneralizedNewtonianModel",
    "PowerLawViscosity",
    "BirdCarreauViscosity",
    "CrossViscosity",
    "CassonViscosity",
    "HerschelBulkleyViscosity",
    # Viscoelastic models
    "ViscoelasticModel",
    "MaxwellModel",
    "GiesekusModel",
    "PTTModel",
    "ViscoelasticConstants",
    # LES delta models
    "LESDelta",
    "CubeRootVolDelta",
    "MaxDeltaXYZ",
    "VanDriestDelta",
    # LES spatial filters
    "LESFilter",
    "SimpleFilter",
    "LaplaceFilter",
    # Wall distance calculators
    "WallDistanceCalculator",
    "ExactWallDistance",
    "ApproximateWallDistance",
    # Compressibility corrections
    "CompressibilityCorrection",
    "SarkarModel",
    "ZemanModel",
    # Scalar transport models
    "ScalarTransportModel",
    "SGDH",
    "GGDH",
    # Turbulence time scale models
    "TurbulenceTimeScale",
    "KolmogorovTimeScale",
    "IntegralTimeScale",
    # Production limiters
    "ProductionLimiter",
    "StandardLimiter",
    "KatoLimiter",
    # Wall treatment models
    "WallTreatment",
    "StandardWallTreatment",
    "AutomaticWallTreatment",
]
