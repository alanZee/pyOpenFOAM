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

**Non-Linear Viscosity Models:**
- :class:`NonLinearViscosityModel` — Abstract base for non-linear viscosity with RTS registry
- :class:`PowerLawViscosity` (non_linear_viscosity) — Power-law viscosity model (RTS-registered)
- :class:`BirdCarreauViscosity` (non_linear_viscosity) — Bird-Carreau four-parameter model (RTS-registered)
- :class:`CrossPowerLawViscosity` — Cross power-law model (RTS-registered)

**Compressible Turbulence Models:**
- :class:`CompressibleTurbulenceModel` — Abstract base for compressible turbulence models with RTS registry
- :class:`KOmegaSSTCompressible` — Compressible k-omega SST with density-weighted transport

**Compressible Wall Functions:**
- :class:`CompressibleWallFunction` — Abstract base for compressible wall functions
- :class:`CompressibleNutWallFunction` — Compressible nut wall function with Van Driest damping
- :class:`CompressibleKWallFunction` — Compressible k wall function with local equilibrium and Van Driest correction
- :class:`CompressibleEpsilonWallFunction` — Compressible epsilon wall function (log-law + viscous sublayer)
- :class:`CompressibleOmegaWallFunction` — Compressible omega wall function (log-law + viscous sublayer)

**Turbulence Inlet Models:**
- :class:`TurbulenceInletModel` — Abstract base for turbulence inlet models with RTS registry
- :class:`FixedTurbulenceInlet` — Fixed (uniform) turbulence quantities at inlet
- :class:`MappedTurbulenceInlet` — Mapped turbulence quantities from reference data

**Kato-Launder Production Limiter:**
- :class:`KatoLaunderDamping` — Kato-Launder production limiter (|S|*|Omega| instead of |S|^2)

**Enhanced Turbulence Inlet Models (v2):**
- :class:`TurbulenceInletModel2` — Enhanced base class with spatial correlation support
- :class:`DigitalFilterInlet` — Digital filter turbulence generation (Klein et al., 2003)
- :class:`SyntheticEddyInlet` — Synthetic eddy method (Jarrin et al., 2006)

**LES Spatial Filters:**
- :class:`SimpleFilter` — simple spatial filter
- :class:`LaplaceFilter` — Laplacian spatial filter

**Generalised Newtonian Viscosity Models:**
- :class:`GeneralizedNewtonianViscosity` — Abstract base for generalised Newtonian viscosity with RTS registry
- :class:`CassonModel` — Casson yield-stress model (RTS-registered)
- :class:`HerschelBulkleyModel` — Herschel-Bulkley yield-stress + power-law (RTS-registered)
- :class:`BinghamModel` — Bingham plastic model (RTS-registered)
- :class:`QuemadaModel` — Quemada suspension model (RTS-registered)
- :class:`StrainRateFunctionModel` — user-defined arbitrary viscosity function (RTS-registered)

**Enhanced RANS/LES Models (Phase 10):**
- :class:`KEpsilonEnhancedModel` — Enhanced realizable k-epsilon with dynamic C_mu and production limiter
- :class:`KOmegaEnhancedModel` — Enhanced k-omega (Wilcox 2006) with cross-diffusion and low-Re correction
- :class:`KOmegaSSTEnhancedModel` — Enhanced k-omega SST (Menter 2003) with c1 cross-diffusion limiter
- :class:`SpalartAllmarasEnhancedModel` — SA-noft2 variant (no trip term, standard production code variant)
- :class:`ImprovedSmagorinskyModel` — Smagorinsky with Van Driest wall damping and SGS energy
- :class:`ImprovedWALEModel` — WALE with clipping, SGS energy, and SGS time scale

**Enhanced RANS/LES Models (Phase 11):**
- :class:`KEpsilonEnhanced2Model` — Realizable k-epsilon v2 with improved C_mu and wall-reflection correction
- :class:`KOmegaEnhanced2Model` — k-omega v2 with improved cross-diffusion and stress limiter
- :class:`KOmegaSSTEnhanced2Model` — SST v2 with F4 blending and Kato-Launder option
- :class:`SpalartAllmarasEnhanced2Model` — SA v2 with QCR and rotational correction (SARC)
- :class:`DynamicLikeSmagorinskyModel` — Smagorinsky with dynamic-like Cs from strain/vorticity ratio
- :class:`ImprovedWALE2Model` — WALE v2 with improved near-wall scaling and wall distance damping
- :class:`EnhancedWallTreatment` — Enhanced wall treatment with tanh blending
- :class:`ThreeLayerWallTreatment` — Three-layer (viscous/buffer/log-law) wall treatment

**Enhanced RANS/LES Models (Phase 12):**
- :class:`KEpsilonEnhanced3Model` — Realizable k-epsilon v3 with SSS correction and elliptic relaxation
- :class:`KOmegaEnhanced3Model` — k-omega v3 with strain-dependent beta and omega clipping
- :class:`KOmegaSSTEnhanced3Model` — SST v3 with F3 blending, rotation-curvature correction, adaptive sigma_k
- :class:`SpalartAllmarasEnhanced3Model` — SA v3 with adaptive Cb1, ft2 correction, separated flow length scale
- :class:`WallAdaptiveSmagorinskyModel` — Smagorinsky with wall-adaptive Cs and rotation suppression
- :class:`ImprovedWALE3Model` — WALE v3 with trace-free Sd, rotation correction, SGS dissipation
- :class:`EnhancedWallTreatment2` — Enhanced wall treatment v2 with roughness support
- :class:`FourLayerWallTreatment` — Four-layer (viscous/buffer/transition/log-law) wall treatment

**Enhanced RANS/LES Models (Phase 13):**
- :class:`KEpsilonEnhanced4Model` — Realizable k-epsilon v4 with v2-f, Yap correction, anisotropy-aware C_mu
- :class:`KOmegaEnhanced4Model` — k-omega v4 with low-Re damping (Wilcox 2008) and viscosity-dependent cross-diffusion
- :class:`KOmegaSSTEnhanced4Model` — SST v4 with intermittency transition, Spalart-Shur curvature correction
- :class:`SpalartAllmarasEnhanced4Model` — SA v4 with QCR2013, curvature correction, improved ft2
- :class:`VremanModel` — Vreman SGS model (zero viscosity in laminar, no wall damping needed)
- :class:`SigmaModel` — Sigma SGS model (singular values of velocity gradient, Nicoud et al. 2011)
- :class:`EnhancedWallTreatment3` — Enhanced wall treatment v3 with Jayatilleke blending and heat transfer
- :class:`AdaptiveWallTreatment` — Adaptive wall treatment with hysteresis-based regime switching

**Enhanced RANS/LES Models (Phase 14):**
- :class:`KEpsilonEnhanced5Model` — Realizable k-epsilon v5 with elliptic blending and RNG correction
- :class:`KOmegaEnhanced5Model` — k-omega v5 with shear-layer adapted beta and SST-like blending
- :class:`KOmegaSSTEnhanced5Model` — SST v5 with amplification factor and improved rotation-curvature
- :class:`SpalartAllmarasEnhanced5Model` — SA v5 with hybrid RANS-LES and controlled decay
- :class:`DynamicLagrangianSGS` — Dynamic Smagorinsky with Lagrangian averaging (Meneveau et al. 1996)
- :class:`AMDModel` — Anisotropic Minimum Dissipation model (Rozema et al. 2015)
- :class:`EnhancedWallTreatment4` — Enhanced wall treatment v4 with Lewis number and Van Driest damping
- :class:`CompressibleWallTreatment` — Compressible wall treatment with Van Driest damping

**Enhanced RANS/LES Models (Phase 16):**
- :class:`KEpsilonEnhanced6Model` — Realizable k-epsilon v6 with anisotropy-aware C_mu and curvature limiter
- :class:`KOmegaEnhanced6Model` — k-omega v6 with compressible cross-diffusion and adaptive beta
- :class:`KOmegaSSTEnhanced6Model` — SST v6 with roughness, SAS, and adaptive sigma
- :class:`SpalartAllmarasEnhanced6Model` — SA v6 with separated-flow correction and adaptive cv1
- :class:`WMLEModel` — Wall-Modeled LES SGS model
- :class:`TensorViscositySGS` — Tensor-viscosity SGS model
- :class:`EnhancedWallTreatment5` — Enhanced wall treatment v5 with conjugate HT and adaptive y+ tracking
- :class:`ConjugateHeatTransfer` — Conjugate heat transfer calculator

**Enhanced RANS/LES Models (Phase 17):**
- :class:`KEpsilonEnhanced7Model` — Realizable k-epsilon v7 with wall-integrated epsilon and dynamic near-wall correction
- :class:`KOmegaEnhanced7Model` — k-omega v7 with buoyancy production and density-gradient cross-diffusion
- :class:`KOmegaSSTEnhanced7Model` — SST v7 with shock-capturing, Sarkar compressibility, and adaptive blending
- :class:`SpalartAllmarasEnhanced7Model` — SA v7 with trip-free formulation and vorticity amplification
- :class:`AnisotropicMDModel` — Anisotropic Minimum Dissipation SGS model
- :class:`StructureFunctionSGS` — Structure-Function SGS model
- :class:`EnhancedWallTreatment6` — Enhanced wall treatment v6 with species transport and Colebrook-White roughness
- :class:`RoughnessCorrelation` — Standalone roughness correlation calculator

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
    compute_nut_u_wall,
    compute_nut_u_rough_wall,
    compute_nut_u_spalding_wall,
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
    SmoothDelta,
)

# LES spatial filters
from pyfoam.turbulence.les_filters import (
    LESFilter,
    SimpleFilter,
    LaplaceFilter,
    AnisotropicFilter,
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

# Non-linear viscosity models (RTS-registered)
from pyfoam.turbulence.non_linear_viscosity import (
    NonLinearViscosityModel,
    CrossPowerLawViscosity,
)

# Compressible turbulence models
from pyfoam.turbulence.compressible_turbulence import (
    CompressibleTurbulenceModel,
    KOmegaSSTCompressible,
    KOmegaSSTCompressibleConstants,
)

# Compressible wall functions
from pyfoam.turbulence.compressible_wall_functions import (
    CompressibleWallFunction,
    CompressibleNutWallFunction,
    CompressibleKWallFunction,
)

# Compressible epsilon and omega wall functions
from pyfoam.turbulence.compressible_wal_functions_2 import (
    CompressibleEpsilonWallFunction,
    CompressibleOmegaWallFunction,
)

# Turbulence inlet models
from pyfoam.turbulence.turbulence_inlet_models import (
    TurbulenceInletModel,
    FixedTurbulenceInlet,
    MappedTurbulenceInlet,
)

# Kato-Launder production limiter
from pyfoam.turbulence.turbulence_kato_launder import KatoLaunderDamping

# Enhanced turbulence inlet models
from pyfoam.turbulence.turbulence_inlet_models_2 import (
    TurbulenceInletModel2,
    DigitalFilterInlet,
    SyntheticEddyInlet,
)

# SGS models (standardised interface)
from pyfoam.turbulence.les_sgs_models import (
    SGSModel,
    DynamicSmagorinskySGS,
    WALE_SGS,
)

# Generalised Newtonian viscosity models (RTS-registered)
from pyfoam.turbulence.generalized_newtonian import (
    GeneralizedNewtonianViscosity,
    CassonModel,
    HerschelBulkleyModel,
    BinghamModel,
    QuemadaModel,
    StrainRateFunctionModel,
)

# Enhanced RANS/LES models (Phase 10)
from pyfoam.turbulence.k_epsilon_enhanced import KEpsilonEnhancedModel, KEpsilonEnhancedConstants
from pyfoam.turbulence.k_omega_enhanced import KOmegaEnhancedModel, KOmegaEnhancedConstants
from pyfoam.turbulence.k_omega_sst_enhanced import KOmegaSSTEnhancedModel, KOmegaSSTEnhancedConstants
from pyfoam.turbulence.spalart_allmaras_enhanced import SpalartAllmarasEnhancedModel, SpalartAllmarasEnhancedConstants
from pyfoam.turbulence.les_model_enhanced import ImprovedSmagorinskyModel, ImprovedWALEModel

# Enhanced RANS/LES models (Phase 11)
from pyfoam.turbulence.k_epsilon_enhanced_2 import KEpsilonEnhanced2Model, KEpsilonEnhanced2Constants
from pyfoam.turbulence.k_omega_enhanced_2 import KOmegaEnhanced2Model, KOmegaEnhanced2Constants
from pyfoam.turbulence.k_omega_sst_enhanced_2 import KOmegaSSTEnhanced2Model, KOmegaSSTEnhanced2Constants
from pyfoam.turbulence.spalart_allmaras_enhanced_2 import SpalartAllmarasEnhanced2Model, SpalartAllmarasEnhanced2Constants
from pyfoam.turbulence.les_model_enhanced_2 import DynamicLikeSmagorinskyModel, ImprovedWALE2Model
from pyfoam.turbulence.wall_treatment_enhanced import EnhancedWallTreatment, ThreeLayerWallTreatment

# Enhanced RANS/LES models (Phase 12)
from pyfoam.turbulence.k_epsilon_enhanced_3 import KEpsilonEnhanced3Model, KEpsilonEnhanced3Constants
from pyfoam.turbulence.k_omega_enhanced_3 import KOmegaEnhanced3Model, KOmegaEnhanced3Constants
from pyfoam.turbulence.k_omega_sst_enhanced_3 import KOmegaSSTEnhanced3Model, KOmegaSSTEnhanced3Constants
from pyfoam.turbulence.spalart_allmaras_enhanced_3 import SpalartAllmarasEnhanced3Model, SpalartAllmarasEnhanced3Constants
from pyfoam.turbulence.les_model_enhanced_3 import WallAdaptiveSmagorinskyModel, ImprovedWALE3Model
from pyfoam.turbulence.wall_treatment_enhanced_2 import EnhancedWallTreatment2, FourLayerWallTreatment

# Enhanced RANS/LES models (Phase 13)
from pyfoam.turbulence.k_epsilon_enhanced_4 import KEpsilonEnhanced4Model, KEpsilonEnhanced4Constants
from pyfoam.turbulence.k_omega_enhanced_4 import KOmegaEnhanced4Model, KOmegaEnhanced4Constants
from pyfoam.turbulence.k_omega_sst_enhanced_4 import KOmegaSSTEnhanced4Model, KOmegaSSTEnhanced4Constants
from pyfoam.turbulence.spalart_allmaras_enhanced_4 import SpalartAllmarasEnhanced4Model, SpalartAllmarasEnhanced4Constants
from pyfoam.turbulence.les_model_enhanced_4 import VremanModel, SigmaModel
from pyfoam.turbulence.wall_treatment_enhanced_3 import EnhancedWallTreatment3, AdaptiveWallTreatment

# Enhanced RANS/LES models (Phase 14)
from pyfoam.turbulence.k_epsilon_enhanced_5 import KEpsilonEnhanced5Model, KEpsilonEnhanced5Constants
from pyfoam.turbulence.k_omega_enhanced_5 import KOmegaEnhanced5Model, KOmegaEnhanced5Constants
from pyfoam.turbulence.k_omega_sst_enhanced_5 import KOmegaSSTEnhanced5Model, KOmegaSSTEnhanced5Constants
from pyfoam.turbulence.spalart_allmaras_enhanced_5 import SpalartAllmarasEnhanced5Model, SpalartAllmarasEnhanced5Constants
from pyfoam.turbulence.les_model_enhanced_5 import DynamicLagrangianSGS, AMDModel
from pyfoam.turbulence.wall_treatment_enhanced_4 import EnhancedWallTreatment4, CompressibleWallTreatment

# Enhanced RANS/LES models (Phase 16)
from pyfoam.turbulence.k_epsilon_enhanced_6 import KEpsilonEnhanced6Model, KEpsilonEnhanced6Constants
from pyfoam.turbulence.k_omega_enhanced_6 import KOmegaEnhanced6Model, KOmegaEnhanced6Constants
from pyfoam.turbulence.k_omega_sst_enhanced_6 import KOmegaSSTEnhanced6Model, KOmegaSSTEnhanced6Constants
from pyfoam.turbulence.spalart_allmaras_enhanced_6 import SpalartAllmarasEnhanced6Model, SpalartAllmarasEnhanced6Constants
from pyfoam.turbulence.les_model_enhanced_6 import WMLEModel, TensorViscositySGS
from pyfoam.turbulence.wall_treatment_enhanced_5 import EnhancedWallTreatment5, ConjugateHeatTransfer

# Enhanced RANS/LES models (Phase 17)
from pyfoam.turbulence.k_epsilon_enhanced_7 import KEpsilonEnhanced7Model, KEpsilonEnhanced7Constants
from pyfoam.turbulence.k_omega_enhanced_7 import KOmegaEnhanced7Model, KOmegaEnhanced7Constants
from pyfoam.turbulence.k_omega_sst_enhanced_7 import KOmegaSSTEnhanced7Model, KOmegaSSTEnhanced7Constants
from pyfoam.turbulence.spalart_allmaras_enhanced_7 import SpalartAllmarasEnhanced7Model, SpalartAllmarasEnhanced7Constants
from pyfoam.turbulence.les_model_enhanced_7 import AnisotropicMDModel, StructureFunctionSGS
from pyfoam.turbulence.wall_treatment_enhanced_6 import EnhancedWallTreatment6, RoughnessCorrelation

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
    "compute_nut_u_wall",
    "compute_nut_u_rough_wall",
    "compute_nut_u_spalding_wall",
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
    "SmoothDelta",
    # LES spatial filters
    "LESFilter",
    "SimpleFilter",
    "LaplaceFilter",
    "AnisotropicFilter",
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
    # Non-linear viscosity models (RTS-registered)
    "NonLinearViscosityModel",
    "CrossPowerLawViscosity",
    # Compressible turbulence models
    "CompressibleTurbulenceModel",
    "KOmegaSSTCompressible",
    "KOmegaSSTCompressibleConstants",
    # Compressible wall functions
    "CompressibleWallFunction",
    "CompressibleNutWallFunction",
    "CompressibleKWallFunction",
    # Compressible epsilon and omega wall functions
    "CompressibleEpsilonWallFunction",
    "CompressibleOmegaWallFunction",
    # Turbulence inlet models
    "TurbulenceInletModel",
    "FixedTurbulenceInlet",
    "MappedTurbulenceInlet",
    # Kato-Launder production limiter
    "KatoLaunderDamping",
    # Enhanced turbulence inlet models
    "TurbulenceInletModel2",
    "DigitalFilterInlet",
    "SyntheticEddyInlet",
    # SGS models (standardised interface)
    "SGSModel",
    "DynamicSmagorinskySGS",
    "WALE_SGS",
    # Generalised Newtonian viscosity models (RTS-registered)
    "GeneralizedNewtonianViscosity",
    "CassonModel",
    "HerschelBulkleyModel",
    "BinghamModel",
    "QuemadaModel",
    "StrainRateFunctionModel",
    # Enhanced RANS/LES models (Phase 10)
    "KEpsilonEnhancedModel",
    "KEpsilonEnhancedConstants",
    "KOmegaEnhancedModel",
    "KOmegaEnhancedConstants",
    "KOmegaSSTEnhancedModel",
    "KOmegaSSTEnhancedConstants",
    "SpalartAllmarasEnhancedModel",
    "SpalartAllmarasEnhancedConstants",
    "ImprovedSmagorinskyModel",
    "ImprovedWALEModel",
    # Enhanced RANS/LES models (Phase 11)
    "KEpsilonEnhanced2Model",
    "KEpsilonEnhanced2Constants",
    "KOmegaEnhanced2Model",
    "KOmegaEnhanced2Constants",
    "KOmegaSSTEnhanced2Model",
    "KOmegaSSTEnhanced2Constants",
    "SpalartAllmarasEnhanced2Model",
    "SpalartAllmarasEnhanced2Constants",
    "DynamicLikeSmagorinskyModel",
    "ImprovedWALE2Model",
    "EnhancedWallTreatment",
    "ThreeLayerWallTreatment",
    # Enhanced RANS/LES models (Phase 12)
    "KEpsilonEnhanced3Model",
    "KEpsilonEnhanced3Constants",
    "KOmegaEnhanced3Model",
    "KOmegaEnhanced3Constants",
    "KOmegaSSTEnhanced3Model",
    "KOmegaSSTEnhanced3Constants",
    "SpalartAllmarasEnhanced3Model",
    "SpalartAllmarasEnhanced3Constants",
    "WallAdaptiveSmagorinskyModel",
    "ImprovedWALE3Model",
    "EnhancedWallTreatment2",
    "FourLayerWallTreatment",
    # Enhanced RANS/LES models (Phase 13)
    "KEpsilonEnhanced4Model",
    "KEpsilonEnhanced4Constants",
    "KOmegaEnhanced4Model",
    "KOmegaEnhanced4Constants",
    "KOmegaSSTEnhanced4Model",
    "KOmegaSSTEnhanced4Constants",
    "SpalartAllmarasEnhanced4Model",
    "SpalartAllmarasEnhanced4Constants",
    "VremanModel",
    "SigmaModel",
    "EnhancedWallTreatment3",
    "AdaptiveWallTreatment",
    # Enhanced RANS/LES models (Phase 14)
    "KEpsilonEnhanced5Model",
    "KEpsilonEnhanced5Constants",
    "KOmegaEnhanced5Model",
    "KOmegaEnhanced5Constants",
    "KOmegaSSTEnhanced5Model",
    "KOmegaSSTEnhanced5Constants",
    "SpalartAllmarasEnhanced5Model",
    "SpalartAllmarasEnhanced5Constants",
    "DynamicLagrangianSGS",
    "AMDModel",
    "EnhancedWallTreatment4",
    "CompressibleWallTreatment",
    # Enhanced RANS/LES models (Phase 16)
    "KEpsilonEnhanced6Model",
    "KEpsilonEnhanced6Constants",
    "KOmegaEnhanced6Model",
    "KOmegaEnhanced6Constants",
    "KOmegaSSTEnhanced6Model",
    "KOmegaSSTEnhanced6Constants",
    "SpalartAllmarasEnhanced6Model",
    "SpalartAllmarasEnhanced6Constants",
    "WMLEModel",
    "TensorViscositySGS",
    "EnhancedWallTreatment5",
    "ConjugateHeatTransfer",
    # Enhanced RANS/LES models (Phase 17)
    "KEpsilonEnhanced7Model",
    "KEpsilonEnhanced7Constants",
    "KOmegaEnhanced7Model",
    "KOmegaEnhanced7Constants",
    "KOmegaSSTEnhanced7Model",
    "KOmegaSSTEnhanced7Constants",
    "SpalartAllmarasEnhanced7Model",
    "SpalartAllmarasEnhanced7Constants",
    "AnisotropicMDModel",
    "StructureFunctionSGS",
    "EnhancedWallTreatment6",
    "RoughnessCorrelation",
]
