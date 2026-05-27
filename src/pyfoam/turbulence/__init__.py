"""
pyfoam.turbulence — RANS/LES/DES turbulence models.

Provides:

**RANS Models:**
- :class:`KEpsilonModel` — standard k-ε model
- :class:`RealizableKEpsilonModel` — realizable k-ε model
- :class:`KOmegaSSTModel` — k-ω SST model (Menter 1994)
- :class:`KOmegaModel` — standard k-ω model (Wilcox 2006)
- :class:`KOmega2006Model` — k-ω model with cross-diffusion and low-Re correction (Wilcox 2006)
- :class:`SpalartAllmarasModel` — S-A one-equation model
- :class:`LaunderSharmaKEModel` — low-Re k-ε model
- :class:`V2FModel` — v²-f model (Durbin 1995)
- :class:`RNGkEpsilonModel` — RNG k-ε model

**LES Models:**
- :class:`SmagorinskyModel` — Smagorinsky SGS model
- :class:`WALEModel` — WALE SGS model
- :class:`DynamicSmagorinskyModel` — dynamic Smagorinsky model
- :class:`DynamicLagrangianModel` — Lagrangian dynamic model
- :class:`KEqnModel` — one-equation k SGS model
- :class:`DeardorffDiffStressModel` — Deardorff diffusion stress SGS model

**DES Models:**
- :class:`KOmegaSSTDESModel` — k-ω SST DES model
- :class:`SpalartAllmarasDDESModel` — SA DDES model
- :class:`SpalartAllmarasDESModel` — SA DES model
- :class:`SpalartAllmarasIDDESModel` — SA IDDES model

**Wall Functions:**
- :class:`NutkWallFunctionBC` — k-based wall function for ν_t
- :class:`NutLowReWallFunctionBC` — low-Re wall function (ν_t=0)
- :class:`EpsilonWallFunctionBC` — ε wall function
- :class:`OmegaWallFunctionBC` — ω wall function
- :class:`KqRWallFunctionBC` — k wall function

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
from pyfoam.turbulence.k_omega import KOmegaModel, KOmegaConstants
from pyfoam.turbulence.k_omega_2006 import KOmega2006Model, KOmega2006Constants
from pyfoam.turbulence.spalart_allmaras import SpalartAllmarasModel, SpalartAllmarasConstants
from pyfoam.turbulence.launder_sharma_ke import LaunderSharmaKEModel, LaunderSharmaKEConstants
from pyfoam.turbulence.v2f import V2FModel, V2FConstants
from pyfoam.turbulence.rng_k_epsilon import RNGkEpsilonModel, RNGkEpsilonConstants

# LES models
from pyfoam.turbulence.smagorinsky import SmagorinskyModel
from pyfoam.turbulence.wale import WALEModel
from pyfoam.turbulence.dynamic_smagorinsky import DynamicSmagorinskyModel
from pyfoam.turbulence.dynamic_lagrangian import DynamicLagrangianModel
from pyfoam.turbulence.k_eqn import KEqnModel, KEqnConstants
from pyfoam.turbulence.deardorff_diff_stress import DeardorffDiffStressModel, DeardorffDiffStressConstants

# DES models (each import triggers @TurbulenceModel.register)
from pyfoam.turbulence.k_omega_sst_des import KOmegaSSTDESModel, KOmegaSSTDESConstants
from pyfoam.turbulence.sa_ddes import SpalartAllmarasDDESModel, SpalartAllmarasDDESConstants
from pyfoam.turbulence.sa_des import SpalartAllmarasDESModel, SpalartAllmarasDESConstants
from pyfoam.turbulence.sa_iddes import SpalartAllmarasIDDESModel, SpalartAllmarasIDDESConstants

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

__all__ = [
    # Base
    "TurbulenceModel",
    # RANS Models
    "KEpsilonModel",
    "RealizableKEpsilonModel",
    "KOmegaSSTModel",
    "KOmegaModel",
    "KOmega2006Model",
    "SpalartAllmarasModel",
    "LaunderSharmaKEModel",
    "V2FModel",
    "RNGkEpsilonModel",
    # LES Models
    "SmagorinskyModel",
    "WALEModel",
    "DynamicSmagorinskyModel",
    "DynamicLagrangianModel",
    "KEqnModel",
    "DeardorffDiffStressModel",
    # DES Models
    "KOmegaSSTDESModel",
    "SpalartAllmarasDDESModel",
    "SpalartAllmarasDESModel",
    "SpalartAllmarasIDDESModel",
    # Constants
    "KEpsilonConstants",
    "KOmegaSSTConstants",
    "KOmegaConstants",
    "KOmega2006Constants",
    "SpalartAllmarasConstants",
    "LaunderSharmaKEConstants",
    "V2FConstants",
    "RNGkEpsilonConstants",
    "KEqnConstants",
    "DeardorffDiffStressConstants",
    "KOmegaSSTDESConstants",
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
]
