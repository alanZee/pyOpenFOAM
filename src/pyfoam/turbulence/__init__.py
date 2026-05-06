"""
pyfoam.turbulence — RANS turbulence models.

Provides:

- :class:`TurbulenceModel` — abstract base class with RTS registry
- :class:`KEpsilonModel` — standard k-ε model
- :class:`RealizableKEpsilonModel` — realizable k-ε model
- :class:`KOmegaSSTModel` — k-ω SST model (Menter 1994)
- :class:`SpalartAllmarasModel` — S-A one-equation model
- :class:`RASModel` — RAS wrapper for solver integration
- Wall function utilities: ``compute_nut_wall``, ``compute_k_wall``,
  ``compute_omega_wall``, ``compute_epsilon_wall``

All models register themselves via ``@TurbulenceModel.register(name)``
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

# Concrete models (each import triggers @TurbulenceModel.register)
from pyfoam.turbulence.k_epsilon import KEpsilonModel, RealizableKEpsilonModel, KEpsilonConstants
from pyfoam.turbulence.k_omega_sst import KOmegaSSTModel, KOmegaSSTConstants
from pyfoam.turbulence.spalart_allmaras import SpalartAllmarasModel, SpalartAllmarasConstants

# Wall functions
from pyfoam.turbulence.wall_functions import (
    compute_nut_wall,
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
    # Models
    "KEpsilonModel",
    "RealizableKEpsilonModel",
    "KOmegaSSTModel",
    "SpalartAllmarasModel",
    # Constants
    "KEpsilonConstants",
    "KOmegaSSTConstants",
    "SpalartAllmarasConstants",
    # Wall functions
    "compute_nut_wall",
    "compute_k_wall",
    "compute_omega_wall",
    "compute_epsilon_wall",
    "compute_y_plus",
    # RAS wrapper
    "RASModel",
    "RASConfig",
]
