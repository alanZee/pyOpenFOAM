"""
pyfoam.applications — Application-level solvers.

Provides complete solver applications that read OpenFOAM case directories
and run simulations using the pyOpenFOAM infrastructure.

Available solvers:

- :class:`IcoFoam` — Transient incompressible laminar (PISO algorithm)
- :class:`PimpleFoam` — Transient incompressible (PIMPLE algorithm with turbulence)
- :class:`SimpleFoam` — Steady-state incompressible (SIMPLE algorithm)
- :class:`RhoSimpleFoam` — Steady-state compressible (SIMPLE algorithm)
- :class:`RhoPimpleFoam` — Transient compressible (PIMPLE algorithm)
- :class:`RhoCentralFoam` — Density-based compressible (Kurganov-Tadmor central scheme)
- :class:`InterFoam` — VOF two-phase incompressible
"""

from pyfoam.applications.solver_base import SolverBase
from pyfoam.applications.ico_foam import IcoFoam
from pyfoam.applications.pimple_foam import PimpleFoam
from pyfoam.applications.simple_foam import SimpleFoam
from pyfoam.applications.rho_simple_foam import RhoSimpleFoam
from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam
from pyfoam.applications.rho_central_foam import RhoCentralFoam
from pyfoam.applications.inter_foam import InterFoam
from pyfoam.applications.time_loop import TimeLoop
from pyfoam.applications.convergence import ConvergenceMonitor

__all__ = [
    "SolverBase",
    "IcoFoam",
    "PimpleFoam",
    "SimpleFoam",
    "RhoSimpleFoam",
    "RhoPimpleFoam",
    "RhoCentralFoam",
    "InterFoam",
    "TimeLoop",
    "ConvergenceMonitor",
]
