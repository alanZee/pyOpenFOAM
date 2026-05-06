"""
pyfoam.applications — Application-level solvers.

Provides complete solver applications that read OpenFOAM case directories
and run simulations using the pyOpenFOAM infrastructure.

Available solvers:

- :class:`SimpleFoam` — Steady-state incompressible (SIMPLE algorithm)
- :class:`RhoSimpleFoam` — Steady-state compressible (SIMPLE algorithm)
- :class:`RhoPimpleFoam` — Transient compressible (PIMPLE algorithm)
- :class:`InterFoam` — VOF two-phase incompressible
"""

from pyfoam.applications.solver_base import SolverBase
from pyfoam.applications.simple_foam import SimpleFoam
from pyfoam.applications.rho_simple_foam import RhoSimpleFoam
from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam
from pyfoam.applications.inter_foam import InterFoam
from pyfoam.applications.time_loop import TimeLoop
from pyfoam.applications.convergence import ConvergenceMonitor

__all__ = [
    "SolverBase",
    "SimpleFoam",
    "RhoSimpleFoam",
    "RhoPimpleFoam",
    "InterFoam",
    "TimeLoop",
    "ConvergenceMonitor",
]
