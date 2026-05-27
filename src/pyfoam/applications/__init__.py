"""
pyfoam.applications — Application-level solvers.

Provides complete solver applications that read OpenFOAM case directories
and run simulations using the pyOpenFOAM infrastructure.

Available solvers:

- :class:`IcoFoam` — Transient incompressible laminar (PISO algorithm)
- :class:`PimpleFoam` — Transient incompressible (PIMPLE algorithm with turbulence)
- :class:`SimpleFoam` — Steady-state incompressible (SIMPLE algorithm)
- :class:`RhoSimpleFoam` — Steady-state compressible (SIMPLE algorithm)
- :class:`BuoyantSimpleFoam` — Steady-state buoyant compressible (SIMPLE algorithm)
- :class:`RhoPimpleFoam` — Transient compressible (PIMPLE algorithm)
- :class:`RhoCentralFoam` — Density-based compressible (Kurganov-Tadmor central scheme)
- :class:`InterFoam` — VOF two-phase incompressible
- :class:`MultiphaseInterFoam` — N-phase VOF incompressible
- :class:`CompressibleInterFoam` — Compressible two-phase VOF
- :class:`TwoPhaseEulerFoam` — Two-fluid Euler-Euler
- :class:`MultiphaseEulerFoam` — N-phase Euler-Euler
- :class:`CavitatingFoam` — Cavitation solver (Schnerr-Sauer)
- :class:`PisoFoam` — Transient incompressible laminar (PISO algorithm)
- :class:`PotentialFoam` — Potential flow initialisation
- :class:`ScalarTransportFoam` — Passive scalar transport
- :class:`LaplacianFoam` — Steady-state diffusion (Laplacian equation)
- :class:`SonicFoam` — Transient compressible (sonic)
- :class:`SrfSimpleFoam` — Steady-state single rotating frame incompressible
- :class:`BuoyantPimpleFoam` — Transient buoyant compressible (PIMPLE algorithm)
- :class:`CHTMultiRegionFoam` — Conjugate heat transfer multi-region
- :class:`ReactingFoam` — Reactive flow solver
- :class:`SolidDisplacementFoam` — Solid mechanics displacement solver
"""

from pyfoam.applications.solver_base import SolverBase
from pyfoam.applications.boundary_foam import BoundaryFoam
from pyfoam.applications.ico_foam import IcoFoam
from pyfoam.applications.pimple_foam import PimpleFoam
from pyfoam.applications.simple_foam import SimpleFoam
from pyfoam.applications.rho_simple_foam import RhoSimpleFoam
from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam
from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam
from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam
from pyfoam.applications.rho_central_foam import RhoCentralFoam
from pyfoam.applications.inter_foam import InterFoam
from pyfoam.applications.porous_simple_foam import PorousSimpleFoam
from pyfoam.applications.multiphase_inter_foam import MultiphaseInterFoam
from pyfoam.applications.compressible_inter_foam import CompressibleInterFoam
from pyfoam.applications.two_phase_euler_foam import TwoPhaseEulerFoam
from pyfoam.applications.multiphase_euler_foam import MultiphaseEulerFoam
from pyfoam.applications.cavitating_foam import CavitatingFoam
from pyfoam.applications.piso_foam import PisoFoam
from pyfoam.applications.potential_foam import PotentialFoam
from pyfoam.applications.scalar_transport_foam import ScalarTransportFoam
from pyfoam.applications.laplacian_foam import LaplacianFoam
from pyfoam.applications.sonic_foam import SonicFoam
from pyfoam.applications.srf_simple_foam import SrfSimpleFoam
from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam
from pyfoam.applications.cht_multi_region_foam import CHTMultiRegionFoam
from pyfoam.applications.reacting_foam import ReactingFoam
from pyfoam.applications.solid_displacement_foam import SolidDisplacementFoam
from pyfoam.applications.time_loop import TimeLoop
from pyfoam.applications.convergence import ConvergenceMonitor

__all__ = [
    "SolverBase",
    "BoundaryFoam",
    "IcoFoam",
    "PimpleFoam",
    "SimpleFoam",
    "RhoSimpleFoam",
    "BuoyantSimpleFoam",
    "BuoyantBoussinesqSimpleFoam",
    "RhoPimpleFoam",
    "RhoCentralFoam",
    "InterFoam",
    "PorousSimpleFoam",
    "MultiphaseInterFoam",
    "CompressibleInterFoam",
    "TwoPhaseEulerFoam",
    "MultiphaseEulerFoam",
    "CavitatingFoam",
    "PisoFoam",
    "PotentialFoam",
    "ScalarTransportFoam",
    "LaplacianFoam",
    "SonicFoam",
    "SrfSimpleFoam",
    "BuoyantPimpleFoam",
    "CHTMultiRegionFoam",
    "ReactingFoam",
    "SolidDisplacementFoam",
    "TimeLoop",
    "ConvergenceMonitor",
]
