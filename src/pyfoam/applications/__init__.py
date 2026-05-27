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
- :class:`IncompressibleFluidFoam` — Unified incompressible solver (SIMPLE/PISO/PIMPLE auto-detection)
- :class:`ShallowWaterFoam` — 2D shallow water equations (Coriolis + bottom friction)
- :class:`RhoPorousSimpleFoam` — Steady-state compressible with porous media (SIMPLE algorithm)
- :class:`ChemFoam` — 0D chemistry solver (Arrhenius kinetics, forward Euler)
- :class:`IncompressibleVoFFoam` — Modern VOF two-phase incompressible (PIMPLE + MULES)
- :class:`IsothermalFluidFoam` — Transient compressible isothermal (PIMPLE algorithm)
- :class:`CompressibleVoFFoam` — Compressible two-phase VOF (modern interface, PIMPLE + energy)
- :class:`IncompressibleDriftFluxFoam` — Incompressible drift-flux with algebraic slip model
- :class:`ElectrostaticFoam` — Electrostatics solver (Laplace/Poisson for V)
- :class:`MagneticFoam` — Magnetostatics solver (vector Poisson for A)
- :class:`MhdFoam` — Magnetohydrodynamics solver (coupled NS + induction)
- :class:`FluidFoam` — Unified compressible solver with full energy equation (PIMPLE)
- :class:`MulticomponentFluidFoam` — Multi-species compressible PIMPLE solver
- :class:`PDRFoam` — Premixed combustion solver with b-Xi model (PIMPLE)
- :class:`DsmcFoam` — Direct Simulation Monte Carlo for rarefied gas dynamics
- :class:`CHTMultiRegionEnhancedFoam` — Enhanced conjugate heat transfer multi-region
- :class:`AdjointFoam` — Continuous adjoint shape optimization solver
- :class:`FinancialFoam` — Black-Scholes equation for option pricing
- :class:`MdFoam` — Lennard-Jones molecular dynamics (Velocity Verlet)
- :class:`ReactingFoamEnhanced` — Enhanced reacting flow with detailed kinetics
- :class:`SprayFoam` — Lagrangian spray solver with two-way Euler-Lagrange coupling
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
from pyfoam.applications.incompressible_fluid_foam import IncompressibleFluidFoam, Algorithm
from pyfoam.applications.shallow_water_foam import ShallowWaterFoam
from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam
from pyfoam.applications.chem_foam import ChemFoam
from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam
from pyfoam.applications.incompressible_vof_foam import IncompressibleVoFFoam
from pyfoam.applications.compressible_vof_foam import CompressibleVoFFoam
from pyfoam.applications.incompressible_drift_flux_foam import IncompressibleDriftFluxFoam
from pyfoam.applications.electrostatic_foam import ElectrostaticFoam
from pyfoam.applications.magnetic_foam import MagneticFoam
from pyfoam.applications.mhd_foam import MhdFoam
from pyfoam.applications.fluid_foam import FluidFoam
from pyfoam.applications.multicomponent_fluid_foam import MulticomponentFluidFoam
from pyfoam.applications.pdr_foam import PDRFoam
from pyfoam.applications.dsmc_foam import DsmcFoam
from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam
from pyfoam.applications.adjoint_foam import AdjointFoam
from pyfoam.applications.financial_foam import FinancialFoam
from pyfoam.applications.md_foam import MdFoam
from pyfoam.applications.reacting_foam_enhanced import ReactingFoamEnhanced
from pyfoam.applications.spray_foam import SprayFoam
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
    "IncompressibleFluidFoam",
    "Algorithm",
    "ShallowWaterFoam",
    "RhoPorousSimpleFoam",
    "ChemFoam",
    "IsothermalFluidFoam",
    "IncompressibleVoFFoam",
    "CompressibleVoFFoam",
    "IncompressibleDriftFluxFoam",
    "ElectrostaticFoam",
    "MagneticFoam",
    "MhdFoam",
    "FluidFoam",
    "MulticomponentFluidFoam",
    "PDRFoam",
    "DsmcFoam",
    "CHTMultiRegionEnhancedFoam",
    "AdjointFoam",
    "FinancialFoam",
    "MdFoam",
    "ReactingFoamEnhanced",
    "SprayFoam",
    "TimeLoop",
    "ConvergenceMonitor",
]
