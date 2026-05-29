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
- :class:`CHTSolver` — Simplified conjugate heat transfer solver with iterative coupling
- :class:`PorousInterFoam` — Porous media two-phase VOF solver (Darcy-Forchheimer)
- :class:`AdjointShapeFoam` — Enhanced adjoint shape optimization with mesh morphing
- :class:`DieselFoam` — Diesel spray combustion solver (compressible PIMPLE + Lagrangian spray + Arrhenius chemistry)
- :class:`AdjointTurbulenceFoam` — Adjoint turbulence optimisation solver
- :class:`ReactingMultiphaseFoam` — Reacting multiphase Euler-Euler solver
- :class:`CombustionFoam` — General combustion solver with multiple reaction mechanisms
- :class:`HeatTransferFoam` — Enhanced heat transfer solver with radiation, convection, and conduction coupling
- :class:`ViscousFoam` — Steady-state viscous flow solver for high-viscosity fluids (non-Newtonian)
- :class:`EnergyFoam` — Enhanced energy equation solver with viscous dissipation, compressibility work, radiation coupling
- :class:`MultiphaseReactingFoam` — Multiphase reacting solver with Euler-Euler + combustion
- :class:`AcousticFoam` — Acoustic wave propagation solver (linearized Euler equations)
- :class:`FinancialFoam2` — Enhanced Black-Scholes with Greeks and American options
- :class:`ReactingFoam2` — Enhanced reacting flow with multi-step mechanisms and ISAT
- :class:`CompressibleInterFoam2` — Enhanced compressible VOF with energy equation and variable Cp
- :class:`TwoPhaseEulerFoam2` — Enhanced Euler-Euler with kinetic theory of granular flow
- :class:`MultiphaseEulerFoam2` — Enhanced N-phase with population balance (MUSIG)
- :class:`SolidFoam` — Solid mechanics solver with thermal stress analysis
- :class:`FilmFoam` — Thin film flow solver with surface tension
- :class:`SprayFoam2` — Enhanced Lagrangian spray with KH-RT breakup
- :class:`PisoFoamEnhanced` — Enhanced PISO solver (Rhie-Chow, non-orthogonal corrections)
- :class:`PimpleFoamEnhanced` — Enhanced PIMPLE solver (Aitken relaxation, warm-up)
- :class:`SimpleFoamEnhanced` — Enhanced SIMPLE solver (SIMPLEC, dynamic relaxation)
- :class:`IcoFoamEnhanced` — Enhanced ICO solver (adaptive dt, Crank-Nicolson)
- :class:`RhoPimpleFoamEnhanced` — Enhanced compressible PIMPLE (coupled energy, Mach-aware)
- :class:`BuoyantSimpleFoamEnhanced` — Enhanced buoyant SIMPLE (Boussinesq, Richardson-aware)
- :class:`BuoyantPimpleFoamEnhanced` — Enhanced buoyant PIMPLE (temp-dependent relaxation)
- :class:`ReactingFoamEnhanced3` — Enhanced reacting solver v3 (stiff chemistry, Strang splitting)
- :class:`IcoFoamEnhanced2` — Enhanced ICO solver v2 (BDF2, multi-stage CFL)
- :class:`SimpleFoamEnhanced2` — Enhanced SIMPLE solver v2 (residual smoothing, adaptive switching)
- :class:`PisoFoamEnhanced2` — Enhanced PISO solver v2 (higher-order Rhie-Chow, adaptive correctors)
- :class:`PimpleFoamEnhanced2` — Enhanced PIMPLE solver v2 (SOR-Aitken, residual prediction)
- :class:`RhoPimpleFoamEnhanced2` — Enhanced compressible PIMPLE v2 (energy predictor-corrector, density correction)
- :class:`BuoyantSimpleFoamEnhanced2` — Enhanced buoyant SIMPLE v2 (implicit Boussinesq, gradient Ri)
- :class:`BuoyantPimpleFoamEnhanced2` — Enhanced buoyant PIMPLE v2 (Brunt-Vaisala limiting, T bounds)
- :class:`ReactingFoamEnhanced4` — Enhanced reacting solver v4 (topological ordering, per-species adaptive)
- :class:`SolidFoamEnhanced` — Enhanced solid mechanics (iterative thermal-mechanical, stress smoothing)
- :class:`FilmFoamEnhanced` — Enhanced thin film (disjoining pressure, adaptive dt, Cox-Voinov)
- :class:`SprayFoamEnhanced` — Enhanced spray (Reitz-Diwakar, parcels, turbulence coupling)
- :class:`MultiphaseEulerFoamEnhanced2` — Enhanced multiphase Euler v2 (QMOM, Saffman-Turner)
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
from pyfoam.applications.cht_solver import CHTSolver, CHTConfig
from pyfoam.applications.porous_inter_foam import PorousInterFoam
from pyfoam.applications.adjoint_shape_foam import AdjointShapeFoam
from pyfoam.applications.diesel_foam import DieselFoam
from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam
from pyfoam.applications.reacting_multiphase_foam import ReactingMultiphaseFoam
from pyfoam.applications.combustion_foam import CombustionFoam
from pyfoam.applications.heat_transfer_foam import HeatTransferFoam
from pyfoam.applications.viscous_foam import ViscousFoam
from pyfoam.applications.energy_foam import EnergyFoam
from pyfoam.applications.multiphase_reacting_foam import MultiphaseReactingFoam
from pyfoam.applications.acoustic_foam import AcousticFoam
from pyfoam.applications.financial_foam_2 import FinancialFoam2
from pyfoam.applications.reacting_foam_enhanced_2 import ReactingFoam2
from pyfoam.applications.compressible_inter_foam_2 import CompressibleInterFoam2
from pyfoam.applications.two_phase_euler_foam_2 import TwoPhaseEulerFoam2
from pyfoam.applications.multiphase_euler_foam_2 import MultiphaseEulerFoam2
from pyfoam.applications.solid_foam import SolidFoam
from pyfoam.applications.film_foam import FilmFoam
from pyfoam.applications.spray_foam_2 import SprayFoam2
from pyfoam.applications.piso_foam_enhanced import PisoFoamEnhanced
from pyfoam.applications.pimple_foam_enhanced import PimpleFoamEnhanced
from pyfoam.applications.simple_foam_enhanced import SimpleFoamEnhanced
from pyfoam.applications.ico_foam_enhanced import IcoFoamEnhanced
from pyfoam.applications.rho_pimple_foam_enhanced import RhoPimpleFoamEnhanced
from pyfoam.applications.buoyant_simple_foam_enhanced import BuoyantSimpleFoamEnhanced
from pyfoam.applications.buoyant_pimple_foam_enhanced import BuoyantPimpleFoamEnhanced
from pyfoam.applications.reacting_foam_enhanced_3 import ReactingFoamEnhanced3
from pyfoam.applications.ico_foam_enhanced_2 import IcoFoamEnhanced2
from pyfoam.applications.simple_foam_enhanced_2 import SimpleFoamEnhanced2
from pyfoam.applications.piso_foam_enhanced_2 import PisoFoamEnhanced2
from pyfoam.applications.pimple_foam_enhanced_2 import PimpleFoamEnhanced2
from pyfoam.applications.rho_pimple_foam_enhanced_2 import RhoPimpleFoamEnhanced2
from pyfoam.applications.buoyant_simple_foam_enhanced_2 import BuoyantSimpleFoamEnhanced2
from pyfoam.applications.buoyant_pimple_foam_enhanced_2 import BuoyantPimpleFoamEnhanced2
from pyfoam.applications.reacting_foam_enhanced_4 import ReactingFoamEnhanced4
from pyfoam.applications.solid_foam_enhanced import SolidFoamEnhanced
from pyfoam.applications.film_foam_enhanced import FilmFoamEnhanced
from pyfoam.applications.spray_foam_enhanced import SprayFoamEnhanced
from pyfoam.applications.multiphase_euler_foam_enhanced_2 import MultiphaseEulerFoamEnhanced2
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
    "CHTSolver",
    "CHTConfig",
    "PorousInterFoam",
    "AdjointShapeFoam",
    "DieselFoam",
    "AdjointTurbulenceFoam",
    "ReactingMultiphaseFoam",
    "CombustionFoam",
    "HeatTransferFoam",
    "ViscousFoam",
    "EnergyFoam",
    "MultiphaseReactingFoam",
    "AcousticFoam",
    "FinancialFoam2",
    "ReactingFoam2",
    "CompressibleInterFoam2",
    "TwoPhaseEulerFoam2",
    "MultiphaseEulerFoam2",
    "SolidFoam",
    "FilmFoam",
    "SprayFoam2",
    "PisoFoamEnhanced",
    "PimpleFoamEnhanced",
    "SimpleFoamEnhanced",
    "IcoFoamEnhanced",
    "RhoPimpleFoamEnhanced",
    "BuoyantSimpleFoamEnhanced",
    "BuoyantPimpleFoamEnhanced",
    "ReactingFoamEnhanced3",
    "IcoFoamEnhanced2",
    "SimpleFoamEnhanced2",
    "PisoFoamEnhanced2",
    "PimpleFoamEnhanced2",
    "RhoPimpleFoamEnhanced2",
    "BuoyantSimpleFoamEnhanced2",
    "BuoyantPimpleFoamEnhanced2",
    "ReactingFoamEnhanced4",
    "SolidFoamEnhanced",
    "FilmFoamEnhanced",
    "SprayFoamEnhanced",
    "MultiphaseEulerFoamEnhanced2",
    "TimeLoop",
    "ConvergenceMonitor",
]
