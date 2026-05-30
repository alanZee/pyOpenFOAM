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
- :class:`IcoFoamEnhanced3` — Enhanced ICO solver v3 (SSP-RK3, temporal error control)
- :class:`SimpleFoamEnhanced3` — Enhanced SIMPLE solver v3 (Anderson mixing, multi-level smoothing)
- :class:`PisoFoamEnhanced3` — Enhanced PISO solver v3 (skewness-corrected Rhie-Chow, momentum balance)
- :class:`PimpleFoamEnhanced3` — Enhanced PIMPLE solver v3 (Newton-Krylov, adaptive outer count)
- :class:`RhoPimpleFoamEnhanced3` — Enhanced compressible PIMPLE v3 (variable Cp, sonic-number relaxation)
- :class:`BuoyantSimpleFoamEnhanced3` — Enhanced buoyant SIMPLE v3 (quadratic Boussinesq, regime classification)
- :class:`BuoyantPimpleFoamEnhanced3` — Enhanced buoyant PIMPLE v3 (semi-implicit buoyancy, wave filtering)
- :class:`ReactingFoamEnhanced5` — Enhanced reacting solver v5 (ISAT, Troe falloff)
- :class:`SolidFoamEnhanced2` — Enhanced solid mechanics v2 (Norton-Bailey creep, fatigue)
- :class:`FilmFoamEnhanced2` — Enhanced film solver v2 (height-function curvature, spinodal)
- :class:`SprayFoamEnhanced2` — Enhanced spray solver v2 (TAB breakup, O'Rourke collision)
- :class:`MultiphaseEulerFoamEnhanced3` — Enhanced multiphase Euler v3 (IAC transport, turbulence modulation)
- :class:`IcoFoamEnhanced4` — Enhanced ICO solver v4 (Lax-Wendroff, multi-stage CFL, conservative reconstruction)
- :class:`SimpleFoamEnhanced4` — Enhanced SIMPLE solver v4 (POD acceleration, SFD, consistent flux)
- :class:`PisoFoamEnhanced4` — Enhanced PISO solver v4 (deferred correction, adaptive correctors, pressure precondition)
- :class:`PimpleFoamEnhanced4` — Enhanced PIMPLE solver v4 (multi-grid, adaptive outer-inner, BDF2-TVD)
- :class:`RhoPimpleFoamEnhanced4` — Enhanced compressible PIMPLE v4 (implicit EOS, shock-capturing, Mach-adaptive)
- :class:`BuoyantSimpleFoamEnhanced4` — Enhanced buoyant SIMPLE v4 (adaptive Boussinesq, turb-regime, stratification)
- :class:`BuoyantPimpleFoamEnhanced4` — Enhanced buoyant PIMPLE v4 (adaptive thermal, rad-buoy coupling, Co-scaled)
- :class:`ReactingFoamEnhanced6` — Enhanced reacting solver v6 (adaptive ISAT, Lie-Trotter, species sub-cycling)
- :class:`SolidFoamEnhanced3` — Enhanced solid mechanics v3 (anisotropy, thermal contact, kinetic damping)
- :class:`FilmFoamEnhanced3` — Enhanced film solver v3 (evaporation, AMR marking, wetting-drying)
- :class:`SprayFoamEnhanced3` — Enhanced spray solver v3 (Ranz-Marshall evaporation, dynamic drag, population balance)
- :class:`MultiphaseEulerFoamEnhanced4` — Enhanced multiphase Euler v4 (Zuber-Findlay drag, two-way turb, pw pressure)
- :class:`IcoFoamEnhanced5` — Enhanced ICO solver v5 (characteristic flux, error-controlled dt, momentum limiter)
- :class:`SimpleFoamEnhanced5` — Enhanced SIMPLE solver v5 (feature precondition, spectral switching, momentum conservation)
- :class:`PisoFoamEnhanced5` — Enhanced PISO solver v5 (temporal error control, anisotropic Rhie-Chow, bounded transport)
- :class:`PimpleFoamEnhanced5` — Enhanced PIMPLE solver v5 (SIMPLEC inner, adaptive coarsening, residual smoothing)
- :class:`RhoPimpleFoamEnhanced5` — Enhanced compressible PIMPLE v5 (low-Mach precondition, energy conservation, acoustic dt)
- :class:`BuoyantSimpleFoamEnhanced5` — Enhanced buoyant SIMPLE v5 (implicit buoyancy, Peclet relaxation, Robin BC)
- :class:`BuoyantPimpleFoamEnhanced5` — Enhanced buoyant PIMPLE v5 (semi-implicit buoyancy, buoyancy dt, TKE production)
- :class:`ReactingFoamEnhanced7` — Enhanced reacting solver v7 (JFNK coupling, mass-consistent velocity, adaptive refinement)
- :class:`SolidFoamEnhanced4` — Enhanced solid mechanics v4 (block-Gauss-Seidel, failure criterion, element deletion)
- :class:`FilmFoamEnhanced4` — Enhanced film solver v4 (surfactant transport, Marangoni flow, rupture dynamics)
- :class:`SprayFoamEnhanced4` — Enhanced spray solver v4 (multi-component evaporation, KH-RT-REF, turbulent dispersion)
- :class:`MultiphaseEulerFoamEnhanced5` — Enhanced multiphase Euler v5 (poly-dispersed, per-phase turbulence, boundedness)
- :class:`IcoFoamEnhanced6` — Enhanced ICO solver v6 (vorticity stabilisation, compact reconstruction, spectral-element time)
- :class:`SimpleFoamEnhanced6` — Enhanced SIMPLE solver v6 (tensorial viscosity, pseudo-transient, residual weighting)
- :class:`PisoFoamEnhanced6` — Enhanced PISO solver v6 (adaptive correctors, entropy-stable flux, compact Rhie-Chow)
- :class:`PimpleFoamEnhanced6` — Enhanced PIMPLE solver v6 (back-substitution, residual scaling, POD pressure)
- :class:`RhoPimpleFoamEnhanced6` — Enhanced compressible PIMPLE v6 (density-velocity coupling, baroclinic torque, entropy variables)
- :class:`BuoyantSimpleFoamEnhanced6` — Enhanced buoyant SIMPLE v6 (strong coupling, energy-momentum interchange, GGDH)
- :class:`BuoyantPimpleFoamEnhanced6` — Enhanced buoyant PIMPLE v6 (projection split, gravity-wave filter, coupled k-epsilon)
- :class:`ReactingFoamEnhanced8` — Enhanced reacting solver v8 (DRG reduction, NN time stepping, WENO transport)
- :class:`SolidFoamEnhanced5` — Enhanced solid mechanics v5 (phase-field fracture, HMM, domain-decomposition)
- :class:`FilmFoamEnhanced5` — Enhanced film solver v5 (foam drainage, thermal viscosity, non-Newtonian rheology)
- :class:`SprayFoamEnhanced5` — Enhanced spray solver v5 (multi-physics parcels, ML collision, spray-combustion)
- :class:`MultiphaseEulerFoamEnhanced6` — Enhanced multiphase Euler v6 (phase-resolved LES, interfacial forces, adaptive QMOM)
- :class:`IcoFoamEnhanced7` — Enhanced ICO solver v7 (wavelet AMR, energy-stable convection, Schur preconditioner)
- :class:`SimpleFoamEnhanced7` — Enhanced SIMPLE solver v7 (VMS turbulence, Anderson restart, convex splitting)
- :class:`PisoFoamEnhanced7` — Enhanced PISO solver v7 (dual-weighted error, CMI interpolation, Hessian precondition)
- :class:`PimpleFoamEnhanced7` — Enhanced PIMPLE solver v7 (block-coupled, semi-implicit, hierarchical multi-grid)
- :class:`RhoPimpleFoamEnhanced7` — Enhanced compressible PIMPLE v7 (pressure density, acoustic splitting, energy switching)
- :class:`BuoyantSimpleFoamEnhanced7` — Enhanced buoyant SIMPLE v7 (quadratic Boussinesq, overset, radiation acceleration)
- :class:`BuoyantPimpleFoamEnhanced7` — Enhanced buoyant PIMPLE v7 (implicit buoyancy, thermal LES, adaptive BL)
- :class:`ReactingFoamEnhanced9` — Enhanced reacting solver v9 (adaptive splitting, NTC, block-Jacobi)
- :class:`SolidFoamEnhanced6` — Enhanced solid mechanics v6 (XFEM enrichment, coupled fatigue, MLPG recovery)
- :class:`FilmFoamEnhanced6` — Enhanced film solver v6 (EHD destabilisation, phase-change, viscoelastic rheology)
- :class:`SprayFoamEnhanced6` — Enhanced spray solver v6 (stochastic breakup, electrostatic charging, wall film)
- :class:`MultiphaseEulerFoamEnhanced7` — Enhanced multiphase Euler v7 (poly-dispersed IAC, DES turbulence, coupled PV)
- :class:`IcoFoamEnhanced8` — Enhanced ICO solver v8 (metric adaptation, BFBt preconditioner, semi-Lagrangian)
- :class:`SimpleFoamEnhanced8` — Enhanced SIMPLE solver v8 (JFNK acceleration, spectral preconditioner, Armijo line search)
- :class:`PisoFoamEnhanced8` — Enhanced PISO solver v8 (embedded RK, skew-symmetric advection, GMRES pressure)
- :class:`PimpleFoamEnhanced8` — Enhanced PIMPLE solver v8 (OIF stepping, adaptive AMG, SIMPLENGA)
- :class:`RhoPimpleFoamEnhanced8` — Enhanced compressible PIMPLE v8 (JST dissipation, dual-time, mixture-averaged)
- :class:`BuoyantSimpleFoamEnhanced8` — Enhanced buoyant SIMPLE v8 (variable Boussinesq, conjugate HTC, Rossby switching)
- :class:`BuoyantPimpleFoamEnhanced8` — Enhanced buoyant PIMPLE v8 (density precondition, entropy-stable thermal, GW CFL)
- :class:`ReactingFoamEnhanced10` — Enhanced reacting solver v10 (HAK hierarchy, IMEX integration, ML combustion)
- :class:`SolidFoamEnhanced7` — Enhanced solid mechanics v7 (spectral damage, topology optimisation, implicit contact)
- :class:`FilmFoamEnhanced7` — Enhanced film solver v7 (Cahn-Hilliard, Marangoni, DLVO disjoining pressure)
- :class:`SprayFoamEnhanced7` — Enhanced spray solver v7 (spray AMR, DNS-calibrated breakup, Langevin dispersion)
- :class:`MultiphaseEulerFoamEnhanced8` — Enhanced multiphase Euler v8 (hyperbolic moments, filtered LES forces, implicit coupling)
- :class:`IcoFoamEnhanced9` — Enhanced ICO solver v9 (NN preconditioner, p-refinement, symplectic integrator)
- :class:`SimpleFoamEnhanced9` — Enhanced SIMPLE solver v9 (reduced-basis acceleration, PAUR, anisotropic diffusion)
- :class:`PisoFoamEnhanced9` — Enhanced PISO solver v9 (wavelet dt, compact coupling, entropy-viscosity)
- :class:`PimpleFoamEnhanced9` — Enhanced PIMPLE solver v9 (PINO stepping, tensor-train pressure, adaptive linearisation)
- :class:`RhoPimpleFoamEnhanced9` — Enhanced compressible PIMPLE v9 (compact WENO, AP IMEX, real-gas EOS)
- :class:`BuoyantSimpleFoamEnhanced9` — Enhanced buoyant SIMPLE v9 (buoyant LES, anisotropic pressure, DO radiation)
- :class:`BuoyantPimpleFoamEnhanced9` — Enhanced buoyant PIMPLE v9 (Boussinesq filter, BTIM, adaptive BL)
- :class:`ReactingFoamEnhanced11` — Enhanced reacting solver v11 (TPDF closure, dynamic ISAT, TFM)
- :class:`SolidFoamEnhanced8` — Enhanced solid mechanics v8 (CZM crack, geometric nonlinearity, CDM fatigue)
- :class:`FilmFoamEnhanced8` — Enhanced film solver v8 (slip BC, surfactant dynamics, film AMR)
- :class:`SprayFoamEnhanced8` — Enhanced spray solver v8 (multicomponent evaporation, CT coalescence, LES coupling)
- :class:`MultiphaseEulerFoamEnhanced9` — Enhanced multiphase Euler v9 (QMOM population balance, Antal wall-lub, turb modulation)
- :class:`IcoFoamEnhanced10` — Enhanced ICO solver v10 (h-multigrid preconditioner, space-time Galerkin, adaptive artificial compressibility)
- :class:`SimpleFoamEnhanced10` — Enhanced SIMPLE solver v10 (operator-learning pressure, spectral viscosity, data-driven under-relaxation)
- :class:`PisoFoamEnhanced10` — Enhanced PISO solver v10 (ILES-MPDATA, pressure-Hodge projection, multirate time stepping)
- :class:`PimpleFoamEnhanced10` — Enhanced PIMPLE solver v10 (VMS pressure, OINN corrector, energy-budget preservation)
- :class:`RhoPimpleFoamEnhanced10` — Enhanced compressible PIMPLE v10 (hybrid RANS-DES, thermodynamic consistency, acoustic-hybrid)
- :class:`BuoyantSimpleFoamEnhanced10` — Enhanced buoyant SIMPLE v10 (non-Boussinesq, shell HTC, Richardson damping)
- :class:`BuoyantPimpleFoamEnhanced10` — Enhanced buoyant PIMPLE v10 (CBPVS block solve, RBTIM triple coupling, temporal filtering)
- :class:`ReactingFoamEnhanced12` — Enhanced reacting solver v12 (MLRC autoencoder, FGM tabulation, MOM soot)
- :class:`SolidFoamEnhanced9` — Enhanced solid mechanics v9 (peridynamics, crystal plasticity, natural element interpolation)
- :class:`FilmFoamEnhanced9` — Enhanced film solver v9 (inertial lubrication, wetting-drying precursor, Marangoni instability)
- :class:`SprayFoamEnhanced9` — Enhanced spray solver v9 (APMS parcel management, DNS-informed drag, Langevin-memory dispersion)
- :class:`MultiphaseEulerFoamEnhanced10` — Enhanced multiphase Euler v10 (MUSIG size classes, BIRS Reynolds stress, IATE interfacial area)
- :class:`SimpleFoamEnhanced11` — Enhanced SIMPLE solver v11 (extended non-orthogonal correction, consistent non-orthogonal correction, over-relaxed stabilisation)
- :class:`PimpleFoamEnhanced11` — Enhanced PIMPLE solver v11 (extended non-orthogonal pressure, consistent non-orthogonal momentum, over-relaxed stabilisation)
- :class:`PisoFoamEnhanced11` — Enhanced PISO solver v11 (extended non-orthogonal projection, consistent Rhie-Chow, over-relaxed stabilisation)
- :class:`IcoFoamEnhanced11` — Enhanced ICO solver v11 (extended non-orthogonal Poisson, consistent velocity-pressure coupling, over-relaxed stabilisation)
- :class:`BuoyantPimpleFoamEnhanced11` — Enhanced buoyant PIMPLE v11 (extended buoyancy-pressure correction, consistent thermal-momentum, over-relaxed stabilisation)
- :class:`BuoyantSimpleFoamEnhanced11` — Enhanced buoyant SIMPLE v11 (extended buoyant-pressure correction, consistent buoyant-momentum, over-relaxed stabilisation)
- :class:`CompressibleInterFoamEnhanced11` — Enhanced compressible VOF v11 (extended VOF pressure correction, consistent phase-fraction, over-relaxed stabilisation)
- :class:`SprayFoamEnhanced11` — Enhanced spray solver v11 (extended spray-pressure correction, consistent parcel-velocity, over-relaxed stabilisation)
- :class:`MultiphaseEulerFoamEnhanced11` — Enhanced multiphase Euler v11 (extended multi-phase pressure, consistent phase-momentum, over-relaxed stabilisation)
- :class:`ReactingFoamEnhanced13` — Enhanced reacting solver v13 (SIMPLEC-consistent coupling, coupled reacting system, pressure-velocity-chemistry coupling)
- :class:`SimpleFoamEnhanced12` — Enhanced SIMPLE solver v12 (adaptive under-relaxation, Aitken acceleration, field-based relaxation)
- :class:`PimpleFoamEnhanced12` — Enhanced PIMPLE solver v12 (adaptive under-relaxation, Aitken acceleration, field-based relaxation)
- :class:`PisoFoamEnhanced12` — Enhanced PISO solver v12 (adaptive under-relaxation, Aitken acceleration, field-based relaxation)
- :class:`IcoFoamEnhanced12` — Enhanced ICO solver v12 (adaptive dt relaxation, Aitken correction, field-based relaxation)
- :class:`BuoyantPimpleFoamEnhanced12` — Enhanced buoyant PIMPLE v12 (adaptive under-relaxation, Aitken correction, field-based relaxation)
- :class:`BuoyantSimpleFoamEnhanced12` — Enhanced buoyant SIMPLE v12 (adaptive under-relaxation, Aitken correction, field-based relaxation)
- :class:`CompressibleInterFoamEnhanced12` — Enhanced compressible VOF v12 (adaptive under-relaxation, Aitken correction, field-based relaxation)
- :class:`SprayFoamEnhanced12` — Enhanced spray solver v12 (adaptive under-relaxation, Aitken correction, field-based relaxation)
- :class:`MultiphaseEulerFoamEnhanced12` — Enhanced multiphase Euler v12 (adaptive under-relaxation, Aitken phase correction, field-based relaxation)
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
from pyfoam.applications.ico_foam_enhanced_3 import IcoFoamEnhanced3
from pyfoam.applications.simple_foam_enhanced_3 import SimpleFoamEnhanced3
from pyfoam.applications.piso_foam_enhanced_3 import PisoFoamEnhanced3
from pyfoam.applications.pimple_foam_enhanced_3 import PimpleFoamEnhanced3
from pyfoam.applications.rho_pimple_foam_enhanced_3 import RhoPimpleFoamEnhanced3
from pyfoam.applications.buoyant_simple_foam_enhanced_3 import BuoyantSimpleFoamEnhanced3
from pyfoam.applications.buoyant_pimple_foam_enhanced_3 import BuoyantPimpleFoamEnhanced3
from pyfoam.applications.reacting_foam_enhanced_5 import ReactingFoamEnhanced5
from pyfoam.applications.solid_foam_enhanced_2 import SolidFoamEnhanced2
from pyfoam.applications.film_foam_enhanced_2 import FilmFoamEnhanced2
from pyfoam.applications.spray_foam_enhanced_2 import SprayFoamEnhanced2
from pyfoam.applications.multiphase_euler_foam_enhanced_3 import MultiphaseEulerFoamEnhanced3
from pyfoam.applications.ico_foam_enhanced_4 import IcoFoamEnhanced4
from pyfoam.applications.simple_foam_enhanced_4 import SimpleFoamEnhanced4
from pyfoam.applications.piso_foam_enhanced_4 import PisoFoamEnhanced4
from pyfoam.applications.pimple_foam_enhanced_4 import PimpleFoamEnhanced4
from pyfoam.applications.rho_pimple_foam_enhanced_4 import RhoPimpleFoamEnhanced4
from pyfoam.applications.buoyant_simple_foam_enhanced_4 import BuoyantSimpleFoamEnhanced4
from pyfoam.applications.buoyant_pimple_foam_enhanced_4 import BuoyantPimpleFoamEnhanced4
from pyfoam.applications.reacting_foam_enhanced_6 import ReactingFoamEnhanced6
from pyfoam.applications.solid_foam_enhanced_3 import SolidFoamEnhanced3
from pyfoam.applications.film_foam_enhanced_3 import FilmFoamEnhanced3
from pyfoam.applications.spray_foam_enhanced_3 import SprayFoamEnhanced3
from pyfoam.applications.multiphase_euler_foam_enhanced_4 import MultiphaseEulerFoamEnhanced4
from pyfoam.applications.ico_foam_enhanced_5 import IcoFoamEnhanced5
from pyfoam.applications.simple_foam_enhanced_5 import SimpleFoamEnhanced5
from pyfoam.applications.piso_foam_enhanced_5 import PisoFoamEnhanced5
from pyfoam.applications.pimple_foam_enhanced_5 import PimpleFoamEnhanced5
from pyfoam.applications.rho_pimple_foam_enhanced_5 import RhoPimpleFoamEnhanced5
from pyfoam.applications.buoyant_simple_foam_enhanced_5 import BuoyantSimpleFoamEnhanced5
from pyfoam.applications.buoyant_pimple_foam_enhanced_5 import BuoyantPimpleFoamEnhanced5
from pyfoam.applications.reacting_foam_enhanced_7 import ReactingFoamEnhanced7
from pyfoam.applications.solid_foam_enhanced_4 import SolidFoamEnhanced4
from pyfoam.applications.film_foam_enhanced_4 import FilmFoamEnhanced4
from pyfoam.applications.spray_foam_enhanced_4 import SprayFoamEnhanced4
from pyfoam.applications.multiphase_euler_foam_enhanced_5 import MultiphaseEulerFoamEnhanced5
from pyfoam.applications.ico_foam_enhanced_6 import IcoFoamEnhanced6
from pyfoam.applications.simple_foam_enhanced_6 import SimpleFoamEnhanced6
from pyfoam.applications.piso_foam_enhanced_6 import PisoFoamEnhanced6
from pyfoam.applications.pimple_foam_enhanced_6 import PimpleFoamEnhanced6
from pyfoam.applications.rho_pimple_foam_enhanced_6 import RhoPimpleFoamEnhanced6
from pyfoam.applications.buoyant_simple_foam_enhanced_6 import BuoyantSimpleFoamEnhanced6
from pyfoam.applications.buoyant_pimple_foam_enhanced_6 import BuoyantPimpleFoamEnhanced6
from pyfoam.applications.reacting_foam_enhanced_8 import ReactingFoamEnhanced8
from pyfoam.applications.solid_foam_enhanced_5 import SolidFoamEnhanced5
from pyfoam.applications.film_foam_enhanced_5 import FilmFoamEnhanced5
from pyfoam.applications.spray_foam_enhanced_5 import SprayFoamEnhanced5
from pyfoam.applications.multiphase_euler_foam_enhanced_6 import MultiphaseEulerFoamEnhanced6
from pyfoam.applications.ico_foam_enhanced_7 import IcoFoamEnhanced7
from pyfoam.applications.simple_foam_enhanced_7 import SimpleFoamEnhanced7
from pyfoam.applications.piso_foam_enhanced_7 import PisoFoamEnhanced7
from pyfoam.applications.pimple_foam_enhanced_7 import PimpleFoamEnhanced7
from pyfoam.applications.rho_pimple_foam_enhanced_7 import RhoPimpleFoamEnhanced7
from pyfoam.applications.buoyant_simple_foam_enhanced_7 import BuoyantSimpleFoamEnhanced7
from pyfoam.applications.buoyant_pimple_foam_enhanced_7 import BuoyantPimpleFoamEnhanced7
from pyfoam.applications.reacting_foam_enhanced_9 import ReactingFoamEnhanced9
from pyfoam.applications.solid_foam_enhanced_6 import SolidFoamEnhanced6
from pyfoam.applications.film_foam_enhanced_6 import FilmFoamEnhanced6
from pyfoam.applications.spray_foam_enhanced_6 import SprayFoamEnhanced6
from pyfoam.applications.multiphase_euler_foam_enhanced_7 import MultiphaseEulerFoamEnhanced7
from pyfoam.applications.ico_foam_enhanced_8 import IcoFoamEnhanced8
from pyfoam.applications.simple_foam_enhanced_8 import SimpleFoamEnhanced8
from pyfoam.applications.piso_foam_enhanced_8 import PisoFoamEnhanced8
from pyfoam.applications.pimple_foam_enhanced_8 import PimpleFoamEnhanced8
from pyfoam.applications.rho_pimple_foam_enhanced_8 import RhoPimpleFoamEnhanced8
from pyfoam.applications.buoyant_simple_foam_enhanced_8 import BuoyantSimpleFoamEnhanced8
from pyfoam.applications.buoyant_pimple_foam_enhanced_8 import BuoyantPimpleFoamEnhanced8
from pyfoam.applications.reacting_foam_enhanced_10 import ReactingFoamEnhanced10
from pyfoam.applications.solid_foam_enhanced_7 import SolidFoamEnhanced7
from pyfoam.applications.film_foam_enhanced_7 import FilmFoamEnhanced7
from pyfoam.applications.spray_foam_enhanced_7 import SprayFoamEnhanced7
from pyfoam.applications.multiphase_euler_foam_enhanced_8 import MultiphaseEulerFoamEnhanced8
from pyfoam.applications.ico_foam_enhanced_9 import IcoFoamEnhanced9
from pyfoam.applications.simple_foam_enhanced_9 import SimpleFoamEnhanced9
from pyfoam.applications.piso_foam_enhanced_9 import PisoFoamEnhanced9
from pyfoam.applications.pimple_foam_enhanced_9 import PimpleFoamEnhanced9
from pyfoam.applications.rho_pimple_foam_enhanced_9 import RhoPimpleFoamEnhanced9
from pyfoam.applications.buoyant_simple_foam_enhanced_9 import BuoyantSimpleFoamEnhanced9
from pyfoam.applications.buoyant_pimple_foam_enhanced_9 import BuoyantPimpleFoamEnhanced9
from pyfoam.applications.reacting_foam_enhanced_11 import ReactingFoamEnhanced11
from pyfoam.applications.solid_foam_enhanced_8 import SolidFoamEnhanced8
from pyfoam.applications.film_foam_enhanced_8 import FilmFoamEnhanced8
from pyfoam.applications.spray_foam_enhanced_8 import SprayFoamEnhanced8
from pyfoam.applications.multiphase_euler_foam_enhanced_9 import MultiphaseEulerFoamEnhanced9
from pyfoam.applications.ico_foam_enhanced_10 import IcoFoamEnhanced10
from pyfoam.applications.simple_foam_enhanced_10 import SimpleFoamEnhanced10
from pyfoam.applications.piso_foam_enhanced_10 import PisoFoamEnhanced10
from pyfoam.applications.pimple_foam_enhanced_10 import PimpleFoamEnhanced10
from pyfoam.applications.rho_pimple_foam_enhanced_10 import RhoPimpleFoamEnhanced10
from pyfoam.applications.buoyant_simple_foam_enhanced_10 import BuoyantSimpleFoamEnhanced10
from pyfoam.applications.buoyant_pimple_foam_enhanced_10 import BuoyantPimpleFoamEnhanced10
from pyfoam.applications.reacting_foam_enhanced_12 import ReactingFoamEnhanced12
from pyfoam.applications.solid_foam_enhanced_9 import SolidFoamEnhanced9
from pyfoam.applications.film_foam_enhanced_9 import FilmFoamEnhanced9
from pyfoam.applications.spray_foam_enhanced_9 import SprayFoamEnhanced9
from pyfoam.applications.multiphase_euler_foam_enhanced_10 import MultiphaseEulerFoamEnhanced10
from pyfoam.applications.simple_foam_enhanced_11 import SimpleFoamEnhanced11
from pyfoam.applications.pimple_foam_enhanced_11 import PimpleFoamEnhanced11
from pyfoam.applications.piso_foam_enhanced_11 import PisoFoamEnhanced11
from pyfoam.applications.ico_foam_enhanced_11 import IcoFoamEnhanced11
from pyfoam.applications.buoyant_pimple_foam_enhanced_11 import BuoyantPimpleFoamEnhanced11
from pyfoam.applications.buoyant_simple_foam_enhanced_11 import BuoyantSimpleFoamEnhanced11
from pyfoam.applications.compressible_inter_foam_enhanced_11 import CompressibleInterFoamEnhanced11
from pyfoam.applications.spray_foam_enhanced_11 import SprayFoamEnhanced11
from pyfoam.applications.multiphase_euler_foam_enhanced_11 import MultiphaseEulerFoamEnhanced11
from pyfoam.applications.reacting_foam_enhanced_13 import ReactingFoamEnhanced13
from pyfoam.applications.simple_foam_enhanced_12 import SimpleFoamEnhanced12
from pyfoam.applications.pimple_foam_enhanced_12 import PimpleFoamEnhanced12
from pyfoam.applications.piso_foam_enhanced_12 import PisoFoamEnhanced12
from pyfoam.applications.ico_foam_enhanced_12 import IcoFoamEnhanced12
from pyfoam.applications.buoyant_pimple_foam_enhanced_12 import BuoyantPimpleFoamEnhanced12
from pyfoam.applications.buoyant_simple_foam_enhanced_12 import BuoyantSimpleFoamEnhanced12
from pyfoam.applications.compressible_inter_foam_enhanced_12 import CompressibleInterFoamEnhanced12
from pyfoam.applications.spray_foam_enhanced_12 import SprayFoamEnhanced12
from pyfoam.applications.multiphase_euler_foam_enhanced_12 import MultiphaseEulerFoamEnhanced12
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
    "IcoFoamEnhanced3",
    "SimpleFoamEnhanced3",
    "PisoFoamEnhanced3",
    "PimpleFoamEnhanced3",
    "RhoPimpleFoamEnhanced3",
    "BuoyantSimpleFoamEnhanced3",
    "BuoyantPimpleFoamEnhanced3",
    "ReactingFoamEnhanced5",
    "SolidFoamEnhanced2",
    "FilmFoamEnhanced2",
    "SprayFoamEnhanced2",
    "MultiphaseEulerFoamEnhanced3",
    "IcoFoamEnhanced4",
    "SimpleFoamEnhanced4",
    "PisoFoamEnhanced4",
    "PimpleFoamEnhanced4",
    "RhoPimpleFoamEnhanced4",
    "BuoyantSimpleFoamEnhanced4",
    "BuoyantPimpleFoamEnhanced4",
    "ReactingFoamEnhanced6",
    "SolidFoamEnhanced3",
    "FilmFoamEnhanced3",
    "SprayFoamEnhanced3",
    "MultiphaseEulerFoamEnhanced4",
    "IcoFoamEnhanced5",
    "SimpleFoamEnhanced5",
    "PisoFoamEnhanced5",
    "PimpleFoamEnhanced5",
    "RhoPimpleFoamEnhanced5",
    "BuoyantSimpleFoamEnhanced5",
    "BuoyantPimpleFoamEnhanced5",
    "ReactingFoamEnhanced7",
    "SolidFoamEnhanced4",
    "FilmFoamEnhanced4",
    "SprayFoamEnhanced4",
    "MultiphaseEulerFoamEnhanced5",
    "IcoFoamEnhanced6",
    "SimpleFoamEnhanced6",
    "PisoFoamEnhanced6",
    "PimpleFoamEnhanced6",
    "RhoPimpleFoamEnhanced6",
    "BuoyantSimpleFoamEnhanced6",
    "BuoyantPimpleFoamEnhanced6",
    "ReactingFoamEnhanced8",
    "SolidFoamEnhanced5",
    "FilmFoamEnhanced5",
    "SprayFoamEnhanced5",
    "MultiphaseEulerFoamEnhanced6",
    "IcoFoamEnhanced7",
    "SimpleFoamEnhanced7",
    "PisoFoamEnhanced7",
    "PimpleFoamEnhanced7",
    "RhoPimpleFoamEnhanced7",
    "BuoyantSimpleFoamEnhanced7",
    "BuoyantPimpleFoamEnhanced7",
    "ReactingFoamEnhanced9",
    "SolidFoamEnhanced6",
    "FilmFoamEnhanced6",
    "SprayFoamEnhanced6",
    "MultiphaseEulerFoamEnhanced7",
    "IcoFoamEnhanced8",
    "SimpleFoamEnhanced8",
    "PisoFoamEnhanced8",
    "PimpleFoamEnhanced8",
    "RhoPimpleFoamEnhanced8",
    "BuoyantSimpleFoamEnhanced8",
    "BuoyantPimpleFoamEnhanced8",
    "ReactingFoamEnhanced10",
    "SolidFoamEnhanced7",
    "FilmFoamEnhanced7",
    "SprayFoamEnhanced7",
    "MultiphaseEulerFoamEnhanced8",
    "IcoFoamEnhanced9",
    "SimpleFoamEnhanced9",
    "PisoFoamEnhanced9",
    "PimpleFoamEnhanced9",
    "RhoPimpleFoamEnhanced9",
    "BuoyantSimpleFoamEnhanced9",
    "BuoyantPimpleFoamEnhanced9",
    "ReactingFoamEnhanced11",
    "SolidFoamEnhanced8",
    "FilmFoamEnhanced8",
    "SprayFoamEnhanced8",
    "MultiphaseEulerFoamEnhanced9",
    "IcoFoamEnhanced10",
    "SimpleFoamEnhanced10",
    "PisoFoamEnhanced10",
    "PimpleFoamEnhanced10",
    "RhoPimpleFoamEnhanced10",
    "BuoyantSimpleFoamEnhanced10",
    "BuoyantPimpleFoamEnhanced10",
    "ReactingFoamEnhanced12",
    "SolidFoamEnhanced9",
    "FilmFoamEnhanced9",
    "SprayFoamEnhanced9",
    "MultiphaseEulerFoamEnhanced10",
    "SimpleFoamEnhanced11",
    "PimpleFoamEnhanced11",
    "PisoFoamEnhanced11",
    "IcoFoamEnhanced11",
    "BuoyantPimpleFoamEnhanced11",
    "BuoyantSimpleFoamEnhanced11",
    "CompressibleInterFoamEnhanced11",
    "SprayFoamEnhanced11",
    "MultiphaseEulerFoamEnhanced11",
    "ReactingFoamEnhanced13",
    "SimpleFoamEnhanced12",
    "PimpleFoamEnhanced12",
    "PisoFoamEnhanced12",
    "IcoFoamEnhanced12",
    "BuoyantPimpleFoamEnhanced12",
    "BuoyantSimpleFoamEnhanced12",
    "CompressibleInterFoamEnhanced12",
    "SprayFoamEnhanced12",
    "MultiphaseEulerFoamEnhanced12",
    "TimeLoop",
    "ConvergenceMonitor",
]
