"""
pyfoam.boundary — Boundary condition hierarchy with RTS selection.

Provides:

- :class:`BoundaryCondition` — abstract base class with RTS registry
- :class:`BoundaryField` — collection of BCs for a field
- Concrete BCs: fixedValue, zeroGradient, noSlip, fixedGradient,
  symmetry, symmetryPlane, cyclic, nutkWallFunction, nutLowReWallFunction,
  nutUWallFunction, nutURoughWallFunction, nutUSpaldingWallFunction,
  kqRWallFunction, epsilonWallFunction, omegaWallFunction, inletOutlet,
  alphaContactAngle, constantAlphaContactAngle
- Velocity BCs: flowRateInletVelocity, pressureInletOutletVelocity,
  rotatingWallVelocity
- Pressure BCs: totalPressure, fixedFluxPressure, prghPressure,
  waveTransmissive
- Turbulence BCs: turbulentIntensityKineticEnergyInlet,
  turbulentMixingLengthDissipationRateInlet,
  turbulentMixingLengthFrequencyInlet
- Energy BCs: fixedEnergy, gradientEnergy, mixedEnergy
- Buoyancy BCs: buoyantPressure
- Turbulent inlet: turbulentInlet
- Matched flow rate: matchedFlowRateOutlet
- Fixed shear stress: fixedShearStress
- fixedNormal: fixes normal component of vector field
- slip: free-slip wall (removes normal, preserves tangential)
- pressureInletOutlet: pressure-based inlet/outlet
- volumeFlowRate: volume-flow-rate based velocity BC
- massFlowRate: mass-flow-rate based velocity BC
- timeVarying: time-varying table-interpolated value BC
- coded: user-defined Python function BC
- gradientEnergy: gradient-based energy BC (in gradient_energy module)
- oscillatingVelocity: oscillating velocity inlet (sinusoidal)
- turbulentMixingLength: general-purpose mixing length turbulence BC
- wallLubrication: wall lubrication force for multiphase Euler-Euler
- dragCoefficient: drag coefficient BC for particle-laden flows
- scaledVelocityInlet: scaled velocity inlet from reference patch
- movingWall: moving wall with translation + rotation
- turbulentIntensityKE: turbulent intensity-based kinetic energy inlet
- fixedTurbulentKE: fixed turbulent kinetic energy
- antalWallLubrication: Antal distance-dependent wall lubrication force
- fixedFluxPressure2: Enhanced fixed flux pressure with buoyancy correction
- inletOutlet3: Enhanced inlet/outlet with turbulence-aware treatment
- tomiyamaWallLubrication: Tomiyama Eo-dependent wall lubrication force for bubble flows
- interfacialMomentum: Interfacial momentum transfer for Euler-Euler multiphase
- nonLinearRoughness: Non-linear roughness wall function with sand-grain model
- waveGeneration: Wave generation inlet using Airy wave theory
- saturatedTemperature: Saturation temperature from pressure (Clausius-Clapeyron)
- subcooling: Subcooling temperature for boiling/condensation models
- stagnationInlet: Stagnation pressure inlet (Bernoulli velocity)
- symmetrySlip: Combined symmetry/slip with plane validation
- adjointInlet: Adjoint velocity inlet for adjoint optimization solvers
- wallFilm: Wall film boundary for Lagrangian film models
- phaseMeanVelocity: Phase-mean velocity for Euler-Euler multiphase models
- convectiveHeatTransfer: Robin BC for conjugate heat transfer
- generic2: Enhanced generic BC with time-varying coefficients and blending functions
- pressureNormalInlet: Pressure inlet with normal velocity for compressible flows
- scaledHeatFlux: Scaled heat flux BC (q = scale * q_ref)
- outletPhaseMeanVelocity: Outlet phase mean velocity for Euler-Euler multiphase
- heatExchanger: Multi-zone heat exchanger with effectiveness-NTU model
- turbulentInlet2: Enhanced turbulent inlet with digital filter method
- mappedVelocityInternal: Maps velocity from internal field to boundary
- variableHeight2: Enhanced variable height with momentum-consistent velocity correction
- tomiyamaWallLubrication2: Enhanced Tomiyama Eo-dependent + distance-dependent wall lubrication
- interfacialHeatTransfer: Interfacial heat transfer for boiling/condensation at interfaces
- antalWallLubrication2: Enhanced Antal wall lubrication with exponent and interface damping
- pressureInletOutlet2: Enhanced pressure inlet/outlet with turbulence-aware treatment
- coupledVelocity: Coupled velocity BC for conjugate heat transfer interfaces
- pressureTransmissive: Transmissive (non-reflecting) pressure BC for outflow
- pressureDirectedInletVelocity: Velocity inlet driven by pressure gradient (Bernoulli)
- swirlFlowRateInletVelocity: Flow rate inlet with superimposed swirl component
- mappedConvectiveHeatTransfer: Mapped conjugate heat transfer with coupled patch temperature
- uniformTotalPressure: Uniform total pressure BC (p_total = p + 0.5*rho*|U|^2)
- directedInletOutlet: Inlet/outlet with directed velocity direction
- externalCoupled: Coupled with external solver via file-based data exchange
- mixedTemperature: Mixed (Robin) temperature BC blending fixed value and gradient
- turbulentTemperatureCoupled: Coupled temperature with turbulent thermal diffusivity
- mappedVelocityAdjustedPressure: Adjusts velocity from mapped coupled patch pressure
- flowRateInletVelocity2: Enhanced flow rate inlet with power-law profile
- pressureInletOutletVelocity2: Enhanced pressure inlet/outlet with direction blending
- surfaceNormalFixedValue2: Enhanced surface normal fixed value with tangential preservation
- turbulentKineticEnergyInlet: k = 1.5 * (I * |U|)^2 inlet
- turbulentDissipationRateInlet: epsilon = C_mu^0.75 * k^1.5 / l_mix inlet
- turbulentSpecificDissipationRateInlet: omega = k^0.5 / (C_mu^0.25 * l_mix) inlet
- compressibleTurbulentTemperatureCoupled: Compressible coupled temperature with turbulent diffusivity
- waveTransmissive2: Enhanced wave transmissive with NSCBC blending
- advectiveDiffusive: Combined advective-diffusive outflow
- pressureInterpolationAMG: AMG-stabilised pressure interpolation
- codedFixedValue: User-coded expression evaluated as Dirichlet BC
- cyclicAMI2: Enhanced cyclic AMI with conservation and non-orthogonal correction
- processorCyclic: Processor cyclic for parallel decompositions with coordinate transforms
- mappedFlowRate: Mapped mass flow rate from coupled patch
- pressureWaveTransmissive: Non-reflecting wave transmissive pressure BC
- turbulentViscosityInlet: Turbulent viscosity inlet (nut = C_mu * k^2 / epsilon)
- turbulentLengthScaleInlet: Turbulent length scale inlet (l_mix = C_mu^0.75 * k^1.5 / epsilon)
- turbulentIntensityInlet: Turbulent intensity inlet (k = 1.5 * (I * |U|)^2)
- turbulentDissipationInlet: Turbulent dissipation inlet with explicit k/l_mix override
- turbulentFrequencyInlet: Turbulent frequency inlet with explicit k/l_mix override
- turbulentKineticEnergyInlet2: Enhanced k inlet with intensity and length scale clamping
- turbulentDissipationInlet2: Enhanced epsilon inlet from intensity and length scale
- turbulentFrequencyInlet2: Enhanced omega inlet from intensity and length scale
- mappedFlowRate2: Enhanced mapped flow rate with power-law radial profile
- pressureWaveTransmissive2: Enhanced wave transmissive with NSCBC blending
- turbulentViscosityInlet2: Enhanced turbulent viscosity inlet with clamping
- turbulentLengthScaleInlet2: Enhanced turbulent length scale inlet with clamping
- turbulentIntensityInlet2: Enhanced turbulent intensity inlet with clamping
- turbulentKineticEnergyInlet3: v3 enhanced k inlet with blended intensity/length-scale
- turbulentDissipationInlet3: v3 enhanced epsilon inlet with blending
- turbulentFrequencyInlet3: v3 enhanced omega inlet with blending
- mappedFlowRate3: Enhanced mapped flow rate with adaptive profile and iterative correction
- pressureWaveTransmissive3: Enhanced wave transmissive with Mach-dependent blending and turbulent damping
- turbulentViscosityInlet3: Enhanced turbulent viscosity inlet with ratio limiter and blending
- turbulentLengthScaleInlet3: Enhanced turbulent length scale inlet with hybrid blending
- turbulentIntensityInlet3: Enhanced turbulent intensity inlet with adaptive Re-based scaling
- turbulentKineticEnergyInlet4: v4 enhanced k inlet with adaptive Re-based blending and clamping
- turbulentDissipationInlet4: v4 enhanced epsilon inlet with adaptive blending and clamping
- turbulentFrequencyInlet4: v4 enhanced omega inlet with adaptive blending and clamping
- turbulentDissipationInlet5: v5 enhanced epsilon inlet with two-layer buffer/log-law model
- turbulentFrequencyInlet5: v5 enhanced omega inlet with two-layer buffer/log-law model
- mappedFlowRate4: Enhanced mapped flow rate with thermal expansion and swirl correction
- pressureWaveTransmissive4: Enhanced wave transmissive with frequency-dependent damping
- turbulentViscosityInlet4: Enhanced turbulent viscosity inlet with pressure-gradient correction
- turbulentLengthScaleInlet4: Enhanced turbulent length scale inlet with adaptive blending
- turbulentIntensityInlet4: Enhanced turbulent intensity inlet with transition model
- turbulentDissipationInlet6: v6 enhanced epsilon inlet with production limiter
- turbulentFrequencyInlet6: v6 enhanced omega inlet with production limiter
- turbulentKineticEnergyInlet5: v5 enhanced k inlet with buoyancy production
- turbulentDissipationInlet7: v7 enhanced epsilon inlet with strain-rate production limiter
- turbulentFrequencyInlet7: v7 enhanced omega inlet with strain-rate production limiter
- mappedFlowRate5: Enhanced mapped flow rate with swirl exponent and variable Cp
- pressureWaveTransmissive5: Enhanced wave transmissive with improved NSCBC
- turbulentViscosityInlet5: Enhanced turbulent viscosity inlet with wall-distance blending
- turbulentLengthScaleInlet5: Enhanced turbulent length scale inlet with two-regime wall model
- turbulentIntensityInlet5: Enhanced turbulent intensity inlet with wall-distance model
- turbulentKineticEnergyInlet6: v6 enhanced k inlet with production-limited buoyancy
- turbulentDissipationInlet8: v8 enhanced epsilon inlet with Kolmogorov-scale limiter
- turbulentFrequencyInlet8: v8 enhanced omega inlet with Kolmogorov-scale limiter
- outletPhaseMeanVelocity2: Enhanced outlet phase mean velocity with pressure-gradient correction
- scaledHeatFlux2: Enhanced scaled heat flux with temperature-dependent scaling
- mappedFlowRate6: Enhanced mapped flow rate with swirl correction and angular momentum conservation
- pressureWaveTransmissive6: Enhanced wave transmissive with improved multi-wave NSCBC
- turbulentViscosityInlet6: Enhanced turbulent viscosity inlet with pressure-gradient correction
- turbulentLengthScaleInlet6: Enhanced turbulent length scale inlet with anisotropy and strain-rate correction
- turbulentIntensityInlet6: Enhanced turbulent intensity inlet with production-to-dissipation limiter
- turbulentKineticEnergyInlet7: v7 enhanced k inlet with compressibility correction and Ma_t limiter
- turbulentDissipationInlet9: v9 enhanced epsilon inlet with realizability constraint
- turbulentFrequencyInlet9: v9 enhanced omega inlet with realizability and SST limiter
- outletPhaseMeanVelocity3: Enhanced outlet phase mean velocity with turbulent-flux weighting
- scaledHeatFlux3: Enhanced scaled heat flux with radiative loss and spatial weighting
- mappedFlowRate7: Enhanced mapped flow rate with swirl correction and profile blending
- pressureWaveTransmissive7: Enhanced wave transmissive with adaptive NSCBC and multi-scale damping
- turbulentViscosityInlet7: Enhanced turbulent viscosity inlet with wall-distance blending
- turbulentLengthScaleInlet7: Enhanced turbulent length scale inlet with two-regime wall model
- turbulentIntensityInlet7: Enhanced turbulent intensity inlet with cascade limiter
- turbulentKineticEnergyInlet8: v8 enhanced k inlet with dynamic production/dissipation balance
- turbulentDissipationInlet10: v10 enhanced epsilon inlet with dynamic time-scale limiter
- turbulentFrequencyInlet10: v10 enhanced omega inlet with dynamic correction and SST CD term
- outletPhaseMeanVelocity4: Enhanced outlet phase mean velocity with TKE coupling
- scaledHeatFlux4: Enhanced scaled heat flux with conjugate interface coupling
- mappedFlowRate8: Enhanced mapped flow rate with radial swirl decay and time-averaged correction
- pressureWaveTransmissive8: Enhanced wave transmissive with entropy wave correction
- turbulentViscosityInlet8: Enhanced turbulent viscosity inlet with anisotropy correction
- turbulentLengthScaleInlet8: Enhanced turbulent length scale inlet with wake-function correction
- turbulentIntensityInlet8: Enhanced turbulent intensity inlet with anisotropy and strain-rate coupling
- turbulentKineticEnergyInlet9: v9 enhanced k inlet with spectral energy correction
- turbulentDissipationInlet11: v11 enhanced epsilon inlet with anisotropic dissipation and cascade model
- turbulentFrequencyInlet11: v11 enhanced omega inlet with frequency-dependent blending
- outletPhaseMeanVelocity5: Enhanced outlet phase mean velocity with turbulent Prandtl correction
- scaledHeatFlux5: Enhanced scaled heat flux with temperature-dependent emissivity
- mappedFlowRate9: Enhanced mapped flow rate with wall-distance profile and adaptive swirl damping
- pressureWaveTransmissive9: Enhanced wave transmissive with viscous damping correction
- turbulentViscosityInlet9: Enhanced turbulent viscosity inlet with temperature correction
- turbulentLengthScaleInlet9: Enhanced turbulent length scale inlet with pressure-gradient correction
- turbulentIntensityInlet9: Enhanced turbulent intensity inlet with spectral and pressure-gradient corrections
- turbulentKineticEnergyInlet10: v10 enhanced k inlet with compressibility and enhanced production limiter
- turbulentDissipationInlet12: v12 enhanced epsilon inlet with compressibility and Q-criterion vortex-stretching
- turbulentFrequencyInlet12: v12 enhanced omega inlet with compressibility and pressure-gradient sensitivity
- outletPhaseMeanVelocity6: Enhanced outlet phase mean velocity with wall-distance correction
- scaledHeatFlux6: Enhanced scaled heat flux with transient thermal inertia correction
- mappedFlowRate10: Enhanced mapped flow rate with anisotropic swirl and coriolis correction
- pressureWaveTransmissive10: Enhanced wave transmissive with acoustic impedance correction
- turbulentViscosityInlet10: Enhanced turbulent viscosity inlet with RST correction and dynamic wall transition
- turbulentLengthScaleInlet10: Enhanced turbulent length scale inlet with dynamic wall transition and production correction
- turbulentIntensityInlet10: Enhanced turbulent intensity inlet with dynamic production limit and gradient correction
- turbulentKineticEnergyInlet11: v11 enhanced k inlet with wall-normal flux and time-scale ratio limiter
- turbulentDissipationInlet13: v13 enhanced epsilon inlet with wall-pressure-fluctuation and Kolmogorov blending
- turbulentFrequencyInlet13: v13 enhanced omega inlet with wall-pressure-fluctuation and Kolmogorov blending
- outletPhaseMeanVelocity7: Enhanced outlet phase mean velocity with nut gradient correction
- scaledHeatFlux7: Enhanced scaled heat flux with history-aware inertia and spatial periodicity
"""

# Import base and collection first
from pyfoam.boundary.boundary_condition import BoundaryCondition, Patch
from pyfoam.boundary.boundary_field import BoundaryField

# Import concrete BCs — each import triggers @BoundaryCondition.register(...)
from pyfoam.boundary.fixed_value import FixedValueBC
from pyfoam.boundary.zero_gradient import ZeroGradientBC
from pyfoam.boundary.no_slip import NoSlipBC
from pyfoam.boundary.fixed_gradient import FixedGradientBC
from pyfoam.boundary.symmetry import SymmetryBC
from pyfoam.boundary.cyclic import CyclicBC
from pyfoam.boundary.wall_function import (
    KqRWallFunctionBC,
    NutkWallFunctionBC,
    NutLowReWallFunctionBC,
    NutUWallFunctionBC,
    NutURoughWallFunctionBC,
    NutUSpaldingWallFunctionBC,
    EpsilonWallFunctionBC,
    OmegaWallFunctionBC,
)
from pyfoam.boundary.inlet_outlet import InletOutletBC
from pyfoam.boundary.alpha_contact_angle import AlphaContactAngleBC

# Phase 9: New boundary conditions
# Velocity BCs
from pyfoam.boundary.velocity_bcs import (
    FlowRateInletVelocityBC,
    PressureInletOutletVelocityBC,
    RotatingWallVelocityBC,
)

# Pressure BCs
from pyfoam.boundary.pressure_bcs import (
    TotalPressureBC,
    FixedFluxPressureBC,
    PrghPressureBC,
    WaveTransmissiveBC,
)

# Turbulence BCs
from pyfoam.boundary.turbulence_bcs import (
    TurbulentIntensityKineticEnergyInletBC,
    TurbulentMixingLengthDissipationRateInletBC,
    TurbulentMixingLengthFrequencyInletBC,
)

# VOF BCs
from pyfoam.boundary.vof_bcs import ConstantAlphaContactAngleBC

# Phase 10: Additional critical BCs
from pyfoam.boundary.empty import EmptyBC
from pyfoam.boundary.advective import AdvectiveBC
from pyfoam.boundary.uniform_fixed_value import UniformFixedValueBC
from pyfoam.boundary.surface_normal_fixed_value import SurfaceNormalFixedValueBC

# Phase 11: Energy / buoyancy / turbulent inlet BCs
from pyfoam.boundary.energy_bcs import (
    FixedEnergyBC,
    GradientEnergyBC,
    MixedEnergyBC,
)
from pyfoam.boundary.buoyant_pressure import BuoyantPressureBC
from pyfoam.boundary.turbulent_inlet import TurbulentInletBC

# Phase 14: Matched flow rate / fixed shear stress BCs
from pyfoam.boundary.matched_flow_rate import MatchedFlowRateOutletBC
from pyfoam.boundary.fixed_shear_stress import FixedShearStressBC

# Phase 15: Wedge / generic / calculated BCs
from pyfoam.boundary.wedge import WedgeBC
from pyfoam.boundary.generic import GenericBC
from pyfoam.boundary.calculated import CalculatedBC

# Phase 5: fixedNormal / slip / pressureInletOutlet BCs
from pyfoam.boundary.fixed_normal import FixedNormalBC
from pyfoam.boundary.slip import SlipBC
from pyfoam.boundary.inlet_outlet_2 import PressureInletOutletBC

# Phase 16: Non-uniform / uniform / mapped BCs
from pyfoam.boundary.nonuniform import NonUniformBC
from pyfoam.boundary.uniform import UniformBC
from pyfoam.boundary.mapped import MappedBC

# Phase 17: Time-varying / coded / gradient-energy BCs
from pyfoam.boundary.time_varying import TimeVaryingBC
from pyfoam.boundary.coded import CodedBC
from pyfoam.boundary.gradient_energy import GradientEnergyBC

# Phase 18: Symmetry plane / processor BCs
from pyfoam.boundary.symmetry_plane import SymmetryPlaneBC
from pyfoam.boundary.processor import ProcessorBC

# Phase 19: FixedValue2 / ZeroGradient2 / CyclicAMI BCs
from pyfoam.boundary.fixed_value_2 import FixedValue2BC
from pyfoam.boundary.zero_gradient_2 import ZeroGradient2BC
from pyfoam.boundary.cyclic_ami import CyclicAMI

# Phase 5 continued: Volume / mass flow rate BCs
from pyfoam.boundary.volume_flow_rate import VolumeFlowRateBC
from pyfoam.boundary.mass_flow_rate import MassFlowRateBC

# Hydrostatic pressure / outlet-inlet BCs
from pyfoam.boundary.hydrostatic_pressure import HydrostaticPressureBC
from pyfoam.boundary.outlet_inlet import OutletInletBC

# Phase 5 continued: Fan / porous medium BCs
from pyfoam.boundary.fan import FanBC
from pyfoam.boundary.porous_medium import PorousMediumBC

# Phase 5: Mapped wall CHT BC
from pyfoam.boundary.mapped_wall import MappedWallBC

# Phase 5: Enhanced rotating wall BC
from pyfoam.boundary.rotating_wall import RotatingWallVelocity2BC

# Phase 5 continued: Oscillating velocity / turbulent mixing length BCs
from pyfoam.boundary.oscillating_velocity import OscillatingVelocityBC
from pyfoam.boundary.turbulent_mixing_length import TurbulentMixingLengthBC

# Multiphase wall lubrication / drag coefficient BCs
from pyfoam.boundary.wall_lubrication import WallLubricationBC
from pyfoam.boundary.drag_coefficient import DragCoefficientBC

# Phase 4-9: Enhanced contact angle and wall function BCs
from pyfoam.boundary.alpha_contact_angle_2 import AlphaContactAngle2BC
from pyfoam.boundary.wall_function_2 import EnhancedWallFunctionBC

# Phase 5-8: Non-conformal couple / mapped pressure inlet BCs
from pyfoam.boundary.non_conformal_couple import NonConformalCoupleBC
from pyfoam.boundary.mapped_pressure_inlet import MappedPressureInletBC

# Phase 5-7: Scaled velocity inlet / moving wall BCs
from pyfoam.boundary.scaled_velocity_inlet import ScaledVelocityInletBC
from pyfoam.boundary.moving_wall import MovingWallBC

# Phase 5: Turbulent kinetic energy BCs
from pyfoam.boundary.turbulent_kinetic_energy_bcs import (
    TurbulentIntensityKEBC,
    FixedTurbulentKEBC,
)

# Phase 5: Enhanced Antal wall lubrication BC
from pyfoam.boundary.wall_lubrication_2 import AntalWallLubricationBC

# Phase 4-9: Enhanced fixed flux pressure and inlet/outlet BCs
from pyfoam.boundary.fixed_flux_pressure_2 import FixedFluxPressure2BC
from pyfoam.boundary.inlet_outlet_3 import InletOutlet3BC

# Phase 5-8: Tomiyama wall lubrication and interfacial momentum BCs
from pyfoam.boundary.wall_lubrication_3 import TomiyamaWallLubricationBC
from pyfoam.boundary.interfacial_momentum import InterfacialMomentumBC

# Phase 5: Non-linear roughness / wave generation BCs
from pyfoam.boundary.nonlinear_roughness import NonLinearRoughnessBC
from pyfoam.boundary.wave_generation import WaveGenerationBC

# Phase 5: Saturated temperature / subcooling BCs
from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC
from pyfoam.boundary.subcooling import SubcoolingBC

# Phase 5-7: Stagnation inlet / symmetry-slip BCs
from pyfoam.boundary.stagnation_inlet import StagnationInletBC
from pyfoam.boundary.symmetry_slip import SymmetrySlipBC

# Phase 5: Enhanced flow rate inlet and wall heat flux BCs
from pyfoam.boundary.flow_rate_inlet_2 import FlowRateInlet2BC
from pyfoam.boundary.wall_heat_flux import WallHeatFluxBC

# Phase 5: Adjoint inlet and wall film BCs
from pyfoam.boundary.adjoint_inlet import AdjointInletBC
from pyfoam.boundary.wall_film import WallFilmBC

# Phase 5: Phase-mean velocity and convective heat transfer BCs
from pyfoam.boundary.phase_mean_velocity import PhaseMeanVelocityBC
from pyfoam.boundary.convective_heat_transfer import ConvectiveHeatTransferBC

# Phase 5: Enhanced generic and pressure normal inlet BCs
from pyfoam.boundary.generic_2 import Generic2BC
from pyfoam.boundary.pressure_normal_inlet import PressureNormalInletBC

# Phase 5-7: Variable height / mass outlet BCs
from pyfoam.boundary.variable_height import VariableHeightBC
from pyfoam.boundary.mass_outlet import MassOutletBC

# Phase 5: Total enthalpy and translating boundary BCs
from pyfoam.boundary.total_enthalpy import TotalEnthalpyBC
from pyfoam.boundary.translating_boundary import TranslatingBoundaryBC

# Phase 5: Scaled heat flux and outlet phase mean velocity BCs
from pyfoam.boundary.scaled_heat_flux import ScaledHeatFluxBC
from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC

# Phase 5: Heat exchanger BC
from pyfoam.boundary.heat_exchanger_bc import HeatExchangerBC

# Phase 5: Enhanced turbulent inlet BC
from pyfoam.boundary.turbulent_inlet_2 import TurbulentInlet2BC

# Phase 5: Mapped velocity internal / enhanced variable height BCs
from pyfoam.boundary.mapped_velocity_internal import MappedVelocityInternalBC
from pyfoam.boundary.variable_height_2 import VariableHeight2BC

# Phase 4-7: Enhanced Tomiyama wall lubrication and interfacial heat transfer BCs
from pyfoam.boundary.wall_lubrication_tomiyama import TomiyamaWallLubrication2BC
from pyfoam.boundary.interfacial_heat_transfer import InterfacialHeatTransferBC

# Phase 5-7: Enhanced Antal wall lubrication and pressure inlet/outlet 2 BCs
from pyfoam.boundary.wall_lubrication_antal import AntalWallLubrication2BC
from pyfoam.boundary.pressure_inlet_outlet_2 import PressureInletOutlet2BC

# Phase 5: Coupled velocity and transmissive pressure BCs
from pyfoam.boundary.coupled_velocity import CoupledVelocityBC
from pyfoam.boundary.pressure_transmissive import PressureTransmissiveBC

# Phase 20: Pressure-directed inlet / swirl flow rate / mapped convective HT BCs
from pyfoam.boundary.pressure_directed_inlet_velocity import PressureDirectedInletVelocityBC
from pyfoam.boundary.swirl_flow_rate_inlet_velocity import SwirlFlowRateInletVelocityBC
from pyfoam.boundary.mapped_convective_heat_transfer import MappedConvectiveHeatTransferBC

# Phase 21: Uniform total pressure / directed inlet-outlet / external coupled /
#           mixed temperature / turbulent temperature coupled BCs
from pyfoam.boundary.uniform_total_pressure import UniformTotalPressureBC
from pyfoam.boundary.directed_inlet_outlet import DirectedInletOutletBC
from pyfoam.boundary.external_coupled import ExternalCoupledBC
from pyfoam.boundary.mixed_temperature import MixedTemperatureBC
from pyfoam.boundary.turbulent_temperature_coupled import TurbulentTemperatureCoupledBC

# Phase 22: Mapped velocity adjusted pressure / enhanced flow rate /
#          enhanced pressure inlet/outlet velocity / enhanced surface normal /
#          turbulent inlet BCs (k, epsilon, omega)
from pyfoam.boundary.mapped_velocity_adjusted_pressure import MappedVelocityAdjustedPressureBC
from pyfoam.boundary.flow_rate_inlet_velocity_2 import FlowRateInletVelocity2BC
from pyfoam.boundary.pressure_inlet_outlet_velocity_2 import PressureInletOutletVelocity2BC
from pyfoam.boundary.surface_normal_fixed_value_2 import SurfaceNormalFixedValue2BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet import TurbulentKineticEnergyInletBC
from pyfoam.boundary.turbulent_dissipation_rate_inlet import TurbulentDissipationRateInletBC
from pyfoam.boundary.turbulent_specific_dissipation_rate_inlet import TurbulentSpecificDissipationRateInletBC

# Phase 23: Compressible coupled HT / enhanced wave transmissive /
#          advective-diffusive / AMG pressure interpolation /
#          coded fixed value / enhanced cyclic AMI / processor cyclic BCs
from pyfoam.boundary.compressible_turbulent_temperature_coupled import (
    CompressibleTurbulentTemperatureCoupledBC,
)
from pyfoam.boundary.wave_transmissive_2 import WaveTransmissive2BC
from pyfoam.boundary.advective_diffusive import AdvectiveDiffusiveBC
from pyfoam.boundary.pressure_interpolation_amg import PressureInterpolationAMGBC
from pyfoam.boundary.coded_fixed_value import CodedFixedValueBC
from pyfoam.boundary.cyclic_ami_2 import CyclicAMI2BC
from pyfoam.boundary.processor_cyclic import ProcessorCyclicBC

# Phase 24: Mapped flow rate / wave transmissive / turbulent inlet BCs
from pyfoam.boundary.mapped_flow_rate import MappedFlowRateBC
from pyfoam.boundary.pressure_wave_transmissive import PressureWaveTransmissiveBC
from pyfoam.boundary.turbulent_viscosity_inlet import TurbulentViscosityInletBC
from pyfoam.boundary.turbulent_length_scale_inlet import TurbulentLengthScaleInletBC
from pyfoam.boundary.turbulent_intensity_inlet import TurbulentIntensityInletBC
from pyfoam.boundary.turbulent_dissipation_inlet import TurbulentDissipationInletBC
from pyfoam.boundary.turbulent_frequency_inlet import TurbulentFrequencyInletBC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_2 import TurbulentKineticEnergyInlet2BC
from pyfoam.boundary.turbulent_dissipation_inlet_2 import TurbulentDissipationInlet2BC
from pyfoam.boundary.turbulent_frequency_inlet_2 import TurbulentFrequencyInlet2BC

# Phase 25: Enhanced mapped flow rate / wave transmissive / turbulent inlet BCs (v2 + v3)
from pyfoam.boundary.mapped_flow_rate_2 import MappedFlowRate2BC
from pyfoam.boundary.pressure_wave_transmissive_2 import PressureWaveTransmissive2BC
from pyfoam.boundary.turbulent_viscosity_inlet_2 import TurbulentViscosityInlet2BC
from pyfoam.boundary.turbulent_length_scale_inlet_2 import TurbulentLengthScaleInlet2BC
from pyfoam.boundary.turbulent_intensity_inlet_2 import TurbulentIntensityInlet2BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_3 import TurbulentKineticEnergyInlet3BC
from pyfoam.boundary.turbulent_dissipation_inlet_3 import TurbulentDissipationInlet3BC
from pyfoam.boundary.turbulent_frequency_inlet_3 import TurbulentFrequencyInlet3BC

# Phase 26: Enhanced mapped flow rate / wave transmissive / turbulent inlet BCs (v3 + v4 + v5)
from pyfoam.boundary.mapped_flow_rate_3 import MappedFlowRate3BC
from pyfoam.boundary.pressure_wave_transmissive_3 import PressureWaveTransmissive3BC
from pyfoam.boundary.turbulent_viscosity_inlet_3 import TurbulentViscosityInlet3BC
from pyfoam.boundary.turbulent_length_scale_inlet_3 import TurbulentLengthScaleInlet3BC
from pyfoam.boundary.turbulent_intensity_inlet_3 import TurbulentIntensityInlet3BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_4 import TurbulentKineticEnergyInlet4BC
from pyfoam.boundary.turbulent_dissipation_inlet_4 import TurbulentDissipationInlet4BC
from pyfoam.boundary.turbulent_frequency_inlet_4 import TurbulentFrequencyInlet4BC
from pyfoam.boundary.turbulent_dissipation_inlet_5 import TurbulentDissipationInlet5BC
from pyfoam.boundary.turbulent_frequency_inlet_5 import TurbulentFrequencyInlet5BC

# Phase 27: Enhanced v4 mapped/wave/turbulent inlets + v5-v7 epsilon/omega/k variants
from pyfoam.boundary.mapped_flow_rate_4 import MappedFlowRate4BC
from pyfoam.boundary.pressure_wave_transmissive_4 import PressureWaveTransmissive4BC
from pyfoam.boundary.turbulent_viscosity_inlet_4 import TurbulentViscosityInlet4BC
from pyfoam.boundary.turbulent_length_scale_inlet_4 import TurbulentLengthScaleInlet4BC
from pyfoam.boundary.turbulent_intensity_inlet_4 import TurbulentIntensityInlet4BC
from pyfoam.boundary.turbulent_dissipation_inlet_6 import TurbulentDissipationInlet6BC
from pyfoam.boundary.turbulent_frequency_inlet_6 import TurbulentFrequencyInlet6BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_5 import TurbulentKineticEnergyInlet5BC
from pyfoam.boundary.turbulent_dissipation_inlet_7 import TurbulentDissipationInlet7BC
from pyfoam.boundary.turbulent_frequency_inlet_7 import TurbulentFrequencyInlet7BC

# Phase 28: Enhanced v5 mapped/wave/turbulent inlets + v6 k + v8 epsilon/omega +
#          enhanced outlet phase mean velocity + enhanced scaled heat flux
from pyfoam.boundary.mapped_flow_rate_5 import MappedFlowRate5BC
from pyfoam.boundary.pressure_wave_transmissive_5 import PressureWaveTransmissive5BC
from pyfoam.boundary.turbulent_viscosity_inlet_5 import TurbulentViscosityInlet5BC
from pyfoam.boundary.turbulent_length_scale_inlet_5 import TurbulentLengthScaleInlet5BC
from pyfoam.boundary.turbulent_intensity_inlet_5 import TurbulentIntensityInlet5BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_6 import TurbulentKineticEnergyInlet6BC
from pyfoam.boundary.turbulent_dissipation_inlet_8 import TurbulentDissipationInlet8BC
from pyfoam.boundary.turbulent_frequency_inlet_8 import TurbulentFrequencyInlet8BC
from pyfoam.boundary.outlet_phase_mean_velocity_2 import OutletPhaseMeanVelocity2BC
from pyfoam.boundary.scaled_heat_flux_2 import ScaledHeatFlux2BC

# Phase 29: Enhanced v6 mapped/wave/turbulent inlets + v7 k + v9 epsilon/omega +
#          enhanced v3 outlet phase mean velocity + enhanced v3 scaled heat flux
from pyfoam.boundary.mapped_flow_rate_6 import MappedFlowRate6BC
from pyfoam.boundary.pressure_wave_transmissive_6 import PressureWaveTransmissive6BC
from pyfoam.boundary.turbulent_viscosity_inlet_6 import TurbulentViscosityInlet6BC
from pyfoam.boundary.turbulent_length_scale_inlet_6 import TurbulentLengthScaleInlet6BC
from pyfoam.boundary.turbulent_intensity_inlet_6 import TurbulentIntensityInlet6BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_7 import TurbulentKineticEnergyInlet7BC
from pyfoam.boundary.turbulent_dissipation_inlet_9 import TurbulentDissipationInlet9BC
from pyfoam.boundary.turbulent_frequency_inlet_9 import TurbulentFrequencyInlet9BC
from pyfoam.boundary.outlet_phase_mean_velocity_3 import OutletPhaseMeanVelocity3BC
from pyfoam.boundary.scaled_heat_flux_3 import ScaledHeatFlux3BC

# Phase 30: Enhanced v7 mapped/wave/turbulent inlets + v8 k + v10 epsilon/omega +
#          enhanced v4 outlet phase mean velocity + enhanced v4 scaled heat flux
from pyfoam.boundary.mapped_flow_rate_7 import MappedFlowRate7BC
from pyfoam.boundary.pressure_wave_transmissive_7 import PressureWaveTransmissive7BC
from pyfoam.boundary.turbulent_viscosity_inlet_7 import TurbulentViscosityInlet7BC
from pyfoam.boundary.turbulent_length_scale_inlet_7 import TurbulentLengthScaleInlet7BC
from pyfoam.boundary.turbulent_intensity_inlet_7 import TurbulentIntensityInlet7BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_8 import TurbulentKineticEnergyInlet8BC
from pyfoam.boundary.turbulent_dissipation_inlet_10 import TurbulentDissipationInlet10BC
from pyfoam.boundary.turbulent_frequency_inlet_10 import TurbulentFrequencyInlet10BC
from pyfoam.boundary.outlet_phase_mean_velocity_4 import OutletPhaseMeanVelocity4BC
from pyfoam.boundary.scaled_heat_flux_4 import ScaledHeatFlux4BC

# Phase 31: Enhanced v8 mapped/wave/turbulent inlets + v9 k + v11 epsilon/omega +
#          enhanced v5 outlet phase mean velocity + enhanced v5 scaled heat flux
from pyfoam.boundary.mapped_flow_rate_8 import MappedFlowRate8BC
from pyfoam.boundary.pressure_wave_transmissive_8 import PressureWaveTransmissive8BC
from pyfoam.boundary.turbulent_viscosity_inlet_8 import TurbulentViscosityInlet8BC
from pyfoam.boundary.turbulent_length_scale_inlet_8 import TurbulentLengthScaleInlet8BC
from pyfoam.boundary.turbulent_intensity_inlet_8 import TurbulentIntensityInlet8BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_9 import TurbulentKineticEnergyInlet9BC
from pyfoam.boundary.turbulent_dissipation_inlet_11 import TurbulentDissipationInlet11BC
from pyfoam.boundary.turbulent_frequency_inlet_11 import TurbulentFrequencyInlet11BC
from pyfoam.boundary.outlet_phase_mean_velocity_5 import OutletPhaseMeanVelocity5BC
from pyfoam.boundary.scaled_heat_flux_5 import ScaledHeatFlux5BC

# Phase 32: Enhanced v9 mapped/wave/turbulent inlets + v10 k + v12 epsilon/omega +
#          enhanced v6 outlet phase mean velocity + enhanced v6 scaled heat flux
from pyfoam.boundary.mapped_flow_rate_9 import MappedFlowRate9BC
from pyfoam.boundary.pressure_wave_transmissive_9 import PressureWaveTransmissive9BC
from pyfoam.boundary.turbulent_viscosity_inlet_9 import TurbulentViscosityInlet9BC
from pyfoam.boundary.turbulent_length_scale_inlet_9 import TurbulentLengthScaleInlet9BC
from pyfoam.boundary.turbulent_intensity_inlet_9 import TurbulentIntensityInlet9BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_10 import TurbulentKineticEnergyInlet10BC
from pyfoam.boundary.turbulent_dissipation_inlet_12 import TurbulentDissipationInlet12BC
from pyfoam.boundary.turbulent_frequency_inlet_12 import TurbulentFrequencyInlet12BC
from pyfoam.boundary.outlet_phase_mean_velocity_6 import OutletPhaseMeanVelocity6BC
from pyfoam.boundary.scaled_heat_flux_6 import ScaledHeatFlux6BC

# Phase 33: Enhanced v10 mapped/wave/turbulent inlets + v11 k + v13 epsilon/omega +
#          enhanced v7 outlet phase mean velocity + enhanced v7 scaled heat flux
from pyfoam.boundary.mapped_flow_rate_10 import MappedFlowRate10BC
from pyfoam.boundary.pressure_wave_transmissive_10 import PressureWaveTransmissive10BC
from pyfoam.boundary.turbulent_viscosity_inlet_10 import TurbulentViscosityInlet10BC
from pyfoam.boundary.turbulent_length_scale_inlet_10 import TurbulentLengthScaleInlet10BC
from pyfoam.boundary.turbulent_intensity_inlet_10 import TurbulentIntensityInlet10BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_11 import TurbulentKineticEnergyInlet11BC
from pyfoam.boundary.turbulent_dissipation_inlet_13 import TurbulentDissipationInlet13BC
from pyfoam.boundary.turbulent_frequency_inlet_13 import TurbulentFrequencyInlet13BC
from pyfoam.boundary.outlet_phase_mean_velocity_7 import OutletPhaseMeanVelocity7BC
from pyfoam.boundary.scaled_heat_flux_7 import ScaledHeatFlux7BC

# Phase 34: Enhanced BCs (v14-v16 existing + v2-v10 new)
from pyfoam.boundary.mapped_flow_rate_11 import MappedFlowRate11BC
from pyfoam.boundary.mapped_flow_rate_12 import MappedFlowRate12BC
from pyfoam.boundary.mapped_flow_rate_13 import MappedFlowRate13BC
from pyfoam.boundary.mapped_flow_rate_14 import MappedFlowRate14BC
from pyfoam.boundary.mapped_flow_rate_15 import MappedFlowRate15BC
from pyfoam.boundary.mapped_flow_rate_16 import MappedFlowRate16BC
from pyfoam.boundary.pressure_wave_transmissive_11 import PressureWaveTransmissive11BC
from pyfoam.boundary.pressure_wave_transmissive_12 import PressureWaveTransmissive12BC
from pyfoam.boundary.pressure_wave_transmissive_13 import PressureWaveTransmissive13BC
from pyfoam.boundary.pressure_wave_transmissive_14 import PressureWaveTransmissive14BC
from pyfoam.boundary.pressure_wave_transmissive_15 import PressureWaveTransmissive15BC
from pyfoam.boundary.pressure_wave_transmissive_16 import PressureWaveTransmissive16BC
from pyfoam.boundary.turbulent_intensity_inlet_11 import TurbulentIntensityInlet11BC
from pyfoam.boundary.turbulent_intensity_inlet_12 import TurbulentIntensityInlet12BC
from pyfoam.boundary.turbulent_intensity_inlet_13 import TurbulentIntensityInlet13BC
from pyfoam.boundary.turbulent_intensity_inlet_14 import TurbulentIntensityInlet14BC
from pyfoam.boundary.turbulent_intensity_inlet_15 import TurbulentIntensityInlet15BC
from pyfoam.boundary.turbulent_intensity_inlet_16 import TurbulentIntensityInlet16BC
from pyfoam.boundary.turbulent_viscosity_inlet_11 import TurbulentViscosityInlet11BC
from pyfoam.boundary.turbulent_viscosity_inlet_12 import TurbulentViscosityInlet12BC
from pyfoam.boundary.turbulent_viscosity_inlet_13 import TurbulentViscosityInlet13BC
from pyfoam.boundary.turbulent_viscosity_inlet_14 import TurbulentViscosityInlet14BC
from pyfoam.boundary.turbulent_viscosity_inlet_15 import TurbulentViscosityInlet15BC
from pyfoam.boundary.turbulent_viscosity_inlet_16 import TurbulentViscosityInlet16BC
from pyfoam.boundary.turbulent_length_scale_inlet_11 import TurbulentLengthScaleInlet11BC
from pyfoam.boundary.turbulent_length_scale_inlet_12 import TurbulentLengthScaleInlet12BC
from pyfoam.boundary.turbulent_length_scale_inlet_13 import TurbulentLengthScaleInlet13BC
from pyfoam.boundary.turbulent_length_scale_inlet_14 import TurbulentLengthScaleInlet14BC
from pyfoam.boundary.turbulent_length_scale_inlet_15 import TurbulentLengthScaleInlet15BC
from pyfoam.boundary.turbulent_length_scale_inlet_16 import TurbulentLengthScaleInlet16BC
from pyfoam.boundary.turbulent_dissipation_inlet_14 import TurbulentDissipationInlet14BC
from pyfoam.boundary.turbulent_dissipation_inlet_15 import TurbulentDissipationInlet15BC
from pyfoam.boundary.turbulent_dissipation_inlet_16 import TurbulentDissipationInlet16BC
from pyfoam.boundary.turbulent_frequency_inlet_14 import TurbulentFrequencyInlet14BC
from pyfoam.boundary.turbulent_frequency_inlet_15 import TurbulentFrequencyInlet15BC
from pyfoam.boundary.turbulent_frequency_inlet_16 import TurbulentFrequencyInlet16BC
from pyfoam.boundary.outlet_phase_mean_velocity_8 import OutletPhaseMeanVelocity8BC
from pyfoam.boundary.outlet_phase_mean_velocity_9 import OutletPhaseMeanVelocity9BC
from pyfoam.boundary.outlet_phase_mean_velocity_10 import OutletPhaseMeanVelocity10BC
from pyfoam.boundary.outlet_phase_mean_velocity_11 import OutletPhaseMeanVelocity11BC
from pyfoam.boundary.outlet_phase_mean_velocity_12 import OutletPhaseMeanVelocity12BC
from pyfoam.boundary.outlet_phase_mean_velocity_13 import OutletPhaseMeanVelocity13BC
from pyfoam.boundary.outlet_phase_mean_velocity_14 import OutletPhaseMeanVelocity14BC
from pyfoam.boundary.outlet_phase_mean_velocity_15 import OutletPhaseMeanVelocity15BC
from pyfoam.boundary.outlet_phase_mean_velocity_16 import OutletPhaseMeanVelocity16BC
from pyfoam.boundary.scaled_heat_flux_8 import ScaledHeatFlux8BC
from pyfoam.boundary.scaled_heat_flux_9 import ScaledHeatFlux9BC
from pyfoam.boundary.scaled_heat_flux_10 import ScaledHeatFlux10BC
from pyfoam.boundary.scaled_heat_flux_11 import ScaledHeatFlux11BC
from pyfoam.boundary.scaled_heat_flux_12 import ScaledHeatFlux12BC
from pyfoam.boundary.scaled_heat_flux_13 import ScaledHeatFlux13BC
from pyfoam.boundary.scaled_heat_flux_14 import ScaledHeatFlux14BC
from pyfoam.boundary.scaled_heat_flux_15 import ScaledHeatFlux15BC
from pyfoam.boundary.scaled_heat_flux_16 import ScaledHeatFlux16BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_12 import TurbulentKineticEnergyInlet12BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_13 import TurbulentKineticEnergyInlet13BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_14 import TurbulentKineticEnergyInlet14BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_15 import TurbulentKineticEnergyInlet15BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_16 import TurbulentKineticEnergyInlet16BC
from pyfoam.boundary.non_conformal_couple_2 import NonConformalCouple2BC
from pyfoam.boundary.non_conformal_couple_3 import NonConformalCouple3BC
from pyfoam.boundary.non_conformal_couple_4 import NonConformalCouple4BC
from pyfoam.boundary.non_conformal_couple_5 import NonConformalCouple5BC
from pyfoam.boundary.non_conformal_couple_6 import NonConformalCouple6BC
from pyfoam.boundary.non_conformal_couple_7 import NonConformalCouple7BC
from pyfoam.boundary.non_conformal_couple_8 import NonConformalCouple8BC
from pyfoam.boundary.non_conformal_couple_9 import NonConformalCouple9BC
from pyfoam.boundary.non_conformal_couple_10 import NonConformalCouple10BC
from pyfoam.boundary.processor_cyclic_2 import ProcessorCyclic2BC
from pyfoam.boundary.processor_cyclic_3 import ProcessorCyclic3BC
from pyfoam.boundary.processor_cyclic_4 import ProcessorCyclic4BC
from pyfoam.boundary.processor_cyclic_5 import ProcessorCyclic5BC
from pyfoam.boundary.processor_cyclic_6 import ProcessorCyclic6BC
from pyfoam.boundary.processor_cyclic_7 import ProcessorCyclic7BC
from pyfoam.boundary.processor_cyclic_8 import ProcessorCyclic8BC
from pyfoam.boundary.processor_cyclic_9 import ProcessorCyclic9BC
from pyfoam.boundary.processor_cyclic_10 import ProcessorCyclic10BC
from pyfoam.boundary.wedge_bc_2 import Wedge2BC
from pyfoam.boundary.wedge_bc_3 import Wedge3BC
from pyfoam.boundary.wedge_bc_4 import Wedge4BC
from pyfoam.boundary.wedge_bc_5 import Wedge5BC
from pyfoam.boundary.wedge_bc_6 import Wedge6BC
from pyfoam.boundary.wedge_bc_7 import Wedge7BC
from pyfoam.boundary.wedge_bc_8 import Wedge8BC
from pyfoam.boundary.wedge_bc_9 import Wedge9BC
from pyfoam.boundary.wedge_bc_10 import Wedge10BC
from pyfoam.boundary.slip_wall_bc_2 import Slip2BC
from pyfoam.boundary.slip_wall_bc_3 import Slip3BC
from pyfoam.boundary.slip_wall_bc_4 import Slip4BC
from pyfoam.boundary.slip_wall_bc_5 import Slip5BC
from pyfoam.boundary.slip_wall_bc_6 import Slip6BC
from pyfoam.boundary.slip_wall_bc_7 import Slip7BC
from pyfoam.boundary.slip_wall_bc_8 import Slip8BC
from pyfoam.boundary.slip_wall_bc_9 import Slip9BC
from pyfoam.boundary.slip_wall_bc_10 import Slip10BC
from pyfoam.boundary.phase_mean_velocity_2 import PhaseMeanVelocity2BC
from pyfoam.boundary.phase_mean_velocity_3 import PhaseMeanVelocity3BC
from pyfoam.boundary.phase_mean_velocity_4 import PhaseMeanVelocity4BC
from pyfoam.boundary.phase_mean_velocity_5 import PhaseMeanVelocity5BC
from pyfoam.boundary.phase_mean_velocity_6 import PhaseMeanVelocity6BC
from pyfoam.boundary.phase_mean_velocity_7 import PhaseMeanVelocity7BC
from pyfoam.boundary.phase_mean_velocity_8 import PhaseMeanVelocity8BC
from pyfoam.boundary.phase_mean_velocity_9 import PhaseMeanVelocity9BC
from pyfoam.boundary.phase_mean_velocity_10 import PhaseMeanVelocity10BC
from pyfoam.boundary.coupled_thermal_bc_2 import CoupledTemperature2BC
from pyfoam.boundary.coupled_thermal_bc_3 import CoupledTemperature3BC
from pyfoam.boundary.coupled_thermal_bc_4 import CoupledTemperature4BC
from pyfoam.boundary.coupled_thermal_bc_5 import CoupledTemperature5BC
from pyfoam.boundary.coupled_thermal_bc_6 import CoupledTemperature6BC
from pyfoam.boundary.coupled_thermal_bc_7 import CoupledTemperature7BC
from pyfoam.boundary.coupled_thermal_bc_8 import CoupledTemperature8BC
from pyfoam.boundary.coupled_thermal_bc_9 import CoupledTemperature9BC
from pyfoam.boundary.coupled_thermal_bc_10 import CoupledTemperature10BC
from pyfoam.boundary.registered_missing_bcs import _registered_count  # noqa: F401 — triggers RTS registration

__all__ = [
    # Base
    "BoundaryCondition",
    "Patch",
    "BoundaryField",
    # Original concrete BCs
    "FixedValueBC",
    "ZeroGradientBC",
    "NoSlipBC",
    "FixedGradientBC",
    "SymmetryBC",
    "CyclicBC",
    "NutkWallFunctionBC",
    "NutLowReWallFunctionBC",
    "NutUWallFunctionBC",
    "NutURoughWallFunctionBC",
    "NutUSpaldingWallFunctionBC",
    "KqRWallFunctionBC",
    "EpsilonWallFunctionBC",
    "OmegaWallFunctionBC",
    "InletOutletBC",
    "AlphaContactAngleBC",
    # Phase 9: Velocity BCs
    "FlowRateInletVelocityBC",
    "PressureInletOutletVelocityBC",
    "RotatingWallVelocityBC",
    # Phase 9: Pressure BCs
    "TotalPressureBC",
    "FixedFluxPressureBC",
    "PrghPressureBC",
    "WaveTransmissiveBC",
    # Phase 9: Turbulence BCs
    "TurbulentIntensityKineticEnergyInletBC",
    "TurbulentMixingLengthDissipationRateInletBC",
    "TurbulentMixingLengthFrequencyInletBC",
    # Phase 9: VOF BCs
    "ConstantAlphaContactAngleBC",
    # Phase 10: Additional critical BCs
    "EmptyBC",
    "AdvectiveBC",
    "UniformFixedValueBC",
    "SurfaceNormalFixedValueBC",
    # Phase 11: Energy / buoyancy / turbulent inlet BCs
    "FixedEnergyBC",
    "GradientEnergyBC",
    "MixedEnergyBC",
    "BuoyantPressureBC",
    "TurbulentInletBC",
    # Phase 14: Matched flow rate / fixed shear stress BCs
    "MatchedFlowRateOutletBC",
    "FixedShearStressBC",
    # Phase 15: Wedge / generic / calculated BCs
    "WedgeBC",
    "GenericBC",
    "CalculatedBC",
    # Phase 5: fixedNormal / slip / pressureInletOutlet BCs
    "FixedNormalBC",
    "SlipBC",
    "PressureInletOutletBC",
    # Phase 16: Non-uniform / uniform / mapped BCs
    "NonUniformBC",
    "UniformBC",
    "MappedBC",
    # Phase 17: Time-varying / coded / gradient-energy BCs
    "TimeVaryingBC",
    "CodedBC",
    "GradientEnergyBC",
    # Phase 18: Symmetry plane / processor BCs
    "SymmetryPlaneBC",
    "ProcessorBC",
    # Phase 19: FixedValue2 / ZeroGradient2 / CyclicAMI BCs
    "FixedValue2BC",
    "ZeroGradient2BC",
    "CyclicAMI",
    # Phase 5 continued: Volume / mass flow rate BCs
    "VolumeFlowRateBC",
    "MassFlowRateBC",
    # Hydrostatic pressure / outlet-inlet BCs
    "HydrostaticPressureBC",
    "OutletInletBC",
    # Phase 5 continued: Fan / porous medium BCs
    "FanBC",
    "PorousMediumBC",
    # Phase 5: Mapped wall CHT BC
    "MappedWallBC",
    # Phase 5: Enhanced rotating wall BC
    "RotatingWallVelocity2BC",
    # Phase 5 continued: Oscillating velocity / turbulent mixing length BCs
    "OscillatingVelocityBC",
    "TurbulentMixingLengthBC",
    # Multiphase wall lubrication / drag coefficient BCs
    "WallLubricationBC",
    "DragCoefficientBC",
    # Phase 4-9: Enhanced contact angle and wall function BCs
    "AlphaContactAngle2BC",
    "EnhancedWallFunctionBC",
    # Phase 5-8: Non-conformal couple / mapped pressure inlet BCs
    "NonConformalCoupleBC",
    "MappedPressureInletBC",
    # Phase 5-7: Scaled velocity inlet / moving wall BCs
    "ScaledVelocityInletBC",
    "MovingWallBC",
    # Phase 5: Turbulent kinetic energy BCs
    "TurbulentIntensityKEBC",
    "FixedTurbulentKEBC",
    # Phase 5: Enhanced Antal wall lubrication BC
    "AntalWallLubricationBC",
    # Phase 4-9: Enhanced fixed flux pressure and inlet/outlet BCs
    "FixedFluxPressure2BC",
    "InletOutlet3BC",
    # Phase 5-8: Tomiyama wall lubrication and interfacial momentum BCs
    "TomiyamaWallLubricationBC",
    "InterfacialMomentumBC",
    # Phase 5: Non-linear roughness / wave generation BCs
    "NonLinearRoughnessBC",
    "WaveGenerationBC",
    # Phase 5: Saturated temperature / subcooling BCs
    "SaturatedTemperatureBC",
    "SubcoolingBC",
    # Phase 5-7: Stagnation inlet / symmetry-slip BCs
    "StagnationInletBC",
    "SymmetrySlipBC",
    # Phase 5: Enhanced flow rate inlet and wall heat flux BCs
    "FlowRateInlet2BC",
    "WallHeatFluxBC",
    # Phase 5: Adjoint inlet and wall film BCs
    "AdjointInletBC",
    "WallFilmBC",
    # Phase 5: Phase-mean velocity and convective heat transfer BCs
    "PhaseMeanVelocityBC",
    "ConvectiveHeatTransferBC",
    # Phase 5-7: Variable height / mass outlet BCs
    "VariableHeightBC",
    "MassOutletBC",
    # Phase 5: Total enthalpy and translating boundary BCs
    "TotalEnthalpyBC",
    "TranslatingBoundaryBC",
    # Phase 5: Enhanced generic and pressure normal inlet BCs
    "Generic2BC",
    "PressureNormalInletBC",
    # Phase 5: Scaled heat flux and outlet phase mean velocity BCs
    "ScaledHeatFluxBC",
    "OutletPhaseMeanVelocityBC",
    # Phase 5: Heat exchanger BC
    "HeatExchangerBC",
    # Phase 5: Enhanced turbulent inlet BC
    "TurbulentInlet2BC",
    # Phase 5: Mapped velocity internal / enhanced variable height BCs
    "MappedVelocityInternalBC",
    "VariableHeight2BC",
    # Phase 4-7: Enhanced Tomiyama and interfacial HT BCs
    "TomiyamaWallLubrication2BC",
    "InterfacialHeatTransferBC",
    # Phase 5-7: Enhanced Antal and pressure inlet/outlet 2 BCs
    "AntalWallLubrication2BC",
    "PressureInletOutlet2BC",
    # Phase 5: Coupled velocity and transmissive pressure BCs
    "CoupledVelocityBC",
    "PressureTransmissiveBC",
    # Phase 20: Pressure-directed inlet / swirl flow rate / mapped convective HT BCs
    "PressureDirectedInletVelocityBC",
    "SwirlFlowRateInletVelocityBC",
    "MappedConvectiveHeatTransferBC",
    # Phase 21: Uniform total pressure / directed inlet-outlet / external coupled /
    #           mixed temperature / turbulent temperature coupled BCs
    "UniformTotalPressureBC",
    "DirectedInletOutletBC",
    "ExternalCoupledBC",
    "MixedTemperatureBC",
    "TurbulentTemperatureCoupledBC",
    # Phase 22: Mapped velocity adjusted pressure / enhanced flow rate /
    #          enhanced pressure inlet/outlet velocity / enhanced surface normal /
    #          turbulent inlet BCs (k, epsilon, omega)
    "MappedVelocityAdjustedPressureBC",
    "FlowRateInletVelocity2BC",
    "PressureInletOutletVelocity2BC",
    "SurfaceNormalFixedValue2BC",
    "TurbulentKineticEnergyInletBC",
    "TurbulentDissipationRateInletBC",
    "TurbulentSpecificDissipationRateInletBC",
    # Phase 23: Compressible coupled HT / enhanced wave transmissive /
    #          advective-diffusive / AMG pressure interpolation /
    #          coded fixed value / enhanced cyclic AMI / processor cyclic BCs
    "CompressibleTurbulentTemperatureCoupledBC",
    "WaveTransmissive2BC",
    "AdvectiveDiffusiveBC",
    "PressureInterpolationAMGBC",
    "CodedFixedValueBC",
    "CyclicAMI2BC",
    "ProcessorCyclicBC",
    # Phase 24: Mapped flow rate / wave transmissive / turbulent inlet BCs
    "MappedFlowRateBC",
    "PressureWaveTransmissiveBC",
    "TurbulentViscosityInletBC",
    "TurbulentLengthScaleInletBC",
    "TurbulentIntensityInletBC",
    "TurbulentDissipationInletBC",
    "TurbulentFrequencyInletBC",
    "TurbulentKineticEnergyInlet2BC",
    "TurbulentDissipationInlet2BC",
    "TurbulentFrequencyInlet2BC",
    # Phase 25: Enhanced mapped flow rate / wave transmissive / turbulent inlet BCs (v2 + v3)
    "MappedFlowRate2BC",
    "PressureWaveTransmissive2BC",
    "TurbulentViscosityInlet2BC",
    "TurbulentLengthScaleInlet2BC",
    "TurbulentIntensityInlet2BC",
    "TurbulentKineticEnergyInlet3BC",
    "TurbulentDissipationInlet3BC",
    "TurbulentFrequencyInlet3BC",
    # Phase 26: Enhanced mapped flow rate / wave transmissive / turbulent inlet BCs (v3 + v4 + v5)
    "MappedFlowRate3BC",
    "PressureWaveTransmissive3BC",
    "TurbulentViscosityInlet3BC",
    "TurbulentLengthScaleInlet3BC",
    "TurbulentIntensityInlet3BC",
    "TurbulentKineticEnergyInlet4BC",
    "TurbulentDissipationInlet4BC",
    "TurbulentFrequencyInlet4BC",
    "TurbulentDissipationInlet5BC",
    "TurbulentFrequencyInlet5BC",
    # Phase 27: Enhanced v4 mapped/wave/turbulent inlets + v5-v7 variants
    "MappedFlowRate4BC",
    "PressureWaveTransmissive4BC",
    "TurbulentViscosityInlet4BC",
    "TurbulentLengthScaleInlet4BC",
    "TurbulentIntensityInlet4BC",
    "TurbulentDissipationInlet6BC",
    "TurbulentFrequencyInlet6BC",
    "TurbulentKineticEnergyInlet5BC",
    "TurbulentDissipationInlet7BC",
    "TurbulentFrequencyInlet7BC",
    # Phase 28: Enhanced v5 mapped/wave/turbulent inlets + v6 k + v8 epsilon/omega +
    #          enhanced outlet phase mean velocity + enhanced scaled heat flux
    "MappedFlowRate5BC",
    "PressureWaveTransmissive5BC",
    "TurbulentViscosityInlet5BC",
    "TurbulentLengthScaleInlet5BC",
    "TurbulentIntensityInlet5BC",
    "TurbulentKineticEnergyInlet6BC",
    "TurbulentDissipationInlet8BC",
    "TurbulentFrequencyInlet8BC",
    "OutletPhaseMeanVelocity2BC",
    "ScaledHeatFlux2BC",
    # Phase 29: Enhanced v6 mapped/wave/turbulent inlets + v7 k + v9 epsilon/omega +
    #          enhanced v3 outlet phase mean velocity + enhanced v3 scaled heat flux
    "MappedFlowRate6BC",
    "PressureWaveTransmissive6BC",
    "TurbulentViscosityInlet6BC",
    "TurbulentLengthScaleInlet6BC",
    "TurbulentIntensityInlet6BC",
    "TurbulentKineticEnergyInlet7BC",
    "TurbulentDissipationInlet9BC",
    "TurbulentFrequencyInlet9BC",
    "OutletPhaseMeanVelocity3BC",
    "ScaledHeatFlux3BC",
    # Phase 30: Enhanced v7 mapped/wave/turbulent inlets + v8 k + v10 epsilon/omega +
    #          enhanced v4 outlet phase mean velocity + enhanced v4 scaled heat flux
    "MappedFlowRate7BC",
    "PressureWaveTransmissive7BC",
    "TurbulentViscosityInlet7BC",
    "TurbulentLengthScaleInlet7BC",
    "TurbulentIntensityInlet7BC",
    "TurbulentKineticEnergyInlet8BC",
    "TurbulentDissipationInlet10BC",
    "TurbulentFrequencyInlet10BC",
    "OutletPhaseMeanVelocity4BC",
    "ScaledHeatFlux4BC",
    # Phase 31: Enhanced v8 mapped/wave/turbulent inlets + v9 k + v11 epsilon/omega +
    #          enhanced v5 outlet phase mean velocity + enhanced v5 scaled heat flux
    "MappedFlowRate8BC",
    "PressureWaveTransmissive8BC",
    "TurbulentViscosityInlet8BC",
    "TurbulentLengthScaleInlet8BC",
    "TurbulentIntensityInlet8BC",
    "TurbulentKineticEnergyInlet9BC",
    "TurbulentDissipationInlet11BC",
    "TurbulentFrequencyInlet11BC",
    "OutletPhaseMeanVelocity5BC",
    "ScaledHeatFlux5BC",
    # Phase 32: Enhanced v9 mapped/wave/turbulent inlets + v10 k + v12 epsilon/omega +
    #          enhanced v6 outlet phase mean velocity + enhanced v6 scaled heat flux
    "MappedFlowRate9BC",
    "PressureWaveTransmissive9BC",
    "TurbulentViscosityInlet9BC",
    "TurbulentLengthScaleInlet9BC",
    "TurbulentIntensityInlet9BC",
    "TurbulentKineticEnergyInlet10BC",
    "TurbulentDissipationInlet12BC",
    "TurbulentFrequencyInlet12BC",
    "OutletPhaseMeanVelocity6BC",
    "ScaledHeatFlux6BC",
    # Phase 33: Enhanced v10 mapped/wave/turbulent inlets + v11 k + v13 epsilon/omega +
    #          enhanced v7 outlet phase mean velocity + enhanced v7 scaled heat flux
    "MappedFlowRate10BC",
    "PressureWaveTransmissive10BC",
    "TurbulentViscosityInlet10BC",
    "TurbulentLengthScaleInlet10BC",
    "TurbulentIntensityInlet10BC",
    "TurbulentKineticEnergyInlet11BC",
    "TurbulentDissipationInlet13BC",
    "TurbulentFrequencyInlet13BC",
    "OutletPhaseMeanVelocity7BC",
    "ScaledHeatFlux7BC",
    "CoupledTemperature10BC",
    "CoupledTemperature9BC",
    "CoupledTemperature8BC",
    "CoupledTemperature7BC",
    "CoupledTemperature6BC",
    "CoupledTemperature5BC",
    "CoupledTemperature4BC",
    "CoupledTemperature3BC",
    "CoupledTemperature2BC",
    "PhaseMeanVelocity10BC",
    "PhaseMeanVelocity9BC",
    "PhaseMeanVelocity8BC",
    "PhaseMeanVelocity7BC",
    "PhaseMeanVelocity6BC",
    "PhaseMeanVelocity5BC",
    "PhaseMeanVelocity4BC",
    "PhaseMeanVelocity3BC",
    "PhaseMeanVelocity2BC",
    "Slip10BC",
    "Slip9BC",
    "Slip8BC",
    "Slip7BC",
    "Slip6BC",
    "Slip5BC",
    "Slip4BC",
    "Slip3BC",
    "Slip2BC",
    "Wedge10BC",
    "Wedge9BC",
    "Wedge8BC",
    "Wedge7BC",
    "Wedge6BC",
    "Wedge5BC",
    "Wedge4BC",
    "Wedge3BC",
    "Wedge2BC",
    "ProcessorCyclic10BC",
    "ProcessorCyclic9BC",
    "ProcessorCyclic8BC",
    "ProcessorCyclic7BC",
    "ProcessorCyclic6BC",
    "ProcessorCyclic5BC",
    "ProcessorCyclic4BC",
    "ProcessorCyclic3BC",
    "ProcessorCyclic2BC",
    "NonConformalCouple10BC",
    "NonConformalCouple9BC",
    "NonConformalCouple8BC",
    "NonConformalCouple7BC",
    "NonConformalCouple6BC",
    "NonConformalCouple5BC",
    "NonConformalCouple4BC",
    "NonConformalCouple3BC",
    "NonConformalCouple2BC",
    "TurbulentKineticEnergyInlet16BC",
    "TurbulentKineticEnergyInlet15BC",
    "TurbulentKineticEnergyInlet14BC",
    "TurbulentKineticEnergyInlet13BC",
    "TurbulentKineticEnergyInlet12BC",
    "ScaledHeatFlux16BC",
    "ScaledHeatFlux15BC",
    "ScaledHeatFlux14BC",
    "ScaledHeatFlux13BC",
    "ScaledHeatFlux12BC",
    "ScaledHeatFlux11BC",
    "ScaledHeatFlux10BC",
    "ScaledHeatFlux9BC",
    "ScaledHeatFlux8BC",
    "OutletPhaseMeanVelocity16BC",
    "OutletPhaseMeanVelocity15BC",
    "OutletPhaseMeanVelocity14BC",
    "OutletPhaseMeanVelocity13BC",
    "OutletPhaseMeanVelocity12BC",
    "OutletPhaseMeanVelocity11BC",
    "OutletPhaseMeanVelocity10BC",
    "OutletPhaseMeanVelocity9BC",
    "OutletPhaseMeanVelocity8BC",
    "TurbulentFrequencyInlet16BC",
    "TurbulentFrequencyInlet15BC",
    "TurbulentFrequencyInlet14BC",
    "TurbulentDissipationInlet16BC",
    "TurbulentDissipationInlet15BC",
    "TurbulentDissipationInlet14BC",
    "TurbulentLengthScaleInlet16BC",
    "TurbulentLengthScaleInlet15BC",
    "TurbulentLengthScaleInlet14BC",
    "TurbulentLengthScaleInlet13BC",
    "TurbulentLengthScaleInlet12BC",
    "TurbulentLengthScaleInlet11BC",
    "TurbulentViscosityInlet16BC",
    "TurbulentViscosityInlet15BC",
    "TurbulentViscosityInlet14BC",
    "TurbulentViscosityInlet13BC",
    "TurbulentViscosityInlet12BC",
    "TurbulentViscosityInlet11BC",
    "TurbulentIntensityInlet16BC",
    "TurbulentIntensityInlet15BC",
    "TurbulentIntensityInlet14BC",
    "TurbulentIntensityInlet13BC",
    "TurbulentIntensityInlet12BC",
    "TurbulentIntensityInlet11BC",
    "PressureWaveTransmissive16BC",
    "PressureWaveTransmissive15BC",
    "PressureWaveTransmissive14BC",
    "PressureWaveTransmissive13BC",
    "PressureWaveTransmissive12BC",
    "PressureWaveTransmissive11BC",
    "MappedFlowRate16BC",
    "MappedFlowRate15BC",
    "MappedFlowRate14BC",
    "MappedFlowRate13BC",
    "MappedFlowRate12BC",
    "MappedFlowRate11BC",
]
