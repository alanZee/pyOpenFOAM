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
]
