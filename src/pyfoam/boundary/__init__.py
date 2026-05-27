"""
pyfoam.boundary — Boundary condition hierarchy with RTS selection.

Provides:

- :class:`BoundaryCondition` — abstract base class with RTS registry
- :class:`BoundaryField` — collection of BCs for a field
- Concrete BCs: fixedValue, zeroGradient, noSlip, fixedGradient,
  symmetry, symmetryPlane, cyclic, nutkWallFunction, nutLowReWallFunction,
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
]
