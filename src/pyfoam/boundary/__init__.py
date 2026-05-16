"""
pyfoam.boundary — Boundary condition hierarchy with RTS selection.

Provides:

- :class:`BoundaryCondition` — abstract base class with RTS registry
- :class:`BoundaryField` — collection of BCs for a field
- Concrete BCs: fixedValue, zeroGradient, noSlip, fixedGradient,
  symmetryPlane, cyclic, nutkWallFunction, kqRWallFunction, inletOutlet,
  alphaContactAngle, constantAlphaContactAngle
- Velocity BCs: flowRateInletVelocity, pressureInletOutletVelocity,
  rotatingWallVelocity
- Pressure BCs: totalPressure, fixedFluxPressure, prghPressure,
  waveTransmissive
- Turbulence BCs: turbulentIntensityKineticEnergyInlet,
  turbulentMixingLengthDissipationRateInlet,
  turbulentMixingLengthFrequencyInlet
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
from pyfoam.boundary.wall_function import KqRWallFunctionBC, NutkWallFunctionBC
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
    "KqRWallFunctionBC",
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
]
