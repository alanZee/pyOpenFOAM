"""
pyfoam.multiphase — Multiphase flow models.

Provides:

- :class:`VOFAdvection` — Volume of Fluid advection with interface compression
- :class:`MULESLimiter` — Bounded scalar transport limiter
- :class:`SurfaceTensionModel` — Continuum Surface Force (CSF) model
- Interphase models: SchillerNaumannDrag, WenYuDrag, GidaspowDrag,
  TomiyamaLift, VirtualMassForce
- Drag model ABC: :class:`DragModel` with RTS registry
  (``pyfoam.multiphase.drag_models``)
- Lift model ABC: :class:`LiftModel` with RTS registry
  (``pyfoam.multiphase.lift_models``)
- :class:`SaffmanLift` — Saffman shear-induced lift force
- Cavitation models: SchnerrSauer, Merkle, ZGB
- Enhanced cavitation models: ZGBModel, MerkleModel (with convergence enhancements)
- Interface reconstruction: PLICReconstruction
- :class:`PopulationBalanceModel` — Population balance equation solver
  (method of classes, droplet/bubble size distribution tracking)
- :class:`BubbleModel` — Abstract bubble diameter model with RTS registry
- :class:`ConstantBubble` — Constant (fixed) bubble diameter
- :class:`BubbleBreakup` — Bubble breakup/coalescence equilibrium model
- :class:`PhaseChangeModel` — Abstract phase change model with RTS registry
- :class:`LeeModel` — Lee empirical phase change model
- :class:`SchnerrSauerEnhanced` — Enhanced Schnerr-Sauer with convergence improvements
- :class:`TurbulenceDampingModel` — Abstract turbulence damping model with RTS registry
- :class:`InterfaceDamping` — Damps k and epsilon near free surface
- :class:`InterfacialAreaModel` — Abstract interfacial area density model with RTS registry
- :class:`ConstantInterfacialArea` — Constant interfacial area density
- :class:`VariableInterfacialArea` — Alpha-dependent interfacial area density
- :class:`RelativeVelocityModel` — Abstract base for relative velocity models
- :class:`ManninenRelativeVelocity` — Manninen et al. algebraic slip model
- :class:`GraceRelativeVelocity` — Grace drag correlation for bubbles/particles
- :class:`TurbulenceInteractionModel` — Abstract interphase turbulence interaction model
- :class:`StandardInteraction` — Standard interphase turbulence interaction (Lopez de Bertodano)
- :class:`WallLubricationModel` — Abstract wall lubrication force model with RTS registry
- :class:`AntalWallLubrication` — Antal et al. (1991) distance-dependent wall lubrication
- :class:`TomiyamaWallLubrication` — Tomiyama et al. (1998) Eo-dependent wall lubrication
- :class:`TurbulenceDamping2Model` — Enhanced turbulence damping with y+ awareness
- :class:`WolfhardtDamping` — Wolfhardt model for near-wall turbulence damping
- :class:`TurbulenceWallDampingModel` — Abstract wall damping model for VOF with RTS registry
- :class:`BrackbillDamping` — Brackbill near-wall damping for VOF simulations
- :class:`VirtualMassModel` — Abstract virtual mass force model with RTS registry
- :class:`ConstantVirtualMass` — Constant virtual mass coefficient
- :class:`LambVirtualMass` — Lamb's inviscid virtual mass (C_vm = 0.5)
"""

from pyfoam.multiphase.volume_of_fluid import VOFAdvection
from pyfoam.multiphase.mules import MULESLimiter
from pyfoam.multiphase.surface_tension import SurfaceTensionModel
from pyfoam.multiphase.interphase_models import (
    SchillerNaumannDrag,
    WenYuDrag,
    GidaspowDrag,
    TomiyamaLift,
    VirtualMassForce,
)
from pyfoam.multiphase.cavitation import SchnerrSauer, Merkle, ZGB
from pyfoam.multiphase.interface_reconstruction import (
    InterfaceReconstruction,
    PLICReconstruction,
)
from pyfoam.multiphase.interface_compression import InterfaceCompression
from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension
from pyfoam.multiphase.population_balance import (
    PopulationBalanceModel,
    PBEBin,
    ConstantCoalescence,
    ShearCoalescence,
    ConstantBreakup,
    WeberBreakup,
    ShearBreakup,
)
from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel, MerkleModel
from pyfoam.multiphase.bubble_models import (
    BubbleModel,
    ConstantBubble,
    BubbleBreakup,
)
from pyfoam.multiphase.drift_flux_models import (
    DriftFluxModel,
    SimpleDriftFlux,
    GeneralDriftFlux,
)
from pyfoam.multiphase.phase_change import (
    PhaseChangeModel,
    LeeModel,
    SchnerrSauerEnhanced,
)
from pyfoam.multiphase.turbulence_damping import (
    TurbulenceDampingModel,
    InterfaceDamping,
)
from pyfoam.multiphase.mass_transfer import (
    MassTransferModel,
    LeeMassTransfer,
    ThermalPhaseChange,
)
from pyfoam.multiphase.interfacial_area import (
    InterfacialAreaModel,
    ConstantInterfacialArea,
    VariableInterfacialArea,
)
from pyfoam.multiphase.relative_velocity import (
    RelativeVelocityModel,
    ManninenRelativeVelocity,
    GraceRelativeVelocity,
)

# Phase 7: Drag and lift model ABC hierarchies
from pyfoam.multiphase.drag_models import DragModel
from pyfoam.multiphase.lift_models import LiftModel, SaffmanLift

# Phase 7: Turbulence transfer models
from pyfoam.multiphase.turbulence_transfer import (
    TurbulenceTransferModel,
    ContinuousTurbulenceTransfer,
    DispersedTurbulenceTransfer,
)

# Phase 7: Turbulence interaction models
from pyfoam.multiphase.turbulence_interaction import (
    TurbulenceInteractionModel,
    StandardInteraction,
)

# Phase 7: Wall lubrication models
from pyfoam.multiphase.wall_lubrication_models import (
    WallLubricationModel,
    AntalWallLubrication,
    TomiyamaWallLubrication,
)

# Phase 7: Enhanced turbulence damping models
from pyfoam.multiphase.turbulence_damping_2 import (
    TurbulenceDamping2Model,
    WolfhardtDamping,
)

# Phase 7: Wall damping models for VOF
from pyfoam.multiphase.turbulence_wall_damping import (
    TurbulenceWallDampingModel,
    BrackbillDamping,
)

# Phase 7: Virtual mass models
from pyfoam.multiphase.virtual_mass_models import (
    VirtualMassModel,
    ConstantVirtualMass,
    LambVirtualMass,
)

__all__ = [
    "VOFAdvection",
    "MULESLimiter",
    "SurfaceTensionModel",
    "SchillerNaumannDrag",
    "WenYuDrag",
    "GidaspowDrag",
    "TomiyamaLift",
    "VirtualMassForce",
    "SchnerrSauer",
    "Merkle",
    "ZGB",
    "InterfaceReconstruction",
    "PLICReconstruction",
    "InterfaceCompression",
    "CSFSurfaceTension",
    "PopulationBalanceModel",
    "PBEBin",
    "ConstantCoalescence",
    "ShearCoalescence",
    "ConstantBreakup",
    "WeberBreakup",
    "ShearBreakup",
    # Enhanced cavitation models
    "ZGBModel",
    "MerkleModel",
    # Bubble models
    "BubbleModel",
    "ConstantBubble",
    "BubbleBreakup",
    # Drift-flux models
    "DriftFluxModel",
    "SimpleDriftFlux",
    "GeneralDriftFlux",
    # Phase change models
    "PhaseChangeModel",
    "LeeModel",
    "SchnerrSauerEnhanced",
    # Turbulence damping models
    "TurbulenceDampingModel",
    "InterfaceDamping",
    # Mass transfer models
    "MassTransferModel",
    "LeeMassTransfer",
    "ThermalPhaseChange",
    # Interfacial area models
    "InterfacialAreaModel",
    "ConstantInterfacialArea",
    "VariableInterfacialArea",
    # Relative velocity models
    "RelativeVelocityModel",
    "ManninenRelativeVelocity",
    "GraceRelativeVelocity",
    # Phase 7: Drag and lift model ABCs
    "DragModel",
    "LiftModel",
    "SaffmanLift",
    # Phase 7: Turbulence transfer models
    "TurbulenceTransferModel",
    "ContinuousTurbulenceTransfer",
    "DispersedTurbulenceTransfer",
    # Phase 7: Turbulence interaction models
    "TurbulenceInteractionModel",
    "StandardInteraction",
    # Phase 7: Wall lubrication models
    "WallLubricationModel",
    "AntalWallLubrication",
    "TomiyamaWallLubrication",
    # Phase 7: Enhanced turbulence damping models
    "TurbulenceDamping2Model",
    "WolfhardtDamping",
    # Phase 7: Wall damping models for VOF
    "TurbulenceWallDampingModel",
    "BrackbillDamping",
    # Phase 7: Virtual mass models
    "VirtualMassModel",
    "ConstantVirtualMass",
    "LambVirtualMass",
]
