"""
pyfoam.multiphase — Multiphase flow models.

Provides:

- :class:`VOFAdvection` — Volume of Fluid advection with interface compression
- :class:`MULESLimiter` — Bounded scalar transport limiter
- :class:`SurfaceTensionModel` — Continuum Surface Force (CSF) model
- Interphase models: SchillerNaumannDrag, WenYuDrag, GidaspowDrag,
  TomiyamaLift, VirtualMassForce
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
]
