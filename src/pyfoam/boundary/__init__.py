"""
pyfoam.boundary — Boundary condition hierarchy with RTS selection.

Provides:

- :class:`BoundaryCondition` — abstract base class with RTS registry
- :class:`BoundaryField` — collection of BCs for a field
- Concrete BCs: fixedValue, zeroGradient, noSlip, fixedGradient,
  symmetryPlane, cyclic, nutkWallFunction, kqRWallFunction, inletOutlet
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

__all__ = [
    # Base
    "BoundaryCondition",
    "Patch",
    "BoundaryField",
    # Concrete BCs
    "FixedValueBC",
    "ZeroGradientBC",
    "NoSlipBC",
    "FixedGradientBC",
    "SymmetryBC",
    "CyclicBC",
    "NutkWallFunctionBC",
    "KqRWallFunctionBC",
    "InletOutletBC",
]
