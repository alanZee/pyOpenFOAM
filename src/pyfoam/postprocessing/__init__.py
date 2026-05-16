"""
pyfoam.postprocessing — Post-processing tools and function objects.

Provides:

- :class:`FunctionObject` — base class for all function objects
- :class:`Forces` — force and moment calculation on patches
- :class:`ForceCoeffs` — force coefficient calculation (Cd, Cl, Cm)
- :class:`WallShearStress` — wall shear stress computation
- :class:`YPlus` — y+ calculation for wall-bounded flows
- :class:`FieldOperations` — grad, div, curl field operations
- :class:`Probes` — point probe sampling
- :class:`LineSample` — line sampling (sets)
- :class:`SurfaceSample` — surface sampling
- :class:`VTKWriter` — VTK file output
- :class:`FoamToVTK` — case-level VTK conversion

All function objects follow OpenFOAM's function object API and can be
configured via dictionary entries in ``system/controlDict``.
"""

from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry
from pyfoam.postprocessing.forces import Forces, ForceCoeffs
from pyfoam.postprocessing.wall_shear_stress import WallShearStress
from pyfoam.postprocessing.y_plus import YPlus
from pyfoam.postprocessing.field_operations import FieldOperations
from pyfoam.postprocessing.sampling import Probes, LineSample, SurfaceSample
from pyfoam.postprocessing.vtk_output import VTKWriter, FoamToVTK

__all__ = [
    # Framework
    "FunctionObject",
    "FunctionObjectRegistry",
    # Forces
    "Forces",
    "ForceCoeffs",
    # Wall quantities
    "WallShearStress",
    "YPlus",
    # Field operations
    "FieldOperations",
    # Sampling
    "Probes",
    "LineSample",
    "SurfaceSample",
    # VTK output
    "VTKWriter",
    "FoamToVTK",
]
