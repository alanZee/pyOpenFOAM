"""
pyfoam.tools — Mesh quality checking, field initialisation, and utility tools.

Mirrors OpenFOAM command-line tools:

- :func:`check_mesh` — validate mesh quality (orthogonality, skewness, etc.)
- :func:`set_fields` — initialise field values based on geometric regions
- :func:`foam_dictionary` — query or modify dictionary entries
- :func:`renumber_mesh` — renumber cells using Reverse Cuthill-McKee
- :func:`transform_points` — transform mesh vertex coordinates
- :func:`foam_list_times` — list time directories in a case
"""

from pyfoam.tools.check_mesh import CheckMeshResult, check_mesh
from pyfoam.tools.foam_dictionary import foam_dictionary
from pyfoam.tools.foam_list_times import foam_list_times
from pyfoam.tools.renumber_mesh import RenumberResult, renumber_mesh
from pyfoam.tools.set_fields import BoxRegion, CylinderRegion, set_fields
from pyfoam.tools.transform_points import transform_points

__all__ = [
    "CheckMeshResult",
    "check_mesh",
    "BoxRegion",
    "CylinderRegion",
    "set_fields",
    "foam_dictionary",
    "foam_list_times",
    "RenumberResult",
    "renumber_mesh",
    "transform_points",
]
