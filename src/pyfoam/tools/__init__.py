"""
pyfoam.tools — Mesh quality checking, field initialisation, and utility tools.

Mirrors OpenFOAM command-line tools:

- :func:`check_mesh` — validate mesh quality (orthogonality, skewness, etc.)
- :func:`set_fields` — initialise field values based on geometric regions
- :func:`foam_dictionary` — query or modify dictionary entries
- :func:`renumber_mesh` — renumber cells using Reverse Cuthill-McKee
"""

from pyfoam.tools.check_mesh import CheckMeshResult, check_mesh
from pyfoam.tools.foam_dictionary import foam_dictionary
from pyfoam.tools.renumber_mesh import RenumberResult, renumber_mesh
from pyfoam.tools.set_fields import BoxRegion, CylinderRegion, set_fields

__all__ = [
    "CheckMeshResult",
    "check_mesh",
    "BoxRegion",
    "CylinderRegion",
    "set_fields",
    "foam_dictionary",
    "RenumberResult",
    "renumber_mesh",
]
