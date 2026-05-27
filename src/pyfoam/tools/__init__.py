"""
pyfoam.tools — Mesh quality checking and field initialisation utilities.

Mirrors OpenFOAM command-line tools:

- :func:`check_mesh` — validate mesh quality (orthogonality, skewness, etc.)
- :func:`set_fields` — initialise field values based on geometric regions
"""

from pyfoam.tools.check_mesh import CheckMeshResult, check_mesh
from pyfoam.tools.set_fields import BoxRegion, CylinderRegion, set_fields

__all__ = [
    "CheckMeshResult",
    "check_mesh",
    "BoxRegion",
    "CylinderRegion",
    "set_fields",
]
