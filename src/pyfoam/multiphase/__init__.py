"""
pyfoam.multiphase — Multiphase flow models.

Provides:

- :class:`VOFAdvection` — Volume of Fluid advection with interface compression
"""

from pyfoam.multiphase.volume_of_fluid import VOFAdvection

__all__ = [
    "VOFAdvection",
]
