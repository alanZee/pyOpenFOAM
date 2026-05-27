"""
pyfoam.waves — Wave models for coastal and offshore engineering.

Implements wave theories used in wave generation and absorption:

- :class:`~pyfoam.waves.wave_model.WaveModel` — abstract base with RTS registry.
- :class:`~pyfoam.waves.airy.AiryWave` — linear Airy wave theory.
- :class:`~pyfoam.waves.stokes.StokesWave` — 2nd-order Stokes wave theory.
- :class:`~pyfoam.waves.cnoidal.CnoidalWave` — cnoidal wave theory (shallow water).
- :class:`~pyfoam.waves.regular_wave.RegularWave` — multi-component superposition.
"""

from pyfoam.waves.wave_model import WaveModel
from pyfoam.waves.airy import AiryWave
from pyfoam.waves.stokes import StokesWave
from pyfoam.waves.cnoidal import CnoidalWave
from pyfoam.waves.regular_wave import RegularWave

__all__ = [
    "WaveModel",
    "AiryWave",
    "StokesWave",
    "CnoidalWave",
    "RegularWave",
]
