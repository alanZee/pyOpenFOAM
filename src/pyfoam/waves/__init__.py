"""
pyfoam.waves — Wave models for coastal and offshore engineering.

Implements wave theories used in wave generation and absorption:

- :class:`~pyfoam.waves.wave_model.WaveModel` — abstract base with RTS registry.
- :class:`~pyfoam.waves.airy.AiryWave` — linear Airy wave theory.
- :class:`~pyfoam.waves.stokes.StokesWave` — 2nd-order Stokes wave theory.
- :class:`~pyfoam.waves.cnoidal.CnoidalWave` — cnoidal wave theory (shallow water).
- :class:`~pyfoam.waves.regular_wave.RegularWave` — multi-component superposition.

Enhanced wave models:

- :mod:`~pyfoam.waves.enhanced_2` — IrregularWave, DirectionalWave, SolitaryWave.
- :mod:`~pyfoam.waves.enhanced_3` — StreamFunctionWave, BoussinesqWave, MildSlopeWave.
- :mod:`~pyfoam.waves.enhanced_4` — SpectralWave, WaveTrain, RogueWave.
- :mod:`~pyfoam.waves.enhanced_5` — ReflectedWave, DiffractedWave, AbsorptionModel.
- :mod:`~pyfoam.waves.enhanced_6` — ActiveAbsorption, PassiveAbsorption.
- :mod:`~pyfoam.waves.enhanced_7` — RelaxationZone, WaveGenerationModel.
- :mod:`~pyfoam.waves.enhanced_8` — PistonType, FlapType.
- :mod:`~pyfoam.waves.enhanced_9` — PressureType, IrregularGeneration.
- :mod:`~pyfoam.waves.enhanced_10` — AbsorptionGeneration, FlapDiffraction.
"""

from pyfoam.waves.wave_model import WaveModel
from pyfoam.waves.airy import AiryWave
from pyfoam.waves.stokes import StokesWave
from pyfoam.waves.cnoidal import CnoidalWave
from pyfoam.waves.regular_wave import RegularWave

# 增强型波浪模型（触发 RTS 注册）
from pyfoam.waves.enhanced_2 import IrregularWave, DirectionalWave, SolitaryWave
from pyfoam.waves.enhanced_3 import StreamFunctionWave, BoussinesqWave, MildSlopeWave
from pyfoam.waves.enhanced_4 import SpectralWave, WaveTrain, RogueWave
from pyfoam.waves.enhanced_5 import ReflectedWave, DiffractedWave, AbsorptionModel
from pyfoam.waves.enhanced_6 import ActiveAbsorption, PassiveAbsorption
from pyfoam.waves.enhanced_7 import RelaxationZone, WaveGenerationModel
from pyfoam.waves.enhanced_8 import PistonType, FlapType
from pyfoam.waves.enhanced_9 import PressureType, IrregularGeneration
from pyfoam.waves.enhanced_10 import AbsorptionGeneration, FlapDiffraction

__all__ = [
    "WaveModel",
    "AiryWave",
    "StokesWave",
    "CnoidalWave",
    "RegularWave",
    # enhanced_2
    "IrregularWave",
    "DirectionalWave",
    "SolitaryWave",
    # enhanced_3
    "StreamFunctionWave",
    "BoussinesqWave",
    "MildSlopeWave",
    # enhanced_4
    "SpectralWave",
    "WaveTrain",
    "RogueWave",
    # enhanced_5
    "ReflectedWave",
    "DiffractedWave",
    "AbsorptionModel",
    # enhanced_6
    "ActiveAbsorption",
    "PassiveAbsorption",
    # enhanced_7
    "RelaxationZone",
    "WaveGenerationModel",
    # enhanced_8
    "PistonType",
    "FlapType",
    # enhanced_9
    "PressureType",
    "IrregularGeneration",
    # enhanced_10
    "AbsorptionGeneration",
    "FlapDiffraction",
]
