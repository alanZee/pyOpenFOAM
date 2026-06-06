"""
Tutorial validation: parallel and wave smoke tests.

验证并行计算和波浪模型的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestParallelSmoke:
    """并行计算 smoke 测试。"""

    def test_decomposition_import(self):
        """Decomposition 可导入。"""
        from pyfoam.parallel import Decomposition
        assert Decomposition is not None

    def test_halo_exchange_import(self):
        """HaloExchange 可导入。"""
        from pyfoam.parallel import HaloExchange
        assert HaloExchange is not None


class TestWaveSmoke:
    """波浪模型 smoke 测试。"""

    def test_airy_wave_import(self):
        """AiryWave 可导入。"""
        from pyfoam.waves import AiryWave
        assert AiryWave is not None

    def test_stokes_wave_import(self):
        """StokesWave 可导入。"""
        from pyfoam.waves import StokesWave
        assert StokesWave is not None

    def test_cnoidal_wave_import(self):
        """CnoidalWave 可导入。"""
        from pyfoam.waves import CnoidalWave
        assert CnoidalWave is not None
