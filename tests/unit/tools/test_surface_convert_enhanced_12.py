"""Tests for surface_convert_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_convert_enhanced_12 import ConvertEnhanced12Result, CompressionResult, MultiLODResult, surface_convert_enhanced_12


class TestConvertEnhanced12Result:
    def test_returns_result(self):
        r = surface_convert_enhanced_12()
        assert isinstance(r, ConvertEnhanced12Result)

    def test_compression(self):
        r = surface_convert_enhanced_12(enable_compression=True)
        assert isinstance(r.compression, CompressionResult)
        assert r.compression.enabled is True

    def test_multi_lod(self):
        r = surface_convert_enhanced_12(enable_multi_lod=True)
        assert isinstance(r.multi_lod, MultiLODResult)
        assert r.multi_lod.enabled is True
