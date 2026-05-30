"""Tests for surface_convert_enhanced_8 — enhanced format conversion v8."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_convert_enhanced_8 import (
    ConvertEnhanced8Result, FormatDetection, CompressionInfo,
    surface_convert_enhanced_8,
)


class TestSurfaceConvert8:
    def test_returns_result_type(self):
        r = ConvertEnhanced8Result()
        assert isinstance(r, ConvertEnhanced8Result)

    def test_format_detection_type(self):
        fd = FormatDetection()
        assert fd.detected_format == ""
        assert fd.confidence == 0.0

    def test_compression_type(self):
        ci = CompressionInfo()
        assert ci.codec == "none"
        assert ci.compression_ratio == 1.0

    def test_default_fields(self):
        r = ConvertEnhanced8Result()
        assert r.n_parallel_inputs == 0
        assert isinstance(r.format_detection, FormatDetection)
        assert isinstance(r.compression, CompressionInfo)
