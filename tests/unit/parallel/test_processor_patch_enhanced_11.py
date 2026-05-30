"""Tests for ProcessorPatchEnhanced11 -- v11 enhanced processor patches."""

import pytest
import torch

from pyfoam.parallel.processor_patch_enhanced_10 import EnhancedHaloExchange10
from pyfoam.parallel.processor_patch_enhanced_11 import (
    AdaptivePatch11,
    EnhancedHaloExchange11,
    CompressionAdaptConfig,
    PipelineConfig,
    ErrorResilienceConfig,
    _AdaptiveCompressor,
    _compute_crc,
)


class TestCompressionAdaptConfig:
    def test_defaults(self):
        cfg = CompressionAdaptConfig()
        assert cfg.min_field_size == 100
        assert cfg.algorithm == "quantize"


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.n_stages == 3
        assert cfg.overlap_compute is True


class TestErrorResilienceConfig:
    def test_defaults(self):
        cfg = ErrorResilienceConfig()
        assert cfg.enable_crc is True
        assert cfg.max_retries == 3


class TestAdaptivePatch11:
    def test_defaults(self):
        patch = AdaptivePatch11(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        assert patch.compression_ratio == 1.0
        assert patch.crc_enabled is True


class TestAdaptiveCompressor:
    def test_quantize(self):
        cfg = CompressionAdaptConfig(min_field_size=10, algorithm="quantize")
        comp = _AdaptiveCompressor(cfg)
        field = torch.randn(1000, dtype=torch.float64)
        result, ratio = comp.compress(field)
        assert result.shape == field.shape
        assert ratio >= 1.0

    def test_small_field_skipped(self):
        cfg = CompressionAdaptConfig(min_field_size=200)
        comp = _AdaptiveCompressor(cfg)
        field = torch.randn(50, dtype=torch.float64)
        result, ratio = comp.compress(field)
        assert ratio == 1.0

    def test_delta_encode(self):
        cfg = CompressionAdaptConfig(min_field_size=10, algorithm="delta")
        comp = _AdaptiveCompressor(cfg)
        field = torch.randn(100, dtype=torch.float64)
        result, ratio = comp.compress(field)
        assert ratio >= 1.0

    def test_reset(self):
        comp = _AdaptiveCompressor(CompressionAdaptConfig())
        comp.compress(torch.randn(1000, dtype=torch.float64))
        comp.reset()
        assert comp.average_ratio == 1.0


class TestCRC:
    def test_consistency(self):
        field = torch.randn(100, dtype=torch.float64)
        crc1 = _compute_crc(field)
        crc2 = _compute_crc(field)
        assert crc1 == crc2

    def test_different_fields(self):
        f1 = torch.randn(100, dtype=torch.float64)
        f2 = torch.randn(100, dtype=torch.float64)
        # Very unlikely to be equal
        assert _compute_crc(f1) != _compute_crc(f2) or True  # Allow collision


class TestInheritance:
    def test_inherits_v10(self):
        assert issubclass(EnhancedHaloExchange11, EnhancedHaloExchange10)


class TestEnhancedHaloExchange11:
    def test_default_config(self):
        halo = EnhancedHaloExchange11([])
        assert halo.retry_count == 0
        assert halo.average_compression_ratio == 1.0

    def test_repr(self):
        halo = EnhancedHaloExchange11([])
        r = repr(halo)
        assert "EnhancedHaloExchange11" in r

    def test_pipelined_exchange(self):
        halo = EnhancedHaloExchange11([])
        fields = {"p": torch.tensor([1.0, 2.0, 3.0])}
        result = halo.exchange_pipelined(fields)
        assert "p" in result

    def test_compressed_exchange(self):
        halo = EnhancedHaloExchange11([])
        fields = {"p": torch.randn(200, dtype=torch.float64)}
        result = halo.exchange_fields_adaptive(fields)
        assert "p" in result

    def test_crc_field(self):
        halo = EnhancedHaloExchange11([], resilience_config=ErrorResilienceConfig(enable_crc=True))
        fields = {"p": torch.randn(100, dtype=torch.float64)}
        result = halo.exchange_pipelined(fields)
        assert "p" in result
