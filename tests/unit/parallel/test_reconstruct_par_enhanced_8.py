"""Tests for ReconstructParEnhanced8 -- v8 enhanced parallel reconstruction."""

import pytest
import torch

from pyfoam.parallel.reconstruct_par_enhanced_7 import ReconstructParEnhanced7
from pyfoam.parallel.reconstruct_par_enhanced_8 import (
    ReconstructParEnhanced8,
    V8ReconstructResult,
    StreamingConfig,
    EntropyConfig,
    FieldCorrelation,
)


class TestStreamingConfig:
    """Test StreamingConfig dataclass."""

    def test_defaults(self):
        cfg = StreamingConfig()
        assert cfg.chunk_size == 10000
        assert cfg.enable_prefetch is True


class TestEntropyConfig:
    """Test EntropyConfig dataclass."""

    def test_defaults(self):
        cfg = EntropyConfig()
        assert cfg.entropy_threshold == 3.0
        assert cfg.adaptive is True


class TestFieldCorrelation:
    """Test FieldCorrelation dataclass."""

    def test_defaults(self):
        fc = FieldCorrelation()
        assert fc.correlation == 0.0
        assert fc.shared_structure is True


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v7(self):
        assert issubclass(ReconstructParEnhanced8, ReconstructParEnhanced7)


class TestEntropyComputation:
    """Test field entropy computation."""

    def test_constant_field_zero_entropy(self):
        field = torch.ones(100, dtype=torch.float64)
        entropy = ReconstructParEnhanced8.compute_field_entropy(field)
        assert entropy == 0.0

    def test_random_field_positive_entropy(self):
        torch.manual_seed(42)
        field = torch.randn(1000, dtype=torch.float64)
        entropy = ReconstructParEnhanced8.compute_field_entropy(field)
        assert entropy > 0.0


class TestCorrelation:
    """Test field correlation computation."""

    def test_perfect_correlation(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a * 2.0
        corr = ReconstructParEnhanced8.compute_correlation(a, b)
        assert abs(corr - 1.0) < 1e-6

    def test_negative_correlation(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        b = -a
        corr = ReconstructParEnhanced8.compute_correlation(a, b)
        assert abs(corr - (-1.0)) < 1e-6


class TestAdaptiveCompression:
    """Test adaptive compression level selection."""

    def test_low_entropy_high_compression(self):
        recon = ReconstructParEnhanced8("/tmp/test")
        level = recon._adaptive_compression_level(0.5)
        assert level == recon._entropy_config.max_compression_level

    def test_high_entropy_low_compression(self):
        recon = ReconstructParEnhanced8("/tmp/test")
        level = recon._adaptive_compression_level(10.0)
        assert level == recon._entropy_config.min_compression_level


class TestV8Reconstruction:
    """Test v8 reconstruction method."""

    def test_returns_v8_result(self, tmp_path):
        # Create minimal processor directory structure
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced8(tmp_path)
        result = recon.reconstruct_case_v8(
            field_names=["p"],
            streaming=True,
        )
        assert isinstance(result, V8ReconstructResult)


class TestRepr:
    """Test string representations."""

    def test_repr(self, tmp_path):
        recon = ReconstructParEnhanced8(tmp_path)
        r = repr(recon)
        assert "ReconstructParEnhanced8" in r
