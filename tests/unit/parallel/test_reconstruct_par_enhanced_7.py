"""Tests for ReconstructParEnhanced7 -- v7 enhanced parallel reconstruction."""

import pytest
import torch
import math

from pyfoam.parallel.reconstruct_par_enhanced_7 import (
    ReconstructParEnhanced7,
    V7ReconstructResult,
    WaveletCompressionConfig,
    FieldQualityMetrics,
    AMRLevelInfo,
)
from pyfoam.parallel.reconstruct_par_enhanced_6 import ReconstructParEnhanced6


class TestWaveletCompressionConfig:
    """Test WaveletCompressionConfig dataclass."""

    def test_defaults(self):
        cfg = WaveletCompressionConfig()
        assert cfg.level == 2
        assert cfg.threshold == 0.01
        assert cfg.preserve_mean is True


class TestFieldQualityMetrics:
    """Test FieldQualityMetrics dataclass."""

    def test_defaults(self):
        m = FieldQualityMetrics()
        assert m.conservation_error == 0.0
        assert m.boundedness is True


class TestAMRLevelInfo:
    """Test AMRLevelInfo dataclass."""

    def test_defaults(self):
        info = AMRLevelInfo()
        assert info.level == 0
        assert info.n_cells == 0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v6(self):
        assert issubclass(ReconstructParEnhanced7, ReconstructParEnhanced6)


class TestWaveletCompress:
    """Test wavelet compression."""

    def test_constant_field_unchanged(self):
        """Constant field should be preserved after compression."""
        field = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float64)
        result, nnz = ReconstructParEnhanced7.wavelet_compress(
            field, level=1, threshold=0.01, preserve_mean=True,
        )
        assert result.shape == field.shape
        assert abs(result.mean().item() - 5.0) < 1e-10

    def test_small_field_no_error(self):
        """Single-element field should not error."""
        field = torch.tensor([42.0], dtype=torch.float64)
        result, nnz = ReconstructParEnhanced7.wavelet_compress(field, level=2)
        assert result.numel() == 1

    def test_compression_reduces_nonzeros(self):
        """Compression should reduce nonzero coefficients."""
        # 带噪声的信号
        torch.manual_seed(42)
        n = 64
        signal = torch.sin(torch.linspace(0, 2 * math.pi, n, dtype=torch.float64))
        noise = 0.01 * torch.randn(n, dtype=torch.float64)
        field = signal + noise

        result, nnz = ReconstructParEnhanced7.wavelet_compress(
            field, level=2, threshold=0.05,
        )
        # 大多数小波系数应被截断
        assert nnz < n


class TestFieldQuality:
    """Test field quality computation."""

    def test_identical_fields(self):
        field = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        m = ReconstructParEnhanced7.compute_field_quality(field, field.clone())
        assert m.conservation_error == pytest.approx(0.0)
        assert m.boundedness is True

    def test_shifted_field_conservation_error(self):
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        shifted = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        m = ReconstructParEnhanced7.compute_field_quality(original, shifted)
        assert m.conservation_error > 0


class TestV7Reconstruction:
    """Test v7 case reconstruction."""

    def test_repr(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced7(case_dir)
        r = repr(recon)
        assert "ReconstructParEnhanced7" in r
        assert "wavelet_level" in r

    def test_set_wavelet_config(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced7(case_dir)
        cfg = WaveletCompressionConfig(level=3, threshold=0.05)
        recon.set_wavelet_config(cfg)
        assert recon._wavelet_config.level == 3

    def test_set_amr_levels(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced7(case_dir)
        levels = [AMRLevelInfo(level=0, n_cells=100), AMRLevelInfo(level=1, n_cells=400)]
        recon.set_amr_levels(levels)
        assert len(recon._amr_levels) == 2

    def test_reconstruct_case_v7(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        # 创建 processor 目录避免 discover 报错
        for i in range(2):
            proc_dir = tmp_path / "case" / f"processor{i}"
            proc_dir.mkdir()
        recon = ReconstructParEnhanced7(case_dir)
        result = recon.reconstruct_case_v7(
            field_names=["p", "U"],
            compression_level=2,
        )
        assert isinstance(result, V7ReconstructResult)
        assert result.n_compressed == 2
