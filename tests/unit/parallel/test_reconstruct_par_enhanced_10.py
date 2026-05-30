"""Tests for ReconstructParEnhanced10 -- v10 enhanced parallel reconstruction."""

import pytest
import torch

from pyfoam.parallel.reconstruct_par_enhanced_9 import ReconstructParEnhanced9
from pyfoam.parallel.reconstruct_par_enhanced_10 import (
    ReconstructParEnhanced10,
    V10ReconstructResult,
    SpectralAnalysisConfig,
    CompressionConfig,
    ProvenanceEntry,
    _compute_spectral_quality,
    _lossy_compress,
)


class TestSpectralAnalysisConfig:
    def test_defaults(self):
        cfg = SpectralAnalysisConfig()
        assert cfg.n_modes == 64
        assert cfg.energy_threshold == 0.99


class TestCompressionConfig:
    def test_defaults(self):
        cfg = CompressionConfig()
        assert cfg.error_bound == 1e-6
        assert cfg.min_compression_ratio == 2.0


class TestProvenanceEntry:
    def test_defaults(self):
        entry = ProvenanceEntry()
        assert entry.field_name == ""
        assert entry.checksum_before == 0.0


class TestInheritance:
    def test_inherits_v9(self):
        assert issubclass(ReconstructParEnhanced10, ReconstructParEnhanced9)


class TestSpectralQuality:
    def test_compute_spectral_quality(self):
        field = torch.randn(100, dtype=torch.float64)
        quality, energy = _compute_spectral_quality(field, n_modes=16)
        assert 0.0 <= quality <= 1.0
        assert 0.0 <= energy <= 1.0

    def test_small_field(self):
        field = torch.tensor([1.0, 2.0], dtype=torch.float64)
        quality, energy = _compute_spectral_quality(field)
        assert quality == 1.0


class TestLossyCompress:
    def test_compression(self):
        field = torch.randn(1000, dtype=torch.float64)
        compressed, ratio = _lossy_compress(field, error_bound=1e-3)
        assert compressed.shape == field.shape
        assert ratio >= 1.0

    def test_uniform_field(self):
        field = torch.ones(100, dtype=torch.float64)
        compressed, ratio = _lossy_compress(field)
        assert ratio == 1.0


class TestFieldPruning:
    def test_should_prune_identical(self):
        f1 = torch.tensor([1.0, 2.0, 3.0])
        f2 = torch.tensor([1.0, 2.0, 3.0])
        assert ReconstructParEnhanced10.should_prune_field(f1, f2) is True

    def test_should_not_prune_different(self):
        f1 = torch.tensor([1.0, 2.0, 3.0])
        f2 = torch.tensor([1.0, 2.0, 4.0])
        assert ReconstructParEnhanced10.should_prune_field(f1, f2) is False


class TestV10Reconstruction:
    def test_returns_v10_result(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced10(tmp_path)
        result = recon.reconstruct_case_v10(field_names=["p", "U"])
        assert isinstance(result, V10ReconstructResult)

    def test_with_spectral_analysis(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced10(tmp_path)
        result = recon.reconstruct_case_v10(
            field_names=["p"],
            spectral_analysis=True,
        )
        assert result.spectral_quality >= 0

    def test_with_compression(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced10(tmp_path)
        result = recon.reconstruct_case_v10(
            field_names=["p"],
            error_bounded_compression=True,
            error_bound=1e-3,
        )
        assert result.compression_ratio >= 1.0

    def test_with_provenance(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced10(tmp_path)
        result = recon.reconstruct_case_v10(
            field_names=["p"],
            compute_provenance=True,
        )
        assert len(result.provenance) > 0


class TestRepr:
    def test_repr(self, tmp_path):
        recon = ReconstructParEnhanced10(tmp_path)
        r = repr(recon)
        assert "ReconstructParEnhanced10" in r
