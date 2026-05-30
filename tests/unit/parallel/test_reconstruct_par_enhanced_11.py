"""Tests for ReconstructParEnhanced11 -- v11 enhanced parallel reconstruction."""

import pytest
import torch

from pyfoam.parallel.reconstruct_par_enhanced_10 import ReconstructParEnhanced10
from pyfoam.parallel.reconstruct_par_enhanced_11 import (
    ReconstructParEnhanced11,
    V11ReconstructResult,
    WaveletAnalysisConfig,
    CheckpointAdaptConfig,
    FieldCorrelationResult,
    _haar_wavelet_decompose,
    _wavelet_energy_ratio,
    _estimate_checkpoint_frequency,
)


class TestWaveletAnalysisConfig:
    def test_defaults(self):
        cfg = WaveletAnalysisConfig()
        assert cfg.n_levels == 4
        assert cfg.wavelet_type == "haar"
        assert cfg.energy_threshold == 0.95


class TestCheckpointAdaptConfig:
    def test_defaults(self):
        cfg = CheckpointAdaptConfig()
        assert cfg.base_frequency == 10
        assert cfg.change_rate_threshold == 0.01


class TestFieldCorrelationResult:
    def test_defaults(self):
        result = FieldCorrelationResult()
        assert result.group_count == 0
        assert result.avg_correlation == 0.0


class TestInheritance:
    def test_inherits_v10(self):
        assert issubclass(ReconstructParEnhanced11, ReconstructParEnhanced10)


class TestHaarWaveletDecompose:
    def test_decomposition_levels(self):
        field = torch.randn(64, dtype=torch.float64)
        details, approx = _haar_wavelet_decompose(field, n_levels=4)
        assert len(details) == 4

    def test_approx_size_halved(self):
        field = torch.randn(64, dtype=torch.float64)
        details, approx = _haar_wavelet_decompose(field, n_levels=1)
        assert approx.shape[0] == 32

    def test_small_field(self):
        field = torch.tensor([1.0, 2.0], dtype=torch.float64)
        details, approx = _haar_wavelet_decompose(field, n_levels=4)
        assert len(details) >= 1


class TestWaveletEnergyRatio:
    def test_full_retention(self):
        field = torch.randn(64, dtype=torch.float64)
        details, approx = _haar_wavelet_decompose(field, n_levels=4)
        ratio = _wavelet_energy_ratio(details, approx, keep_levels=4)
        assert ratio >= 0.99  # All levels kept

    def test_partial_retention(self):
        field = torch.randn(64, dtype=torch.float64)
        details, approx = _haar_wavelet_decompose(field, n_levels=4)
        ratio = _wavelet_energy_ratio(details, approx, keep_levels=1)
        assert 0.0 <= ratio <= 1.0


class TestCheckpointFrequency:
    def test_default_frequency(self):
        cfg = CheckpointAdaptConfig()
        freq = _estimate_checkpoint_frequency([], cfg)
        assert freq == cfg.base_frequency

    def test_single_field(self):
        cfg = CheckpointAdaptConfig()
        history = [torch.randn(50, dtype=torch.float64)]
        freq = _estimate_checkpoint_frequency(history, cfg)
        assert freq == cfg.base_frequency


class TestFieldCorrelation:
    def test_self_correlation(self):
        fields = {"p": torch.randn(100, dtype=torch.float64)}
        result = ReconstructParEnhanced11.compute_field_correlation(fields)
        assert result.group_count == 1

    def test_correlated_fields(self):
        base = torch.randn(100, dtype=torch.float64)
        fields = {
            "p": base,
            "p_copy": base + 0.001 * torch.randn(100, dtype=torch.float64),
        }
        result = ReconstructParEnhanced11.compute_field_correlation(fields, threshold=0.5)
        assert result.group_count <= 2


class TestV11Reconstruction:
    def test_returns_v11_result(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced11(tmp_path)
        result = recon.reconstruct_case_v11(field_names=["p", "U"])
        assert isinstance(result, V11ReconstructResult)

    def test_with_wavelet(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced11(tmp_path)
        result = recon.reconstruct_case_v11(
            field_names=["p"],
            wavelet_analysis=True,
        )
        assert result.wavelet_levels >= 0
        assert result.wavelet_energy_retained >= 0

    def test_with_correlation(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced11(tmp_path)
        result = recon.reconstruct_case_v11(
            field_names=["p", "U"],
            cross_correlation=True,
        )
        assert result.field_correlation is not None


class TestRepr:
    def test_repr(self, tmp_path):
        recon = ReconstructParEnhanced11(tmp_path)
        r = repr(recon)
        assert "ReconstructParEnhanced11" in r
