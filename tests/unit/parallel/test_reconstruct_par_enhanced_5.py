"""Tests for ReconstructParEnhanced5 — v5 enhanced parallel reconstruction."""

import pytest
import torch

from pyfoam.parallel.reconstruct_par_enhanced_5 import (
    ReconstructParEnhanced5,
    V5ReconstructResult,
    SmoothingConfig,
)
from pyfoam.parallel.reconstruct_par_enhanced_4 import ReconstructParEnhanced4


class TestSmoothingConfig:
    """Test SmoothingConfig dataclass."""

    def test_defaults(self):
        cfg = SmoothingConfig()
        assert cfg.n_passes == 3
        assert cfg.diffusion_coeff == 0.1
        assert cfg.adaptive is True

    def test_custom(self):
        cfg = SmoothingConfig(n_passes=5, adaptive=False)
        assert cfg.n_passes == 5
        assert cfg.adaptive is False


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v4(self):
        assert issubclass(ReconstructParEnhanced5, ReconstructParEnhanced4)


class TestLaplacianSmooth:
    """Test Laplacian smoothing."""

    def test_constant_field_unchanged(self):
        """Constant field should remain unchanged after smoothing."""
        field = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float64)
        adjacency = torch.tensor([
            [1, 2, -1],
            [0, 3, -1],
            [0, 3, -1],
            [1, 2, -1],
        ], dtype=torch.long)

        smoothed = ReconstructParEnhanced5.laplacian_smooth(
            field, adjacency, n_passes=3, diffusion_coeff=0.5,
        )
        assert torch.allclose(smoothed, field, atol=1e-10)

    def test_smoothing_reduces_oscillation(self):
        """Smoothing should reduce oscillations."""
        field = torch.tensor([0.0, 10.0, 0.0, 10.0], dtype=torch.float64)
        adjacency = torch.tensor([
            [1, -1, -1],
            [0, 2, -1],
            [1, 3, -1],
            [2, -1, -1],
        ], dtype=torch.long)

        smoothed = ReconstructParEnhanced5.laplacian_smooth(
            field, adjacency, n_passes=10, diffusion_coeff=0.3,
        )
        # Range should be smaller
        assert smoothed.max() - smoothed.min() < field.max() - field.min()


class TestAdaptiveSmooth:
    """Test adaptive smoothing."""

    def test_preserves_sharp_features(self):
        """Adaptive smoothing preserves features where gradient exceeds threshold."""
        field = torch.tensor([0.0, 0.0, 100.0, 100.0], dtype=torch.float64)
        adjacency = torch.tensor([
            [1, -1, -1],
            [0, 2, -1],
            [1, 3, -1],
            [2, -1, -1],
        ], dtype=torch.long)

        cfg = SmoothingConfig(
            n_passes=3, diffusion_coeff=0.5,
            adaptive=True, gradient_threshold=0.01,
        )
        smoothed = ReconstructParEnhanced5.adaptive_smooth(
            field, adjacency, cfg,
        )
        # The sharp feature should be largely preserved
        assert smoothed[2].item() > 50.0


class TestClipBounds:
    """Test field bounds clipping."""

    def test_clip_basic(self):
        field = torch.tensor([-5.0, 0.0, 5.0, 15.0], dtype=torch.float64)
        clipped, n = ReconstructParEnhanced5.clip_field_bounds(
            field, lower=0.0, upper=10.0,
        )
        assert n == 2  # -5 and 15 clipped
        assert clipped.min().item() == pytest.approx(0.0)
        assert clipped.max().item() == pytest.approx(10.0)

    def test_no_clipping_needed(self):
        field = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        clipped, n = ReconstructParEnhanced5.clip_field_bounds(
            field, lower=0.0, upper=10.0,
        )
        assert n == 0
        assert torch.allclose(clipped, field)


class TestFieldQuality:
    """Test quality metric computation."""

    def test_perfect_quality(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced5(case_dir)

        merged = torch.tensor([10.0, 20.0], dtype=torch.float64)
        per_proc = {0: torch.tensor([10.0], dtype=torch.float64)}
        proc_map = {0: torch.tensor([0])}

        q = recon.compute_field_quality(merged, per_proc, proc_map)
        assert q == pytest.approx(1.0)

    def test_imperfect_quality(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced5(case_dir)

        merged = torch.tensor([10.0, 20.0], dtype=torch.float64)
        per_proc = {0: torch.tensor([15.0], dtype=torch.float64)}
        proc_map = {0: torch.tensor([0])}

        q = recon.compute_field_quality(merged, per_proc, proc_map)
        assert q < 1.0
        assert q > 0.0


class TestV5Reconstruction:
    """Test v5 case reconstruction."""

    def test_smoothing_config(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced5(case_dir)
        cfg = SmoothingConfig(n_passes=5)
        recon.set_smoothing_config(cfg)
        assert recon._smoothing_config.n_passes == 5

    def test_repr(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced5(case_dir)
        r = repr(recon)
        assert "ReconstructParEnhanced5" in r
        assert "smoothing=3" in r
