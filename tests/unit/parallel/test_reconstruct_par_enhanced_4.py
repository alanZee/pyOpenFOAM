"""Tests for ReconstructParEnhanced4 — v4 enhanced parallel reconstruction."""

import pytest
import torch

from pyfoam.parallel.reconstruct_par_enhanced_4 import (
    ReconstructParEnhanced4,
    V4ReconstructResult,
    GradientMergeConfig,
)
from pyfoam.parallel.reconstruct_par_enhanced_3 import ReconstructParEnhanced3


class TestGradientMergeConfig:
    """Test GradientMergeConfig dataclass."""

    def test_defaults(self):
        cfg = GradientMergeConfig()
        assert cfg.use_gradient is True
        assert cfg.gradient_radius == 1.0
        assert cfg.consistency_weight == 0.5

    def test_custom(self):
        cfg = GradientMergeConfig(use_gradient=False, gradient_radius=2.0)
        assert cfg.use_gradient is False
        assert cfg.gradient_radius == 2.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v3(self):
        assert issubclass(ReconstructParEnhanced4, ReconstructParEnhanced3)


class TestGradientEstimation:
    """Test least-squares gradient estimation."""

    def test_constant_field_zero_gradient(self):
        """Constant field has zero gradient."""
        positions = torch.tensor([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=torch.float64)
        values = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float64)

        grad = ReconstructParEnhanced4.estimate_gradient_ls(positions, values)
        assert torch.allclose(grad, torch.zeros(3, dtype=torch.float64), atol=1e-10)

    def test_linear_field_correct_gradient(self):
        """Linear field f = 2*x + 3*y recovers correct gradient."""
        positions = torch.tensor([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        ], dtype=torch.float64)
        values = torch.tensor([0.0, 2.0, 3.0, 5.0], dtype=torch.float64)

        grad = ReconstructParEnhanced4.estimate_gradient_ls(positions, values)
        assert abs(grad[0].item() - 2.0) < 1e-8
        assert abs(grad[1].item() - 3.0) < 1e-8
        assert abs(grad[2].item()) < 1e-8

    def test_insufficient_points(self):
        """Less than 4 points returns zero gradient."""
        positions = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float64)
        values = torch.tensor([1.0, 2.0], dtype=torch.float64)

        grad = ReconstructParEnhanced4.estimate_gradient_ls(positions, values)
        assert torch.allclose(grad, torch.zeros(3, dtype=torch.float64))


class TestGradientCorrectedMerge:
    """Test gradient-corrected field merge."""

    def test_merge_basic(self, tmp_path):
        """Gradient merge produces valid output."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced4(case_dir)

        proc_fields = {
            0: torch.tensor([10.0, 20.0], dtype=torch.float64),
            1: torch.tensor([40.0, 50.0], dtype=torch.float64),
        }
        proc_map = {
            0: torch.tensor([0, 1]),
            1: torch.tensor([1, 2]),
        }
        centres = torch.tensor([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
        ], dtype=torch.float64)

        result, n_corr = recon.merge_field_gradient_corrected(
            proc_fields, proc_map, centres, n_global_cells=3,
        )
        assert result.shape == (3,)
        assert n_corr >= 0

    def test_merge_no_overlap(self, tmp_path):
        """No overlapping cells: no corrections."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced4(case_dir)

        proc_fields = {
            0: torch.tensor([10.0], dtype=torch.float64),
            1: torch.tensor([20.0], dtype=torch.float64),
        }
        proc_map = {
            0: torch.tensor([0]),
            1: torch.tensor([1]),
        }
        centres = torch.tensor([
            [0, 0, 0], [1, 0, 0],
        ], dtype=torch.float64)

        result, n_corr = recon.merge_field_gradient_corrected(
            proc_fields, proc_map, centres, n_global_cells=2,
        )
        assert n_corr == 0
        assert result[0].item() == pytest.approx(10.0)
        assert result[1].item() == pytest.approx(20.0)


class TestConsistencyError:
    """Test boundary consistency error computation."""

    def test_perfect_consistency(self, tmp_path):
        """Consistent fields have zero error."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced4(case_dir)

        merged = torch.tensor([10.0, 20.0], dtype=torch.float64)
        proc_fields = {0: torch.tensor([10.0], dtype=torch.float64)}
        proc_map = {0: torch.tensor([0])}

        err = recon.compute_consistency_error(merged, proc_fields, proc_map)
        assert err == pytest.approx(0.0)

    def test_inconsistent_fields(self, tmp_path):
        """Inconsistent fields have nonzero error."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced4(case_dir)

        merged = torch.tensor([10.0, 20.0], dtype=torch.float64)
        proc_fields = {0: torch.tensor([12.0], dtype=torch.float64)}
        proc_map = {0: torch.tensor([0])}

        err = recon.compute_consistency_error(merged, proc_fields, proc_map)
        assert err > 0


class TestV4Reconstruction:
    """Test v4 case reconstruction."""

    def test_reconstruct_case_v4_skip(self, tmp_path):
        """v4 reconstruction handles empty case."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced4(case_dir)
        with pytest.raises(FileNotFoundError):
            recon.reconstruct_case_v4(zone_aware=False, gradient_merge=False)

    def test_set_gradient_config(self, tmp_path):
        """Gradient config can be set."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced4(case_dir)
        cfg = GradientMergeConfig(use_gradient=False)
        recon.set_gradient_config(cfg)
        assert recon._gradient_config.use_gradient is False

    def test_repr(self, tmp_path):
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced4(case_dir)
        r = repr(recon)
        assert "ReconstructParEnhanced4" in r
        assert "gradient=on" in r
