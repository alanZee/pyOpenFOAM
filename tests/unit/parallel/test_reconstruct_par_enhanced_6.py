"""Tests for ReconstructParEnhanced6 -- v6 enhanced parallel reconstruction."""

import pytest
import torch

from pyfoam.parallel.reconstruct_par_enhanced_6 import (
    ReconstructParEnhanced6,
    V6ReconstructResult,
    AnisotropicSmoothingConfig,
    CheckpointInfo,
)
from pyfoam.parallel.reconstruct_par_enhanced_5 import ReconstructParEnhanced5


class TestAnisotropicSmoothingConfig:
    """Test AnisotropicSmoothingConfig dataclass."""

    def test_defaults(self):
        cfg = AnisotropicSmoothingConfig()
        assert cfg.n_passes == 3
        assert cfg.adaptive is True
        assert cfg.diffusion_coeffs.shape == (3,)

    def test_custom(self):
        cfg = AnisotropicSmoothingConfig(
            n_passes=5,
            diffusion_coeffs=torch.tensor([0.1, 0.2, 0.05]),
        )
        assert cfg.n_passes == 5


class TestCheckpointInfo:
    """Test CheckpointInfo dataclass."""

    def test_defaults(self):
        info = CheckpointInfo()
        assert info.n_fields == 0
        assert info.checkpoint_id == ""


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v5(self):
        assert issubclass(ReconstructParEnhanced6, ReconstructParEnhanced5)


class TestAnisotropicSmooth:
    """Test anisotropic smoothing."""

    def test_constant_field_unchanged(self):
        """Constant field should remain unchanged after anisotropic smoothing."""
        field = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float64)
        adjacency = torch.tensor([
            [1, 2, -1],
            [0, 3, -1],
            [0, 3, -1],
            [1, 2, -1],
        ], dtype=torch.long)
        centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=torch.float64)

        smoothed = ReconstructParEnhanced6.anisotropic_smooth(
            field, adjacency, centres, n_passes=3,
        )
        assert torch.allclose(smoothed, field, atol=1e-10)

    def test_smoothing_reduces_oscillation(self):
        """Anisotropic smoothing should reduce oscillations."""
        field = torch.tensor([0.0, 10.0, 0.0, 10.0], dtype=torch.float64)
        adjacency = torch.tensor([
            [1, -1, -1],
            [0, 2, -1],
            [1, 3, -1],
            [2, -1, -1],
        ], dtype=torch.long)
        centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=torch.float64)

        smoothed = ReconstructParEnhanced6.anisotropic_smooth(
            field, adjacency, centres, n_passes=10,
        )
        assert smoothed.max() - smoothed.min() < field.max() - field.min()


class TestNormaliseField:
    """Test field normalisation."""

    def test_normalise_basic(self):
        field = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        normalised, orig_mean, orig_std = ReconstructParEnhanced6.normalise_field(
            field, target_mean=0.0, target_std=1.0,
        )
        assert abs(normalised.mean().item()) < 1e-10
        assert abs(normalised.std().item() - 1.0) < 1e-10

    def test_normalise_constant_field(self):
        """Constant field should be returned unchanged."""
        field = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
        normalised, _, _ = ReconstructParEnhanced6.normalise_field(field)
        assert torch.allclose(normalised, field)


class TestCheckpointing:
    """Test checkpoint creation."""

    def test_create_checkpoint(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced6(case_dir)

        fields = {
            "p": torch.tensor([1.0, 2.0, 3.0]),
            "U": torch.tensor([4.0, 5.0]),
        }
        cp = recon.create_checkpoint(fields)
        assert cp.n_fields == 2
        assert cp.checkpoint_id == "cp_0000"

    def test_multiple_checkpoints(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced6(case_dir)

        recon.create_checkpoint({"p": torch.tensor([1.0])})
        recon.create_checkpoint({"U": torch.tensor([2.0])})
        assert len(recon.checkpoints) == 2
        assert recon.checkpoints[1].checkpoint_id == "cp_0001"


class TestV6Reconstruction:
    """Test v6 case reconstruction."""

    def test_repr(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced6(case_dir)
        r = repr(recon)
        assert "ReconstructParEnhanced6" in r
        assert "checkpoints=0" in r

    def test_anisotropic_config(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        recon = ReconstructParEnhanced6(case_dir)
        cfg = AnisotropicSmoothingConfig(n_passes=5)
        recon.set_anisotropic_config(cfg)
        assert recon._aniso_config.n_passes == 5
