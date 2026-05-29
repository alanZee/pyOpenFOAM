"""Tests for ReconstructParEnhanced3 — v3 enhanced parallel reconstruction."""

import pytest
import torch

from pyfoam.parallel.reconstruct_par_enhanced_3 import (
    ReconstructParEnhanced3,
    V3ReconstructResult,
    ZoneMergeResult,
)
from pyfoam.parallel.reconstruct_par_enhanced_2 import (
    ReconstructParEnhanced2,
    MergeStrategy,
)


class TestZoneMergeResult:
    """Test ZoneMergeResult dataclass."""

    def test_creation(self):
        zr = ZoneMergeResult(zone_name="interior", zone_id=0, n_cells=100)
        assert zr.zone_name == "interior"
        assert zr.n_cells == 100
        assert zr.merge_method == "first_occurrence"

    def test_defaults(self):
        zr = ZoneMergeResult(zone_name="wall")
        assert zr.zone_id == 0
        assert zr.n_cells == 0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v2(self):
        assert issubclass(ReconstructParEnhanced3, ReconstructParEnhanced2)


class TestZoneAwareReconstruction:
    """Test zone-aware reconstruction features."""

    def test_zone_names_empty(self, tmp_path):
        """No zone map returns empty list."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced3(case_dir)
        assert recon.zone_names == []

    def test_set_zone_map(self, tmp_path):
        """Zone map sets correctly."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced3(case_dir)
        recon.set_zone_map({
            "interior": torch.tensor([0, 1, 2]),
            "wall": torch.tensor([3, 4]),
        })
        assert len(recon.zone_names) == 2
        assert "interior" in recon.zone_names
        assert "wall" in recon.zone_names

    def test_merge_field_zone_aware(self, tmp_path):
        """Zone-aware merge produces correct results."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced3(case_dir)

        proc_fields = {
            0: torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
            1: torch.tensor([40.0, 50.0], dtype=torch.float64),
        }
        proc_map = {
            0: torch.tensor([0, 1, 2]),
            1: torch.tensor([2, 3]),
        }

        recon.set_zone_map({
            "zone0": torch.tensor([0, 1]),
            "zone1": torch.tensor([2, 3]),
        })

        result, zone_results = recon.merge_field_zone_aware(
            proc_fields, proc_map, n_global_cells=4
        )
        assert result.shape == (4,)
        assert result[0].item() == pytest.approx(10.0)
        assert result[1].item() == pytest.approx(20.0)
        # Cell 2: (30 + 40) / 2 = 35
        assert result[2].item() == pytest.approx(35.0)
        assert result[3].item() == pytest.approx(50.0)
        assert len(zone_results) == 2

    def test_merge_field_zone_aware_no_zones(self, tmp_path):
        """Zone-aware merge without zone map works."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced3(case_dir)

        proc_fields = {0: torch.tensor([1.0, 2.0], dtype=torch.float64)}
        proc_map = {0: torch.tensor([0, 1])}

        result, zone_results = recon.merge_field_zone_aware(
            proc_fields, proc_map, n_global_cells=2
        )
        assert len(zone_results) == 0
        assert torch.allclose(result, torch.tensor([1.0, 2.0], dtype=torch.float64))


class TestAdaptiveStrategy:
    """Test adaptive merge strategy selection."""

    def test_uniform_field(self):
        """Uniform field selects first_occurrence."""
        data = torch.ones(100, dtype=torch.float64) * 5.0
        strategy = ReconstructParEnhanced3._select_merge_strategy(data)
        assert strategy == MergeStrategy.FIRST_OCCURRENCE

    def test_nonuniform_field(self):
        """Non-uniform field selects volume_weighted."""
        data = torch.arange(100, dtype=torch.float64)
        strategy = ReconstructParEnhanced3._select_merge_strategy(data)
        assert strategy == MergeStrategy.VOLUME_WEIGHTED

    def test_single_value(self):
        """Single value selects first_occurrence."""
        data = torch.tensor([42.0], dtype=torch.float64)
        strategy = ReconstructParEnhanced3._select_merge_strategy(data)
        assert strategy == MergeStrategy.FIRST_OCCURRENCE


class TestV3Reconstruction:
    """Test v3 case reconstruction."""

    def test_reconstruct_case_v3_basic(self, tmp_path):
        """Basic v3 reconstruction handles empty case gracefully."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced3(case_dir)
        # Empty case has no processor dirs, so reconstruct_case_v3
        # delegates to v2 which triggers discover() and raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            recon.reconstruct_case_v3(zone_aware=False)

    def test_reconstruct_case_v3_with_zones(self, tmp_path):
        """v3 reconstruction with zone map handles empty case."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced3(case_dir)
        recon.set_zone_map({"zoneA": torch.tensor([0, 1])})
        with pytest.raises(FileNotFoundError):
            recon.reconstruct_case_v3(zone_aware=True)


class TestGhostInterpolation:
    """Test ghost cell interpolation."""

    def test_interpolate_ghost_values(self, tmp_path):
        """Ghost values are interpolated correctly."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced3(case_dir)

        field = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0], dtype=torch.float64)
        ghost_idx = torch.tensor([3, 4])
        src_idx = torch.tensor([[0, 1], [1, 2]])
        weights = torch.tensor([[0.3, 0.7], [0.5, 0.5]], dtype=torch.float64)

        result = recon.interpolate_ghost_values(field, ghost_idx, src_idx, weights)
        # Cell 3: 0.3*1.0 + 0.7*2.0 = 1.7
        assert result[3].item() == pytest.approx(1.7)
        # Cell 4: 0.5*2.0 + 0.5*3.0 = 2.5
        assert result[4].item() == pytest.approx(2.5)

    def test_repr(self, tmp_path):
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced3(case_dir)
        r = repr(recon)
        assert "ReconstructParEnhanced3" in r
        assert "zones=0" in r
