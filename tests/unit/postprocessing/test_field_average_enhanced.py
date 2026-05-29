"""Tests for FieldAverageEnhanced — time-weighted averaging and Reynolds decomposition."""

import pytest
import torch

from pyfoam.postprocessing.field_average_enhanced import FieldAverageEnhanced


# ---------------------------------------------------------------------------
# FieldAverageEnhanced tests
# ---------------------------------------------------------------------------


class TestFieldAverageEnhanced:
    """Test FieldAverageEnhanced function object."""

    def test_creation(self):
        """Create FieldAverageEnhanced with config."""
        fa = FieldAverageEnhanced("testEnh", {
            "fields": ["p"],
            "mean": True,
            "prime3Mean": True,
        })
        assert fa.name == "testEnh"
        assert fa._compute_prime3 is True
        assert fa._time_weighted is True

    def test_default_config(self):
        """Default config: prime3Mean=False, timeWeighted=True."""
        fa = FieldAverageEnhanced()
        assert fa._compute_prime3 is False
        assert fa._time_weighted is True

    def test_execute_single_step(self, fv_mesh, sample_fields):
        """Execute at a single time step."""
        fa = FieldAverageEnhanced("test", {"fields": ["p"], "mean": True})
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(0.0)

        assert fa.n_samples == 1
        assert "p" in fa.mean_fields

    def test_execute_time_weighted(self, fv_mesh, sample_fields):
        """Time-weighted averaging uses dt as weight."""
        fa = FieldAverageEnhanced("test", {
            "fields": ["p"],
            "mean": True,
            "timeWeighted": True,
        })
        fa.initialise(fv_mesh, sample_fields)

        fa.execute(0.0)
        fa.execute(1.0)
        fa.execute(3.0)  # dt=2.0

        assert fa._cumulative_dt == pytest.approx(4.0)  # 1.0 (first, dt=0) + 1.0 + 2.0

    def test_execute_non_time_weighted(self, fv_mesh, sample_fields):
        """Non-time-weighted: each sample has weight 1."""
        fa = FieldAverageEnhanced("test", {
            "fields": ["p"],
            "mean": True,
            "timeWeighted": False,
        })
        fa.initialise(fv_mesh, sample_fields)

        fa.execute(0.0)
        fa.execute(1.0)
        fa.execute(3.0)

        assert fa._cumulative_dt == 3.0  # 1+1+1

    def test_prime3_mean(self, fv_mesh, sample_fields):
        """Third-order central moment accumulation."""
        fa = FieldAverageEnhanced("test", {
            "fields": ["p"],
            "mean": True,
            "prime3Mean": True,
        })
        fa.initialise(fv_mesh, sample_fields)

        fa.execute(1.0)
        fa.execute(2.0)

        assert "p" in fa.prime3_fields
        assert fa.prime3_fields["p"].shape == (2,)

    def test_skewness(self, fv_mesh, sample_fields):
        """Skewness requires prime2 and prime3."""
        fa = FieldAverageEnhanced("test", {
            "fields": ["p"],
            "mean": True,
            "prime2Mean": True,
            "prime3Mean": True,
        })
        fa.initialise(fv_mesh, sample_fields)

        fa.execute(1.0)
        fa.execute(2.0)

        skew = fa.get_skewness("p")
        assert skew is not None
        assert skew.shape == (2,)

    def test_skewness_no_prime3(self, fv_mesh, sample_fields):
        """Skewness returns None without prime3."""
        fa = FieldAverageEnhanced("test", {
            "fields": ["p"],
            "mean": True,
            "prime2Mean": True,
        })
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        assert fa.get_skewness("p") is None

    def test_skewness_no_prime2(self, fv_mesh, sample_fields):
        """Skewness returns None without prime2."""
        fa = FieldAverageEnhanced("test", {
            "fields": ["p"],
            "mean": True,
            "prime3Mean": True,
        })
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        assert fa.get_skewness("p") is None

    def test_reynolds_decomposition(self, fv_mesh, sample_fields):
        """Reynolds decomposition: q = <q> + q'."""
        fa = FieldAverageEnhanced("test", {
            "fields": ["p"],
            "mean": True,
        })
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        decomp = fa.decompose("p", torch.tensor([102000.0, 100000.0], dtype=torch.float64))
        assert "mean" in decomp
        assert "fluctuation" in decomp
        # mean + fluctuation should equal instantaneous
        reconstructed = decomp["mean"] + decomp["fluctuation"]
        expected = torch.tensor([102000.0, 100000.0], dtype=torch.float64)
        assert torch.allclose(reconstructed, expected)

    def test_decompose_missing_mean(self, fv_mesh, sample_fields):
        """decompose raises KeyError for missing field."""
        fa = FieldAverageEnhanced("test", {"fields": ["p"], "mean": False})
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        with pytest.raises(KeyError):
            fa.decompose("p", torch.tensor([1.0, 2.0]))

    def test_inherits_field_average(self):
        """FieldAverageEnhanced is a subclass of FieldAverage."""
        from pyfoam.postprocessing.field_average import FieldAverage
        assert issubclass(FieldAverageEnhanced, FieldAverage)

    def test_write_includes_prime3(self, fv_mesh, sample_fields, tmp_path):
        """Write includes prime3Mean file."""
        fa = FieldAverageEnhanced("test", {
            "fields": ["p"],
            "mean": True,
            "prime3Mean": True,
        })
        fa.set_output_path(tmp_path)
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)
        fa.write()

        assert (tmp_path / "p_prime3Mean").exists()

    def test_finalise_clears_prime3(self, fv_mesh, sample_fields):
        """finalise clears prime3 accumulators."""
        fa = FieldAverageEnhanced("test", {
            "fields": ["p"],
            "prime3Mean": True,
        })
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        assert len(fa._prime3_accum) > 0
        fa.finalise()
        assert len(fa._prime3_accum) == 0

    def test_non_positive_dt_skipped(self, fv_mesh, sample_fields):
        """Non-positive dt is skipped."""
        fa = FieldAverageEnhanced("test", {"fields": ["p"], "mean": True})
        fa.initialise(fv_mesh, sample_fields)

        fa.execute(1.0)
        fa.execute(1.0)  # dt=0, should be skipped

        assert fa.n_samples == 1
