"""Tests for ForcesEnhanced.

Tests cover:
- Force decomposition (pressure/viscous moment separation)
- Extra CoR moments
- Per-patch forces
- Fluctuation statistics
"""

import pytest
import torch

from pyfoam.postprocessing.forces import Forces
from pyfoam.postprocessing.forces_enhanced import (
    ForcesEnhanced,
    ForceDecomposition,
    FluctuationStats,
)


class TestForcesEnhanced:
    """Tests for ForcesEnhanced."""

    def test_inherits_from_forces(self):
        fo = ForcesEnhanced("test", {"patches": ["bottom"]})
        assert isinstance(fo, Forces)

    def test_default_params(self):
        fo = ForcesEnhanced("test", {"patches": ["bottom"]})
        assert fo._compute_fluctuations is False
        assert fo._per_patch_stats is True
        assert len(fo._extra_cofr) == 0

    def test_custom_params(self):
        fo = ForcesEnhanced("test", {
            "patches": ["bottom"],
            "computeFluctuations": True,
            "extraCofR": [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        })
        assert fo._compute_fluctuations is True
        assert len(fo._extra_cofr) == 2

    def test_execute(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced("test", {"patches": ["bottom"]})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(1.0)
        assert len(fo.decompositions) == 2
        assert len(fo.force_total) == 2

    def test_force_decomposition(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced("test", {"patches": ["bottom"]})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        decomp = fo.decompositions[0]
        assert isinstance(decomp, ForceDecomposition)
        assert decomp.F_pressure is not None
        assert decomp.F_viscous is not None
        assert decomp.F_total is not None
        assert decomp.M_pressure is not None
        assert decomp.M_viscous is not None
        assert decomp.M_total is not None

    def test_extra_cor_moments(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced("test", {
            "patches": ["bottom"],
            "extraCofR": [[0.5, 0.0, 0.0]],
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.extra_moments[0]) == 1

    def test_per_patch_forces(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced("test", {"patches": ["bottom"]})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert "bottom" in fo.per_patch_forces

    def test_fluctuation_stats_insufficient_data(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced("test", {"patches": ["bottom"], "computeFluctuations": True})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        stats = fo.compute_fluctuation_stats()
        assert stats is None  # Only 1 data point

    def test_fluctuation_stats(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced("test", {"patches": ["bottom"], "computeFluctuations": True})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(1.0)
        fo.execute(2.0)
        stats = fo.compute_fluctuation_stats()
        assert stats is not None
        assert isinstance(stats, FluctuationStats)
        assert stats.n_samples == 3
        assert stats.rms is not None
        assert stats.peak_to_peak is not None

    def test_execute_skips_disabled(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced("test", {"patches": ["bottom"], "enabled": False})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.decompositions) == 0


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
