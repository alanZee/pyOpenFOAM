"""Tests for YPlusEnhanced3.

Tests cover:
- Adaptive wall law selection
- Regime classification
- Time evolution tracking
- Execute with mesh
"""

import pytest
import torch

from pyfoam.postprocessing.y_plus_enhanced_3 import (
    YPlusEnhanced3,
    RegimeClassification,
    YPlusEvolution,
)
from pyfoam.postprocessing.y_plus_enhanced_2 import YPlusEnhanced2


class TestYPlusEnhanced3:
    """Tests for YPlusEnhanced3."""

    def test_inherits_from_enhanced2(self):
        fo = YPlusEnhanced3("test", {"rho": 1.0, "mu": 1e-5})
        assert isinstance(fo, YPlusEnhanced2)

    def test_default_params(self):
        fo = YPlusEnhanced3("test", {"rho": 1.0, "mu": 1e-5})
        assert fo._adaptive_wall_law is True
        assert fo._track_evolution is True
        assert fo._regime_history_flag is True

    def test_custom_params(self):
        fo = YPlusEnhanced3("test", {
            "rho": 1.0, "mu": 1e-5,
            "adaptiveWallLaw": False,
            "trackEvolution": False,
        })
        assert fo._adaptive_wall_law is False
        assert fo._track_evolution is False

    def test_select_wall_law(self):
        fo = YPlusEnhanced3("test", {"rho": 1.0, "mu": 1e-5})
        # Viscous sublayer
        assert fo._select_wall_law(2.0) == "spalding"
        # Buffer layer
        assert fo._select_wall_law(15.0) == "spalding"
        # Log-law
        assert fo._select_wall_law(100.0) == "wernerWengle"

    def test_execute(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced3("test", {
            "rho": 1.0,
            "mu": 1e-3,
            "patches": ["bottom"],
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.regime_history) == 1
        assert len(fo.patch_history) == 1

    def test_regime_classification(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced3("test", {
            "rho": 1.0,
            "mu": 1e-3,
            "patches": ["bottom"],
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        regime = fo.get_latest_regime("bottom")
        assert regime is not None
        assert isinstance(regime, RegimeClassification)
        assert regime.patch_name == "bottom"
        assert regime.time == pytest.approx(0.0)
        # Fractions should sum to 1.0
        total = regime.viscous_fraction + regime.buffer_fraction + regime.log_law_fraction
        assert total == pytest.approx(1.0, abs=0.01)
        assert regime.recommended_wall_law in ("spalding", "wernerWengle", "mixed")

    def test_evolution_tracking(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced3("test", {
            "rho": 1.0,
            "mu": 1e-3,
            "patches": ["bottom"],
            "trackEvolution": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(0.1)

        evol = fo.get_evolution("bottom")
        assert evol is not None
        assert isinstance(evol, YPlusEvolution)
        assert len(evol.times) == 2
        assert len(evol.y_plus_mean_history) == 2
        assert len(evol.y_plus_max_history) == 2

    def test_convergence_rate(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced3("test", {
            "rho": 1.0,
            "mu": 1e-3,
            "patches": ["bottom"],
            "trackEvolution": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(0.1)

        evol = fo.get_evolution("bottom")
        assert evol is not None
        # Convergence rate should be computed
        assert isinstance(evol.convergence_rate, float)

    def test_get_latest_regime_no_data(self):
        fo = YPlusEnhanced3("test", {"rho": 1.0, "mu": 1e-5})
        assert fo.get_latest_regime("bottom") is None

    def test_get_evolution_no_data(self):
        fo = YPlusEnhanced3("test", {"rho": 1.0, "mu": 1e-5})
        assert fo.get_evolution("bottom") is None

    def test_execute_no_field(self, fv_mesh):
        """Should handle missing U field gracefully."""
        fo = YPlusEnhanced3("test", {
            "rho": 1.0, "mu": 1e-3,
            "patches": ["bottom"],
        })
        fo.initialise(fv_mesh, {})
        fo.execute(0.0)
        assert len(fo.regime_history) == 0


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
