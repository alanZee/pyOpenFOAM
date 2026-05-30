"""Tests for YPlusEnhanced5.

Tests cover:
- AMR suggestions
- y+ budget analysis
- Wall model consistency check
- Custom parameters
"""

import pytest
import torch

from pyfoam.postprocessing.y_plus_enhanced_5 import (
    YPlusEnhanced5,
    AMRSuggestion,
    YPlusBudget,
    WallModelConsistency,
)
from pyfoam.postprocessing.y_plus_enhanced_4 import YPlusEnhanced4


class TestYPlusEnhanced5:
    """Tests for YPlusEnhanced5."""

    def test_inherits_from_enhanced4(self):
        yp = YPlusEnhanced5("test", {
            "rho": 1.0,
            "mu": 1e-5,
            "Uref": 10.0,
        })
        assert isinstance(yp, YPlusEnhanced4)

    def test_default_params(self):
        yp = YPlusEnhanced5("test", {
            "rho": 1.0,
            "mu": 1e-5,
            "Uref": 10.0,
        })
        assert yp._suggest_amr is False
        assert yp._budget_analysis is True
        assert yp._consistency_check is True
        assert yp._y_plus_ideal == [0.5, 2.0]
        assert yp._current_wall_model == "automatic"

    def test_custom_params(self):
        yp = YPlusEnhanced5("test", {
            "rho": 1.0,
            "mu": 1e-5,
            "Uref": 10.0,
            "suggestAMR": True,
            "budgetAnalysis": False,
            "consistencyCheck": False,
            "yPlusIdealRange": [1.0, 5.0],
            "currentWallModel": "standardWallFunction",
        })
        assert yp._suggest_amr is True
        assert yp._budget_analysis is False
        assert yp._consistency_check is False
        assert yp._y_plus_ideal == [1.0, 5.0]
        assert yp._current_wall_model == "standardWallFunction"

    def test_amr_suggestion_dataclass(self):
        amr = AMRSuggestion(
            patch_name="wall",
            time=1.0,
            needs_refinement=True,
            refinement_factor=0.5,
            cells_to_refine=100,
            y_plus_target=1.0,
        )
        assert amr.patch_name == "wall"
        assert amr.needs_refinement is True

    def test_y_plus_budget_dataclass(self):
        budget = YPlusBudget(
            patch_name="wall",
            time=1.0,
            y_plus_from_velocity=5.0,
            y_plus_from_distance=2.0,
            y_plus_from_viscosity=1.0,
            y_plus_total=8.0,
        )
        assert budget.y_plus_total == pytest.approx(8.0)

    def test_wall_model_consistency_dataclass(self):
        cc = WallModelConsistency(
            patch_name="wall",
            time=1.0,
            current_model="standardWallFunction",
            y_plus_mean=30.0,
            y_plus_range=(20.0, 40.0),
            is_consistent=True,
            recommended_model="standardWallFunction",
            confidence=0.95,
        )
        assert cc.is_consistent is True
        assert cc.confidence == pytest.approx(0.95)

    def test_empty_amr_suggestions(self):
        yp = YPlusEnhanced5("test", {
            "rho": 1.0,
            "mu": 1e-5,
        })
        assert yp.amr_suggestions == []
        assert yp.get_latest_amr("wall") is None

    def test_empty_budgets(self):
        yp = YPlusEnhanced5("test", {
            "rho": 1.0,
            "mu": 1e-5,
        })
        assert yp.y_plus_budgets == []

    def test_empty_consistency(self):
        yp = YPlusEnhanced5("test", {
            "rho": 1.0,
            "mu": 1e-5,
        })
        assert yp.consistency_results == []
        assert yp.get_latest_consistency("wall") is None
