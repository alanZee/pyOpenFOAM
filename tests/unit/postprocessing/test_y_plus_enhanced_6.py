"""Tests for YPlusEnhanced6.

Tests cover:
- Wall heat transfer
- Adaptive wall function selection
- y+ prediction
- Custom parameters
"""

import pytest
import torch

from pyfoam.postprocessing.y_plus_enhanced_6 import (
    YPlusEnhanced6,
    WallHeatTransfer,
    AdaptiveWallFunction,
    YPlusPrediction,
)
from pyfoam.postprocessing.y_plus_enhanced_5 import YPlusEnhanced5


class TestYPlusEnhanced6:
    """Tests for YPlusEnhanced6."""

    def test_inherits_from_enhanced5(self):
        yp = YPlusEnhanced6("test", {
            "rho": 1.0,
            "mu": 1e-5,
            "Uref": 10.0,
        })
        assert isinstance(yp, YPlusEnhanced5)

    def test_default_params(self):
        yp = YPlusEnhanced6("test", {
            "rho": 1.0,
            "mu": 1e-5,
            "Uref": 10.0,
        })
        assert yp._heat_transfer_enabled is False
        assert yp._adaptive_wf is False
        assert yp._prediction_enabled is False
        assert yp._kappa_fluid == pytest.approx(0.6)
        assert yp._Pr == pytest.approx(0.71)

    def test_custom_params(self):
        yp = YPlusEnhanced6("test", {
            "rho": 1.0,
            "mu": 1e-5,
            "Uref": 10.0,
            "wallHeatTransfer": True,
            "adaptiveWallFunction": True,
            "predictionEnabled": True,
            "thermalConductivity": 0.025,
            "Pr": 0.7,
        })
        assert yp._heat_transfer_enabled is True
        assert yp._adaptive_wf is True
        assert yp._prediction_enabled is True
        assert yp._kappa_fluid == pytest.approx(0.025)
        assert yp._Pr == pytest.approx(0.7)

    def test_wall_heat_transfer_dataclass(self):
        ht = WallHeatTransfer(
            patch_name="wall",
            time=1.0,
            y_plus_mean=30.0,
            stanton_number=0.001,
            nusselt_number=100.0,
            heat_transfer_coeff=50.0,
        )
        assert ht.patch_name == "wall"
        assert ht.y_plus_mean == pytest.approx(30.0)

    def test_adaptive_wall_function_dataclass(self):
        wf = AdaptiveWallFunction(
            patch_name="wall",
            time=1.0,
            y_plus_mean=30.0,
            y_plus_std=5.0,
            selected_function="standardWallFunction",
            confidence=0.9,
            alternative="scalableWallFunction",
        )
        assert wf.selected_function == "standardWallFunction"
        assert wf.confidence == pytest.approx(0.9)

    def test_y_plus_prediction_dataclass(self):
        pred = YPlusPrediction(
            patch_name="wall",
            time=1.0,
            predicted_y_plus=35.0,
            trend=0.5,
            ar_coeff=0.8,
        )
        assert pred.predicted_y_plus == pytest.approx(35.0)
        assert pred.ar_coeff == pytest.approx(0.8)

    def test_empty_heat_transfer(self):
        yp = YPlusEnhanced6("test", {
            "rho": 1.0,
            "mu": 1e-5,
        })
        assert yp.heat_transfer == []
        assert yp.get_latest_heat_transfer("wall") is None

    def test_empty_wall_function_selections(self):
        yp = YPlusEnhanced6("test", {
            "rho": 1.0,
            "mu": 1e-5,
        })
        assert yp.wall_function_selections == []
        assert yp.get_latest_wall_function("wall") is None

    def test_empty_y_plus_predictions(self):
        yp = YPlusEnhanced6("test", {
            "rho": 1.0,
            "mu": 1e-5,
        })
        assert yp.y_plus_predictions == []
