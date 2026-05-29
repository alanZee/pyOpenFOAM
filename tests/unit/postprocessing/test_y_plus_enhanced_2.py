"""Tests for YPlusEnhanced2.

Tests cover:
- Wall law models (Spalding, Werner-Wengle, mixed)
- Mesh quality metrics
- Friction velocity history
- Enhanced wall distance
"""

import pytest
import torch

from pyfoam.postprocessing.y_plus_enhanced import YPlusEnhanced, WallTreatment
from pyfoam.postprocessing.y_plus_enhanced_2 import (
    YPlusEnhanced2,
    MeshQualityMetrics,
    WallLawType,
)


class TestWallLawType:
    """Tests for WallLawType constants."""

    def test_types(self):
        assert WallLawType.SPALDING == "spalding"
        assert WallLawType.WERNER_WENGLE == "wernerWengle"
        assert WallLawType.MIXED == "mixed"
        assert len(WallLawType.ALL_TYPES) == 3


class TestYPlusEnhanced2:
    """Tests for YPlusEnhanced2."""

    def test_inherits_from_enhanced(self):
        fo = YPlusEnhanced2("test", {"rho": 1.0, "mu": 1e-5})
        assert isinstance(fo, YPlusEnhanced)

    def test_default_params(self):
        fo = YPlusEnhanced2("test", {"rho": 1.0, "mu": 1e-5})
        assert fo._wall_law == "spalding"
        assert fo._u_ref == pytest.approx(1.0)
        assert fo._compute_mesh_quality is True
        assert fo._target_yplus == pytest.approx(1.0)

    def test_custom_params(self):
        fo = YPlusEnhanced2("test", {
            "rho": 1.0, "mu": 1e-5,
            "wallLaw": "wernerWengle",
            "Uref": 10.0,
            "targetYplus": 5.0,
        })
        assert fo._wall_law == "wernerWengle"
        assert fo._target_yplus == pytest.approx(5.0)

    def test_execute(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced2("test", {
            "rho": 1.0, "mu": 1e-3,
            "patches": ["bottom"],
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.patch_history) == 1
        assert len(fo.u_tau_history) == 1

    def test_mesh_quality_metrics(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced2("test", {
            "rho": 1.0, "mu": 1e-3,
            "patches": ["bottom"],
            "computeMeshQuality": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert len(fo.mesh_quality_history) == 1
        mq = fo.get_latest_mesh_quality("bottom")
        assert mq is not None
        assert isinstance(mq, MeshQualityMetrics)
        assert mq.quality_grade in ("A", "B", "C", "D")
        assert mq.y_plus_ratio >= 1.0
        assert mq.recommended_refinement > 0

    def test_u_tau_history(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced2("test", {
            "rho": 1.0, "mu": 1e-3,
            "patches": ["bottom"],
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        utau = fo.u_tau_history[0]
        assert "bottom" in utau
        assert (utau["bottom"] >= 0).all()

    def test_spalding_wall_law(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced2("test", {
            "rho": 1.0, "mu": 1e-3,
            "patches": ["bottom"],
            "wallLaw": "spalding",
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.patch_history) == 1

    def test_werner_wengle_wall_law(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced2("test", {
            "rho": 1.0, "mu": 1e-3,
            "patches": ["bottom"],
            "wallLaw": "wernerWengle",
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.patch_history) == 1

    def test_mixed_wall_law(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced2("test", {
            "rho": 1.0, "mu": 1e-3,
            "patches": ["bottom"],
            "wallLaw": "mixed",
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.patch_history) == 1

    def test_get_latest_mesh_quality_no_data(self):
        fo = YPlusEnhanced2("test", {"rho": 1.0, "mu": 1e-5})
        assert fo.get_latest_mesh_quality("bottom") is None

    def test_wall_law_u_plus_spalding(self):
        """Spalding u+ should be positive and increasing."""
        fo = YPlusEnhanced2("test", {"rho": 1.0, "mu": 1e-5, "wallLaw": "spalding"})
        y_plus = torch.tensor([1.0, 5.0, 30.0, 100.0], dtype=torch.float64)
        u_plus = fo._wall_law_u_plus(y_plus)
        assert (u_plus > 0).all()
        assert (u_plus[1:] >= u_plus[:-1]).all()

    def test_wall_law_u_plus_werner_wengle(self):
        """Werner-Wengle u+ should be positive."""
        fo = YPlusEnhanced2("test", {"rho": 1.0, "mu": 1e-5, "wallLaw": "wernerWengle"})
        y_plus = torch.tensor([1.0, 5.0, 30.0, 100.0], dtype=torch.float64)
        u_plus = fo._wall_law_u_plus(y_plus)
        assert (u_plus > 0).all()

    def test_wall_law_u_plus_mixed(self):
        """Mixed wall law u+ should be positive."""
        fo = YPlusEnhanced2("test", {"rho": 1.0, "mu": 1e-5, "wallLaw": "mixed"})
        y_plus = torch.tensor([1.0, 5.0, 30.0, 100.0], dtype=torch.float64)
        u_plus = fo._wall_law_u_plus(y_plus)
        assert (u_plus > 0).all()


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
