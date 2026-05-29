"""Tests for WallShearStressEnhanced.

Tests cover:
- Enhanced stats (tau_w avg, max, min, std)
- Friction velocity (u_tau)
- Friction coefficient (Cf)
- Properties and inheritance
"""

import pytest
import torch

from pyfoam.postprocessing.wall_shear_stress import WallShearStress
from pyfoam.postprocessing.wall_shear_stress_enhanced import (
    WallShearStressEnhanced,
    WSSPatchStats,
)


class TestWallShearStressEnhanced:
    """Tests for WallShearStressEnhanced."""

    def test_inherits_from_wss(self):
        fo = WallShearStressEnhanced("test", {"patches": ["bottom"]})
        assert isinstance(fo, WallShearStress)

    def test_default_params(self):
        fo = WallShearStressEnhanced("test", {"patches": ["bottom"]})
        assert fo._mu == pytest.approx(1.0)
        assert fo._u_ref == pytest.approx(1.0)
        assert fo._compute_cf is False
        assert fo._compute_utau is True

    def test_custom_params(self):
        fo = WallShearStressEnhanced("test", {
            "patches": ["bottom"],
            "mu": 1e-3,
            "Uref": 10.0,
            "computeCf": True,
        })
        assert fo._mu == pytest.approx(1e-3)
        assert fo._u_ref == pytest.approx(10.0)
        assert fo._compute_cf is True

    def test_execute(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced("test", {"patches": ["bottom"]})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(1.0)
        assert len(fo.patch_stats_history) == 2
        assert len(fo.utau_history) == 2

    def test_patch_stats(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced("test", {"patches": ["bottom"]})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        stats = fo.patch_stats_history[0]
        assert "bottom" in stats
        ps = stats["bottom"]
        assert isinstance(ps, WSSPatchStats)
        assert ps.tau_w_avg >= 0
        assert ps.tau_w_max >= ps.tau_w_min
        assert ps.n_faces > 0

    def test_utau_history(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced("test", {"patches": ["bottom"]})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        utau = fo.utau_history[0]
        assert "bottom" in utau
        assert (utau["bottom"] >= 0).all()

    def test_cf_history(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced("test", {
            "patches": ["bottom"],
            "computeCf": True,
            "Uref": 1.0,
            "rho": 1.0,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        cf = fo.cf_history[0]
        assert "bottom" in cf
        assert cf["bottom"] >= 0

    def test_get_latest_stats(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced("test", {"patches": ["bottom"]})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        ps = fo.get_latest_stats("bottom")
        assert ps is not None
        assert isinstance(ps, WSSPatchStats)

    def test_get_latest_stats_no_data(self):
        fo = WallShearStressEnhanced("test", {"patches": ["bottom"]})
        assert fo.get_latest_stats("bottom") is None

    def test_execute_skips_no_U(self, fv_mesh):
        from pyfoam.fields.vol_fields import volScalarField
        fo = WallShearStressEnhanced("test", {"patches": ["bottom"]})
        p = volScalarField(fv_mesh, "p")
        p.assign(torch.tensor([101325.0, 101300.0], dtype=torch.float64))
        fo.initialise(fv_mesh, {"p": p})
        fo.execute(0.0)
        assert len(fo.patch_stats_history) == 0


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
