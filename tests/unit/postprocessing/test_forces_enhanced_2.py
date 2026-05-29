"""Tests for ForcesEnhanced2.

Tests cover:
- Projected forces
- Direction vector normalization
- Coefficient computation
- Execute with mesh
"""

import pytest
import torch

from pyfoam.postprocessing.forces_enhanced_2 import (
    ForcesEnhanced2,
    ProjectedForces,
)
from pyfoam.postprocessing.forces_enhanced import ForcesEnhanced


class TestForcesEnhanced2:
    """Tests for ForcesEnhanced2."""

    def test_inherits_from_enhanced(self):
        fo = ForcesEnhanced2("test", {
            "patches": ["bottom"],
            "rhoInf": 1.0,
        })
        assert isinstance(fo, ForcesEnhanced)

    def test_default_directions(self):
        fo = ForcesEnhanced2("test", {"patches": ["bottom"]})
        assert fo.drag_dir.norm().item() == pytest.approx(1.0)
        assert fo.lift_dir.norm().item() == pytest.approx(1.0)

    def test_custom_directions(self):
        fo = ForcesEnhanced2("test", {
            "patches": ["bottom"],
            "dragDir": [0.0, 0.0, 1.0],
            "liftDir": [0.0, 1.0, 0.0],
        })
        assert fo.drag_dir[2].item() == pytest.approx(1.0)

    def test_direction_orthogonality(self):
        """Side direction should be perpendicular to drag and lift."""
        fo = ForcesEnhanced2("test", {"patches": ["bottom"]})
        dot_drag = torch.dot(fo.drag_dir, fo.side_dir).abs()
        dot_lift = torch.dot(fo.lift_dir, fo.side_dir).abs()
        assert dot_drag.item() < 1e-10
        assert dot_lift.item() < 1e-10

    def test_default_reference_params(self):
        fo = ForcesEnhanced2("test", {"patches": ["bottom"]})
        assert fo._L_ref == pytest.approx(1.0)
        assert fo._A_ref == pytest.approx(1.0)
        assert fo._compute_coefficients is True

    def test_execute(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced2("test", {
            "patches": ["bottom"],
            "rhoInf": 1.0,
            "Uref": 1.0,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.projected_forces) == 1

    def test_projected_forces_values(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced2("test", {
            "patches": ["bottom"],
            "rhoInf": 1.0,
            "Uref": 1.0,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        pf = fo.get_latest_projected()
        assert pf is not None
        assert isinstance(pf, ProjectedForces)
        assert pf.time == pytest.approx(0.0)
        assert isinstance(pf.drag, float)
        assert isinstance(pf.lift, float)

    def test_get_latest_no_data(self):
        fo = ForcesEnhanced2("test", {"patches": ["bottom"]})
        assert fo.get_latest_projected() is None


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
