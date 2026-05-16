"""Tests for WallShearStress and YPlus function objects."""

from __future__ import annotations

import pytest
import torch

from pyfoam.postprocessing.wall_shear_stress import WallShearStress
from pyfoam.postprocessing.y_plus import YPlus


class TestWallShearStress:
    def test_init_defaults(self):
        wss = WallShearStress()
        assert wss.name == "wallShearStress"
        assert wss._rho == 1.0

    def test_init_with_config(self):
        config = {"patches": ["movingWall"], "rho": 1.225}
        wss = WallShearStress("wss1", config)
        assert wss._patches == ["movingWall"]
        assert wss._rho == 1.225

    def test_initialise(self, fv_mesh, sample_fields):
        wss = WallShearStress("wss", {"patches": ["bottom"]})
        wss.initialise(fv_mesh, sample_fields)
        assert wss.mesh is fv_mesh

    def test_auto_detect_wall_patches(self, fv_mesh, sample_fields):
        wss = WallShearStress("wss")
        wss.initialise(fv_mesh, sample_fields)
        # Both patches are type "wall"
        assert len(wss._patches) == 2

    def test_execute(self, fv_mesh, sample_fields):
        wss = WallShearStress("wss", {"patches": ["bottom"]})
        wss.initialise(fv_mesh, sample_fields)

        wss.execute(0.0)
        assert len(wss.tau_w_history) == 1
        assert len(wss.times) == 1

    def test_tau_w_shape(self, fv_mesh, sample_fields):
        wss = WallShearStress("wss", {"patches": ["bottom"]})
        wss.initialise(fv_mesh, sample_fields)
        wss.execute(0.0)

        # bottom patch has 5 faces
        tau = wss.tau_w_history[0]["bottom"]
        assert tau.shape == (5, 3)

    def test_tau_w_non_negative_magnitude(self, fv_mesh, sample_fields):
        wss = WallShearStress("wss", {"patches": ["bottom"]})
        wss.initialise(fv_mesh, sample_fields)
        wss.execute(0.0)

        tau = wss.tau_w_history[0]["bottom"]
        tau_mag = tau.norm(dim=1)
        assert (tau_mag >= 0).all()

    def test_write(self, fv_mesh, sample_fields, tmp_path):
        wss = WallShearStress("wss", {"patches": ["bottom"]})
        wss.set_output_path(tmp_path)
        wss.initialise(fv_mesh, sample_fields)
        wss.execute(0.0)
        wss.write()

        wss_file = tmp_path / "wallShearStress.dat"
        assert wss_file.exists()


class TestYPlus:
    def test_init_defaults(self):
        yp = YPlus()
        assert yp.name == "yPlus"
        assert yp._rho == 1.0
        assert yp._mu == 1.0

    def test_init_with_config(self):
        config = {"patches": ["wall"], "rho": 1.225, "mu": 1e-3}
        yp = YPlus("yp1", config)
        assert yp._rho == 1.225
        assert yp._mu == 1e-3

    def test_initialise(self, fv_mesh, sample_fields):
        yp = YPlus("yp", {"patches": ["bottom"]})
        yp.initialise(fv_mesh, sample_fields)
        assert yp.mesh is fv_mesh

    def test_auto_detect_wall_patches(self, fv_mesh, sample_fields):
        yp = YPlus("yp")
        yp.initialise(fv_mesh, sample_fields)
        assert len(yp._patches) == 2

    def test_execute(self, fv_mesh, sample_fields):
        yp = YPlus("yp", {"patches": ["bottom"]})
        yp.initialise(fv_mesh, sample_fields)

        yp.execute(0.0)
        assert len(yp.y_plus_history) == 1
        assert len(yp.times) == 1

    def test_y_plus_shape(self, fv_mesh, sample_fields):
        yp = YPlus("yp", {"patches": ["bottom"]})
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(0.0)

        # bottom patch has 5 faces
        y_plus = yp.y_plus_history[0]["bottom"]
        assert y_plus.shape == (5,)

    def test_y_plus_positive(self, fv_mesh, sample_fields):
        yp = YPlus("yp", {"patches": ["bottom"]})
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(0.0)

        y_plus = yp.y_plus_history[0]["bottom"]
        assert (y_plus >= 0).all()

    def test_y_plus_depends_on_viscosity(self, fv_mesh, sample_fields):
        # Lower viscosity → higher y+
        yp_low = YPlus("yp", {"patches": ["bottom"], "mu": 0.001})
        yp_high = YPlus("yp", {"patches": ["bottom"], "mu": 1.0})

        yp_low.initialise(fv_mesh, sample_fields)
        yp_high.initialise(fv_mesh, sample_fields)

        yp_low.execute(0.0)
        yp_high.execute(0.0)

        y_low = yp_low.y_plus_history[0]["bottom"].mean()
        y_high = yp_high.y_plus_history[0]["bottom"].mean()

        # Lower viscosity → higher y+ (y+ = y * u_tau / nu)
        assert y_low > y_high

    def test_write(self, fv_mesh, sample_fields, tmp_path):
        yp = YPlus("yp", {"patches": ["bottom"]})
        yp.set_output_path(tmp_path)
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(0.0)
        yp.write()

        yp_file = tmp_path / "yPlus.dat"
        assert yp_file.exists()


class TestRegistration:
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure modules are imported and registered."""
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        FunctionObjectRegistry.clear()
        # Force re-import to trigger registration
        import importlib
        from pyfoam.postprocessing import wall_shear_stress, y_plus
        importlib.reload(wall_shear_stress)
        importlib.reload(y_plus)
        yield
        FunctionObjectRegistry.clear()

    def test_wall_shear_stress_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "wallShearStress" in FunctionObjectRegistry.list_registered()

    def test_y_plus_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "yPlus" in FunctionObjectRegistry.list_registered()
