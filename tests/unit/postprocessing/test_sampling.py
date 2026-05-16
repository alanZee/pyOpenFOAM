"""Tests for Probes, LineSample, and SurfaceSample function objects."""

from __future__ import annotations

import pytest
import torch

from pyfoam.postprocessing.sampling import Probes, LineSample, SurfaceSample, _find_cell_for_point


class TestFindCellForPoint:
    def test_find_cell_centre(self, fv_mesh):
        # Point at cell 0 centre (0.5, 0.5, 0.5)
        point = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        idx = _find_cell_for_point(point, fv_mesh)
        assert idx == 0

    def test_find_cell_1(self, fv_mesh):
        # Point at cell 1 centre (0.5, 0.5, 1.5)
        point = torch.tensor([0.5, 0.5, 1.5], dtype=torch.float64)
        idx = _find_cell_for_point(point, fv_mesh)
        assert idx == 1


class TestProbes:
    def test_init_defaults(self):
        probes = Probes()
        assert probes.name == "probes"
        assert probes._field_names == []
        assert probes._locations == []

    def test_init_with_config(self):
        config = {
            "fields": ["p", "U"],
            "probeLocations": [[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]],
        }
        probes = Probes("p1", config)
        assert probes._field_names == ["p", "U"]
        assert len(probes._locations) == 2

    def test_initialise(self, fv_mesh, sample_fields):
        config = {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        }
        probes = Probes("p1", config)
        probes.initialise(fv_mesh, sample_fields)

        assert len(probes._cell_indices) == 1
        assert probes._cell_indices[0] == 0

    def test_execute_scalar(self, fv_mesh, sample_fields):
        config = {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]],
        }
        probes = Probes("p1", config)
        probes.initialise(fv_mesh, sample_fields)

        probes.execute(0.0)
        assert len(probes.times) == 1
        assert len(probes.results["p"][0]) == 1
        assert len(probes.results["p"][1]) == 1

    def test_execute_vector(self, fv_mesh, sample_fields):
        config = {
            "fields": ["U"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        }
        probes = Probes("p1", config)
        probes.initialise(fv_mesh, sample_fields)

        probes.execute(0.0)
        assert len(probes.vector_results["U"][0]) == 1

    def test_execute_multiple_times(self, fv_mesh, sample_fields):
        config = {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        }
        probes = Probes("p1", config)
        probes.initialise(fv_mesh, sample_fields)

        for t in [0.0, 0.1, 0.2]:
            probes.execute(t)

        assert len(probes.times) == 3
        assert len(probes.results["p"][0]) == 3

    def test_write(self, fv_mesh, sample_fields, tmp_path):
        config = {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        }
        probes = Probes("p1", config)
        probes.set_output_path(tmp_path)
        probes.initialise(fv_mesh, sample_fields)
        probes.execute(0.0)
        probes.write()

        probe_file = tmp_path / "p_probe.dat"
        assert probe_file.exists()


class TestLineSample:
    def test_init_defaults(self):
        ls = LineSample()
        assert ls.name == "lineSample"
        assert ls._n_points == 100

    def test_init_with_config(self):
        config = {
            "fields": ["p"],
            "start": [0.0, 0.5, 0.5],
            "end": [1.0, 0.5, 0.5],
            "nPoints": 50,
        }
        ls = LineSample("ls1", config)
        assert ls._n_points == 50

    def test_initialise(self, fv_mesh, sample_fields):
        config = {
            "fields": ["p"],
            "start": [0.0, 0.5, 0.5],
            "end": [1.0, 0.5, 0.5],
            "nPoints": 10,
        }
        ls = LineSample("ls1", config)
        ls.initialise(fv_mesh, sample_fields)

        assert len(ls.sample_points) == 10
        assert len(ls._cell_indices) == 10

    def test_execute(self, fv_mesh, sample_fields):
        config = {
            "fields": ["p"],
            "start": [0.0, 0.5, 0.5],
            "end": [1.0, 0.5, 0.5],
            "nPoints": 5,
        }
        ls = LineSample("ls1", config)
        ls.initialise(fv_mesh, sample_fields)

        ls.execute(0.0)
        assert len(ls.times) == 1
        assert len(ls.results["p"]) == 1

    def test_write(self, fv_mesh, sample_fields, tmp_path):
        config = {
            "fields": ["p"],
            "start": [0.0, 0.5, 0.5],
            "end": [1.0, 0.5, 0.5],
            "nPoints": 5,
        }
        ls = LineSample("ls1", config)
        ls.set_output_path(tmp_path)
        ls.initialise(fv_mesh, sample_fields)
        ls.execute(0.0)
        ls.write()

        line_file = tmp_path / "p_line.dat"
        assert line_file.exists()


class TestSurfaceSample:
    def test_init_defaults(self):
        ss = SurfaceSample()
        assert ss.name == "surfaceSample"
        assert ss._surface_type == "plane"

    def test_init_with_config(self):
        config = {
            "fields": ["p"],
            "surfaceType": "plane",
            "surfaceParams": {"point": [0.5, 0.5, 0.5], "normal": [0, 0, 1]},
            "nPoints": 100,
        }
        ss = SurfaceSample("ss1", config)
        assert ss._surface_type == "plane"

    def test_initialise_plane(self, fv_mesh, sample_fields):
        config = {
            "fields": ["p"],
            "surfaceType": "plane",
            "surfaceParams": {"point": [0.5, 0.5, 0.5], "normal": [0, 0, 1]},
            "nPoints": 25,
        }
        ss = SurfaceSample("ss1", config)
        ss.initialise(fv_mesh, sample_fields)

        assert len(ss.sample_points) > 0
        assert len(ss._cell_indices) > 0

    def test_execute(self, fv_mesh, sample_fields):
        config = {
            "fields": ["p"],
            "surfaceType": "plane",
            "surfaceParams": {"point": [0.5, 0.5, 0.5], "normal": [0, 0, 1]},
            "nPoints": 25,
        }
        ss = SurfaceSample("ss1", config)
        ss.initialise(fv_mesh, sample_fields)

        ss.execute(0.0)
        assert len(ss.times) == 1
        assert len(ss.results["p"]) == 1

    def test_results_have_stats(self, fv_mesh, sample_fields):
        config = {
            "fields": ["p"],
            "surfaceType": "plane",
            "surfaceParams": {"point": [0.5, 0.5, 0.5], "normal": [0, 0, 1]},
            "nPoints": 25,
        }
        ss = SurfaceSample("ss1", config)
        ss.initialise(fv_mesh, sample_fields)
        ss.execute(0.0)

        stats = ss.results["p"][0]
        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats

    def test_write(self, fv_mesh, sample_fields, tmp_path):
        config = {
            "fields": ["p"],
            "surfaceType": "plane",
            "surfaceParams": {"point": [0.5, 0.5, 0.5], "normal": [0, 0, 1]},
            "nPoints": 25,
        }
        ss = SurfaceSample("ss1", config)
        ss.set_output_path(tmp_path)
        ss.initialise(fv_mesh, sample_fields)
        ss.execute(0.0)
        ss.write()

        surf_file = tmp_path / "p_surface.dat"
        assert surf_file.exists()


class TestSamplingRegistration:
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure modules are imported and registered."""
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        FunctionObjectRegistry.clear()
        # Force re-import to trigger registration
        import importlib
        from pyfoam.postprocessing import sampling
        importlib.reload(sampling)
        yield
        FunctionObjectRegistry.clear()

    def test_probes_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "probes" in FunctionObjectRegistry.list_registered()

    def test_sets_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "sets" in FunctionObjectRegistry.list_registered()

    def test_surfaces_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "surfaces" in FunctionObjectRegistry.list_registered()
