"""Tests for FunctionObject base class and FunctionObjectRegistry."""

from __future__ import annotations

import pytest
import torch

from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry


# ---------------------------------------------------------------------------
# Concrete test implementation
# ---------------------------------------------------------------------------


class MockFunctionObject(FunctionObject):
    """Mock function object for testing."""

    def __init__(self, name: str = "mock", config=None):
        super().__init__(name, config)
        self.initialised = False
        self.executed_times = []
        self.written = False
        self.finalised = False

    def initialise(self, mesh, fields):
        self._mesh = mesh
        self._fields = fields
        self.initialised = True

    def execute(self, time):
        self.executed_times.append(time)

    def write(self):
        self.written = True

    def finalise(self):
        self.finalised = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFunctionObject:
    def test_init_defaults(self):
        fo = MockFunctionObject()
        assert fo.name == "mock"
        assert fo.config == {}
        assert fo._enabled is True
        assert fo.mesh is None
        assert fo.fields == {}

    def test_init_with_config(self):
        config = {"enabled": False, "key": "value"}
        fo = MockFunctionObject("test", config)
        assert fo.name == "test"
        assert fo.config == config
        assert fo._enabled is False

    def test_set_output_path(self, tmp_path):
        fo = MockFunctionObject()
        output_dir = tmp_path / "output"
        fo.set_output_path(output_dir)
        assert fo.output_path == output_dir
        assert output_dir.exists()

    def test_lifecycle(self, fv_mesh, sample_fields):
        fo = MockFunctionObject()

        # Initialisation
        fo.initialise(fv_mesh, sample_fields)
        assert fo.initialised is True
        assert fo.mesh is fv_mesh
        assert fo.fields == sample_fields

        # Execute
        fo.execute(0.0)
        fo.execute(0.1)
        assert fo.executed_times == [0.0, 0.1]

        # Write
        fo.write()
        assert fo.written is True

        # Finalise
        fo.finalise()
        assert fo.finalised is True

    def test_repr(self):
        fo = MockFunctionObject("test")
        assert "MockFunctionObject" in repr(fo)
        assert "test" in repr(fo)


class TestFunctionObjectRegistry:
    @pytest.fixture(autouse=False)
    def clear_registry(self):
        """Clear registry before each test."""
        FunctionObjectRegistry.clear()
        yield
        FunctionObjectRegistry.clear()

    def test_register_and_create(self, clear_registry):
        FunctionObjectRegistry.register("mock", MockFunctionObject)

        config = {"name": "test_mock", "enabled": True}
        fo = FunctionObjectRegistry.create("mock", config)

        assert isinstance(fo, MockFunctionObject)
        assert fo.name == "test_mock"
        assert fo._enabled is True

    def test_register_default_name(self, clear_registry):
        FunctionObjectRegistry.register("mock", MockFunctionObject)

        fo = FunctionObjectRegistry.create("mock", {})
        assert fo.name == "mock"

    def test_register_invalid_class(self, clear_registry):
        with pytest.raises(TypeError):
            FunctionObjectRegistry.register("invalid", dict)

    def test_create_unknown_type(self, clear_registry):
        with pytest.raises(KeyError, match="Unknown function object type"):
            FunctionObjectRegistry.create("nonexistent", {})

    def test_list_registered(self, clear_registry):
        FunctionObjectRegistry.register("mock1", MockFunctionObject)
        FunctionObjectRegistry.register("mock2", MockFunctionObject)

        registered = FunctionObjectRegistry.list_registered()
        assert "mock1" in registered
        assert "mock2" in registered

    def test_clear(self, clear_registry):
        FunctionObjectRegistry.register("mock", MockFunctionObject)
        assert len(FunctionObjectRegistry.list_registered()) == 1

        FunctionObjectRegistry.clear()
        assert len(FunctionObjectRegistry.list_registered()) == 0
