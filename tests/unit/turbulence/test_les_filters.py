"""Tests for LES spatial filters.

Tests cover:
- LESFilter RTS registry
- SimpleFilter: top-hat (box) filter
- LaplaceFilter: Laplacian-based filter
"""

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.turbulence.les_filters import (
    LESFilter,
    SimpleFilter,
    LaplaceFilter,
)


# ---------------------------------------------------------------------------
# Mesh fixture (2-cell hex, same as smagorinsky tests)
# ---------------------------------------------------------------------------

_POINTS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 2.0],
    [1.0, 0.0, 2.0],
    [1.0, 1.0, 2.0],
    [0.0, 1.0, 2.0],
]

_FACES = [
    [4, 5, 6, 7],
    [0, 3, 2, 1],
    [0, 1, 5, 4],
    [3, 7, 6, 2],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
    [8, 9, 10, 11],
    [4, 5, 9, 8],
    [7, 11, 10, 6],
    [4, 8, 11, 7],
    [5, 6, 10, 9],
]

_OWNER = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_NEIGHBOUR = [1]

_BOUNDARY = [
    {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
    {"name": "top", "type": "wall", "startFace": 6, "nFaces": 5},
]


@pytest.fixture
def mesh():
    """2-cell hex FvMesh with geometry computed."""
    m = FvMesh(
        points=torch.tensor(_POINTS, dtype=torch.float64),
        faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in _FACES],
        owner=torch.tensor(_OWNER, dtype=INDEX_DTYPE),
        neighbour=torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE),
        boundary=_BOUNDARY,
    )
    m.compute_geometry()
    return m


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestLESFilterRegistry:
    """LESFilter RTS registration tests."""

    def test_simple_filter_registered(self):
        assert "simpleFilter" in LESFilter.available_types()

    def test_laplace_filter_registered(self):
        assert "laplaceFilter" in LESFilter.available_types()

    def test_factory_create_simple(self):
        f = LESFilter.create("simpleFilter", n_passes=2)
        assert isinstance(f, SimpleFilter)
        assert f.n_passes == 2

    def test_factory_create_laplace(self):
        f = LESFilter.create("laplaceFilter", n_iterations=3)
        assert isinstance(f, LaplaceFilter)
        assert f.n_iterations == 3

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown LESFilter"):
            LESFilter.create("nonexistentFilter")

    def test_available_types_sorted(self):
        types = LESFilter.available_types()
        assert types == sorted(types)

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @LESFilter.register("simpleFilter")
            class _Duplicate:
                pass


# ---------------------------------------------------------------------------
# SimpleFilter tests
# ---------------------------------------------------------------------------


class TestSimpleFilter:
    """Top-hat filter tests."""

    def test_default_n_passes(self):
        f = SimpleFilter()
        assert f.n_passes == 1

    def test_custom_n_passes(self):
        f = SimpleFilter(n_passes=3)
        assert f.n_passes == 3

    def test_n_passes_min_one(self):
        f = SimpleFilter(n_passes=0)
        assert f.n_passes == 1

    def test_scalar_field_shape(self, mesh):
        f = SimpleFilter()
        field = torch.ones(mesh.n_cells, dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        assert filtered.shape == field.shape

    def test_vector_field_shape(self, mesh):
        f = SimpleFilter()
        field = torch.ones(mesh.n_cells, 3, dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        assert filtered.shape == field.shape

    def test_uniform_field_unchanged(self, mesh):
        """A uniform scalar field should pass through the filter unchanged."""
        f = SimpleFilter()
        field = torch.full((mesh.n_cells,), 5.0, dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        assert torch.allclose(filtered, field, atol=1e-12)

    def test_uniform_vector_unchanged(self, mesh):
        """A uniform vector field should pass through the filter unchanged."""
        f = SimpleFilter()
        field = torch.tensor([[1.0, 2.0, 3.0]] * mesh.n_cells, dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        assert torch.allclose(filtered, field, atol=1e-12)

    def test_reduces_variation(self, mesh):
        """Filtering should reduce spatial variation."""
        f = SimpleFilter()
        field = torch.tensor([0.0, 10.0], dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        # After averaging, both cells should be closer to the mean
        mean_orig = field.mean()
        max_dev_orig = (field - mean_orig).abs().max()
        max_dev_filt = (filtered - mean_orig).abs().max()
        assert max_dev_filt < max_dev_orig

    def test_multiple_passes_more_smoothing(self, mesh):
        """More passes should produce more smoothing."""
        field = torch.tensor([0.0, 10.0], dtype=torch.float64)

        f1 = SimpleFilter(n_passes=1)
        f2 = SimpleFilter(n_passes=5)

        filtered1 = f1.apply_filter(field.clone(), mesh)
        filtered2 = f2.apply_filter(field.clone(), mesh)

        # Both cells should be closer to mean after 5 passes
        mean = field.mean()
        dev1 = (filtered1 - mean).abs().sum()
        dev2 = (filtered2 - mean).abs().sum()
        assert dev2 <= dev1 + 1e-10

    def test_finite_output(self, mesh):
        """Output should contain no NaN or Inf."""
        f = SimpleFilter()
        field = torch.tensor([1e10, -1e10], dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        assert torch.isfinite(filtered).all()


# ---------------------------------------------------------------------------
# LaplaceFilter tests
# ---------------------------------------------------------------------------


class TestLaplaceFilter:
    """Laplacian-based filter tests."""

    def test_default_n_iterations(self):
        f = LaplaceFilter()
        assert f.n_iterations == 1

    def test_custom_n_iterations(self):
        f = LaplaceFilter(n_iterations=5)
        assert f.n_iterations == 5

    def test_n_iterations_min_one(self):
        f = LaplaceFilter(n_iterations=0)
        assert f.n_iterations == 1

    def test_scalar_field_shape(self, mesh):
        f = LaplaceFilter()
        field = torch.ones(mesh.n_cells, dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        assert filtered.shape == field.shape

    def test_vector_field_shape(self, mesh):
        f = LaplaceFilter()
        field = torch.ones(mesh.n_cells, 3, dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        assert filtered.shape == field.shape

    def test_uniform_scalar_unchanged(self, mesh):
        """A uniform scalar field should pass through the Laplacian filter unchanged."""
        f = LaplaceFilter()
        field = torch.full((mesh.n_cells,), 7.0, dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        # Laplacian of uniform field = 0 → no change
        assert torch.allclose(filtered, field, atol=1e-12)

    def test_uniform_vector_unchanged(self, mesh):
        """A uniform vector field should pass through the Laplacian filter unchanged."""
        f = LaplaceFilter()
        field = torch.tensor([[3.0, 4.0, 5.0]] * mesh.n_cells, dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        assert torch.allclose(filtered, field, atol=1e-12)

    def test_reduces_variation(self, mesh):
        """Laplacian filter should reduce spatial variation."""
        f = LaplaceFilter()
        field = torch.tensor([0.0, 10.0], dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        mean_orig = field.mean()
        max_dev_orig = (field - mean_orig).abs().max()
        max_dev_filt = (filtered - mean_orig).abs().max()
        assert max_dev_filt < max_dev_orig

    def test_more_iterations_more_smoothing(self, mesh):
        """More iterations should produce more smoothing."""
        field = torch.tensor([0.0, 10.0], dtype=torch.float64)

        f1 = LaplaceFilter(n_iterations=1)
        f5 = LaplaceFilter(n_iterations=5)

        filtered1 = f1.apply_filter(field.clone(), mesh)
        filtered5 = f5.apply_filter(field.clone(), mesh)

        mean = field.mean()
        dev1 = (filtered1 - mean).abs().sum()
        dev5 = (filtered5 - mean).abs().sum()
        assert dev5 <= dev1 + 1e-10

    def test_preserves_mean(self, mesh):
        """Filter should approximately preserve the field mean."""
        f = LaplaceFilter()
        field = torch.tensor([2.0, 8.0], dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        # Mean should be approximately preserved (Laplacian adds/subtracts equally)
        assert torch.allclose(filtered.mean(), field.mean(), atol=1.0)

    def test_finite_output(self, mesh):
        """Output should contain no NaN or Inf."""
        f = LaplaceFilter()
        field = torch.tensor([1e10, -1e10], dtype=torch.float64)
        filtered = f.apply_filter(field, mesh)
        assert torch.isfinite(filtered).all()
