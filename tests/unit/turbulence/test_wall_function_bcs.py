"""Tests for wall function boundary conditions.

Tests cover:
- nutLowReWallFunction
- epsilonWallFunction
- omegaWallFunction
"""

import pytest
import torch

from pyfoam.boundary.wall_function import (
    NutLowReWallFunctionBC,
    EpsilonWallFunctionBC,
    OmegaWallFunctionBC,
    NutkWallFunctionBC,
    KqRWallFunctionBC,
)

from tests.unit.turbulence.conftest import make_fv_mesh


class TestNutLowReWallFunction:
    """Tests for low-Re wall function for ν_t."""

    def test_apply_sets_zero(self):
        """apply() sets wall face ν_t to zero."""
        mesh = make_fv_mesh()
        # Create a simple patch-like object
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutLowReWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64)
        result = bc.apply(field)

        # Wall face values should be zero
        assert (result[:10] == 0).all()

    def test_matrix_contributions_zero(self):
        """matrix_contributions() returns zero diag and source."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutLowReWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)

        assert diag.shape == (20,)
        assert source.shape == (20,)
        assert (diag == 0).all()
        assert (source == 0).all()


class TestEpsilonWallFunction:
    """Tests for ε wall function."""

    def test_compute_epsilon_shape(self):
        """compute_epsilon() returns correct shape."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = EpsilonWallFunctionBC(patch)
        k = torch.ones(10, dtype=torch.float64)
        y = torch.ones(10, dtype=torch.float64) * 0.01

        eps = bc.compute_epsilon(k, y)
        assert eps.shape == (10,)

    def test_compute_epsilon_positive(self):
        """compute_epsilon() returns positive values."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = EpsilonWallFunctionBC(patch)
        k = torch.ones(10, dtype=torch.float64) * 0.01
        y = torch.ones(10, dtype=torch.float64) * 0.01

        eps = bc.compute_epsilon(k, y)
        assert (eps > 0).all()

    def test_apply_with_value(self):
        """apply() with value coefficient sets face values."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = EpsilonWallFunctionBC(patch, {"value": 0.1})
        field = torch.zeros(20, dtype=torch.float64)
        result = bc.apply(field)

        # Wall face values should be 0.1
        assert (result[:10] == 0.1).all()

    def test_apply_without_value(self):
        """apply() without value coefficient leaves field unchanged."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = EpsilonWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64) * 0.5
        result = bc.apply(field)

        # Field should be unchanged
        assert (result == 0.5).all()


class TestOmegaWallFunction:
    """Tests for ω wall function."""

    def test_compute_omega_shape(self):
        """compute_omega() returns correct shape."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = OmegaWallFunctionBC(patch)
        k = torch.ones(10, dtype=torch.float64)
        y = torch.ones(10, dtype=torch.float64) * 0.01

        omega = bc.compute_omega(k, y)
        assert omega.shape == (10,)

    def test_compute_omega_positive(self):
        """compute_omega() returns positive values."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = OmegaWallFunctionBC(patch)
        k = torch.ones(10, dtype=torch.float64) * 0.01
        y = torch.ones(10, dtype=torch.float64) * 0.01

        omega = bc.compute_omega(k, y)
        assert (omega > 0).all()

    def test_apply_with_value(self):
        """apply() with value coefficient sets face values."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = OmegaWallFunctionBC(patch, {"value": 100.0})
        field = torch.zeros(20, dtype=torch.float64)
        result = bc.apply(field)

        # Wall face values should be 100.0
        assert (result[:10] == 100.0).all()

    def test_apply_without_value(self):
        """apply() without value coefficient leaves field unchanged."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = OmegaWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64) * 50.0
        result = bc.apply(field)

        # Field should be unchanged
        assert (result == 50.0).all()
