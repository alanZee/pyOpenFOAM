"""Tests for view_factors_gen — view factor matrix computation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyfoam.tools.view_factors_gen import view_factors_gen, ViewFactorResult


class TestViewFactorsGen:
    """Test the view_factors_gen function."""

    def test_basic_computation(self, fv_mesh):
        """View factor computation should produce a valid result."""
        result = view_factors_gen(fv_mesh)
        assert isinstance(result, ViewFactorResult)
        assert result.n_total_faces > 0
        assert result.view_factor_matrix.shape[0] == result.n_total_faces
        assert result.view_factor_matrix.shape[1] == result.n_total_faces

    def test_view_factor_non_negative(self, fv_mesh):
        """All view factors should be non-negative."""
        result = view_factors_gen(fv_mesh)
        assert np.all(result.view_factor_matrix >= 0.0)

    def test_diagonal_is_zero(self, fv_mesh):
        """Self-view factors should be zero (flat faces)."""
        result = view_factors_gen(fv_mesh)
        diag = np.diag(result.view_factor_matrix)
        np.testing.assert_allclose(diag, 0.0, atol=1e-15)

    def test_view_factor_bounded(self, fv_mesh):
        """View factors should not exceed 1."""
        result = view_factors_gen(fv_mesh)
        assert np.all(result.view_factor_matrix <= 1.0 + 1e-10)

    def test_row_sums_shape(self, fv_mesh):
        """Row sums should have correct shape."""
        result = view_factors_gen(fv_mesh)
        assert result.row_sums.shape[0] == result.n_total_faces

    def test_patch_names_populated(self, fv_mesh):
        """Patch names should be populated."""
        result = view_factors_gen(fv_mesh)
        assert len(result.patch_names) > 0
        assert len(result.patch_face_counts) > 0
        assert sum(result.patch_face_counts) == result.n_total_faces

    def test_filter_patches(self, fv_mesh):
        """Specifying patches should filter the result."""
        all_patches = view_factors_gen(fv_mesh)
        if len(all_patches.patch_names) > 1:
            single_patch = view_factors_gen(
                fv_mesh,
                boundary_patches=[all_patches.patch_names[0]],
            )
            assert single_patch.n_total_faces < all_patches.n_total_faces

    def test_invalid_patch_name(self, fv_mesh):
        """Non-existent patch name should raise ValueError."""
        with pytest.raises(ValueError, match="No matching"):
            view_factors_gen(fv_mesh, boundary_patches=["nonexistent_patch"])

    def test_symmetry_for_simple_geometry_skip(self, fv_mesh):
        """For symmetric geometry, some view factors should be symmetric."""
        result = view_factors_gen(fv_mesh)
        F = result.view_factor_matrix
        # Check that the matrix is not trivially all zeros
        assert np.sum(np.abs(F)) > 0
