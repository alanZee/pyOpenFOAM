"""Tests for interpolation schemes — linear, upwind, linear-upwind, QUICK."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh

from pyfoam.discretisation.weights import (
    compute_centre_weights,
    compute_upwind_weights,
)
from pyfoam.discretisation.interpolation import LinearInterpolation
from pyfoam.discretisation.schemes.upwind import UpwindInterpolation
from pyfoam.discretisation.schemes.linear_upwind import LinearUpwindInterpolation
from pyfoam.discretisation.schemes.quick import QuickInterpolation

from tests.unit.discretisation.conftest import make_fv_mesh


# ---------------------------------------------------------------------------
# Weight computation tests
# ---------------------------------------------------------------------------


class TestComputeCentreWeights:
    """Tests for compute_centre_weights function."""

    def test_weights_match_mesh_weights(self, fv_mesh: FvMesh):
        """Computed weights should match the mesh's cached face_weights."""
        weights = compute_centre_weights(
            fv_mesh.cell_centres,
            fv_mesh.face_centres,
            fv_mesh.owner,
            fv_mesh.neighbour,
            fv_mesh.n_internal_faces,
            fv_mesh.n_faces,
        )
        torch.testing.assert_close(weights, fv_mesh.face_weights)

    def test_boundary_weights_are_one(self, fv_mesh: FvMesh):
        """Boundary face weights should be exactly 1.0."""
        weights = compute_centre_weights(
            fv_mesh.cell_centres,
            fv_mesh.face_centres,
            fv_mesh.owner,
            fv_mesh.neighbour,
            fv_mesh.n_internal_faces,
            fv_mesh.n_faces,
        )
        boundary_weights = weights[fv_mesh.n_internal_faces:]
        torch.testing.assert_close(
            boundary_weights,
            torch.ones_like(boundary_weights),
        )

    def test_internal_weights_in_range(self, fv_mesh: FvMesh):
        """Internal face weights should be in [0, 1]."""
        weights = compute_centre_weights(
            fv_mesh.cell_centres,
            fv_mesh.face_centres,
            fv_mesh.owner,
            fv_mesh.neighbour,
            fv_mesh.n_internal_faces,
            fv_mesh.n_faces,
        )
        internal = weights[:fv_mesh.n_internal_faces]
        assert (internal >= 0).all()
        assert (internal <= 1).all()

    def test_symmetric_mesh_weight_is_half(self, fv_mesh: FvMesh):
        """For the symmetric 2-cell hex mesh, the internal face weight
        should be 0.5 (face is equidistant from both cell centres)."""
        weights = compute_centre_weights(
            fv_mesh.cell_centres,
            fv_mesh.face_centres,
            fv_mesh.owner,
            fv_mesh.neighbour,
            fv_mesh.n_internal_faces,
            fv_mesh.n_faces,
        )
        # Internal face is at z=1, cell centres at z=0.5 and z=1.5
        torch.testing.assert_close(
            weights[0],
            torch.tensor(0.5, dtype=fv_mesh.dtype),
        )

    def test_total_faces(self, fv_mesh: FvMesh):
        """Output should have length equal to total number of faces."""
        weights = compute_centre_weights(
            fv_mesh.cell_centres,
            fv_mesh.face_centres,
            fv_mesh.owner,
            fv_mesh.neighbour,
            fv_mesh.n_internal_faces,
            fv_mesh.n_faces,
        )
        assert weights.shape == (fv_mesh.n_faces,)


class TestComputeUpwindWeights:
    """Tests for compute_upwind_weights function."""

    def test_positive_flux_selects_owner(self, fv_mesh: FvMesh):
        """Positive flux should give owner_weight=1, neighbour_weight=0."""
        n_faces = fv_mesh.n_faces
        flux = torch.ones(n_faces, dtype=torch.float64)  # all positive
        owner_w, nbr_w = compute_upwind_weights(
            flux, fv_mesh.n_internal_faces, n_faces,
        )
        assert owner_w[0] == 1.0
        assert nbr_w[0] == 0.0

    def test_negative_flux_selects_neighbour(self, fv_mesh: FvMesh):
        """Negative flux should give owner_weight=0, neighbour_weight=1."""
        n_faces = fv_mesh.n_faces
        flux = -torch.ones(n_faces, dtype=torch.float64)  # all negative
        owner_w, nbr_w = compute_upwind_weights(
            flux, fv_mesh.n_internal_faces, n_faces,
        )
        assert owner_w[0] == 0.0
        assert nbr_w[0] == 1.0

    def test_boundary_faces_always_owner(self, fv_mesh: FvMesh):
        """Boundary faces should always use owner value."""
        n_faces = fv_mesh.n_faces
        flux = -torch.ones(n_faces, dtype=torch.float64)
        owner_w, nbr_w = compute_upwind_weights(
            flux, fv_mesh.n_internal_faces, n_faces,
        )
        boundary_owner = owner_w[fv_mesh.n_internal_faces:]
        torch.testing.assert_close(
            boundary_owner,
            torch.ones_like(boundary_owner),
        )

    def test_zero_flux_selects_owner(self, fv_mesh: FvMesh):
        """Zero flux should select owner (flux >= 0 condition)."""
        n_faces = fv_mesh.n_faces
        flux = torch.zeros(n_faces, dtype=torch.float64)
        owner_w, nbr_w = compute_upwind_weights(
            flux, fv_mesh.n_internal_faces, n_faces,
        )
        assert owner_w[0] == 1.0
        assert nbr_w[0] == 0.0


# ---------------------------------------------------------------------------
# Linear interpolation tests
# ---------------------------------------------------------------------------


class TestLinearInterpolation:
    """Tests for LinearInterpolation scheme."""

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values."""
        scheme = LinearInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 42.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 42.0, dtype=torch.float64),
        )

    def test_internal_face_value(self, fv_mesh: FvMesh):
        """Internal face value should be weighted average of owner/neighbour."""
        scheme = LinearInterpolation(fv_mesh)
        # Cell 0 = 10.0, Cell 1 = 20.0
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)

        # Internal face weight is 0.5 for this symmetric mesh
        # φ_f = 0.5 * 10 + 0.5 * 20 = 15
        torch.testing.assert_close(
            face_vals[0],
            torch.tensor(15.0, dtype=torch.float64),
        )

    def test_boundary_faces_use_owner(self, fv_mesh: FvMesh):
        """Boundary face values should equal owner cell values."""
        scheme = LinearInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)

        # Boundary faces of cell 0 (faces 1-5)
        for i in range(1, 6):
            torch.testing.assert_close(
                face_vals[i],
                torch.tensor(10.0, dtype=torch.float64),
                msg=f"Face {i}: expected 10.0 (cell 0 owner)",
            )

        # Boundary faces of cell 1 (faces 6-10)
        for i in range(6, 11):
            torch.testing.assert_close(
                face_vals[i],
                torch.tensor(20.0, dtype=torch.float64),
                msg=f"Face {i}: expected 20.0 (cell 1 owner)",
            )

    def test_output_shape(self, fv_mesh: FvMesh):
        """Output should have shape (n_faces,)."""
        scheme = LinearInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_callable(self, fv_mesh: FvMesh):
        """Scheme should be callable."""
        scheme = LinearInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_invalid_dimension_raises(self, fv_mesh: FvMesh):
        """Non-1D input should raise ValueError."""
        scheme = LinearInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_preserves_dtype(self, fv_mesh: FvMesh):
        """Output dtype should match input dtype."""
        scheme = LinearInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert face_vals.dtype == torch.float64


# ---------------------------------------------------------------------------
# Upwind interpolation tests
# ---------------------------------------------------------------------------


class TestUpwindInterpolation:
    """Tests for UpwindInterpolation scheme."""

    def test_positive_flux_uses_owner(self, fv_mesh: FvMesh):
        """Positive flux should select owner cell value."""
        scheme = UpwindInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)  # all positive
        face_vals = scheme.interpolate(phi, face_flux=flux)

        # Internal face: owner is cell 0, flux >= 0 → use cell 0 value
        assert face_vals[0] == 10.0

    def test_negative_flux_uses_neighbour(self, fv_mesh: FvMesh):
        """Negative flux should select neighbour cell value."""
        scheme = UpwindInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = -torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=flux)

        # Internal face: neighbour is cell 1, flux < 0 → use cell 1 value
        assert face_vals[0] == 20.0

    def test_boundary_faces_always_owner(self, fv_mesh: FvMesh):
        """Boundary faces should always use owner value regardless of flux."""
        scheme = UpwindInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = -torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=flux)

        # Boundary of cell 0
        for i in range(1, 6):
            assert face_vals[i] == 10.0, f"Face {i}: expected owner value"
        # Boundary of cell 1
        for i in range(6, 11):
            assert face_vals[i] == 20.0, f"Face {i}: expected owner value"

    def test_no_flux_defaults_to_owner(self, fv_mesh: FvMesh):
        """Without flux, should use owner values."""
        scheme = UpwindInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=None)

        # All faces should be owner values
        assert face_vals[0] == 10.0  # owner is cell 0
        for i in range(1, 6):
            assert face_vals[i] == 10.0
        for i in range(6, 11):
            assert face_vals[i] == 20.0

    def test_output_shape(self, fv_mesh: FvMesh):
        """Output should have shape (n_faces,)."""
        scheme = UpwindInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=flux)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_invalid_dimension_raises(self, fv_mesh: FvMesh):
        """Non-1D input should raise ValueError."""
        scheme = UpwindInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)


# ---------------------------------------------------------------------------
# Linear upwind interpolation tests
# ---------------------------------------------------------------------------


class TestLinearUpwindInterpolation:
    """Tests for LinearUpwindInterpolation scheme."""

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values (zero gradient)."""
        scheme = LinearUpwindInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 7.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=flux)

        # Constant field → zero gradient → face values = cell values
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 7.0, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        """Should raise ValueError when face_flux is None."""
        scheme = LinearUpwindInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_output_shape(self, fv_mesh: FvMesh):
        """Output should have shape (n_faces,)."""
        scheme = LinearUpwindInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=flux)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_invalid_dimension_raises(self, fv_mesh: FvMesh):
        """Non-1D input should raise ValueError."""
        scheme = LinearUpwindInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, face_flux=flux)

    def test_boundary_faces_use_owner_reconstruction(self, fv_mesh: FvMesh):
        """Boundary faces should use owner-based reconstruction."""
        scheme = LinearUpwindInterpolation(fv_mesh)
        # Linear field in z: φ = z
        phi = torch.tensor([0.5, 1.5], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=flux)

        # Boundary faces should be finite
        assert torch.isfinite(face_vals).all()

    def test_output_is_finite(self, fv_mesh: FvMesh):
        """All output values should be finite."""
        scheme = LinearUpwindInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.tensor(
            [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6, -0.3, 0.1, 0.7, -0.2],
            dtype=torch.float64,
        )
        face_vals = scheme.interpolate(phi, face_flux=flux)
        assert torch.isfinite(face_vals).all()


# ---------------------------------------------------------------------------
# QUICK interpolation tests
# ---------------------------------------------------------------------------


class TestQuickInterpolation:
    """Tests for QuickInterpolation scheme."""

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values."""
        scheme = QuickInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 3.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=flux)

        # Constant field → all face values should be 3.0
        # (QUICK correction terms cancel for constant field)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 3.0, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        """Should raise ValueError when face_flux is None."""
        scheme = QuickInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_output_shape(self, fv_mesh: FvMesh):
        """Output should have shape (n_faces,)."""
        scheme = QuickInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=flux)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_invalid_dimension_raises(self, fv_mesh: FvMesh):
        """Non-1D input should raise ValueError."""
        scheme = QuickInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, face_flux=flux)

    def test_fallback_to_linear_when_no_upstream(self, fv_mesh: FvMesh):
        """With only 2 cells, the QUICK stencil cannot find upstream-of-upstream
        for the internal face, so it should fall back to linear interpolation."""
        scheme = QuickInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=flux)

        # Should equal linear interpolation result for this case
        linear_scheme = LinearInterpolation(fv_mesh)
        linear_face = linear_scheme.interpolate(phi)
        torch.testing.assert_close(face_vals, linear_face)

    def test_boundary_faces_are_linear(self, fv_mesh: FvMesh):
        """Boundary faces should use linear interpolation (no QUICK for boundaries)."""
        scheme = QuickInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=flux)

        # Boundary faces of cell 0 should be 10.0
        for i in range(1, 6):
            torch.testing.assert_close(
                face_vals[i],
                torch.tensor(10.0, dtype=torch.float64),
                msg=f"Face {i}: expected 10.0 (boundary of cell 0)",
            )

        # Boundary faces of cell 1 should be 20.0
        for i in range(6, 11):
            torch.testing.assert_close(
                face_vals[i],
                torch.tensor(20.0, dtype=torch.float64),
                msg=f"Face {i}: expected 20.0 (boundary of cell 1)",
            )

    def test_output_is_finite(self, fv_mesh: FvMesh):
        """All output values should be finite."""
        scheme = QuickInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.tensor(
            [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6, -0.3, 0.1, 0.7, -0.2],
            dtype=torch.float64,
        )
        face_vals = scheme.interpolate(phi, face_flux=flux)
        assert torch.isfinite(face_vals).all()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestSchemeIntegration:
    """Integration tests verifying schemes work together."""

    def test_all_schemes_same_for_constant_field(self, fv_mesh: FvMesh):
        """All schemes should produce the same result for a constant field."""
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        linear = LinearInterpolation(fv_mesh)(phi)
        upwind = UpwindInterpolation(fv_mesh)(phi, face_flux=flux)
        lu = LinearUpwindInterpolation(fv_mesh)(phi, face_flux=flux)
        quick = QuickInterpolation(fv_mesh)(phi, face_flux=flux)

        expected = torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64)
        torch.testing.assert_close(linear, expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(upwind, expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(lu, expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(quick, expected, atol=1e-10, rtol=1e-10)

    def test_schemes_produce_different_results(self, fv_mesh: FvMesh):
        """Different schemes should generally produce different results
        for a non-constant field."""
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        linear = LinearInterpolation(fv_mesh)(phi)[0]
        upwind = UpwindInterpolation(fv_mesh)(phi, face_flux=flux)[0]

        # Linear: 15.0, Upwind: 10.0 (positive flux → owner)
        assert linear != upwind

    def test_module_level_imports(self):
        """All scheme classes should be importable from the package."""
        from pyfoam.discretisation import (
            InterpolationScheme,
            LinearInterpolation,
            UpwindInterpolation,
            LinearUpwindInterpolation,
            QuickInterpolation,
            compute_centre_weights,
            compute_upwind_weights,
        )
        # Verify they are the correct types
        assert issubclass(LinearInterpolation, InterpolationScheme)
        assert issubclass(UpwindInterpolation, InterpolationScheme)
        assert issubclass(LinearUpwindInterpolation, InterpolationScheme)
        assert issubclass(QuickInterpolation, InterpolationScheme)

    def test_repr(self, fv_mesh: FvMesh):
        """Scheme repr should include class name."""
        scheme = LinearInterpolation(fv_mesh)
        assert "LinearInterpolation" in repr(scheme)

    def test_mesh_property(self, fv_mesh: FvMesh):
        """Scheme should expose the mesh."""
        scheme = LinearInterpolation(fv_mesh)
        assert scheme.mesh is fv_mesh
