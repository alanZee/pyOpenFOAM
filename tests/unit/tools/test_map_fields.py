"""Tests for map_fields — field mapping between meshes."""

import numpy as np
import pytest
import torch

from pyfoam.tools.map_fields import map_fields, map_fields_from_case


def _make_offset_mesh(offset=(2.0, 0.0, 0.0)):
    """Create a 4x4x1 mesh shifted by *offset* from the origin."""
    from tests.unit.tools.conftest import make_4x4_hex_mesh

    mesh = make_4x4_hex_mesh()
    shift = torch.tensor(offset, dtype=mesh.points.dtype)
    mesh._points = mesh._points + shift.unsqueeze(0)
    # Invalidate cached geometry so it is recomputed with new positions
    mesh._face_centres = None
    mesh._face_area_vectors = None
    mesh._cell_centres = None
    mesh._cell_volumes = None
    mesh._face_weights = None
    mesh._delta_coefficients = None
    mesh.compute_geometry()
    return mesh


class TestMapFields:
    """Test the map_fields function."""

    def test_scalar_field_shape(self, large_mesh):
        """Mapped scalar field should have shape (n_target_cells,)."""
        n_src = large_mesh.n_cells
        src_field = np.ones(n_src) * 42.0

        # Target mesh identical to source (same mesh object)
        result = map_fields(large_mesh, large_mesh, {"p": src_field})

        assert result["p"].shape == (n_src,)
        np.testing.assert_allclose(result["p"], 42.0)

    def test_vector_field_shape(self, large_mesh):
        """Mapped vector field should have shape (n_target_cells, 3)."""
        n_src = large_mesh.n_cells
        src_field = np.zeros((n_src, 3))
        src_field[:, 0] = 1.0

        result = map_fields(large_mesh, large_mesh, {"U": src_field})

        assert result["U"].shape == (n_src, 3)
        np.testing.assert_allclose(result["U"][:, 0], 1.0)

    def test_identity_mapping_preserves_values(self, large_mesh):
        """Mapping to the same mesh should preserve all values."""
        n_src = large_mesh.n_cells
        pressure = np.arange(n_src, dtype=np.float64)

        result = map_fields(large_mesh, large_mesh, {"p": pressure})
        np.testing.assert_allclose(result["p"], pressure)

    def test_nearest_neighbour_selects_closest(self, large_mesh):
        """Nearest-neighbour should pick the closest source cell."""
        n_src = large_mesh.n_cells
        # Give each source cell a unique value
        src_data = np.arange(n_src, dtype=np.float64)
        src_fields = {"label": src_data}

        # Target is same mesh → every cell maps to itself
        result = map_fields(large_mesh, large_mesh, src_fields)
        np.testing.assert_allclose(result["label"], src_data)

    def test_multiple_fields(self, large_mesh):
        """Multiple fields should all be mapped."""
        n_src = large_mesh.n_cells
        result = map_fields(
            large_mesh,
            large_mesh,
            {
                "p": np.ones(n_src),
                "U": np.zeros((n_src, 3)),
                "k": np.full(n_src, 0.5),
            },
        )

        assert set(result.keys()) == {"p", "U", "k"}
        np.testing.assert_allclose(result["p"], 1.0)
        np.testing.assert_allclose(result["k"], 0.5)

    def test_offset_mesh_nearest_neighbour(self, large_mesh):
        """Fields should map correctly between overlapping meshes."""
        offset_mesh = _make_offset_mesh(offset=(0.3, 0.2, 0.0))
        n_src = large_mesh.n_cells
        src_data = np.arange(n_src, dtype=np.float64)

        result = map_fields(large_mesh, offset_mesh, {"label": src_data})

        # Result should have n_target_cells elements
        assert result["label"].shape[0] == offset_mesh.n_cells
        # All values should be valid indices from the source
        assert np.all(result["label"] >= 0)
        assert np.all(result["label"] < n_src)

    def test_wrong_field_size_raises(self, large_mesh):
        """Field with wrong number of values should raise ValueError."""
        wrong_size = np.ones(large_mesh.n_cells + 5)

        with pytest.raises(ValueError, match="unsupported shape|values but"):
            map_fields(large_mesh, large_mesh, {"p": wrong_size})

    def test_unsupported_method_raises(self, large_mesh):
        """Unknown interpolation method should raise ValueError."""
        n_src = large_mesh.n_cells
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            map_fields(large_mesh, large_mesh, {"p": np.ones(n_src)}, method="linear")

    def test_2d_field_wrong_columns_raises(self, large_mesh):
        """2D field with wrong column count should raise ValueError."""
        n_src = large_mesh.n_cells
        bad_field = np.zeros((n_src, 4))  # 4 components, not 3

        with pytest.raises(ValueError, match="unsupported shape"):
            map_fields(large_mesh, large_mesh, {"bad": bad_field})

    def test_map_fields_from_case_with_meshes(self, large_mesh):
        """map_fields_from_case should work when FvMesh objects are passed."""
        n_src = large_mesh.n_cells
        src_data = np.ones(n_src) * 7.0

        result = map_fields_from_case(
            large_mesh, large_mesh, {"p": src_data}
        )
        np.testing.assert_allclose(result["p"], 7.0)

    def test_map_fields_from_case_with_string_raises(self, large_mesh):
        """map_fields_from_case should raise FileNotFoundError for non-existent paths."""
        with pytest.raises(FileNotFoundError):
            map_fields_from_case("/fake/path", large_mesh, {"p": np.ones(1)})
