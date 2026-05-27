"""Tests for MappedWallBC (mapped wall CHT boundary condition)."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition, MappedWallBC
from pyfoam.boundary.boundary_condition import Patch


@pytest.fixture
def cht_patch() -> Patch:
    """A 3-face patch for CHT interface testing."""
    return Patch(
        name="fluidSide",
        face_indices=torch.tensor([0, 1, 2]),
        face_normals=torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2]),
    )


class TestMappedWallBC:
    """Tests for the mappedWall boundary condition."""

    def test_registration(self):
        """mappedWall is registered in RTS."""
        assert "mappedWall" in BoundaryCondition.available_types()

    def test_factory_creation(self, cht_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "mappedWall",
            cht_patch,
            {"neighbourRegion": "solid", "kappa": 50.0},
        )
        assert isinstance(bc, MappedWallBC)

    def test_properties(self, cht_patch):
        """Properties are parsed correctly."""
        bc = MappedWallBC(
            cht_patch,
            {
                "neighbourRegion": "solid",
                "neighbourPatch": "fluidSide",
                "kappa": 50.0,
            },
        )
        assert bc.neighbour_region == "solid"
        assert bc.neighbour_patch == "fluidSide"
        assert bc.kappa == 50.0

    def test_default_values(self, cht_patch):
        """Default values are correct."""
        bc = MappedWallBC(cht_patch)
        assert bc.neighbour_region == ""
        assert bc.neighbour_patch == ""
        assert bc.kappa == 1.0

    def test_kappa_setter(self, cht_patch):
        """Kappa can be updated."""
        bc = MappedWallBC(cht_patch)
        bc.kappa = 100.0
        assert bc.kappa == 100.0

    def test_apply_with_coupled_field(self, cht_patch):
        """apply() sets boundary face values from coupled region."""
        bc = MappedWallBC(cht_patch, {"kappa": 1.0})

        # Coupled region has 5 cells; owners of coupled faces
        coupled_field = torch.tensor([300.0, 310.0, 320.0, 330.0, 340.0], dtype=torch.float64)
        coupled_owners = torch.tensor([0, 1, 2, 3, 4])
        # Map: face 0 -> coupled face 2, face 1 -> coupled face 3, face 2 -> coupled face 4
        coupled_face_map = torch.tensor([2, 3, 4])

        bc.set_coupled_field(coupled_field, coupled_owners, coupled_face_map)

        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)

        # Should read temperatures from coupled cells 2, 3, 4
        assert torch.allclose(field[0:3], torch.tensor([320.0, 330.0, 340.0], dtype=torch.float64))

    def test_apply_without_coupled_field_fallback(self, cht_patch):
        """apply() falls back to zero-gradient when no coupled field is set."""
        bc = MappedWallBC(cht_patch)
        field = torch.zeros(10, dtype=torch.float64)
        field[0:3] = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        bc.apply(field)
        # Should copy owner cell values (zero-gradient)
        assert torch.allclose(field[0:3], torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, cht_patch):
        """apply() with explicit patch_idx works correctly."""
        bc = MappedWallBC(cht_patch)
        coupled_field = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
        coupled_owners = torch.tensor([0, 1, 2, 3])
        coupled_face_map = torch.tensor([0, 1, 2])

        bc.set_coupled_field(coupled_field, coupled_owners, coupled_face_map)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        # Should write at indices 5, 6, 7
        assert torch.allclose(field[5:8], torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))

    def test_matrix_contributions_with_coupled(self, cht_patch):
        """Matrix contributions use kappa-weighted penalty method."""
        bc = MappedWallBC(cht_patch, {"kappa": 50.0})

        coupled_field = torch.tensor([300.0, 310.0, 320.0, 330.0], dtype=torch.float64)
        coupled_owners = torch.tensor([0, 1, 2, 3])
        coupled_face_map = torch.tensor([0, 1, 2])

        bc.set_coupled_field(coupled_field, coupled_owners, coupled_face_map)

        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)

        # Owner cells 0, 1, 2 should have contributions
        assert diag[0] > 0
        assert diag[1] > 0
        assert diag[2] > 0

        # Source should contain kappa * delta * area * T_coupled
        # coeff = 50 * 2.0 * 1.0 = 100 per face
        expected_diag = 100.0
        assert torch.allclose(diag[0:3], torch.full((3,), expected_diag, dtype=torch.float64))
        # source = 100 * T_coupled
        assert torch.allclose(source[0], torch.tensor(100.0 * 300.0, dtype=torch.float64))
        assert torch.allclose(source[1], torch.tensor(100.0 * 310.0, dtype=torch.float64))
        assert torch.allclose(source[2], torch.tensor(100.0 * 320.0, dtype=torch.float64))

    def test_matrix_contributions_without_coupled(self, cht_patch):
        """Matrix contributions are zero-flux when no coupled field set."""
        bc = MappedWallBC(cht_patch, {"kappa": 50.0})
        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)

        # Diag still gets contributions (kappa * delta * area)
        assert diag[0] > 0
        # Source is zero (no coupled data)
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_accumulate_matrix_contributions(self, cht_patch):
        """Matrix contributions accumulate into pre-existing tensors."""
        bc = MappedWallBC(cht_patch, {"kappa": 10.0})
        coupled_field = torch.tensor([500.0, 500.0, 500.0, 500.0], dtype=torch.float64)
        coupled_owners = torch.tensor([0, 1, 2, 3])
        coupled_face_map = torch.tensor([0, 1, 2])
        bc.set_coupled_field(coupled_field, coupled_owners, coupled_face_map)

        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10

        # Pre-existing values
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64) * 5.0

        diag_out, source_out = bc.matrix_contributions(field, n_cells, diag=diag, source=source)

        # Should accumulate, not overwrite
        assert diag_out[0] > 1.0
        assert source_out[0] > 5.0
