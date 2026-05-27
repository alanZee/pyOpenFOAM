"""Tests for cyclicAMI boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.boundary_condition import Patch
from pyfoam.boundary.cyclic_ami import CyclicAMI


@pytest.fixture
def ami_pair():
    """Two non-conformal AMI patches with 3 and 2 faces respectively.

    Patch A: 3 faces, owner cells [0, 1, 2]
    Patch B: 2 faces, owner cells [3, 4]
    """
    patch_a = Patch(
        name="cyclicAMI_half1",
        face_indices=torch.tensor([20, 21, 22]),
        face_normals=torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2]),
        neighbour_patch="cyclicAMI_half2",
    )
    patch_b = Patch(
        name="cyclicAMI_half2",
        face_indices=torch.tensor([30, 31]),
        face_normals=torch.tensor([
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.5, 1.5], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([3, 4]),
        neighbour_patch="cyclicAMI_half1",
    )
    return patch_a, patch_b


@pytest.fixture
def ami_weights():
    """Weight matrix: (3, 2) mapping patch B -> patch A.

    Face 0 on patch A: 70% from B-face-0, 30% from B-face-1
    Face 1 on patch A: 50% from each
    Face 2 on patch A: 20% from B-face-0, 80% from B-face-1
    """
    return torch.tensor([
        [0.7, 0.3],
        [0.5, 0.5],
        [0.2, 0.8],
    ], dtype=torch.float64)


class TestCyclicAMI:
    """Test the cyclicAMI boundary condition."""

    def test_registration(self):
        """cyclicAMI is registered in the RTS registry."""
        assert "cyclicAMI" in BoundaryCondition.available_types()

    def test_factory_creation(self, ami_pair):
        """BC can be created via the factory method."""
        patch_a, _ = ami_pair
        bc = BoundaryCondition.create("cyclicAMI", patch_a)
        assert isinstance(bc, CyclicAMI)

    def test_neighbour_patch_name(self, ami_pair):
        """neighbour_patch_name is read from patch or coeffs."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        assert bc.neighbour_patch_name == "cyclicAMI_half2"

    def test_neighbour_patch_name_from_coeffs(self, ami_pair):
        """neighbour_patch_name can be overridden via coefficients."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a, {"neighbourPatch": "otherPatch"})
        assert bc.neighbour_patch_name == "otherPatch"

    def test_transform_default(self, ami_pair):
        """Default transform is 'noOrdering'."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        assert bc.transform == "noOrdering"

    def test_transform_from_coeffs(self, ami_pair):
        """Transform can be set via coefficients."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a, {"transform": "defaultBehaviour"})
        assert bc.transform == "defaultBehaviour"

    def test_apply_with_ami_weights(self, ami_pair, ami_weights):
        """apply() uses AMI weights to interpolate neighbour values."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        bc.set_ami_weights(ami_weights)
        neighbour_vals = torch.tensor([100.0, 200.0], dtype=torch.float64)
        bc.set_neighbour_field(neighbour_vals)

        field = torch.zeros(25, dtype=torch.float64)
        bc.apply(field)

        # face 0: 0.7*100 + 0.3*200 = 130
        # face 1: 0.5*100 + 0.5*200 = 150
        # face 2: 0.2*100 + 0.8*200 = 180
        assert torch.allclose(field[20], torch.tensor(130.0, dtype=torch.float64))
        assert torch.allclose(field[21], torch.tensor(150.0, dtype=torch.float64))
        assert torch.allclose(field[22], torch.tensor(180.0, dtype=torch.float64))

    def test_apply_without_weights_falls_back_to_owner(self, ami_pair):
        """Without AMI weights, falls back to owner-cell values (zero gradient)."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        bc.set_neighbour_field(torch.tensor([100.0, 200.0], dtype=torch.float64))

        field = torch.zeros(25, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        # Without weights, uses owner cell values
        assert torch.allclose(field[20], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[21], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[22], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_without_neighbour_falls_back_to_owner(self, ami_pair, ami_weights):
        """Without neighbour data, falls back to owner-cell values."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        bc.set_ami_weights(ami_weights)
        # No neighbour field set

        field = torch.zeros(25, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        assert torch.allclose(field[20], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[21], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[22], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, ami_pair, ami_weights):
        """apply() with explicit patch_idx."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        bc.set_ami_weights(ami_weights)
        bc.set_neighbour_field(torch.tensor([50.0, 100.0], dtype=torch.float64))

        field = torch.zeros(35, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        # face 0 at [5]: 0.7*50 + 0.3*100 = 65
        assert torch.allclose(field[5], torch.tensor(65.0, dtype=torch.float64))

    def test_matrix_contributions_with_ami(self, ami_pair, ami_weights):
        """Matrix contributions use AMI-interpolated values."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        bc.set_ami_weights(ami_weights)
        bc.set_neighbour_field(torch.tensor([100.0, 200.0], dtype=torch.float64))

        field = torch.zeros(25, dtype=torch.float64)
        n_cells = 5
        diag, source = bc.matrix_contributions(field, n_cells)

        # coeff = delta * area = 2.0 * 1.0 = 2.0 for each face
        expected_diag = torch.tensor([2.0, 2.0, 2.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(diag, expected_diag)

        # source[c] += coeff * interpolated_value
        # cell 0: 2.0 * 130 = 260
        # cell 1: 2.0 * 150 = 300
        # cell 2: 2.0 * 180 = 360
        expected_source = torch.tensor([260.0, 300.0, 360.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(source, expected_source)

    def test_matrix_contributions_without_ami_data(self, ami_pair):
        """Without AMI data, source contribution is zero."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)

        field = torch.zeros(25, dtype=torch.float64)
        n_cells = 5
        diag, source = bc.matrix_contributions(field, n_cells)

        expected_diag = torch.tensor([2.0, 2.0, 2.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(diag, expected_diag)
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, ami_pair, ami_weights):
        """Matrix contributions accumulate into pre-existing tensors."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        bc.set_ami_weights(ami_weights)
        bc.set_neighbour_field(torch.tensor([100.0, 200.0], dtype=torch.float64))

        field = torch.zeros(25, dtype=torch.float64)
        n_cells = 5
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)

        # diag[0] = 1.0 + 2.0 = 3.0
        assert torch.allclose(diag[0], torch.tensor(3.0, dtype=torch.float64))
        # source[0] = 1.0 + 260.0 = 261.0
        assert torch.allclose(source[0], torch.tensor(261.0, dtype=torch.float64))

    def test_identity_weights_reduces_to_cyclic(self, ami_pair):
        """With identity weights and conformal face count, behaves like cyclic."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        # Use 2x2 identity to make it conformal (matching face count)
        # But patch_a has 3 faces, patch_b has 2 faces, so we use a
        # degenerate 3x2 identity-like matrix
        weights = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ], dtype=torch.float64)
        bc.set_ami_weights(weights)
        bc.set_neighbour_field(torch.tensor([10.0, 20.0], dtype=torch.float64))

        field = torch.zeros(25, dtype=torch.float64)
        bc.apply(field)

        assert torch.allclose(field[20], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[21], torch.tensor(20.0, dtype=torch.float64))
        # face 2: 0.5*10 + 0.5*20 = 15
        assert torch.allclose(field[22], torch.tensor(15.0, dtype=torch.float64))

    def test_repr(self, ami_pair):
        """repr shows class name and patch info."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        r = repr(bc)
        assert "CyclicAMI" in r
        assert "cyclicAMI_half1" in r

    def test_type_name(self, ami_pair):
        """type_name returns the registered name."""
        patch_a, _ = ami_pair
        bc = CyclicAMI(patch_a)
        assert bc.type_name == "cyclicAMI"
