"""Tests for processor boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.boundary_condition import Patch
from pyfoam.boundary.processor import ProcessorBC


@pytest.fixture
def proc_patch():
    """A 3-face processor patch for testing."""
    return Patch(
        name="procPatch0to1",
        face_indices=torch.tensor([10, 11, 12]),
        face_normals=torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2]),
    )


@pytest.fixture
def proc_pair():
    """Two coupled processor patches."""
    patch_a = Patch(
        name="proc0to1",
        face_indices=torch.tensor([20, 21]),
        face_normals=torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1]),
    )
    patch_b = Patch(
        name="proc1to0",
        face_indices=torch.tensor([22, 23]),
        face_normals=torch.tensor([
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([2, 3]),
    )
    return patch_a, patch_b


class TestProcessorBC:
    """Test the processor boundary condition."""

    def test_registration(self):
        """processor is registered in the RTS registry."""
        assert "processor" in BoundaryCondition.available_types()

    def test_factory_creation(self, proc_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("processor", proc_patch)
        assert isinstance(bc, ProcessorBC)

    def test_default_proc_numbers(self, proc_patch):
        """Default processor numbers are 0 and 1."""
        bc = ProcessorBC(proc_patch)
        assert bc._my_proc == 0
        assert bc._neighbour_proc == 1

    def test_custom_proc_numbers(self, proc_patch):
        """Custom processor numbers from coefficients."""
        bc = ProcessorBC(proc_patch, coeffs={"myProcNo": 3, "neighbProcNo": 7})
        assert bc._my_proc == 3
        assert bc._neighbour_proc == 7

    def test_apply_with_neighbour_field(self, proc_patch):
        """apply() copies neighbour field values when available."""
        bc = ProcessorBC(proc_patch)
        neighbour_vals = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        bc.set_neighbour_field(neighbour_vals)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert torch.allclose(field[10], torch.tensor(100.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(200.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(300.0, dtype=torch.float64))

    def test_apply_without_neighbour_field_fallback(self, proc_patch):
        """apply() copies from owner cells when no neighbour data."""
        bc = ProcessorBC(proc_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 10.0
        field[2] = 15.0
        bc.apply(field)

        # Should fallback to owner values
        assert torch.allclose(field[10], torch.tensor(5.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(15.0, dtype=torch.float64))

    def test_apply_vector_field_with_neighbour(self, proc_patch):
        """apply() works for vector fields with neighbour data."""
        bc = ProcessorBC(proc_patch)
        neighbour_vals = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=torch.float64)
        bc.set_neighbour_field(neighbour_vals)

        field = torch.zeros(15, 3, dtype=torch.float64)
        bc.apply(field)

        assert torch.allclose(field[10], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))

    def test_prepare_send_buffer(self, proc_patch):
        """prepare_send_buffer extracts boundary face values."""
        bc = ProcessorBC(proc_patch)
        field = torch.arange(15, dtype=torch.float64)
        send_buf = bc.prepare_send_buffer(field)

        assert send_buf.shape == (3,)
        assert torch.allclose(send_buf, torch.tensor([10.0, 11.0, 12.0], dtype=torch.float64))

    def test_receive_buffer(self, proc_patch):
        """receive_buffer stores neighbour data."""
        bc = ProcessorBC(proc_patch)
        buf = torch.tensor([50.0, 60.0, 70.0], dtype=torch.float64)
        bc.receive_buffer(buf)

        assert bc._neighbour_field is not None
        assert torch.allclose(bc._neighbour_field, buf)

    def test_matrix_contributions_with_neighbour(self, proc_patch):
        """Matrix contributions include neighbour penalty terms."""
        bc = ProcessorBC(proc_patch)
        neighbour_vals = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.set_neighbour_field(neighbour_vals)

        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 5
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Diagonal should have contributions from processor faces
        assert diag.abs().sum() > 0

    def test_matrix_contributions_without_neighbour(self, proc_patch):
        """Matrix contributions use owner values when no neighbour data."""
        bc = ProcessorBC(proc_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 10.0
        field[2] = 15.0
        n_cells = 5
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.abs().sum() > 0
        # Source should use owner values as fallback
        assert source.abs().sum() > 0

    def test_type_name(self, proc_patch):
        """type_name returns 'processor'."""
        bc = ProcessorBC(proc_patch)
        assert bc.type_name == "processor"

    def test_preserves_non_boundary_values(self, proc_patch):
        """apply() does not modify non-boundary cells."""
        bc = ProcessorBC(proc_patch)
        neighbour_vals = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        bc.set_neighbour_field(neighbour_vals)

        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)

        # Non-boundary values should be unchanged
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_proc_pair_data_exchange(self, proc_pair):
        """Two coupled processor patches can exchange data."""
        patch_a, patch_b = proc_pair
        bc_a = ProcessorBC(patch_a, coeffs={"myProcNo": 0, "neighbProcNo": 1})
        bc_b = ProcessorBC(patch_b, coeffs={"myProcNo": 1, "neighbProcNo": 0})

        # Processor A sends values from its boundary faces [20, 21]
        field = torch.zeros(25, dtype=torch.float64)
        field[20] = 1.0
        field[21] = 2.0

        send_buf = bc_a.prepare_send_buffer(field)
        bc_b.receive_buffer(send_buf)

        # Apply B's BC -- should write to face_indices [22, 23]
        field_b = torch.zeros(25, dtype=torch.float64)
        bc_b.apply(field_b)

        assert torch.allclose(field_b[22], torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(field_b[23], torch.tensor(2.0, dtype=torch.float64))
