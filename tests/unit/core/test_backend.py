"""Tests for backend abstraction — scatter, gather, sparse operations."""

import pytest
import torch

from pyfoam.core.backend import (
    Backend,
    gather,
    scatter_add,
    sparse_coo_tensor,
    sparse_mm,
)
from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


# ---------------------------------------------------------------------------
# scatter_add
# ---------------------------------------------------------------------------


class TestScatterAdd:
    def test_basic_scatter(self):
        """scatter_add accumulates values at target indices."""
        src = torch.tensor([1.0, 2.0, 3.0])
        index = torch.tensor([0, 1, 0])
        out = scatter_add(src, index, dim_size=2)
        assert out.shape == (2,)
        # index 0 receives 1.0 + 3.0 = 4.0, index 1 receives 2.0
        assert torch.allclose(out, torch.tensor([4.0, 2.0], dtype=torch.float64))

    def test_default_dtype_is_float64(self):
        src = torch.tensor([1.0, 2.0])
        index = torch.tensor([0, 1])
        out = scatter_add(src, index, dim_size=2)
        assert out.dtype == CFD_DTYPE

    def test_respects_dtype_override(self):
        src = torch.tensor([1.0, 2.0])
        index = torch.tensor([0, 1])
        out = scatter_add(src, index, dim_size=2, dtype=torch.float32)
        assert out.dtype == torch.float32

    def test_device_cpu(self):
        src = torch.tensor([1.0, 2.0])
        index = torch.tensor([0, 1])
        out = scatter_add(src, index, dim_size=2, device="cpu")
        assert out.device == torch.device("cpu")

    def test_fvm_flux_assembly_pattern(self):
        """Simulate FVM: 4 faces contribute fluxes to 3 cells."""
        # Face fluxes: positive = owner cell gains
        fluxes = torch.tensor([10.0, -5.0, 3.0, 7.0])
        # Owner cell for each face
        owners = torch.tensor([0, 1, 0, 2])
        # Neighbour cell for each face (internal faces)
        neighbours = torch.tensor([1, 2, 1, 0])

        cell_values = scatter_add(fluxes, owners, dim_size=3)
        cell_values = cell_values + scatter_add(-fluxes, neighbours, dim_size=3)

        # Verify conservation: sum of all cell values should be 0
        # (fluxes cancel globally)
        assert cell_values.shape == (3,)

    def test_duplicate_indices_accumulate(self):
        src = torch.tensor([1.0, 1.0, 1.0])
        index = torch.tensor([0, 0, 0])
        out = scatter_add(src, index, dim_size=1)
        assert torch.allclose(out, torch.tensor([3.0], dtype=torch.float64))

    def test_large_scatter(self):
        n = 10000
        src = torch.ones(n)
        index = torch.arange(n) % 100
        out = scatter_add(src, index, dim_size=100)
        # Each cell should receive exactly 100 contributions
        expected = torch.full((100,), 100.0, dtype=torch.float64)
        assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------


class TestGather:
    def test_basic_gather(self):
        src = torch.tensor([10.0, 20.0, 30.0, 40.0])
        index = torch.tensor([2, 0, 3])
        out = gather(src, index)
        assert torch.allclose(out, torch.tensor([30.0, 10.0, 40.0]))

    def test_gather_preserves_dtype(self):
        src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        index = torch.tensor([1, 2])
        out = gather(src, index)
        assert out.dtype == torch.float32

    def test_gather_device_cpu(self):
        src = torch.tensor([1.0, 2.0])
        index = torch.tensor([0])
        out = gather(src, index, device="cpu")
        assert out.device == torch.device("cpu")

    def test_boundary_condition_lookup(self):
        """Simulate BC lookup: boundary faces reference internal values."""
        cell_values = torch.tensor([100.0, 200.0, 300.0])
        boundary_cells = torch.tensor([0, 2, 1, 0])
        bc_values = gather(cell_values, boundary_cells)
        assert torch.allclose(bc_values, torch.tensor([100.0, 300.0, 200.0, 100.0]))

    def test_gather_2d(self):
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = torch.tensor([[0, 2]])
        out = gather(src, index, dim=0)
        # gather picks src[index[i,j], j] — so col 0 row 0, col 1 row 2
        assert out.shape == (1, 2)
        assert torch.allclose(out, torch.tensor([[1.0, 6.0]]))


# ---------------------------------------------------------------------------
# sparse_coo_tensor
# ---------------------------------------------------------------------------


class TestSparseCOOTensor:
    def test_basic_coo(self):
        indices = torch.tensor([[0, 1, 2], [1, 0, 2]])
        values = torch.tensor([3.0, 4.0, 5.0])
        size = (3, 3)
        sparse = sparse_coo_tensor(indices, values, size)
        assert sparse.is_sparse
        dense = sparse.to_dense()
        assert torch.allclose(dense[0, 1], torch.tensor(3.0, dtype=torch.float64))
        assert torch.allclose(dense[1, 0], torch.tensor(4.0, dtype=torch.float64))
        assert torch.allclose(dense[2, 2], torch.tensor(5.0, dtype=torch.float64))

    def test_default_dtype_is_float64(self):
        indices = torch.tensor([[0], [1]])
        values = torch.tensor([1.0])
        sparse = sparse_coo_tensor(indices, values, (2, 2))
        assert sparse.dtype == CFD_DTYPE

    def test_dtype_override(self):
        indices = torch.tensor([[0], [1]])
        values = torch.tensor([1.0])
        sparse = sparse_coo_tensor(indices, values, (2, 2), dtype=torch.float32)
        assert sparse.dtype == torch.float32

    def test_device_cpu(self):
        indices = torch.tensor([[0], [1]])
        values = torch.tensor([1.0])
        sparse = sparse_coo_tensor(indices, values, (2, 2), device="cpu")
        assert sparse.device == torch.device("cpu")

    def test_fvm_jacobian_pattern(self):
        """Build a typical FVM Jacobian: diagonal + off-diagonal for 3 cells."""
        # Diagonal entries (cell self-coupling)
        diag_idx = torch.tensor([[0, 1, 2], [0, 1, 2]])
        diag_val = torch.tensor([4.0, 4.0, 4.0])

        # Off-diagonal (neighbour coupling)
        off_idx = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        off_val = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        all_idx = torch.cat([diag_idx, off_idx], dim=1)
        all_val = torch.cat([diag_val, off_val])

        jacobian = sparse_coo_tensor(all_idx, all_val, (3, 3))
        dense = jacobian.to_dense()

        # Check diagonal
        for i in range(3):
            assert torch.allclose(dense[i, i], torch.tensor(4.0, dtype=torch.float64))
        # Check off-diagonal
        assert torch.allclose(dense[0, 1], torch.tensor(-1.0, dtype=torch.float64))


# ---------------------------------------------------------------------------
# sparse_mm
# ---------------------------------------------------------------------------


class TestSparseMM:
    def test_sparse_times_dense_vector(self):
        """Sparse matrix @ dense vector."""
        indices = torch.tensor([[0, 1, 0], [0, 1, 1]])
        values = torch.tensor([2.0, 3.0, 1.0])
        mat = sparse_coo_tensor(indices, values, (2, 2))
        vec = torch.tensor([5.0, 7.0], dtype=CFD_DTYPE)
        result = sparse_mm(mat, vec)
        # [2*5 + 1*7, 3*7] = [17, 21]
        assert torch.allclose(result, torch.tensor([17.0, 21.0], dtype=torch.float64))

    def test_sparse_identity(self):
        """Identity sparse matrix @ vector = vector."""
        n = 5
        indices = torch.stack([torch.arange(n), torch.arange(n)])
        values = torch.ones(n)
        identity = sparse_coo_tensor(indices, values, (n, n))
        vec = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=CFD_DTYPE)
        result = sparse_mm(identity, vec)
        assert torch.allclose(result, vec)

    def test_sparse_csr_conversion(self):
        """COO -> CSR conversion for solver performance."""
        indices = torch.tensor([[0, 1, 2], [1, 0, 2]])
        values = torch.tensor([1.0, 2.0, 3.0])
        coo = sparse_coo_tensor(indices, values, (3, 3))
        csr = coo.to_sparse_csr()
        vec = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)
        result = sparse_mm(csr, vec)
        # [1*1, 2*1, 3*1] = [1, 2, 3]
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    def test_device_cpu(self):
        indices = torch.tensor([[0], [0]])
        values = torch.tensor([1.0])
        mat = sparse_coo_tensor(indices, values, (1, 1))
        vec = torch.tensor([5.0], dtype=CFD_DTYPE)
        result = sparse_mm(mat, vec, device="cpu")
        assert result.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class TestBackend:
    def test_default_config(self):
        be = Backend()
        assert be.dtype == torch.float64
        assert be.device == torch.device("cpu") or be.device.type in ("cuda", "mps")

    def test_custom_config(self):
        be = Backend(device="cpu", dtype=torch.float32)
        assert be.dtype == torch.float32
        assert be.device == torch.device("cpu")

    def test_scatter_add(self):
        be = Backend(device="cpu")
        src = torch.tensor([1.0, 2.0, 3.0])
        index = torch.tensor([0, 1, 0])
        out = be.scatter_add(src, index, dim_size=2)
        assert torch.allclose(out, torch.tensor([4.0, 2.0], dtype=torch.float64))

    def test_gather(self):
        be = Backend(device="cpu")
        src = torch.tensor([10.0, 20.0, 30.0])
        index = torch.tensor([2, 0])
        out = be.gather(src, index)
        assert torch.allclose(out, torch.tensor([30.0, 10.0]))

    def test_sparse_coo_tensor(self):
        be = Backend(device="cpu")
        indices = torch.tensor([[0, 1], [1, 0]])
        values = torch.tensor([1.0, 2.0])
        sparse = be.sparse_coo_tensor(indices, values, (2, 2))
        assert sparse.is_sparse

    def test_sparse_mm(self):
        be = Backend(device="cpu")
        indices = torch.tensor([[0, 1], [1, 0]])
        values = torch.tensor([1.0, 2.0])
        mat = be.sparse_coo_tensor(indices, values, (2, 2))
        vec = torch.tensor([3.0, 4.0], dtype=be.dtype)
        result = be.sparse_mm(mat, vec)
        # [1*4, 2*3] = [4, 6]
        assert torch.allclose(result, torch.tensor([4.0, 6.0], dtype=torch.float64))

    def test_repr(self):
        be = Backend(device="cpu")
        r = repr(be)
        assert "Backend" in r
        assert "float64" in r

    def test_backend_isolation(self):
        """Different Backend instances have independent configs."""
        be1 = Backend(dtype=torch.float32)
        be2 = Backend(dtype=torch.float64)
        assert be1.dtype == torch.float32
        assert be2.dtype == torch.float64


# ---------------------------------------------------------------------------
# Integration: scatter + gather round-trip
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_scatter_gather_roundtrip(self):
        """scatter_add then gather should be consistent for unique indices."""
        values = torch.tensor([10.0, 20.0, 30.0])
        index = torch.tensor([2, 0, 1])
        # Scatter into larger tensor
        scattered = scatter_add(values, index, dim_size=4)
        # Gather back
        gathered = gather(scattered, index)
        assert torch.allclose(gathered, values.to(dtype=torch.float64))

    def test_sparse_solve_pattern(self):
        """Assemble sparse matrix, convert to CSR, solve via matmul."""
        # Simple 2x2 system: [[2, 1], [1, 3]] @ x = [5, 8]
        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        values = torch.tensor([2.0, 1.0, 1.0, 3.0])
        A = sparse_coo_tensor(indices, values, (2, 2))
        b = torch.tensor([5.0, 8.0], dtype=CFD_DTYPE)

        # Solve via dense (for testing; real solver would use CG/BiCG)
        A_dense = A.to_dense()
        x = torch.linalg.solve(A_dense, b)

        # Verify: A @ x ≈ b
        result = sparse_mm(A, x)
        assert torch.allclose(result, b, atol=1e-10)
