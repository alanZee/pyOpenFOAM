"""Tests for LduMatrix — LDU sparse matrix format."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_3cell_chain():
    """Create a 3-cell chain mesh: 0 -- 1 -- 2.

    Two internal faces:
      face 0: owner=0, neighbour=1
      face 1: owner=1, neighbour=2

    Returns (n_cells, owner, neighbour).
    """
    n_cells = 3
    owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
    return n_cells, owner, neighbour


def _make_2cell_mesh():
    """Create a simple 2-cell mesh with 1 internal face.

    Returns (n_cells, owner, neighbour).
    """
    n_cells = 2
    owner = torch.tensor([0], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
    return n_cells, owner, neighbour


@pytest.fixture
def chain_mesh():
    """Fixture: 3-cell chain mesh."""
    return _make_3cell_chain()


@pytest.fixture
def two_cell_mesh():
    """Fixture: 2-cell mesh."""
    return _make_2cell_mesh()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLduMatrixConstruction:
    def test_basic_creation(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        assert mat.n_cells == 2
        assert mat.n_internal_faces == 1

    def test_diag_zeros_initially(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        assert torch.allclose(mat.diag, torch.zeros(2, dtype=CFD_DTYPE))

    def test_lower_zeros_initially(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        assert torch.allclose(mat.lower, torch.zeros(1, dtype=CFD_DTYPE))

    def test_upper_zeros_initially(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        assert torch.allclose(mat.upper, torch.zeros(1, dtype=CFD_DTYPE))

    def test_default_dtype_is_float64(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        assert mat.dtype == CFD_DTYPE
        assert mat.diag.dtype == CFD_DTYPE
        assert mat.lower.dtype == CFD_DTYPE
        assert mat.upper.dtype == CFD_DTYPE

    def test_owner_neighbour_stored(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        assert torch.equal(mat.owner, owner)
        assert torch.equal(mat.neighbour, neighbour)

    def test_device_cpu(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour, device="cpu")
        assert mat.device == torch.device("cpu")
        assert mat.diag.device == torch.device("cpu")

    def test_repr(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        r = repr(mat)
        assert "LduMatrix" in r
        assert "n_cells=2" in r
        assert "n_internal_faces=1" in r


# ---------------------------------------------------------------------------
# Property setters
# ---------------------------------------------------------------------------


class TestLduMatrixSetters:
    def test_set_diag(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0])
        assert torch.allclose(mat.diag, torch.tensor([4.0, 4.0], dtype=CFD_DTYPE))

    def test_set_lower(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.lower = torch.tensor([-1.0])
        assert torch.allclose(mat.lower, torch.tensor([-1.0], dtype=CFD_DTYPE))

    def test_set_upper(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.upper = torch.tensor([-1.0])
        assert torch.allclose(mat.upper, torch.tensor([-1.0], dtype=CFD_DTYPE))

    def test_add_to_diag_scalar(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.add_to_diag(torch.tensor(3.0))
        assert torch.allclose(mat.diag, torch.tensor([3.0, 3.0], dtype=CFD_DTYPE))

    def test_add_to_diag_vector(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.add_to_diag(torch.tensor([1.0, 2.0]))
        assert torch.allclose(mat.diag, torch.tensor([1.0, 2.0], dtype=CFD_DTYPE))


# ---------------------------------------------------------------------------
# Matrix-vector product
# ---------------------------------------------------------------------------


class TestLduMatrixAx:
    def test_identity_matrix(self, two_cell_mesh):
        """Diagonal-only matrix: Ax = diag * x."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([3.0, 5.0])
        x = torch.tensor([2.0, 4.0], dtype=CFD_DTYPE)
        y = mat.Ax(x)
        assert torch.allclose(y, torch.tensor([6.0, 20.0], dtype=CFD_DTYPE))

    def test_symmetric_off_diagonal(self, two_cell_mesh):
        """Symmetric: lower == upper == -1, diag == 2."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([2.0, 2.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-1.0])

        x = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        y = mat.Ax(x)
        # y[0] = 2*1 + (-1)*1 = 1
        # y[1] = 2*1 + (-1)*1 = 1
        assert torch.allclose(y, torch.tensor([1.0, 1.0], dtype=CFD_DTYPE))

    def test_asymmetric_off_diagonal(self, two_cell_mesh):
        """Asymmetric: lower=-1, upper=-2."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-2.0])

        x = torch.tensor([1.0, 3.0], dtype=CFD_DTYPE)
        y = mat.Ax(x)
        # y[0] = 4*1 + (-1)*3 = 1
        # y[1] = 4*3 + (-2)*1 = 10
        assert torch.allclose(y, torch.tensor([1.0, 10.0], dtype=CFD_DTYPE))

    def test_3cell_chain(self, chain_mesh):
        """3-cell chain: 0 -- 1 -- 2."""
        n_cells, owner, neighbour = chain_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 6.0, 4.0])
        mat.lower = torch.tensor([-1.0, -1.0])
        mat.upper = torch.tensor([-1.0, -1.0])

        x = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        y = mat.Ax(x)
        # y[0] = 4*1 + (-1)*2 = 2
        # y[1] = 6*2 + (-1)*1 + (-1)*3 = 8
        # y[2] = 4*3 + (-1)*2 = 10
        assert torch.allclose(y, torch.tensor([2.0, 8.0, 10.0], dtype=CFD_DTYPE))

    def test_zero_matrix(self, two_cell_mesh):
        """Zero matrix: Ax = 0."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        x = torch.tensor([5.0, 7.0], dtype=CFD_DTYPE)
        y = mat.Ax(x)
        assert torch.allclose(y, torch.zeros(2, dtype=CFD_DTYPE))

    def test_wrong_size_raises(self, two_cell_mesh):
        """Ax with wrong vector size should raise."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        with pytest.raises(ValueError, match="3 elements"):
            mat.Ax(x)

    def test_non_floating_raises(self, two_cell_mesh):
        """Ax with integer tensor should raise."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        x = torch.tensor([1, 2], dtype=torch.int64)
        with pytest.raises(TypeError, match="floating-point"):
            mat.Ax(x)


# ---------------------------------------------------------------------------
# COO / CSR conversion
# ---------------------------------------------------------------------------


class TestLduMatrixSparse:
    def test_to_sparse_coo_shape(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-1.0])

        coo = mat.to_sparse_coo()
        assert coo.shape == (2, 2)

    def test_to_sparse_coo_values(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-1.0])

        coo = mat.to_sparse_coo()
        dense = coo.to_dense()
        # Diagonal
        assert torch.allclose(dense[0, 0], torch.tensor(4.0, dtype=CFD_DTYPE))
        assert torch.allclose(dense[1, 1], torch.tensor(4.0, dtype=CFD_DTYPE))
        # Off-diagonal
        assert torch.allclose(dense[0, 1], torch.tensor(-1.0, dtype=CFD_DTYPE))
        assert torch.allclose(dense[1, 0], torch.tensor(-1.0, dtype=CFD_DTYPE))

    def test_to_sparse_csr(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([2.0, 3.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-2.0])

        csr = mat.to_sparse_csr()
        assert csr.layout == torch.sparse_csr

    def test_sparse_matvec_matches_ldu(self, chain_mesh):
        """COO matvec should match LduMatrix.Ax."""
        n_cells, owner, neighbour = chain_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 6.0, 4.0])
        mat.lower = torch.tensor([-1.0, -1.0])
        mat.upper = torch.tensor([-1.0, -1.0])

        x = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        y_ldu = mat.Ax(x)

        coo = mat.to_sparse_coo()
        y_sparse = torch.sparse.mm(coo, x.unsqueeze(1)).squeeze(1)

        assert torch.allclose(y_ldu, y_sparse, atol=1e-12)


# ---------------------------------------------------------------------------
# Optimised sparse operations (Phase 13)
# ---------------------------------------------------------------------------


class TestLduMatrixSparseOptimised:
    """Tests for GPU-optimised sparse matrix operations."""

    def test_to_sparse_coo_matches_original(self, chain_mesh):
        """Optimised to_sparse_coo should produce same result."""
        n_cells, owner, neighbour = chain_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 6.0, 4.0])
        mat.lower = torch.tensor([-1.0, -1.0])
        mat.upper = torch.tensor([-2.0, -2.0])

        coo = mat.to_sparse_coo()
        dense = coo.to_dense()

        # Verify specific entries
        assert torch.allclose(dense[0, 0], torch.tensor(4.0, dtype=CFD_DTYPE))
        assert torch.allclose(dense[1, 1], torch.tensor(6.0, dtype=CFD_DTYPE))
        assert torch.allclose(dense[0, 1], torch.tensor(-1.0, dtype=CFD_DTYPE))
        assert torch.allclose(dense[1, 0], torch.tensor(-2.0, dtype=CFD_DTYPE))

    def test_to_sparse_csr_cached(self, two_cell_mesh):
        """Cached CSR should match non-cached."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([2.0, 3.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-2.0])

        csr1 = mat.to_sparse_csr_cached()
        csr2 = mat.to_sparse_csr_cached()
        # Should be the same cached object
        assert csr1 is csr2

    def test_cache_invalidation_on_diag_set(self, two_cell_mesh):
        """Setting diag should invalidate cache."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([2.0, 3.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-2.0])

        csr1 = mat.to_sparse_csr_cached()
        mat.diag = torch.tensor([5.0, 6.0])  # triggers invalidation
        csr2 = mat.to_sparse_csr_cached()
        # Should be different objects (cache was invalidated)
        assert csr1 is not csr2

    def test_cache_invalidation_on_lower_set(self, two_cell_mesh):
        """Setting lower should invalidate cache."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([2.0, 3.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-2.0])

        csr1 = mat.to_sparse_csr_cached()
        mat.lower = torch.tensor([-3.0])
        csr2 = mat.to_sparse_csr_cached()
        assert csr1 is not csr2

    def test_cache_invalidation_on_upper_set(self, two_cell_mesh):
        """Setting upper should invalidate cache."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([2.0, 3.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-2.0])

        csr1 = mat.to_sparse_csr_cached()
        mat.upper = torch.tensor([-4.0])
        csr2 = mat.to_sparse_csr_cached()
        assert csr1 is not csr2

    def test_cache_invalidation_on_add_to_diag(self, two_cell_mesh):
        """add_to_diag should invalidate cache."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([2.0, 3.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-2.0])

        csr1 = mat.to_sparse_csr_cached()
        mat.add_to_diag(torch.tensor([1.0, 1.0]))
        csr2 = mat.to_sparse_csr_cached()
        assert csr1 is not csr2

    def test_invalidate_cache_explicit(self, two_cell_mesh):
        """Explicit invalidate_cache should clear the cache."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([2.0, 3.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-2.0])

        mat.to_sparse_csr_cached()
        mat.invalidate_cache()
        assert "csr" not in mat._csr_cache

    def test_ax_sparse_matches_ax(self, chain_mesh):
        """Ax_sparse should produce identical results to Ax."""
        n_cells, owner, neighbour = chain_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 6.0, 4.0])
        mat.lower = torch.tensor([-1.0, -1.0])
        mat.upper = torch.tensor([-1.0, -1.0])

        x = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        y_ldu = mat.Ax(x)
        y_sparse = mat.Ax_sparse(x)

        assert torch.allclose(y_ldu, y_sparse, atol=1e-12)

    def test_ax_sparse_asymmetric(self, two_cell_mesh):
        """Ax_sparse with asymmetric matrix."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-2.0])

        x = torch.tensor([1.0, 3.0], dtype=CFD_DTYPE)
        y_ldu = mat.Ax(x)
        y_sparse = mat.Ax_sparse(x)

        assert torch.allclose(y_ldu, y_sparse, atol=1e-12)

    def test_ax_batched(self, chain_mesh):
        """Ax_batched should handle multiple RHS correctly."""
        n_cells, owner, neighbour = chain_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 6.0, 4.0])
        mat.lower = torch.tensor([-1.0, -1.0])
        mat.upper = torch.tensor([-1.0, -1.0])

        # 3 cells, 2 RHS vectors
        x = torch.tensor([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ], dtype=CFD_DTYPE)
        y_batched = mat.Ax_batched(x)

        # Check each column matches single Ax
        y0 = mat.Ax(x[:, 0])
        y1 = mat.Ax(x[:, 1])

        assert torch.allclose(y_batched[:, 0], y0, atol=1e-12)
        assert torch.allclose(y_batched[:, 1], y1, atol=1e-12)

    def test_ax_sparse_zero_off_diagonal(self, two_cell_mesh):
        """Ax_sparse with only diagonal entries."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([3.0, 5.0])
        mat.lower = torch.tensor([0.0])
        mat.upper = torch.tensor([0.0])

        x = torch.tensor([2.0, 4.0], dtype=CFD_DTYPE)
        y = mat.Ax_sparse(x)

        assert torch.allclose(y, torch.tensor([6.0, 20.0], dtype=CFD_DTYPE))


# ---------------------------------------------------------------------------
# Integration: realistic FVM diffusion matrix
# ---------------------------------------------------------------------------


class TestLduMatrixIntegration:
    def test_diffusion_matrix_3cell(self, chain_mesh):
        """Realistic 3-cell diffusion matrix-vector product."""
        n_cells, owner, neighbour = chain_mesh
        mat = LduMatrix(n_cells, owner, neighbour)

        # Diffusion coefficients: deltaCoeff * area = 1.0 for each face
        coeff = torch.tensor([1.0, 1.0])
        mat.lower = -coeff
        mat.upper = -coeff
        # Diagonal: sum of absolute off-diagonal for each cell
        # Cell 0: 1 face, cell 1: 2 faces, cell 2: 1 face
        mat.diag = torch.tensor([1.0, 2.0, 1.0])

        # Uniform field x = 1 → Laplacian is zero (Neumann BCs implicit)
        x = torch.ones(3, dtype=CFD_DTYPE)
        y = mat.Ax(x)
        assert torch.allclose(y, torch.zeros(3, dtype=CFD_DTYPE), atol=1e-12)

        # Verify specific matrix-vector product
        x = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        y = mat.Ax(x)
        # y[0] = 1*1 + (-1)*2 = -1
        # y[1] = 2*2 + (-1)*1 + (-1)*3 = 0
        # y[2] = 1*3 + (-1)*2 = 1
        assert torch.allclose(y, torch.tensor([-1.0, 0.0, 1.0], dtype=CFD_DTYPE), atol=1e-12)
