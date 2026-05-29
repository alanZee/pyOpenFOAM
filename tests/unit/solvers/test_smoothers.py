"""Tests for standalone smoothers (GaussSeidel, Jacobi, DICG)."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.smoothers import (
    GaussSeidelSmoother,
    JacobiSmoother,
    DICGSmoother,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_symmetric_matrix(n_cells, owner, neighbour, coeff=1.0):
    """Build a symmetric positive-definite LDU matrix."""
    mat = LduMatrix(n_cells, owner, neighbour)
    n_internal = int(neighbour.shape[0])
    lower = -coeff * torch.ones(n_internal, dtype=CFD_DTYPE)
    upper = -coeff * torch.ones(n_internal, dtype=CFD_DTYPE)
    mat.lower = lower
    mat.upper = upper
    diag = torch.zeros(n_cells, dtype=CFD_DTYPE)
    for f in range(n_internal):
        p = int(owner[f])
        n = int(neighbour[f])
        diag[p] += coeff
        diag[n] += coeff
    diag += 0.1
    mat.diag = diag
    return mat


def _make_asymmetric_matrix(n_cells, owner, neighbour):
    """Build an asymmetric LDU matrix."""
    mat = LduMatrix(n_cells, owner, neighbour)
    n_internal = int(neighbour.shape[0])
    lower = -1.0 * torch.ones(n_internal, dtype=CFD_DTYPE)
    upper = -1.5 * torch.ones(n_internal, dtype=CFD_DTYPE)
    mat.lower = lower
    mat.upper = upper
    diag = torch.zeros(n_cells, dtype=CFD_DTYPE)
    for f in range(n_internal):
        p = int(owner[f])
        n = int(neighbour[f])
        diag[p] += abs(float(lower[f]))
        diag[n] += abs(float(upper[f]))
    diag += 0.5
    mat.diag = diag
    return mat


@pytest.fixture
def sym_3cell():
    """3-cell symmetric SPD matrix."""
    owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
    return _make_symmetric_matrix(3, owner, neighbour)


@pytest.fixture
def asym_3cell():
    """3-cell asymmetric matrix."""
    owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
    return _make_asymmetric_matrix(3, owner, neighbour)


@pytest.fixture
def sym_10cell():
    """10-cell symmetric SPD matrix."""
    owner = torch.arange(9, dtype=INDEX_DTYPE)
    neighbour = torch.arange(1, 10, dtype=INDEX_DTYPE)
    return _make_symmetric_matrix(10, owner, neighbour)


# ---------------------------------------------------------------------------
#  GaussSeidelSmoother
# ---------------------------------------------------------------------------


class TestGaussSeidelSmoother:

    def test_identity_system(self):
        """GS on identity: one iteration gives exact solution."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        gs = GaussSeidelSmoother(mat)
        x = torch.zeros(3, dtype=CFD_DTYPE)
        b = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)

        x_smooth = gs.smooth(x, b, n_iterations=1)
        assert torch.allclose(x_smooth, b, atol=1e-10)

    def test_2cell_symmetric(self):
        """GS converges on a 2x2 symmetric system."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        # A = [[4,-1],[-1,4]], b = A @ [1,2] = [2, 7]
        b = torch.tensor([2.0, 7.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        gs = GaussSeidelSmoother(mat)
        x = x0.clone()
        for _ in range(50):
            x = gs.smooth(x, b, n_iterations=1)

        r = b - mat.Ax(x)
        assert torch.norm(r) < 1e-4

    def test_residual_decreases(self, sym_3cell):
        """More GS iterations should decrease residual."""
        mat = sym_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        gs = GaussSeidelSmoother(mat)

        x_few = gs.smooth(x0.clone(), b, n_iterations=5)
        r_few = b - mat.Ax(x_few)

        x_many = gs.smooth(x0.clone(), b, n_iterations=50)
        r_many = b - mat.Ax(x_many)

        assert torch.norm(r_many) < torch.norm(r_few)

    def test_10cell_convergence(self, sym_10cell):
        """GS converges on a 10-cell system."""
        mat = sym_10cell
        b = torch.ones(10, dtype=CFD_DTYPE)
        x0 = torch.zeros(10, dtype=CFD_DTYPE)

        gs = GaussSeidelSmoother(mat)
        x = gs.smooth(x0, b, n_iterations=200)

        r = b - mat.Ax(x)
        assert torch.norm(r) < 0.1

    def test_output_not_inplace(self, sym_3cell):
        """Smoothing should not modify the input tensor in-place."""
        mat = sym_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        gs = GaussSeidelSmoother(mat)
        x_orig = x0.clone()
        _ = gs.smooth(x0, b, n_iterations=10)
        assert torch.allclose(x0, x_orig, atol=1e-15)

    def test_under_relaxation(self, sym_3cell):
        """Under-relaxation (omega < 1) should work."""
        mat = sym_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        gs = GaussSeidelSmoother(mat, omega=0.5)
        x = gs.smooth(x0, b, n_iterations=100)
        r = b - mat.Ax(x)
        # Should still converge (slower) with under-relaxation
        assert torch.norm(r) < 1.0


# ---------------------------------------------------------------------------
#  JacobiSmoother
# ---------------------------------------------------------------------------


class TestJacobiSmoother:

    def test_identity_system(self):
        """Jacobi on identity: x = b after one iteration."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        jac = JacobiSmoother(mat, omega=1.0)
        x = torch.zeros(3, dtype=CFD_DTYPE)
        b = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)

        x_smooth = jac.smooth(x, b, n_iterations=1)
        assert torch.allclose(x_smooth, b, atol=1e-10)

    def test_scaled_identity(self):
        """Jacobi on diag(2): x = b/2 after one iteration with omega=1."""
        mat = LduMatrix(2, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([2.0, 2.0], dtype=CFD_DTYPE)

        jac = JacobiSmoother(mat, omega=1.0)
        x = torch.zeros(2, dtype=CFD_DTYPE)
        b = torch.tensor([4.0, 6.0], dtype=CFD_DTYPE)

        x_smooth = jac.smooth(x, b, n_iterations=1)
        assert torch.allclose(x_smooth, torch.tensor([2.0, 3.0], dtype=CFD_DTYPE), atol=1e-10)

    def test_residual_decreases(self, sym_3cell):
        """More Jacobi iterations should decrease residual."""
        mat = sym_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        jac = JacobiSmoother(mat)

        x_few = jac.smooth(x0.clone(), b, n_iterations=10)
        r_few = b - mat.Ax(x_few)

        x_many = jac.smooth(x0.clone(), b, n_iterations=200)
        r_many = b - mat.Ax(x_many)

        assert torch.norm(r_many) < torch.norm(r_few)

    def test_10cell_convergence(self, sym_10cell):
        """Jacobi converges on a 10-cell system with enough iterations."""
        mat = sym_10cell
        b = torch.ones(10, dtype=CFD_DTYPE)
        x0 = torch.zeros(10, dtype=CFD_DTYPE)

        jac = JacobiSmoother(mat, omega=2.0 / 3.0)
        x = jac.smooth(x0, b, n_iterations=500)

        r = b - mat.Ax(x)
        assert torch.norm(r) < 0.5

    def test_output_not_inplace(self, sym_3cell):
        """Smoothing should not modify the input tensor."""
        mat = sym_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        jac = JacobiSmoother(mat)
        x_orig = x0.clone()
        _ = jac.smooth(x0, b, n_iterations=10)
        assert torch.allclose(x0, x_orig, atol=1e-15)

    def test_default_omega(self):
        """Default omega should be 2/3."""
        mat = LduMatrix(2, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)

        jac = JacobiSmoother(mat)
        assert abs(jac._omega - 2.0 / 3.0) < 1e-12


# ---------------------------------------------------------------------------
#  DICGSmoother
# ---------------------------------------------------------------------------


class TestDICGSmoother:

    def test_identity_system(self):
        """DICG on identity should converge immediately."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        dicg = DICGSmoother(mat)
        x = torch.zeros(3, dtype=CFD_DTYPE)
        b = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)

        x_smooth = dicg.smooth(x, b, n_iterations=1)
        assert torch.allclose(x_smooth, b, atol=1e-10)

    def test_2cell_symmetric(self):
        """DICG converges on a 2x2 symmetric system."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        b = torch.tensor([2.0, 7.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        dicg = DICGSmoother(mat)
        x = dicg.smooth(x0, b, n_iterations=20)
        r = b - mat.Ax(x)
        assert torch.norm(r) < 1e-4

    def test_residual_decreases(self, sym_3cell):
        """More DICG iterations should decrease residual."""
        mat = sym_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        dicg = DICGSmoother(mat)

        x_few = dicg.smooth(x0.clone(), b, n_iterations=2)
        r_few = b - mat.Ax(x_few)

        x_many = dicg.smooth(x0.clone(), b, n_iterations=30)
        r_many = b - mat.Ax(x_many)

        assert torch.norm(r_many) < torch.norm(r_few)

    def test_10cell_convergence(self, sym_10cell):
        """DICG converges on a 10-cell system."""
        mat = sym_10cell
        b = torch.ones(10, dtype=CFD_DTYPE)
        x0 = torch.zeros(10, dtype=CFD_DTYPE)

        dicg = DICGSmoother(mat)
        x = dicg.smooth(x0, b, n_iterations=50)

        r = b - mat.Ax(x)
        # DICG should converge faster than plain Jacobi
        assert torch.norm(r) < 0.1

    def test_output_not_inplace(self, sym_3cell):
        """Smoothing should not modify the input tensor."""
        mat = sym_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        dicg = DICGSmoother(mat)
        x_orig = x0.clone()
        _ = dicg.smooth(x0, b, n_iterations=10)
        assert torch.allclose(x0, x_orig, atol=1e-15)

    def test_output_device_dtype(self, sym_3cell):
        """Output matches matrix device/dtype."""
        mat = sym_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        dicg = DICGSmoother(mat)
        x = dicg.smooth(x0, b, n_iterations=1)
        assert x.device == mat.device
        assert x.dtype == mat.dtype


# ---------------------------------------------------------------------------
#  Cross-smoother comparisons
# ---------------------------------------------------------------------------


class TestSmootherComparison:
    """Compare smoother behaviours."""

    def test_gauss_seidel_faster_than_jacobi(self, sym_10cell):
        """GS should converge faster than Jacobi on the same system."""
        mat = sym_10cell
        b = torch.ones(10, dtype=CFD_DTYPE)
        x0 = torch.zeros(10, dtype=CFD_DTYPE)

        gs = GaussSeidelSmoother(mat)
        jac = JacobiSmoother(mat, omega=2.0 / 3.0)

        x_gs = gs.smooth(x0.clone(), b, n_iterations=50)
        r_gs = b - mat.Ax(x_gs)

        x_jac = jac.smooth(x0.clone(), b, n_iterations=50)
        r_jac = b - mat.Ax(x_jac)

        assert torch.norm(r_gs) < torch.norm(r_jac)

    def test_dicg_faster_than_jacobi(self, sym_10cell):
        """DICG should converge faster than Jacobi."""
        mat = sym_10cell
        b = torch.ones(10, dtype=CFD_DTYPE)
        x0 = torch.zeros(10, dtype=CFD_DTYPE)

        dicg = DICGSmoother(mat)
        jac = JacobiSmoother(mat, omega=2.0 / 3.0)

        x_dicg = dicg.smooth(x0.clone(), b, n_iterations=20)
        r_dicg = b - mat.Ax(x_dicg)

        x_jac = jac.smooth(x0.clone(), b, n_iterations=20)
        r_jac = b - mat.Ax(x_jac)

        assert torch.norm(r_dicg) < torch.norm(r_jac)
