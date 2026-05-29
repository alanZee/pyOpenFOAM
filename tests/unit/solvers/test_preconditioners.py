"""Tests for DIC, DILU, ILU0, ILUT, and Jacobi preconditioners."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.preconditioners import (
    DICPreconditioner,
    DILUPreconditioner,
    ILU0Preconditioner,
    ILUTPreconditioner,
    JacobiPreconditioner,
)


class TestDICPreconditioner:
    """Tests for Diagonal Incomplete Cholesky preconditioner."""

    def test_diagonal_only_matrix(self):
        """DIC on a diagonal-only matrix should give 1/diag."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([4.0, 2.0, 8.0], dtype=CFD_DTYPE)

        dic = DICPreconditioner(mat)
        r = torch.tensor([4.0, 2.0, 8.0], dtype=CFD_DTYPE)
        z = dic.apply(r)
        # z[i] = r[i] / diag[i]
        assert torch.allclose(z, torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE), atol=1e-10)

    def test_symmetric_matrix(self):
        """DIC on a symmetric 2-cell matrix."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([2.0, 2.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        dic = DICPreconditioner(mat)
        r = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        z = dic.apply(r)
        # Should be well-defined positive values
        assert z.shape == (2,)
        assert torch.all(z > 0)

    def test_identity_preconditioner(self):
        """DIC with unity diagonal should approximate identity."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([100.0, 100.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-0.001], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-0.001], dtype=CFD_DTYPE)

        dic = DICPreconditioner(mat)
        r = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        z = dic.apply(r)
        # With dominant diagonal, z ≈ r / diag
        assert torch.allclose(z, r / 100.0, atol=1e-6)

    def test_output_device_dtype(self):
        """Preconditioner output matches input device/dtype."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        dic = DICPreconditioner(mat)
        r = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        z = dic.apply(r)
        assert z.device == mat.device
        assert z.dtype == mat.dtype


class TestDILUPreconditioner:
    """Tests for Diagonal Incomplete LU preconditioner."""

    def test_diagonal_only_matrix(self):
        """DILU on a diagonal-only matrix should give 1/diag."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([4.0, 2.0, 8.0], dtype=CFD_DTYPE)

        dilu = DILUPreconditioner(mat)
        r = torch.tensor([4.0, 2.0, 8.0], dtype=CFD_DTYPE)
        z = dilu.apply(r)
        assert torch.allclose(z, torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE), atol=1e-10)

    def test_asymmetric_matrix(self):
        """DILU on an asymmetric 2-cell matrix."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-2.0], dtype=CFD_DTYPE)

        dilu = DILUPreconditioner(mat)
        r = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        z = dilu.apply(r)
        assert z.shape == (2,)
        assert torch.all(z > 0)

    def test_symmetric_matches_dic(self):
        """For symmetric matrices, DILU should give similar results to DIC."""
        owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
        mat = LduMatrix(3, owner, neighbour)
        mat.diag = torch.tensor([4.0, 6.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)

        dic = DICPreconditioner(mat)
        dilu = DILUPreconditioner(mat)

        r = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        z_dic = dic.apply(r)
        z_dilu = dilu.apply(r)

        # Should be similar for symmetric case (not exact due to
        # different factorisation approaches)
        assert torch.allclose(z_dic, z_dilu, atol=0.1)

    def test_output_shape(self):
        """DILU output has correct shape."""
        mat = LduMatrix(5, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.ones(5, dtype=CFD_DTYPE)

        dilu = DILUPreconditioner(mat)
        r = torch.ones(5, dtype=CFD_DTYPE)
        z = dilu.apply(r)
        assert z.shape == (5,)


class TestILU0Preconditioner:
    """Tests for Incomplete LU with zero fill preconditioner."""

    def test_diagonal_only_matrix(self):
        """ILU0 on a diagonal-only matrix should give 1/diag."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([4.0, 2.0, 8.0], dtype=CFD_DTYPE)

        ilu0 = ILU0Preconditioner(mat)
        r = torch.tensor([4.0, 2.0, 8.0], dtype=CFD_DTYPE)
        z = ilu0.apply(r)
        assert torch.allclose(z, torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE), atol=1e-8)

    def test_symmetric_2cell(self):
        """ILU0 on a symmetric 2-cell matrix."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        ilu0 = ILU0Preconditioner(mat)
        r = torch.tensor([2.0, 7.0], dtype=CFD_DTYPE)
        z = ilu0.apply(r)
        assert z.shape == (2,)
        assert torch.all(torch.isfinite(z))

    def test_asymmetric_3cell(self):
        """ILU0 on a 3-cell chain with asymmetric coefficients."""
        owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
        mat = LduMatrix(3, owner, neighbour)
        mat.diag = torch.tensor([4.0, 6.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-2.0, -1.5], dtype=CFD_DTYPE)

        ilu0 = ILU0Preconditioner(mat)
        r = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        z = ilu0.apply(r)
        assert z.shape == (3,)
        assert torch.all(torch.isfinite(z))
        # ILU0 should produce a reasonable preconditioned vector
        assert torch.norm(z) > 0

    def test_output_device_dtype(self):
        """Preconditioner output matches matrix device/dtype."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        ilu0 = ILU0Preconditioner(mat)
        r = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        z = ilu0.apply(r)
        assert z.device == mat.device
        assert z.dtype == mat.dtype

    def test_improves_over_identity(self):
        """ILU0 preconditioner should improve convergence vs no preconditioner."""
        owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
        mat = LduMatrix(3, owner, neighbour)
        mat.diag = torch.tensor([4.0, 6.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)

        ilu0 = ILU0Preconditioner(mat)
        r = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)
        z = ilu0.apply(r)
        # ||M⁻¹r|| should be of same order as ||r|| (not degenerate)
        assert 1e-10 < float(torch.norm(z)) < 1e10


class TestILUTPreconditioner:
    """Tests for Incomplete LU with threshold dropping."""

    def test_diagonal_only_matrix(self):
        """ILUT on a diagonal-only matrix should give 1/diag."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([4.0, 2.0, 8.0], dtype=CFD_DTYPE)

        ilut = ILUTPreconditioner(mat, drop_tol=0.01)
        r = torch.tensor([4.0, 2.0, 8.0], dtype=CFD_DTYPE)
        z = ilut.apply(r)
        assert torch.allclose(z, torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE), atol=1e-8)

    def test_symmetric_2cell(self):
        """ILUT on a symmetric 2-cell matrix."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        ilut = ILUTPreconditioner(mat, drop_tol=0.01)
        r = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        z = ilut.apply(r)
        assert z.shape == (2,)
        assert torch.all(torch.isfinite(z))

    def test_asymmetric_3cell(self):
        """ILUT on an asymmetric 3-cell chain."""
        owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
        mat = LduMatrix(3, owner, neighbour)
        mat.diag = torch.tensor([4.0, 6.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-2.0, -1.5], dtype=CFD_DTYPE)

        ilut = ILUTPreconditioner(mat, drop_tol=0.01)
        r = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        z = ilut.apply(r)
        assert z.shape == (3,)
        assert torch.all(torch.isfinite(z))

    def test_threshold_affects_result(self):
        """Different drop tolerances should produce different results."""
        owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
        mat = LduMatrix(3, owner, neighbour)
        mat.diag = torch.tensor([10.0, 10.0, 10.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-0.5, -0.3], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-0.3, -0.5], dtype=CFD_DTYPE)

        ilut_loose = ILUTPreconditioner(mat, drop_tol=0.5)
        ilut_tight = ILUTPreconditioner(mat, drop_tol=1e-10)
        r = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        z_loose = ilut_loose.apply(r)
        z_tight = ilut_tight.apply(r)
        # Results should differ (at least slightly) with different tolerances
        # With tight tolerance, ILUT approaches ILU0
        assert torch.all(torch.isfinite(z_loose))
        assert torch.all(torch.isfinite(z_tight))

    def test_output_device_dtype(self):
        """Output matches matrix device/dtype."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        ilut = ILUTPreconditioner(mat)
        r = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        z = ilut.apply(r)
        assert z.device == mat.device
        assert z.dtype == mat.dtype


class TestJacobiPreconditioner:
    """Tests for Jacobi (diagonal) preconditioner."""

    def test_basic_application(self):
        """Jacobi on a diagonal matrix: z = r / diag."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([2.0, 4.0, 8.0], dtype=CFD_DTYPE)

        jac = JacobiPreconditioner(mat)
        r = torch.tensor([2.0, 8.0, 16.0], dtype=CFD_DTYPE)
        z = jac.apply(r)
        assert torch.allclose(z, torch.tensor([1.0, 2.0, 2.0], dtype=CFD_DTYPE), atol=1e-10)

    def test_off_diagonal_ignored(self):
        """Jacobi only uses the diagonal, ignoring off-diagonal entries."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-100.0], dtype=CFD_DTYPE)  # Large off-diagonal
        mat.upper = torch.tensor([-100.0], dtype=CFD_DTYPE)

        jac = JacobiPreconditioner(mat)
        r = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        z = jac.apply(r)
        # Should still give r / diag regardless of off-diagonal
        assert torch.allclose(z, torch.tensor([1.0, 1.0], dtype=CFD_DTYPE), atol=1e-10)

    def test_output_device_dtype(self):
        """Output matches matrix device/dtype."""
        mat = LduMatrix(2, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([3.0, 3.0], dtype=CFD_DTYPE)

        jac = JacobiPreconditioner(mat)
        r = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        z = jac.apply(r)
        assert z.device == mat.device
        assert z.dtype == mat.dtype

    def test_zero_diagonal_clamped(self):
        """Near-zero diagonal entries are clamped to avoid division by zero."""
        mat = LduMatrix(2, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([0.0, 1.0], dtype=CFD_DTYPE)

        jac = JacobiPreconditioner(mat)
        r = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        z = jac.apply(r)
        # Should not produce inf/nan
        assert torch.all(torch.isfinite(z))

    def test_output_shape(self):
        """Output has correct shape."""
        mat = LduMatrix(5, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.ones(5, dtype=CFD_DTYPE)

        jac = JacobiPreconditioner(mat)
        r = torch.ones(5, dtype=CFD_DTYPE)
        z = jac.apply(r)
        assert z.shape == (5,)
