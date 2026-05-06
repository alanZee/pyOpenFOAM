"""Tests for DIC and DILU preconditioners."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.preconditioners import DICPreconditioner, DILUPreconditioner


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
