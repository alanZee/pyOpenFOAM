"""Tests for FvMatrix — finite volume matrix with source and relaxation."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.fv_matrix import FvMatrix, LinearSolver
from pyfoam.core.ldu_matrix import LduMatrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_2cell_mesh():
    """2-cell mesh with 1 internal face."""
    n_cells = 2
    owner = torch.tensor([0], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
    return n_cells, owner, neighbour


def _make_3cell_chain():
    """3-cell chain: 0 -- 1 -- 2."""
    n_cells = 3
    owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
    return n_cells, owner, neighbour


def _make_simple_fv_matrix():
    """Create a simple 2-cell FvMatrix with diffusion-like coefficients."""
    n_cells, owner, neighbour = _make_2cell_mesh()
    mat = FvMatrix(n_cells, owner, neighbour)
    mat.diag = torch.tensor([2.0, 2.0])
    mat.lower = torch.tensor([-1.0])
    mat.upper = torch.tensor([-1.0])
    return mat


@pytest.fixture
def two_cell_mesh():
    return _make_2cell_mesh()


@pytest.fixture
def chain_mesh():
    return _make_3cell_chain()


@pytest.fixture
def simple_fv():
    return _make_simple_fv_matrix()


# ---------------------------------------------------------------------------
# Mock boundary condition
# ---------------------------------------------------------------------------


class MockFixedValueBC:
    """Minimal fixedValue-like BC for testing matrix contributions."""

    def __init__(self, owner_cells, face_areas, delta_coeffs, values):
        self._owner_cells = owner_cells
        self._face_areas = face_areas.to(dtype=CFD_DTYPE)
        self._delta_coeffs = delta_coeffs.to(dtype=CFD_DTYPE)
        self._values = values.to(dtype=CFD_DTYPE)

    def matrix_contributions(self, field, n_cells, diag=None, source=None):
        device = field.device
        dtype = field.dtype
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        coeff = (self._delta_coeffs * self._face_areas).to(dtype=dtype)
        diag.scatter_add_(0, self._owner_cells.to(device), coeff)
        source.scatter_add_(0, self._owner_cells.to(device), coeff * self._values.to(dtype=dtype))
        return diag, source


class MockZeroGradientBC:
    """Minimal zeroGradient-like BC — no matrix contribution."""

    def matrix_contributions(self, field, n_cells, diag=None, source=None):
        device = field.device
        dtype = field.dtype
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFvMatrixConstruction:
    def test_basic_creation(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = FvMatrix(n_cells, owner, neighbour)
        assert mat.n_cells == 2
        assert mat.n_internal_faces == 1

    def test_source_zeros_initially(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = FvMatrix(n_cells, owner, neighbour)
        assert torch.allclose(mat.source, torch.zeros(2, dtype=CFD_DTYPE))

    def test_inherits_ldu_matrix(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = FvMatrix(n_cells, owner, neighbour)
        assert isinstance(mat, LduMatrix)

    def test_default_dtype(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = FvMatrix(n_cells, owner, neighbour)
        assert mat.dtype == CFD_DTYPE
        assert mat.source.dtype == CFD_DTYPE

    def test_device_cpu(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = FvMatrix(n_cells, owner, neighbour, device="cpu")
        assert mat.device == torch.device("cpu")
        assert mat.source.device == torch.device("cpu")

    def test_relaxation_factor_default(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = FvMatrix(n_cells, owner, neighbour)
        assert mat.relaxation_factor == 1.0

    def test_repr(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = FvMatrix(n_cells, owner, neighbour)
        r = repr(mat)
        assert "FvMatrix" in r
        assert "n_cells=2" in r


# ---------------------------------------------------------------------------
# Source vector
# ---------------------------------------------------------------------------


class TestFvMatrixSource:
    def test_set_source(self, two_cell_mesh):
        n_cells, owner, neighbour = two_cell_mesh
        mat = FvMatrix(n_cells, owner, neighbour)
        mat.source = torch.tensor([10.0, 20.0])
        assert torch.allclose(mat.source, torch.tensor([10.0, 20.0], dtype=CFD_DTYPE))

    def test_add_explicit_source_vector(self, simple_fv):
        simple_fv.add_explicit_source(torch.tensor([5.0, 10.0]))
        assert torch.allclose(
            simple_fv.source, torch.tensor([5.0, 10.0], dtype=CFD_DTYPE)
        )

    def test_add_explicit_source_scalar(self, simple_fv):
        simple_fv.add_explicit_source(torch.tensor(3.0))
        assert torch.allclose(
            simple_fv.source, torch.tensor([3.0, 3.0], dtype=CFD_DTYPE)
        )

    def test_add_explicit_source_accumulates(self, simple_fv):
        simple_fv.add_explicit_source(torch.tensor([1.0, 2.0]))
        simple_fv.add_explicit_source(torch.tensor([3.0, 4.0]))
        assert torch.allclose(
            simple_fv.source, torch.tensor([4.0, 6.0], dtype=CFD_DTYPE)
        )


# ---------------------------------------------------------------------------
# Boundary condition contributions
# ---------------------------------------------------------------------------


class TestFvMatrixBoundary:
    def test_fixed_value_bc_contribution(self, simple_fv):
        """fixedValue BC adds large diagonal + matching source."""
        owners = torch.tensor([0], dtype=INDEX_DTYPE)
        areas = torch.tensor([1.0])
        deltas = torch.tensor([2.0])
        values = torch.tensor([10.0])

        bc = MockFixedValueBC(owners, areas, deltas, values)
        simple_fv.add_boundary_contribution(bc)

        # Diagonal[0] should have the BC coefficient added
        expected_coeff = deltas[0] * areas[0]  # 2.0
        assert simple_fv.diag[0].item() == pytest.approx(2.0 + expected_coeff)
        # Source[0] should have coeff * value
        assert simple_fv.source[0].item() == pytest.approx(expected_coeff * 10.0)
        # Cell 1 unaffected
        assert simple_fv.diag[1].item() == pytest.approx(2.0)
        assert simple_fv.source[1].item() == pytest.approx(0.0)

    def test_zero_gradient_bc_no_contribution(self, simple_fv):
        """zeroGradient BC adds nothing to matrix."""
        bc = MockZeroGradientBC()
        simple_fv.add_boundary_contribution(bc)

        # Matrix should be unchanged
        assert torch.allclose(simple_fv.diag, torch.tensor([2.0, 2.0], dtype=CFD_DTYPE))
        assert torch.allclose(simple_fv.source, torch.zeros(2, dtype=CFD_DTYPE))

    def test_multiple_boundary_contributions(self, two_cell_mesh):
        """Multiple BCs accumulate into the same matrix."""
        n_cells, owner, neighbour = two_cell_mesh
        mat = FvMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-1.0])

        # BC on cell 0
        bc1 = MockFixedValueBC(
            torch.tensor([0], dtype=INDEX_DTYPE),
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([5.0]),
        )
        mat.add_boundary_contribution(bc1)

        # BC on cell 1
        bc2 = MockFixedValueBC(
            torch.tensor([1], dtype=INDEX_DTYPE),
            torch.tensor([1.0]),
            torch.tensor([3.0]),
            torch.tensor([8.0]),
        )
        mat.add_boundary_contribution(bc2)

        # Check accumulated values
        assert mat.diag[0].item() == pytest.approx(4.0 + 2.0)
        assert mat.diag[1].item() == pytest.approx(4.0 + 3.0)
        assert mat.source[0].item() == pytest.approx(2.0 * 5.0)
        assert mat.source[1].item() == pytest.approx(3.0 * 8.0)

    def test_bc_with_field(self, simple_fv):
        """BC that receives field values."""
        owners = torch.tensor([1], dtype=INDEX_DTYPE)
        areas = torch.tensor([1.0])
        deltas = torch.tensor([1.0])
        values = torch.tensor([0.0])  # value not used by this mock

        bc = MockFixedValueBC(owners, areas, deltas, values)
        field = torch.tensor([100.0, 200.0], dtype=CFD_DTYPE)
        simple_fv.add_boundary_contribution(bc, field=field)

        # Source should be coeff * values (0.0 here)
        assert simple_fv.source[1].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Under-relaxation
# ---------------------------------------------------------------------------


class TestFvMatrixRelaxation:
    def test_no_relaxation(self, simple_fv):
        """Relaxation factor 1.0 means no change."""
        diag_before = simple_fv.diag.clone()
        source_before = simple_fv.source.clone()

        field_old = torch.tensor([10.0, 20.0], dtype=CFD_DTYPE)
        simple_fv.relax(field_old, under_relaxation_factor=1.0)

        assert torch.allclose(simple_fv.diag, diag_before)
        assert torch.allclose(simple_fv.source, source_before)

    def test_explicit_relaxation_modifies_diag(self, simple_fv):
        """Under-relaxation divides diagonal by alpha."""
        field_old = torch.tensor([10.0, 20.0], dtype=CFD_DTYPE)
        alpha = 0.5

        diag_before = simple_fv.diag.clone()
        simple_fv.relax(field_old, under_relaxation_factor=alpha)

        # diag' = diag / alpha
        expected_diag = diag_before / alpha
        assert torch.allclose(simple_fv.diag, expected_diag)

    def test_explicit_relaxation_modifies_source(self, simple_fv):
        """Under-relaxation adds (1-alpha)/alpha * diag * field_old to source."""
        field_old = torch.tensor([10.0, 20.0], dtype=CFD_DTYPE)
        alpha = 0.5

        diag_before = simple_fv.diag.clone()
        source_before = simple_fv.source.clone()

        simple_fv.relax(field_old, under_relaxation_factor=alpha)

        # source' = source + (1-alpha) * diag' * field_old
        # where diag' = diag / alpha
        diag_relaxed = diag_before / alpha
        expected_source = source_before + (1.0 - alpha) * diag_relaxed * field_old
        assert torch.allclose(simple_fv.source, expected_source)

    def test_relaxation_stores_factor(self, simple_fv):
        field_old = torch.tensor([1.0, 2.0], dtype=CFD_DTYPE)
        simple_fv.relax(field_old, under_relaxation_factor=0.7)
        assert simple_fv.relaxation_factor == 0.7

    def test_invalid_relaxation_factor_zero(self, simple_fv):
        field_old = torch.tensor([1.0, 2.0], dtype=CFD_DTYPE)
        with pytest.raises(ValueError, match="must be in"):
            simple_fv.relax(field_old, under_relaxation_factor=0.0)

    def test_invalid_relaxation_factor_negative(self, simple_fv):
        field_old = torch.tensor([1.0, 2.0], dtype=CFD_DTYPE)
        with pytest.raises(ValueError, match="must be in"):
            simple_fv.relax(field_old, under_relaxation_factor=-0.5)

    def test_invalid_relaxation_factor_above_one(self, simple_fv):
        field_old = torch.tensor([1.0, 2.0], dtype=CFD_DTYPE)
        with pytest.raises(ValueError, match="must be in"):
            simple_fv.relax(field_old, under_relaxation_factor=1.5)


# ---------------------------------------------------------------------------
# Reference pressure
# ---------------------------------------------------------------------------


class TestFvMatrixReference:
    def test_set_reference(self, simple_fv):
        """Setting reference pins a cell value."""
        diag_before = simple_fv.diag[0].item()
        simple_fv.set_reference(0, value=101325.0)

        # Diagonal should increase
        assert simple_fv.diag[0].item() > diag_before
        # Source should be large * value
        assert simple_fv.source[0].item() > 0.0

    def test_set_reference_invalid_index(self, simple_fv):
        with pytest.raises(ValueError, match="out of range"):
            simple_fv.set_reference(-1, value=0.0)

    def test_set_reference_invalid_index_high(self, simple_fv):
        with pytest.raises(ValueError, match="out of range"):
            simple_fv.set_reference(10, value=0.0)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


class TestFvMatrixSolve:
    def test_solve_with_mock_solver(self, simple_fv):
        """Solve delegates to the solver callable."""
        simple_fv.source = torch.tensor([10.0, 20.0], dtype=CFD_DTYPE)

        class DirectSolver:
            def __call__(self, matrix, source, x0, tolerance, max_iter):
                # Simple direct solve via dense inverse
                A = matrix.to_sparse_coo().to_dense()
                x = torch.linalg.solve(A, source)
                return x, 1, 0.0

        solver = DirectSolver()
        x0 = torch.zeros(2, dtype=CFD_DTYPE)
        solution, iters, residual = simple_fv.solve(solver, x0)

        # Verify: A @ solution ≈ source
        assert torch.allclose(
            simple_fv.Ax(solution), simple_fv.source, atol=1e-10
        )

    def test_solve_returns_tuple(self, simple_fv):
        """Solve returns (solution, iterations, residual)."""

        class TrivialSolver:
            def __call__(self, matrix, source, x0, tolerance, max_iter):
                return x0, 0, 1.0

        solver = TrivialSolver()
        x0 = torch.zeros(2, dtype=CFD_DTYPE)
        result = simple_fv.solve(solver, x0)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------


class TestFvMatrixResidual:
    def test_residual_zero_for_solution(self):
        """Residual is zero when x solves Ax = b."""
        n_cells, owner, neighbour = _make_2cell_mesh()
        mat = FvMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-1.0])
        mat.source = torch.tensor([3.0, 5.0], dtype=CFD_DTYPE)

        # Solve directly
        A_dense = torch.tensor([[4.0, -1.0], [-1.0, 4.0]], dtype=CFD_DTYPE)
        b = torch.tensor([3.0, 5.0], dtype=CFD_DTYPE)
        x = torch.linalg.solve(A_dense, b)

        r = mat.residual(x)
        assert torch.allclose(r, torch.zeros(2, dtype=CFD_DTYPE), atol=1e-10)

    def test_residual_nonzero_for_wrong_guess(self, simple_fv):
        """Residual is nonzero for a random guess."""
        simple_fv.source = torch.tensor([10.0, 20.0], dtype=CFD_DTYPE)
        x = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        r = simple_fv.residual(x)
        # r = b - Ax = [10, 20] - [1, 1] = [9, 19]
        assert not torch.allclose(r, torch.zeros(2, dtype=CFD_DTYPE))


# ---------------------------------------------------------------------------
# Integration: solve with relaxation + BC
# ---------------------------------------------------------------------------


class TestFvMatrixIntegration:
    def test_relaxation_preserves_solution_direction(self):
        """After relaxation, the system should still solve to the same answer."""
        n_cells, owner, neighbour = _make_2cell_mesh()
        mat = FvMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-1.0])
        mat.source = torch.tensor([10.0, 20.0], dtype=CFD_DTYPE)

        # Apply relaxation
        field_old = torch.tensor([1.0, 2.0], dtype=CFD_DTYPE)
        mat.relax(field_old, under_relaxation_factor=0.5)

        # Solve the relaxed system
        A_dense = mat.to_sparse_coo().to_dense()
        x = torch.linalg.solve(A_dense, mat.source)

        # The relaxed solution should satisfy A_relax @ x = b_relax
        assert torch.allclose(mat.Ax(x), mat.source, atol=1e-10)

    def test_bc_then_relax(self):
        """Apply BC, then relax, then solve."""
        n_cells, owner, neighbour = _make_2cell_mesh()
        mat = FvMatrix(n_cells, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0])
        mat.lower = torch.tensor([-1.0])
        mat.upper = torch.tensor([-1.0])

        # Add boundary contribution
        bc = MockFixedValueBC(
            torch.tensor([0], dtype=INDEX_DTYPE),
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([100.0]),
        )
        mat.add_boundary_contribution(bc)

        # Relax
        field_old = torch.tensor([50.0, 50.0], dtype=CFD_DTYPE)
        mat.relax(field_old, under_relaxation_factor=0.3)

        # Solve
        A_dense = mat.to_sparse_coo().to_dense()
        x = torch.linalg.solve(A_dense, mat.source)
        assert torch.allclose(mat.Ax(x), mat.source, atol=1e-10)
