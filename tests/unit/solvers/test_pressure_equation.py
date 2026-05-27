"""Tests for pressure equation assembly, solve, velocity and flux correction.

Uses small synthetic 1D/2D mesh data (no cavity mesh dependency).
Covers: assemble_pressure_equation, solve_pressure_equation,
        correct_velocity, correct_face_flux.
"""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.solvers.pcg import PCGSolver
from pyfoam.solvers.pressure_equation import (
    assemble_pressure_equation,
    correct_face_flux,
    correct_velocity,
    solve_pressure_equation,
)


# ---------------------------------------------------------------------------
# Minimal synthetic mesh (3-cell 1D chain + 2 boundary faces)
# ---------------------------------------------------------------------------

class _SyntheticMesh:
    """Minimal 3-cell chain mesh: 0 -- 1 -- 2 with boundary faces.

    Geometry (unit spacing, unit area, unit depth):
    - cell centres: 0.5, 1.5, 2.5
    - internal faces: f0 (0-1 at x=1), f1 (1-2 at x=2)
    - boundary faces: f2 (left of cell 0), f3 (right of cell 2)
    """

    def __init__(self):
        self.n_cells = 3
        self.n_internal_faces = 2
        self.n_faces = 4
        self.device = "cpu"
        self.dtype = CFD_DTYPE

        self.owner = torch.tensor([0, 1, 0, 2], dtype=INDEX_DTYPE)
        self.neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)

        # Cell volumes (unit depth in z)
        self.cell_volumes = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        # Face areas: internal faces along x, boundaries also along x
        self.face_areas = torch.tensor(
            [[1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0]],
            dtype=CFD_DTYPE,
        )

        # Delta coefficients: |S_f| / d_PN  (distance between cell centres)
        self.delta_coefficients = torch.tensor(
            [1.0, 1.0, 2.0, 2.0], dtype=CFD_DTYPE,
        )

        # Face weights: owner-side interpolation weight
        self.face_weights = torch.tensor(
            [0.5, 0.5, 0.5, 0.5], dtype=CFD_DTYPE,
        )

        # Cell centres (for gradient tests)
        self.cell_centres = torch.tensor(
            [[0.5, 0.0, 0.0],
             [1.5, 0.0, 0.0],
             [2.5, 0.0, 0.0]],
            dtype=CFD_DTYPE,
        )


@pytest.fixture
def mesh():
    return _SyntheticMesh()


# ---------------------------------------------------------------------------
# assemble_pressure_equation
# ---------------------------------------------------------------------------

class TestAssemblePressureEquation:
    """Unit tests for assemble_pressure_equation."""

    def test_returns_fv_matrix(self, mesh):
        """Assembly returns an FvMatrix instance."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        assert isinstance(mat, FvMatrix)

    def test_shape_lower_upper(self, mesh):
        """lower and upper have n_internal_faces entries."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        assert mat.lower.shape == (mesh.n_internal_faces,)
        assert mat.upper.shape == (mesh.n_internal_faces,)

    def test_shape_diag(self, mesh):
        """diag has n_cells entries."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        assert mat.diag.shape == (mesh.n_cells,)

    def test_shape_source(self, mesh):
        """source has n_cells entries."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        assert mat.source.shape == (mesh.n_cells,)

    def test_zero_flux_zero_source(self, mesh):
        """Zero flux produces zero source vector."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        assert torch.allclose(mat.source, torch.zeros_like(mat.source), atol=1e-14)

    def test_source_matches_manual_divergence(self, mesh):
        """Source equals -div(phiHbyA) / V for known flux values."""
        phi = torch.tensor([0.5, -0.3, 0.0, 0.0], dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)

        # Manual: cell 0 gets -phi[0] from internal face 0, -phi[2] from bnd face 2
        # cell 1 gets +phi[0] from face 0, -phi[1] from face 1
        # cell 2 gets +phi[1] from face 1, -phi[3] from bnd face 3
        expected = torch.zeros(3, dtype=CFD_DTYPE)
        expected[0] = (-phi[0] + -phi[2]) / mesh.cell_volumes[0]
        expected[1] = (phi[0] + -phi[1]) / mesh.cell_volumes[1]
        expected[2] = (phi[1] + -phi[3]) / mesh.cell_volumes[2]
        assert torch.allclose(mat.source, expected, atol=1e-12)

    def test_diagonal_nonnegative(self, mesh):
        """Diagonal entries are non-negative for diffusion operator."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 0.5
        mat = assemble_pressure_equation(phi, A_p, mesh)
        assert (mat.diag >= 0).all()

    def test_diagonal_equals_negative_sum_offdiag(self, mesh):
        """diag[P] = -sum(off-diag per row) — conservation property."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        n_int = mesh.n_internal_faces

        from pyfoam.core.backend import scatter_add
        row_sum = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        row_sum = row_sum + scatter_add(mat.lower, mesh.owner[:n_int], mesh.n_cells)
        row_sum = row_sum + scatter_add(mat.upper, mesh.neighbour, mesh.n_cells)

        assert torch.allclose(mat.diag, -row_sum, atol=1e-12)

    def test_lower_upper_negative(self, mesh):
        """Off-diagonal coefficients are negative (Laplacian structure)."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        assert (mat.lower < 0).all()
        assert (mat.upper < 0).all()

    def test_larger_Ap_smaller_coefficients(self, mesh):
        """Larger A_p yields smaller matrix coefficients (1/A_p scaling)."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p_small = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 0.5
        A_p_large = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 2.0
        mat_small = assemble_pressure_equation(phi, A_p_small, mesh)
        mat_large = assemble_pressure_equation(phi, A_p_large, mesh)
        assert (mat_large.diag < mat_small.diag).all()

    def test_matrix_symmetric_dense(self, mesh):
        """Dense form of pressure matrix is symmetric."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        dense = mat.to_sparse_coo().to_dense()
        assert torch.allclose(dense, dense.T, atol=1e-12)

    def test_Ax_consistency(self, mesh):
        """LDU Ax product matches dense matrix-vector product."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        dense = mat.to_sparse_coo().to_dense()

        x = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        assert torch.allclose(mat.Ax(x), dense @ x, atol=1e-12)

    def test_boundary_flux_in_source(self, mesh):
        """Boundary face flux contributes to owner cell source."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        phi[2] = 7.0  # left boundary face, owner=cell 0
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        # source[0] should include -7.0 / V_0
        assert abs(mat.source[0].item() - (-7.0)) < 1e-12


# ---------------------------------------------------------------------------
# solve_pressure_equation
# ---------------------------------------------------------------------------

class TestSolvePressureEquation:
    """Unit tests for solve_pressure_equation."""

    def test_returns_tuple(self, mesh):
        """Returns (p, iterations, residual) tuple."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        p0 = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        solver = PCGSolver(tolerance=1e-6, max_iter=100)
        result = solve_pressure_equation(mat, p0, solver, tolerance=1e-6, max_iter=100)
        assert len(result) == 3

    def test_output_shape(self, mesh):
        """Solution has correct shape."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        p0 = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        solver = PCGSolver(tolerance=1e-6, max_iter=100)
        p_new, _, _ = solve_pressure_equation(mat, p0, solver, tolerance=1e-6, max_iter=100)
        assert p_new.shape == (mesh.n_cells,)

    def test_zero_flux_zero_pressure(self, mesh):
        """Zero flux gives zero pressure correction."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        p0 = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        solver = PCGSolver(tolerance=1e-10, max_iter=500)
        p_new, _, res = solve_pressure_equation(
            mat, p0, solver, tolerance=1e-10, max_iter=500,
        )
        assert torch.allclose(p_new, torch.zeros_like(p_new), atol=1e-6)

    def test_reference_cell_pinned_to_zero(self, mesh):
        """Reference cell is pinned to zero after solving."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        p0 = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        solver = PCGSolver(tolerance=1e-10, max_iter=500)
        p_new, _, _ = solve_pressure_equation(
            mat, p0, solver, tolerance=1e-10, max_iter=500, reference_cell=1,
        )
        assert abs(p_new[1].item()) < 1e-6

    def test_residual_below_tolerance(self, mesh):
        """Final residual is below specified tolerance."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        mat = assemble_pressure_equation(phi, A_p, mesh)
        p0 = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        solver = PCGSolver(tolerance=1e-8, max_iter=500)
        _, _, res = solve_pressure_equation(
            mat, p0, solver, tolerance=1e-8, max_iter=500,
        )
        assert res < 1e-8


# ---------------------------------------------------------------------------
# correct_velocity
# ---------------------------------------------------------------------------

class TestCorrectVelocity:
    """Unit tests for correct_velocity."""

    def test_output_shape(self, mesh):
        """Corrected velocity has shape (n_cells, 3)."""
        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        U_new = correct_velocity(U, HbyA, p, A_p, mesh)
        assert U_new.shape == (mesh.n_cells, 3)

    def test_zero_pressure_no_correction(self, mesh):
        """Zero pressure leaves velocity unchanged from HbyA."""
        HbyA = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=CFD_DTYPE,
        )
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        U_new = correct_velocity(HbyA, HbyA, p, A_p, mesh)
        assert torch.allclose(U_new, HbyA, atol=1e-10)

    def test_pressure_gradient_opposes_velocity(self, mesh):
        """Velocity correction opposes the pressure gradient direction.

        U = HbyA - (1/A_p) * grad(p); if grad(p) is +x, correction is -x.
        """
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = mesh.cell_centres[:, 0].clone()  # p = x, grad in +x
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        U_new = correct_velocity(HbyA, HbyA, p, A_p, mesh)
        # x-component should be negative (opposing +x gradient)
        assert (U_new[:, 0] < 0).all()

    def test_larger_Ap_smaller_correction(self, mesh):
        """Larger A_p produces smaller velocity correction magnitude."""
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = mesh.cell_centres[:, 0].clone()
        A_p_small = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        A_p_large = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 10.0

        U_small = correct_velocity(HbyA, HbyA, p, A_p_small, mesh)
        U_large = correct_velocity(HbyA, HbyA, p, A_p_large, mesh)
        assert U_large.abs().sum() < U_small.abs().sum()

    def test_finite_output(self, mesh):
        """Output is finite even with extreme pressure values."""
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.tensor([1e6, 0.0, -1e6], dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 0.01
        U_new = correct_velocity(HbyA, HbyA, p, A_p, mesh)
        assert torch.isfinite(U_new).all()


# ---------------------------------------------------------------------------
# correct_face_flux
# ---------------------------------------------------------------------------

class TestCorrectFaceFlux:
    """Unit tests for correct_face_flux."""

    def test_output_shape(self, mesh):
        """Corrected flux has shape (n_faces,)."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        phi_new = correct_face_flux(phi, p, A_p, mesh)
        assert phi_new.shape == (mesh.n_faces,)

    def test_zero_pressure_no_correction(self, mesh):
        """Zero pressure leaves flux unchanged."""
        phi = torch.tensor([0.5, -0.3, 0.1, 0.2], dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        phi_new = correct_face_flux(phi, p, A_p, mesh)
        assert torch.allclose(phi_new, phi, atol=1e-10)

    def test_boundary_faces_unchanged(self, mesh):
        """Boundary face fluxes are not modified by the correction."""
        phi = torch.tensor([0.5, -0.3, 1.0, 2.0], dtype=CFD_DTYPE)
        p = torch.tensor([1.0, 0.0, 0.0], dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        phi_new = correct_face_flux(phi, p, A_p, mesh)
        # Boundary faces (indices 2, 3) should be unchanged
        assert torch.allclose(phi_new[2:], phi[2:], atol=1e-12)

    def test_nonzero_pressure_corrects_internal_flux(self, mesh):
        """Non-zero pressure difference modifies internal face fluxes."""
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        p = torch.tensor([1.0, 0.0, 0.0], dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        phi_new = correct_face_flux(phi, p, A_p, mesh)
        # Face 0: owner=0, neigh=1, p_P=1, p_N=0, correction is non-zero
        assert phi_new[0].abs() > 1e-10

    def test_pressure_sign_consistency(self, mesh):
        """Correction sign matches physical expectation.

        If p_P > p_N, dp = p_P - p_N > 0, correction is positive,
        increasing the flux from owner to neighbour.
        """
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        p = torch.tensor([2.0, 0.0, 0.0], dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        phi_new = correct_face_flux(phi, p, A_p, mesh)
        # Face 0: dp = 2.0 - 0.0 = 2.0, correction > 0
        assert phi_new[0] > 0

    def test_does_not_mutate_input(self, mesh):
        """Original flux tensor is not modified in-place."""
        phi = torch.tensor([0.5, -0.3, 0.1, 0.2], dtype=CFD_DTYPE)
        phi_clone = phi.clone()
        p = torch.tensor([1.0, 0.0, 0.0], dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        _ = correct_face_flux(phi, p, A_p, mesh)
        assert torch.allclose(phi, phi_clone)
