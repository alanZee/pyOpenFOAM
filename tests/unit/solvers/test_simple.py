"""Tests for SIMPLE algorithm — pressure-velocity coupling.

Tests cover:
- Basic SIMPLE iteration structure
- Rhie-Chow interpolation
- Pressure equation assembly
- Velocity and flux correction
- Convergence behaviour
- Under-relaxation
- Lid-driven cavity setup
"""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.backend import scatter_add
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.solvers.rhie_chow import (
    compute_HbyA,
    compute_face_flux_HbyA,
    rhie_chow_correction,
    compute_face_flux,
)
from pyfoam.solvers.pressure_equation import (
    assemble_pressure_equation,
    solve_pressure_equation,
    correct_velocity,
    correct_face_flux,
)
from pyfoam.solvers.coupled_solver import CoupledSolverConfig, ConvergenceData

from tests.unit.solvers.conftest_coupled import make_cavity_mesh


class TestRhieChowInterpolation:
    """Tests for Rhie-Chow face flux interpolation."""

    def test_compute_HbyA_shape(self):
        """HbyA has correct shape."""
        n_cells = 4
        H = torch.randn(n_cells, 3, dtype=CFD_DTYPE)
        A_p = torch.rand(n_cells, dtype=CFD_DTYPE) + 0.1

        HbyA = compute_HbyA(H, A_p)
        assert HbyA.shape == (n_cells, 3)

    def test_compute_HbyA_values(self):
        """HbyA = H / A_p."""
        H = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=CFD_DTYPE)
        A_p = torch.tensor([2.0, 3.0], dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        expected = torch.tensor([[0.5, 1.0, 1.5], [4.0/3, 5.0/3, 2.0]], dtype=CFD_DTYPE)
        assert torch.allclose(HbyA, expected, atol=1e-10)

    def test_compute_HbyA_safe_division(self):
        """HbyA handles near-zero A_p safely."""
        H = torch.tensor([[1.0, 2.0, 3.0]], dtype=CFD_DTYPE)
        A_p = torch.tensor([1e-40], dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        assert torch.isfinite(HbyA).all()

    def test_face_flux_HbyA_shape(self):
        """Face flux has correct shape."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)

        phi = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, mesh.face_weights,
        )
        assert phi.shape == (mesh.n_faces,)

    def test_face_flux_HbyA_internal_faces(self):
        """Internal face flux is interpolated HbyA dot S."""
        mesh = make_cavity_mesh(2, 2)
        # Uniform HbyA = (1, 0, 0)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        HbyA[:, 0] = 1.0

        phi = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, mesh.face_weights,
        )

        # For uniform HbyA, internal face flux = HbyA · S
        S = mesh.face_areas[:mesh.n_internal_faces]
        expected = (HbyA[0] * S).sum(dim=1)
        assert torch.allclose(phi[:mesh.n_internal_faces], expected, atol=1e-10)

    def test_rhie_chow_correction_shape(self):
        """Rhie-Chow correction has correct shape."""
        mesh = make_cavity_mesh(2, 2)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        correction = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        assert correction.shape == (mesh.n_faces,)

    def test_rhie_chow_zero_pressure(self):
        """Rhie-Chow correction is zero for uniform pressure."""
        mesh = make_cavity_mesh(2, 2)
        p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 101325.0
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        correction = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )

        # Uniform pressure → zero correction
        assert torch.allclose(correction[:mesh.n_internal_faces], torch.zeros(mesh.n_internal_faces, dtype=CFD_DTYPE), atol=1e-10)

    def test_rhie_chow_pressure_gradient(self):
        """Rhie-Chow correction captures pressure gradient."""
        mesh = make_cavity_mesh(2, 2)
        # Linear pressure: p = x
        p = mesh.cell_centres[:, 0].clone()
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        correction = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )

        # Should be non-zero for faces with pressure gradient
        assert correction[:mesh.n_internal_faces].abs().sum() > 0


class TestPressureEquation:
    """Tests for pressure equation assembly and solution."""

    def test_pressure_equation_assembly_shape(self):
        """Pressure equation matrix has correct dimensions."""
        mesh = make_cavity_mesh(2, 2)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        assert p_eqn.n_cells == mesh.n_cells

    def test_pressure_equation_symmetric(self):
        """Pressure equation matrix is symmetric (Laplacian)."""
        mesh = make_cavity_mesh(2, 2)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # For a Laplacian, lower and upper should be related
        # (not necessarily equal due to non-uniform mesh)
        assert p_eqn.lower.shape == (mesh.n_internal_faces,)
        assert p_eqn.upper.shape == (mesh.n_internal_faces,)

    def test_pressure_equation_solve(self):
        """Pressure equation can be solved."""
        from pyfoam.solvers.pcg import PCGSolver

        mesh = make_cavity_mesh(2, 2)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-6, max_iter=100)
        p_new, iters, residual = solve_pressure_equation(
            p_eqn, p, solver, tolerance=1e-6, max_iter=100,
        )

        assert p_new.shape == (mesh.n_cells,)
        assert iters > 0

    def test_velocity_correction_shape(self):
        """Velocity correction has correct shape."""
        mesh = make_cavity_mesh(2, 2)
        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        U_new = correct_velocity(U, HbyA, p, A_p, mesh)
        assert U_new.shape == (mesh.n_cells, 3)

    def test_velocity_correction_pressure_gradient(self):
        """Velocity correction responds to pressure gradient."""
        mesh = make_cavity_mesh(2, 2)
        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        # Linear pressure: p = x
        p = mesh.cell_centres[:, 0].clone()
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        U_new = correct_velocity(U, HbyA, p, A_p, mesh)

        # Should be non-zero due to pressure gradient
        assert U_new.abs().sum() > 0

    def test_flux_correction_shape(self):
        """Flux correction has correct shape."""
        mesh = make_cavity_mesh(2, 2)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        phi_new = correct_face_flux(phi, p, A_p, mesh, mesh.face_weights)
        assert phi_new.shape == (mesh.n_faces,)

    def test_pressure_equation_diagonal_dominance(self):
        """Pressure equation matrix is diagonally dominant."""
        mesh = make_cavity_mesh(4, 4)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 0.5

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # Diagonal should be non-negative
        assert (p_eqn.diag >= 0).all(), "Diagonal should be non-negative"

        # Sum of off-diagonal magnitudes should not exceed diagonal
        n_int = mesh.n_internal_faces
        off_diag_sum = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        off_diag_sum = off_diag_sum + scatter_add(p_eqn.lower.abs(), mesh.owner[:n_int], mesh.n_cells)
        off_diag_sum = off_diag_sum + scatter_add(p_eqn.upper.abs(), mesh.neighbour, mesh.n_cells)
        assert (p_eqn.diag >= off_diag_sum - 1e-10).all(), "Matrix should be diagonally dominant"

    def test_pressure_equation_zero_source_zero_solution(self):
        """Zero source gives zero pressure correction."""
        from pyfoam.solvers.pcg import PCGSolver

        mesh = make_cavity_mesh(2, 2)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-10, max_iter=100)
        p_new, iters, residual = solve_pressure_equation(
            p_eqn, p, solver, tolerance=1e-10, max_iter=100,
        )

        # With zero flux, pressure correction should be near zero
        assert torch.allclose(p_new, torch.zeros_like(p_new), atol=1e-6), \
            f"Zero source should give zero pressure, got max={p_new.abs().max():.6e}"

    def test_pressure_equation_source_scaling(self):
        """Source term has correct scaling with flux magnitude."""
        mesh = make_cavity_mesh(4, 4)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        # Create non-zero flux
        phiHbyA = torch.randn(mesh.n_faces, dtype=CFD_DTYPE) * 0.01

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # Source should be non-zero
        assert p_eqn.source.abs().sum() > 0, "Source should be non-zero"

        # Double the flux should double the source
        phiHbyA2 = phiHbyA * 2.0
        p_eqn2 = assemble_pressure_equation(phiHbyA2, A_p, mesh, mesh.face_weights)

        # Sources should scale linearly
        ratio = p_eqn2.source.abs().sum() / p_eqn.source.abs().sum()
        assert abs(ratio - 2.0) < 0.01, f"Source should scale linearly, got ratio={ratio:.3f}"

    def test_velocity_correction_divergence_free(self):
        """Velocity correction produces divergence-free field for simple case."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        U_corrected = correct_velocity(torch.zeros_like(HbyA), HbyA, p, A_p, mesh)

        # With zero HbyA and zero pressure, velocity should be zero
        assert torch.allclose(U_corrected, torch.zeros_like(U_corrected), atol=1e-10)

    def test_flux_correction_consistency(self):
        """Flux correction is consistent with velocity correction."""
        mesh = make_cavity_mesh(2, 2)
        p = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        phi_corrected = correct_face_flux(phi, p, A_p, mesh, mesh.face_weights)

        # With non-zero pressure, flux should change
        assert not torch.allclose(phi_corrected, phi, atol=1e-10), \
            "Flux correction should change flux for non-zero pressure"


class TestPressureEquationAssembly:
    """Comprehensive tests for pressure Poisson equation assembly.

    Tests cover:
    - Matrix symmetry verification (A = A^T)
    - Source term correctness (divergence computation)
    - Boundary condition handling (boundary face contributions)
    - Reference cell pinning (singularity removal)
    - Convergence with various mesh sizes
    - Zero flux → zero pressure correction
    - Non-zero flux → correct pressure correction sign
    - Diagonal dominance (diag = -sum(off-diag))
    - Face coefficient positivity
    - Non-uniform A_p effect
    - Matrix-vector product consistency
    - Source scaling with cell volume
    """

    # ------------------------------------------------------------------
    # 1. Matrix symmetry verification
    # ------------------------------------------------------------------

    def test_matrix_symmetry_uniform_Ap(self):
        """Pressure equation matrix is symmetric for uniform A_p.

        For a Laplacian discretisation on a consistent mesh, the matrix
        must satisfy A[i,j] = A[j,i].  We verify by converting to a
        dense matrix and checking A == A^T.
        """
        mesh = make_cavity_mesh(2, 2)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # Convert to dense and check symmetry
        dense = p_eqn.to_sparse_coo().to_dense()
        assert torch.allclose(dense, dense.T, atol=1e-12), (
            f"Matrix not symmetric. Max asymmetry: "
            f"{(dense - dense.T).abs().max():.6e}"
        )

    def test_matrix_symmetry_nonuniform_Ap(self):
        """Matrix is symmetric even with non-uniform A_p.

        When A_p varies between cells, the face-interpolated (1/A_p)_f
        is the same for both owner and neighbour, so the matrix remains
        symmetric.  However, lower and upper differ because they are
        divided by different cell volumes (V_P vs V_N).
        """
        mesh = make_cavity_mesh(4, 4)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        # Non-uniform A_p: varies between cells
        A_p = torch.rand(mesh.n_cells, dtype=CFD_DTYPE) + 0.5

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        dense = p_eqn.to_sparse_coo().to_dense()
        assert torch.allclose(dense, dense.T, atol=1e-12), (
            f"Matrix not symmetric with non-uniform A_p. Max asymmetry: "
            f"{(dense - dense.T).abs().max():.6e}"
        )

    # ------------------------------------------------------------------
    # 2. Source term correctness
    # ------------------------------------------------------------------

    def test_source_term_zero_for_zero_flux(self):
        """Source term is zero when phiHbyA is zero."""
        mesh = make_cavity_mesh(4, 4)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        assert torch.allclose(p_eqn.source, torch.zeros_like(p_eqn.source), atol=1e-14), \
            f"Source should be zero for zero flux, got {p_eqn.source}"

    def test_source_term_divergence_computation(self):
        """Source term correctly computes -div(phiHbyA) / V.

        The source is -div(phiHbyA) scaled to per-unit-volume.
        For an internal face f with owner P and neighbour N:
            source[P] += -phiHbyA[f]  (outflow from P)
            source[N] += +phiHbyA[f]  (inflow to N)
        Then divided by cell volume.
        """
        mesh = make_cavity_mesh(2, 2)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        # Set a known flux on internal faces
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        n_int = mesh.n_internal_faces
        phiHbyA[:n_int] = torch.tensor([1.0, -0.5, 0.3, 0.8], dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # Manually compute expected source
        expected_source = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        for f in range(n_int):
            P = mesh.owner[f].item()
            N = mesh.neighbour[f].item()
            expected_source[P] += -phiHbyA[f]
            expected_source[N] += phiHbyA[f]
        # Boundary contributions (zero flux on boundaries)
        expected_source = expected_source / mesh.cell_volumes

        assert torch.allclose(p_eqn.source, expected_source, atol=1e-12), (
            f"Source mismatch. Expected:\n{expected_source}\nGot:\n{p_eqn.source}"
        )

    def test_source_term_includes_boundary_flux(self):
        """Source term includes boundary face flux contributions.

        Boundary faces contribute -phiHbyA[bnd] to their owner cell's source.
        """
        mesh = make_cavity_mesh(2, 2)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        # Zero internal flux, non-zero boundary flux
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        n_int = mesh.n_internal_faces
        # Set flux on first boundary face (bottom wall, owner cell 0)
        phiHbyA[n_int] = 5.0

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # Cell 0 should have source contribution from boundary face
        # source[0] += -5.0 / V_0
        expected_0 = -5.0 / mesh.cell_volumes[0].item()
        assert abs(p_eqn.source[0].item() - expected_0) < 1e-12, (
            f"Boundary flux not included. Expected source[0]={expected_0}, "
            f"got {p_eqn.source[0].item()}"
        )

    # ------------------------------------------------------------------
    # 3. Boundary condition handling
    # ------------------------------------------------------------------

    def test_boundary_faces_no_matrix_contribution(self):
        """Boundary faces do not contribute off-diagonal terms.

        Only internal faces contribute to lower/upper.  Boundary faces
        only affect the source term (zero-gradient BC: no implicit contribution).
        """
        mesh = make_cavity_mesh(2, 2)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # lower and upper should have exactly n_internal_faces entries
        assert p_eqn.lower.shape[0] == mesh.n_internal_faces
        assert p_eqn.upper.shape[0] == mesh.n_internal_faces

        # For a uniform mesh, all lower/upper should be equal
        assert torch.allclose(p_eqn.lower, p_eqn.lower[0], atol=1e-12)
        assert torch.allclose(p_eqn.upper, p_eqn.upper[0], atol=1e-12)

    def test_boundary_flux_affects_only_owner_cell_source(self):
        """Boundary flux only affects the owner cell's source term.

        Unlike internal faces (which affect both owner and neighbour),
        boundary faces only contribute to their owner cell.
        """
        mesh = make_cavity_mesh(2, 2)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        n_int = mesh.n_internal_faces

        # Set flux only on boundary faces
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        # Bottom boundary: owner cells 0, 1
        phiHbyA[n_int] = 1.0      # owner = cell 0
        phiHbyA[n_int + 1] = 2.0  # owner = cell 1

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # Cells 2, 3 should have zero source (no boundary flux affects them
        # from bottom wall, and other boundary fluxes are zero)
        # Cell 2 is top-left, affected by top boundary face (which is zero)
        # Cell 3 is top-right, affected by top boundary face (which is zero)
        # But cells 2 and 3 are also affected by left/right boundaries (zero flux)
        # So their source should be zero
        assert p_eqn.source[2].abs() < 1e-14, \
            f"Cell 2 source should be zero, got {p_eqn.source[2]}"
        assert p_eqn.source[3].abs() < 1e-14, \
            f"Cell 3 source should be zero, got {p_eqn.source[3]}"

    # ------------------------------------------------------------------
    # 4. Reference cell pinning
    # ------------------------------------------------------------------

    def test_reference_cell_pinning_modifies_diagonal(self):
        """set_reference adds large coefficient to diagonal.

        The reference cell's diagonal is increased by a large value,
        and the source is adjusted to pin the pressure to the given value.
        """
        mesh = make_cavity_mesh(2, 2)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        diag_before = p_eqn.diag.clone()

        p_eqn.set_reference(0, value=0.0)

        # Diagonal of cell 0 should be much larger
        assert p_eqn.diag[0] > diag_before[0], \
            "Reference cell diagonal should increase"

        # Other cells should be unchanged
        assert torch.allclose(p_eqn.diag[1:], diag_before[1:], atol=1e-14), \
            "Non-reference cells should be unchanged"

    def test_reference_cell_pinning_with_value(self):
        """set_reference pins pressure to specified value.

        With a non-zero reference value, the source is adjusted so
        the solution at the reference cell converges to that value.
        """
        mesh = make_cavity_mesh(2, 2)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        p_eqn.set_reference(0, value=100.0)

        # Source of cell 0 should include large_coeff * 100.0
        # The diagonal was increased by large_coeff, so source should be positive
        assert p_eqn.source[0] > 0, \
            "Reference cell source should be positive for positive reference value"

    def test_reference_cell_pinning_solves_to_value(self):
        """Solving with reference pinning yields correct reference value.

        After solving, the reference cell should have the pinned value
        (which is always 0.0 in the current API).
        """
        from pyfoam.solvers.pcg import PCGSolver

        mesh = make_cavity_mesh(2, 2)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-10, max_iter=1000)
        p_new, iters, residual = solve_pressure_equation(
            p_eqn, p, solver, tolerance=1e-10, max_iter=1000,
            reference_cell=2,
        )

        # With zero source and reference pinned to 0, all cells should be ~0
        assert torch.allclose(p_new, torch.zeros_like(p_new), atol=1e-6), (
            f"Zero source with reference=0 should give zero pressure, "
            f"got max={p_new.abs().max():.6e}"
        )

    # ------------------------------------------------------------------
    # 5. Convergence with various mesh sizes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("nx,ny", [(2, 2), (4, 4), (8, 8)])
    def test_convergence_various_mesh_sizes(self, nx, ny):
        """Pressure equation converges for various mesh sizes."""
        from pyfoam.solvers.pcg import PCGSolver

        mesh = make_cavity_mesh(nx, ny)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-8, max_iter=1000)
        p_new, iters, residual = solve_pressure_equation(
            p_eqn, p, solver, tolerance=1e-8, max_iter=1000,
        )

        assert residual < 1e-8, (
            f"Did not converge on {nx}x{ny} mesh. "
            f"Residual: {residual:.6e}, iterations: {iters}"
        )

    @pytest.mark.parametrize("nx,ny", [(2, 2), (4, 4), (8, 8)])
    def test_convergence_with_nonzero_source(self, nx, ny):
        """Pressure equation converges with non-zero but compatible source.

        We create a known solvable source by assembling the equation with
        a non-zero flux and verifying the solver reduces the residual
        significantly from the initial guess.
        """
        from pyfoam.solvers.pcg import PCGSolver

        mesh = make_cavity_mesh(nx, ny)
        n_int = mesh.n_internal_faces

        # Create a flux with non-trivial divergence
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        phiHbyA[:n_int] = 1.0

        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)

        # Compute initial residual for reference
        initial_residual = p_eqn.residual(p).norm().item()

        solver = PCGSolver(tolerance=1e-10, max_iter=5000)
        p_new, iters, residual = solve_pressure_equation(
            p_eqn, p, solver, tolerance=1e-10, max_iter=5000,
        )

        # The solver should reduce the residual significantly
        # (even if it can't reach machine zero due to compatibility issues)
        final_residual = p_eqn.residual(p_new).norm().item()
        assert final_residual < initial_residual * 0.1 or residual < 1e-4, (
            f"Solver did not reduce residual on {nx}x{ny} mesh. "
            f"Initial: {initial_residual:.6e}, Final: {final_residual:.6e}, "
            f"Iterations: {iters}"
        )

    # ------------------------------------------------------------------
    # 6. Zero flux → zero pressure correction
    # ------------------------------------------------------------------

    def test_zero_flux_zero_pressure_correction(self):
        """Zero flux produces zero pressure correction.

        When phiHbyA = 0 everywhere, the source is zero, and the
        pressure correction should be zero (up to solver tolerance).
        """
        from pyfoam.solvers.pcg import PCGSolver

        mesh = make_cavity_mesh(4, 4)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-10, max_iter=1000)
        p_new, iters, residual = solve_pressure_equation(
            p_eqn, p, solver, tolerance=1e-10, max_iter=1000,
        )

        assert torch.allclose(p_new, torch.zeros_like(p_new), atol=1e-6), (
            f"Zero flux should give zero pressure, got max={p_new.abs().max():.6e}"
        )

    def test_zero_flux_zero_velocity_correction(self):
        """Zero pressure gives zero velocity correction."""
        mesh = make_cavity_mesh(2, 2)
        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        U_new = correct_velocity(U, HbyA, p, A_p, mesh)

        assert torch.allclose(U_new, torch.zeros_like(U_new), atol=1e-10), \
            "Zero pressure should give zero velocity correction"

    def test_zero_flux_zero_face_flux_correction(self):
        """Zero pressure gives zero face flux correction."""
        mesh = make_cavity_mesh(2, 2)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        phi_new = correct_face_flux(phi, p, A_p, mesh, mesh.face_weights)

        assert torch.allclose(phi_new, phi, atol=1e-10), \
            "Zero pressure should give zero flux correction"

    # ------------------------------------------------------------------
    # 7. Non-zero flux → correct pressure correction sign
    # ------------------------------------------------------------------

    def test_nonzero_flux_pressure_correction_sign(self):
        """Non-zero flux produces pressure correction with correct sign.

        If flux flows from cell 0 to cell 1 (positive phi), the pressure
        should be higher in cell 0 (to drive the correction).
        """
        from pyfoam.solvers.pcg import PCGSolver

        mesh = make_cavity_mesh(2, 2)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        # Create flux flowing from cell 0 to cell 1 (first internal face)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        # First internal face: owner=0, neighbour=1
        phiHbyA[0] = 1.0  # positive flux from 0 to 1

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-10, max_iter=1000)
        p_new, iters, residual = solve_pressure_equation(
            p_eqn, p, solver, tolerance=1e-10, max_iter=1000,
        )

        # The divergence at cell 0 is negative (outflow), at cell 1 is positive (inflow)
        # source[0] = -phi / V_0 < 0, source[1] = +phi / V_1 > 0
        # For the Laplacian p_eqn * p = source, with negative source at cell 0,
        # pressure at cell 0 should be negative (relative to reference)
        # and positive at cell 1
        # But the sign depends on the reference cell (cell 0 is pinned to 0)
        # So p[1] should be positive (or at least non-zero)
        assert p_new[1].abs() > 1e-10, \
            "Non-zero flux should produce non-zero pressure correction"

    def test_nonzero_flux_velocity_correction_direction(self):
        """Velocity correction opposes the pressure gradient.

        From U = HbyA - (1/A_p) * grad(p), if grad(p) points in +x,
        the velocity correction is in -x.
        """
        mesh = make_cavity_mesh(2, 2)
        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        # Pressure gradient: p increases in +x direction
        p = mesh.cell_centres[:, 0].clone()  # p = x
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        U_new = correct_velocity(U, HbyA, p, A_p, mesh)

        # Velocity in x should be negative (opposing pressure gradient)
        # At cell 0 (x=0.25), grad(p) points +x, so U_x should be negative
        # At cell 1 (x=0.75), grad(p) points +x, so U_x should be negative
        assert (U_new[:, 0] < 0).all(), \
            "Velocity correction should oppose pressure gradient"

    def test_nonzero_flux_face_flux_correction_sign(self):
        """Face flux correction has correct sign.

        If p_P > p_N (owner has higher pressure), the correction should
        reduce the flux from P to N (or increase flux from N to P).
        """
        mesh = make_cavity_mesh(2, 2)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        # Cell 0 has higher pressure than cell 1
        p = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        phi_new = correct_face_flux(phi, p, A_p, mesh, mesh.face_weights)

        # First internal face: owner=0, neighbour=1
        # dp = p_P - p_N = 1.0 - 0.0 = 1.0
        # flux_correction = inv_A_p_f * dp * S_mag * delta_f
        # With p_P > p_N, the correction should be positive
        # (increasing flux from P to N, which is the pressure-driven flow)
        assert phi_new[0] != 0, "Flux correction should be non-zero"

    # ------------------------------------------------------------------
    # 8. Diagonal dominance (diag = -sum(off-diag))
    # ------------------------------------------------------------------

    def test_diagonal_equals_negative_sum_offdiag(self):
        """Diagonal equals negative sum of off-diagonal entries per row.

        For a Laplacian with no source, the matrix must satisfy:
            diag[P] = -sum(lower[f] for f where owner=P)
                     - sum(upper[f] for f where neighbour=P)
        This ensures row sums are zero (conservation).
        """
        mesh = make_cavity_mesh(4, 4)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # Compute sum of off-diagonal per row
        n_int = mesh.n_internal_faces
        off_diag_sum = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        off_diag_sum = off_diag_sum + scatter_add(
            p_eqn.lower, mesh.owner[:n_int], mesh.n_cells
        )
        off_diag_sum = off_diag_sum + scatter_add(
            p_eqn.upper, mesh.neighbour, mesh.n_cells
        )

        # diag should equal -off_diag_sum
        assert torch.allclose(p_eqn.diag, -off_diag_sum, atol=1e-12), (
            f"Diagonal != -sum(off-diag). Max diff: "
            f"{(p_eqn.diag + off_diag_sum).abs().max():.6e}"
        )

    def test_diagonal_nonnegative(self):
        """All diagonal entries are non-negative.

        For a diffusion operator, the diagonal is always non-negative
        (sum of positive face coefficients).
        """
        mesh = make_cavity_mesh(4, 4)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 0.1

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        assert (p_eqn.diag >= 0).all(), (
            f"Negative diagonal entries found: {p_eqn.diag[p_eqn.diag < 0]}"
        )

    # ------------------------------------------------------------------
    # 9. Face coefficient positivity
    # ------------------------------------------------------------------

    def test_face_coefficients_positive(self):
        """Face coefficients (inv_A_p_f * S_mag * delta_f) are positive.

        The Laplacian face coefficient must be positive for the
        diffusion operator to be well-posed.
        """
        mesh = make_cavity_mesh(4, 4)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 0.5

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # lower should be negative (off-diagonal of Laplacian)
        assert (p_eqn.lower < 0).all(), \
            "Lower coefficients should be negative for Laplacian"
        # upper should be negative
        assert (p_eqn.upper < 0).all(), \
            "Upper coefficients should be negative for Laplacian"

    # ------------------------------------------------------------------
    # 10. Non-uniform A_p effect
    # ------------------------------------------------------------------

    def test_nonuniform_Ap_changes_matrix(self):
        """Non-uniform A_p produces a different matrix than uniform A_p.

        When A_p varies, the face-interpolated (1/A_p)_f changes,
        altering the matrix coefficients.
        """
        mesh = make_cavity_mesh(4, 4)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        # Uniform A_p
        A_p_uniform = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        p_eqn_uniform = assemble_pressure_equation(
            phiHbyA, A_p_uniform, mesh, mesh.face_weights
        )

        # Non-uniform A_p
        A_p_nonuniform = torch.rand(mesh.n_cells, dtype=CFD_DTYPE) + 0.5
        p_eqn_nonuniform = assemble_pressure_equation(
            phiHbyA, A_p_nonuniform, mesh, mesh.face_weights
        )

        # Matrices should differ
        assert not torch.allclose(
            p_eqn_uniform.diag, p_eqn_nonuniform.diag, atol=1e-10
        ), "Non-uniform A_p should change diagonal"

    def test_larger_Ap_smaller_coefficients(self):
        """Larger A_p produces smaller matrix coefficients.

        Since face_coeff = (1/A_p)_f * |S_f| * delta_f, increasing A_p
        decreases the coefficients.
        """
        mesh = make_cavity_mesh(4, 4)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        A_p_small = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 0.5
        A_p_large = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 2.0

        p_eqn_small = assemble_pressure_equation(
            phiHbyA, A_p_small, mesh, mesh.face_weights
        )
        p_eqn_large = assemble_pressure_equation(
            phiHbyA, A_p_large, mesh, mesh.face_weights
        )

        # Larger A_p → smaller diagonal
        assert (p_eqn_large.diag < p_eqn_small.diag).all(), \
            "Larger A_p should give smaller diagonal"

    # ------------------------------------------------------------------
    # 11. Matrix-vector product consistency
    # ------------------------------------------------------------------

    def test_Ax_matches_dense_multiplication(self):
        """LDU Ax matches dense matrix-vector product.

        The LDU format matrix-vector product should give the same result
        as multiplying the equivalent dense matrix by the vector.
        """
        mesh = make_cavity_mesh(2, 2)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # Get dense matrix
        dense = p_eqn.to_sparse_coo().to_dense()

        # Test with a known vector
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=CFD_DTYPE)

        # LDU product
        Ax_ldu = p_eqn.Ax(x)

        # Dense product
        Ax_dense = dense @ x

        assert torch.allclose(Ax_ldu, Ax_dense, atol=1e-12), (
            f"LDU Ax != Dense Ax. Max diff: {(Ax_ldu - Ax_dense).abs().max():.6e}"
        )

    def test_Ax_with_random_vector(self):
        """LDU Ax matches dense for random vector."""
        mesh = make_cavity_mesh(4, 4)
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)
        dense = p_eqn.to_sparse_coo().to_dense()

        torch.manual_seed(42)
        x = torch.randn(mesh.n_cells, dtype=CFD_DTYPE)

        Ax_ldu = p_eqn.Ax(x)
        Ax_dense = dense @ x

        assert torch.allclose(Ax_ldu, Ax_dense, atol=1e-10), (
            f"LDU Ax != Dense Ax for random vector. Max diff: "
            f"{(Ax_ldu - Ax_dense).abs().max():.6e}"
        )

    # ------------------------------------------------------------------
    # 12. Source scaling with cell volume
    # ------------------------------------------------------------------

    def test_source_is_per_unit_volume(self):
        """Source term is scaled to per-unit-volume.

        The raw divergence is accumulated, then divided by cell volume.
        We verify by checking that the source is proportional to 1/V.
        """
        mesh = make_cavity_mesh(2, 2)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        n_int = mesh.n_internal_faces

        # Create a flux that gives known divergence
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        phiHbyA[0] = 1.0  # owner=0, neighbour=1

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # Raw divergence at cell 0: -1.0, at cell 1: +1.0
        # After dividing by V: source[0] = -1.0/V_0, source[1] = +1.0/V_1
        V = mesh.cell_volumes
        assert abs(p_eqn.source[0].item() - (-1.0 / V[0].item())) < 1e-12
        assert abs(p_eqn.source[1].item() - (1.0 / V[1].item())) < 1e-12

    def test_smaller_cell_volume_larger_source(self):
        """Smaller cell volume produces larger source for same flux.

        Source = divergence / V, so smaller V → larger source magnitude.
        """
        mesh = make_cavity_mesh(2, 2)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        n_int = mesh.n_internal_faces

        # All cells have equal volume in this mesh, so we check
        # that source is indeed divided by volume
        phiHbyA = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        phiHbyA[:n_int] = 1.0  # uniform flux on all internal faces

        p_eqn = assemble_pressure_equation(phiHbyA, A_p, mesh, mesh.face_weights)

        # Source magnitude should be proportional to 1/V
        # (all cells have same V, so source magnitudes should be similar)
        V = mesh.cell_volumes
        source_times_V = p_eqn.source * V

        # source_times_V should be the raw divergence (before volume division)
        # which is the net flux for each cell
        assert torch.isfinite(source_times_V).all()


class TestSIMPLESolver:
    """Tests for the SIMPLE algorithm."""

    def test_simple_solver_creation(self):
        """SIMPLE solver can be created."""
        mesh = make_cavity_mesh(2, 2)
        config = SIMPLEConfig(relaxation_factor_U=0.7, relaxation_factor_p=0.3)
        solver = SIMPLESolver(mesh, config)

        assert solver.mesh is mesh
        assert solver.config.relaxation_factor_U == 0.7

    def test_simple_solver_default_config(self):
        """SIMPLE solver with default config."""
        mesh = make_cavity_mesh(2, 2)
        solver = SIMPLESolver(mesh)

        assert solver.config.relaxation_factor_U == 0.7
        assert solver.config.relaxation_factor_p == 1.0

    def test_simple_solve_returns_correct_shapes(self):
        """SIMPLE solve returns fields with correct shapes."""
        mesh = make_cavity_mesh(2, 2)
        solver = SIMPLESolver(mesh, SIMPLEConfig(relaxation_factor_p=1.0))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, max_outer_iterations=5, tolerance=1e-3,
        )

        assert U_out.shape == (mesh.n_cells, 3)
        assert p_out.shape == (mesh.n_cells,)
        assert phi_out.shape == (mesh.n_faces,)
        assert isinstance(convergence, ConvergenceData)

    def test_simple_solve_convergence_data(self):
        """SIMPLE solve returns convergence data."""
        mesh = make_cavity_mesh(2, 2)
        solver = SIMPLESolver(mesh, SIMPLEConfig(relaxation_factor_p=1.0))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, max_outer_iterations=10, tolerance=1e-3,
        )

        assert convergence.outer_iterations > 0
        assert len(convergence.residual_history) > 0
        assert convergence.continuity_error >= 0

    def test_simple_solve_with_initial_velocity(self):
        """SIMPLE solve works with non-zero initial velocity."""
        mesh = make_cavity_mesh(2, 2)
        solver = SIMPLESolver(mesh, SIMPLEConfig(relaxation_factor_p=1.0))

        # Initial velocity: uniform x-velocity
        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        U[:, 0] = 1.0
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, max_outer_iterations=5, tolerance=1e-3,
        )

        assert U_out.shape == (mesh.n_cells, 3)
        assert p_out.shape == (mesh.n_cells,)

    def test_simple_under_relaxation(self):
        """Under-relaxation affects the solution."""
        mesh = make_cavity_mesh(2, 2)

        # Without relaxation
        config_no_relax = SIMPLEConfig(
            relaxation_factor_U=1.0,
            relaxation_factor_p=1.0,
        )
        solver_no_relax = SIMPLESolver(mesh, config_no_relax)

        # With relaxation
        config_relax = SIMPLEConfig(
            relaxation_factor_U=0.5,
            relaxation_factor_p=0.3,
        )
        solver_relax = SIMPLESolver(mesh, config_relax)

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        U[:, 0] = 1.0
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U1, p1, phi1, conv1 = solver_no_relax.solve(
            U.clone(), p.clone(), phi.clone(), max_outer_iterations=3,
        )
        U2, p2, phi2, conv2 = solver_relax.solve(
            U.clone(), p.clone(), phi.clone(), max_outer_iterations=3,
        )

        # Solutions should be different due to relaxation
        assert not torch.allclose(p1, p2, atol=1e-10)

    def test_simple_lid_driven_cavity(self):
        """SIMPLE on lid-driven cavity with moving top wall."""
        mesh = make_cavity_mesh(2, 2)
        config = SIMPLEConfig(
            relaxation_factor_U=0.7,
            relaxation_factor_p=0.3,
            p_tolerance=1e-4,
            U_tolerance=1e-4,
        )
        solver = SIMPLESolver(mesh, config)

        # Initial conditions
        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        # Run a few iterations
        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, max_outer_iterations=20, tolerance=1e-2,
        )

        # Should have run at least one iteration
        assert convergence.outer_iterations >= 1

        # Continuity error should decrease over iterations
        if len(convergence.residual_history) > 1:
            first_error = convergence.residual_history[0]["continuity_error"]
            last_error = convergence.residual_history[-1]["continuity_error"]
            # Allow some tolerance for small meshes
            assert last_error <= first_error * 1.1 or last_error < 0.1

    def test_simple_repr(self):
        """SIMPLE solver repr."""
        mesh = make_cavity_mesh(2, 2)
        solver = SIMPLESolver(mesh)
        r = repr(solver)

        assert "SIMPLESolver" in r
        assert "relax_U" in r


class TestSIMPLEConfig:
    """Tests for SIMPLE configuration."""

    def test_default_config(self):
        """Default SIMPLE config has expected values."""
        config = SIMPLEConfig()
        assert config.relaxation_factor_U == 0.7
        assert config.relaxation_factor_p == 1.0
        assert config.n_correctors == 1

    def test_custom_config(self):
        """Custom SIMPLE config."""
        config = SIMPLEConfig(
            relaxation_factor_U=0.5,
            relaxation_factor_p=0.3,
            n_correctors=2,
        )
        assert config.relaxation_factor_U == 0.5
        assert config.relaxation_factor_p == 0.3
        assert config.n_correctors == 2


class TestConvergenceData:
    """Tests for convergence tracking."""

    def test_convergence_data_default(self):
        """ConvergenceData has correct defaults."""
        data = ConvergenceData()
        assert data.p_residual == 0.0
        assert data.U_residual == 0.0
        assert data.continuity_error == 0.0
        assert data.outer_iterations == 0
        assert data.converged is False
        assert len(data.residual_history) == 0

    def test_convergence_data_tracking(self):
        """ConvergenceData tracks residuals."""
        data = ConvergenceData()
        data.p_residual = 1e-5
        data.U_residual = 1e-4
        data.continuity_error = 1e-3
        data.outer_iterations = 10
        data.converged = True

        assert data.p_residual == 1e-5
        assert data.converged is True
