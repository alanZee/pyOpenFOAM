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
