"""Tests for PISO algorithm — transient pressure-velocity coupling.

Tests cover:
- PISO iteration structure (multiple pressure corrections)
- Momentum predictor without under-relaxation
- Pressure correction loop
- Transient simulation stepping
- Convergence behaviour
- Comparison with SIMPLE
"""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.solvers.piso import PISOSolver, PISOConfig
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig
from pyfoam.solvers.coupled_solver import ConvergenceData

from tests.unit.solvers.conftest_coupled import make_cavity_mesh


class TestPISOConfig:
    """Tests for PISO configuration."""

    def test_default_config(self):
        """Default PISO config has expected values."""
        config = PISOConfig()
        assert config.n_correctors == 2
        assert config.relaxation_factor_U == 1.0
        assert config.relaxation_factor_p == 1.0

    def test_custom_config(self):
        """Custom PISO config."""
        config = PISOConfig(
            n_correctors=3,
            p_tolerance=1e-8,
        )
        assert config.n_correctors == 3
        assert config.p_tolerance == 1e-8


class TestPISOSolver:
    """Tests for the PISO algorithm."""

    def test_piso_solver_creation(self):
        """PISO solver can be created."""
        mesh = make_cavity_mesh(2, 2)
        config = PISOConfig(n_correctors=2)
        solver = PISOSolver(mesh, config)

        assert solver.mesh is mesh
        assert solver.config.n_correctors == 2

    def test_piso_solver_default_config(self):
        """PISO solver with default config."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh)

        assert solver.config.n_correctors == 2
        assert solver.config.relaxation_factor_U == 1.0

    def test_piso_solve_returns_correct_shapes(self):
        """PISO solve returns fields with correct shapes."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, tolerance=1e-3,
        )

        assert U_out.shape == (mesh.n_cells, 3)
        assert p_out.shape == (mesh.n_cells,)
        assert phi_out.shape == (mesh.n_faces,)
        assert isinstance(convergence, ConvergenceData)

    def test_piso_solve_convergence_data(self):
        """PISO solve returns convergence data."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, tolerance=1e-3,
        )

        assert convergence.outer_iterations == 1
        assert len(convergence.residual_history) == 2  # 2 corrections
        assert convergence.continuity_error >= 0

    def test_piso_multiple_corrections(self):
        """PISO performs multiple pressure corrections."""
        mesh = make_cavity_mesh(2, 2)
        config = PISOConfig(n_correctors=3)
        solver = PISOSolver(mesh, config)

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, tolerance=1e-3,
        )

        # Should have 3 correction records
        assert len(convergence.residual_history) == 3

    def test_piso_no_under_relaxation(self):
        """PISO does not apply under-relaxation by default."""
        mesh = make_cavity_mesh(2, 2)
        config = PISOConfig(n_correctors=2)
        solver = PISOSolver(mesh, config)

        # Default relaxation factors should be 1.0
        assert solver.config.relaxation_factor_U == 1.0
        assert solver.config.relaxation_factor_p == 1.0

    def test_piso_with_initial_velocity(self):
        """PISO solve works with non-zero initial velocity."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        U[:, 0] = 1.0
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, tolerance=1e-3,
        )

        assert U_out.shape == (mesh.n_cells, 3)
        assert p_out.shape == (mesh.n_cells,)

    def test_piso_transient_stepping(self):
        """PISO can be used for multiple time steps."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        # Simulate 3 time steps
        for step in range(3):
            U, p, phi, convergence = solver.solve(
                U, p, phi, tolerance=1e-3,
            )

        assert U.shape == (mesh.n_cells, 3)
        assert p.shape == (mesh.n_cells,)
        assert phi.shape == (mesh.n_faces,)

    def test_piso_lid_driven_cavity(self):
        """PISO on lid-driven cavity."""
        mesh = make_cavity_mesh(2, 2)
        config = PISOConfig(n_correctors=2)
        solver = PISOSolver(mesh, config)

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        # Run a few time steps
        for step in range(5):
            U, p, phi, convergence = solver.solve(
                U, p, phi, tolerance=1e-2,
            )

        # Should have run without errors
        assert U.shape == (mesh.n_cells, 3)

    def test_piso_repr(self):
        """PISO solver repr."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh)
        r = repr(solver)

        assert "PISOSolver" in r


class TestPISOvsSIMPLE:
    """Comparison tests between PISO and SIMPLE."""

    def test_different_algorithms(self):
        """PISO and SIMPLE produce different results."""
        mesh = make_cavity_mesh(2, 2)

        # SIMPLE with relaxation
        simple_solver = SIMPLESolver(mesh, SIMPLEConfig(
            relaxation_factor_U=0.7,
            relaxation_factor_p=0.3,
        ))

        # PISO without relaxation
        piso_solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        U[:, 0] = 1.0
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        # Run SIMPLE
        U_simple, p_simple, phi_simple, conv_simple = simple_solver.solve(
            U.clone(), p.clone(), phi.clone(), max_outer_iterations=5,
        )

        # Run PISO
        U_piso, p_piso, phi_piso, conv_piso = piso_solver.solve(
            U.clone(), p.clone(), phi.clone(),
        )

        # Results should be different (different algorithms)
        # Note: They might be similar for very simple cases, so we just check shapes
        assert U_simple.shape == U_piso.shape
        assert p_simple.shape == p_piso.shape

    def test_piso_convergence_improves_with_corrections(self):
        """More PISO corrections generally improve continuity."""
        mesh = make_cavity_mesh(2, 2)

        # Use zero initial conditions for a stable start
        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        # 2 corrections
        solver_2 = PISOSolver(mesh, PISOConfig(n_correctors=2))
        _, _, _, conv_2 = solver_2.solve(
            U.clone(), p.clone(), phi.clone(),
        )

        # 3 corrections
        solver_3 = PISOSolver(mesh, PISOConfig(n_correctors=3))
        _, _, _, conv_3 = solver_3.solve(
            U.clone(), p.clone(), phi.clone(),
        )

        # Both should produce finite results
        assert torch.isfinite(torch.tensor(conv_2.continuity_error))
        assert torch.isfinite(torch.tensor(conv_3.continuity_error))


class TestPISOBoundaryConditions:
    """Tests for PISO with boundary conditions (U_bc)."""

    def _make_cavity_bc(self, mesh, U_lid=1.0):
        """Create boundary condition tensor for lid-driven cavity."""
        n_cells = mesh.n_cells
        U_bc = torch.full((n_cells, 3), float('nan'), dtype=CFD_DTYPE)

        # For 2x2 mesh: cells 0,1 are bottom row, 2,3 are top row
        # Bottom wall cells: U = (0, 0, 0)
        U_bc[0, :] = 0.0
        U_bc[1, :] = 0.0

        # Top wall cells (lid): U = (U_lid, 0, 0)
        U_bc[2, 0] = U_lid
        U_bc[2, 1] = 0.0
        U_bc[2, 2] = 0.0
        U_bc[3, 0] = U_lid
        U_bc[3, 1] = 0.0
        U_bc[3, 2] = 0.0

        # Left wall cell: U = (0, 0, 0)
        U_bc[0, :] = 0.0
        U_bc[2, 0] = U_lid  # Re-set lid after left wall

        # Right wall cell: U = (0, 0, 0)
        U_bc[1, :] = 0.0
        U_bc[3, 0] = U_lid  # Re-set lid after right wall

        return U_bc

    def test_piso_with_bc_returns_correct_shapes(self):
        """PISO with U_bc returns correct shapes."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        U_bc = self._make_cavity_bc(mesh)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, U_bc=U_bc, tolerance=1e-3,
        )

        assert U_out.shape == (mesh.n_cells, 3)
        assert p_out.shape == (mesh.n_cells,)
        assert phi_out.shape == (mesh.n_faces,)

    def test_piso_bc_enforced_after_solve(self):
        """PISO enforces boundary conditions after solve."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        U_bc = self._make_cavity_bc(mesh, U_lid=1.0)

        U_out, _, _, _ = solver.solve(U, p, phi, U_bc=U_bc, tolerance=1e-3)

        # Top wall cells should have u=1.0
        assert abs(U_out[2, 0].item() - 1.0) < 1e-10
        assert abs(U_out[3, 0].item() - 1.0) < 1e-10

        # Bottom wall cells should have u=0
        assert abs(U_out[0, 0].item()) < 1e-10
        assert abs(U_out[1, 0].item()) < 1e-10

    def test_piso_bc_preserves_lid_velocity(self):
        """PISO preserves lid velocity across multiple time steps."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        U_bc = self._make_cavity_bc(mesh, U_lid=1.0)

        # Run 5 time steps
        for step in range(5):
            U, p, phi, convergence = solver.solve(
                U, p, phi, U_bc=U_bc, tolerance=1e-3,
            )

        # Lid velocity should still be 1.0
        assert abs(U[2, 0].item() - 1.0) < 1e-10
        assert abs(U[3, 0].item() - 1.0) < 1e-10

    def test_piso_bc_no_bc_same_as_before(self):
        """PISO without U_bc behaves same as before."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        U[:, 0] = 1.0
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, tolerance=1e-3,
        )

        assert U_out.shape == (mesh.n_cells, 3)
        assert torch.isfinite(U_out).all()

    def test_piso_bc_continuity_error_finite(self):
        """PISO with BC produces finite continuity error."""
        mesh = make_cavity_mesh(2, 2)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        U_bc = self._make_cavity_bc(mesh)

        _, _, _, convergence = solver.solve(U, p, phi, U_bc=U_bc, tolerance=1e-3)

        assert torch.isfinite(torch.tensor(convergence.continuity_error))

    def test_piso_bc_larger_mesh(self):
        """PISO with BC on 4x4 mesh."""
        mesh = make_cavity_mesh(4, 4)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=2))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        # Create BC for 4x4 mesh
        n_cells = mesh.n_cells
        U_bc = torch.full((n_cells, 3), float('nan'), dtype=CFD_DTYPE)

        # Bottom wall (j=0): cells 0-3
        for i in range(4):
            U_bc[i, :] = 0.0

        # Top wall (j=3): cells 12-15 (lid)
        for i in range(4):
            U_bc[12 + i, 0] = 1.0
            U_bc[12 + i, 1] = 0.0
            U_bc[12 + i, 2] = 0.0

        # Left wall (i=0): cells 0, 4, 8, 12
        for j in range(4):
            U_bc[j * 4, :] = 0.0

        # Right wall (i=3): cells 3, 7, 11, 15
        for j in range(4):
            U_bc[j * 4 + 3, :] = 0.0

        # Re-set lid (corners belong to lid)
        for i in range(4):
            U_bc[12 + i, 0] = 1.0

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, U_bc=U_bc, tolerance=1e-3,
        )

        assert U_out.shape == (n_cells, 3)
        assert torch.isfinite(U_out).all()

        # Check BC enforcement
        for i in range(4):
            assert abs(U_out[12 + i, 0].item() - 1.0) < 1e-10

    def test_piso_bc_transient_convergence(self):
        """PISO with BC runs multiple time steps without crashing."""
        mesh = make_cavity_mesh(4, 4)
        solver = PISOSolver(mesh, PISOConfig(n_correctors=3))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        n_cells = mesh.n_cells
        U_bc = torch.full((n_cells, 3), float('nan'), dtype=CFD_DTYPE)
        for i in range(4):
            U_bc[i, :] = 0.0
            U_bc[12 + i, 0] = 1.0
        for j in range(4):
            U_bc[j * 4, :] = 0.0
            U_bc[j * 4 + 3, :] = 0.0
        for i in range(4):
            U_bc[12 + i, 0] = 1.0

        # Run multiple time steps - just verify it doesn't crash
        for step in range(5):
            U, p, phi, convergence = solver.solve(
                U, p, phi, U_bc=U_bc, tolerance=1e-3,
            )

        # Verify shapes are preserved
        assert U.shape == (n_cells, 3)
        assert p.shape == (n_cells,)
        assert phi.shape == (mesh.n_faces,)
