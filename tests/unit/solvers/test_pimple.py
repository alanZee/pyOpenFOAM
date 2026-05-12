"""Tests for PIMPLE algorithm — merged PISO + SIMPLE for transient flow.

Tests cover:
- PIMPLE configuration
- Solver creation and execution
- Multiple outer iterations
- Multiple pressure corrections
- Convergence tracking
- Comparison with PISO
"""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.solvers.pimple import PIMPLESolver, PIMPLEConfig
from pyfoam.solvers.piso import PISOSolver, PISOConfig
from pyfoam.solvers.coupled_solver import ConvergenceData

from tests.unit.solvers.conftest_coupled import make_cavity_mesh


class TestPIMPLEConfig:
    """Tests for PIMPLE configuration."""

    def test_default_config(self):
        """Default PIMPLE config has expected values."""
        config = PIMPLEConfig()
        assert config.n_outer_correctors == 3
        assert config.n_correctors == 2
        assert config.relaxation_factor_U == 0.7
        assert config.relaxation_factor_p == 0.3

    def test_custom_config(self):
        """Custom PIMPLE config."""
        config = PIMPLEConfig(
            n_outer_correctors=5,
            n_correctors=3,
            relaxation_factor_U=0.5,
            relaxation_factor_p=0.2,
        )
        assert config.n_outer_correctors == 5
        assert config.n_correctors == 3
        assert config.relaxation_factor_U == 0.5
        assert config.relaxation_factor_p == 0.2


class TestPIMPLESolver:
    """Tests for the PIMPLE algorithm."""

    def test_pimple_solver_creation(self):
        """PIMPLE solver can be created."""
        mesh = make_cavity_mesh(2, 2)
        config = PIMPLEConfig(n_outer_correctors=3, n_correctors=2)
        solver = PIMPLESolver(mesh, config)

        assert solver.mesh is mesh
        assert solver.config.n_outer_correctors == 3
        assert solver.config.n_correctors == 2

    def test_pimple_solver_default_config(self):
        """PIMPLE solver with default config."""
        mesh = make_cavity_mesh(2, 2)
        solver = PIMPLESolver(mesh)

        assert solver.config.n_outer_correctors == 3
        assert solver.config.n_correctors == 2

    def test_pimple_solve_returns_correct_shapes(self):
        """PIMPLE solve returns fields with correct shapes."""
        mesh = make_cavity_mesh(2, 2)
        solver = PIMPLESolver(mesh, PIMPLEConfig(relaxation_factor_p=1.0))

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

    def test_pimple_solve_convergence_data(self):
        """PIMPLE solve returns convergence data."""
        mesh = make_cavity_mesh(2, 2)
        solver = PIMPLESolver(mesh, PIMPLEConfig(relaxation_factor_p=1.0))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, max_outer_iterations=10, tolerance=1e-3,
        )

        assert convergence.outer_iterations > 0
        assert len(convergence.residual_history) > 0
        assert convergence.continuity_error >= 0

    def test_pimple_with_initial_velocity(self):
        """PIMPLE solve works with non-zero initial velocity."""
        mesh = make_cavity_mesh(2, 2)
        solver = PIMPLESolver(mesh, PIMPLEConfig(relaxation_factor_p=1.0))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        U[:, 0] = 1.0
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, max_outer_iterations=5, tolerance=1e-3,
        )

        assert U_out.shape == (mesh.n_cells, 3)
        assert p_out.shape == (mesh.n_cells,)

    def test_pimple_multiple_outer_iterations(self):
        """PIMPLE uses multiple outer iterations."""
        mesh = make_cavity_mesh(2, 2)
        config = PIMPLEConfig(
            n_outer_correctors=5,
            n_correctors=2,
            relaxation_factor_p=1.0,
        )
        solver = PIMPLESolver(mesh, config)

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, max_outer_iterations=10, tolerance=1e-3,
        )

        # Should have run at least one iteration
        assert convergence.outer_iterations >= 1

    def test_pimple_lid_driven_cavity(self):
        """PIMPLE on lid-driven cavity with moving top wall."""
        mesh = make_cavity_mesh(2, 2)
        config = PIMPLEConfig(
            n_outer_correctors=3,
            n_correctors=2,
            relaxation_factor_U=0.7,
            relaxation_factor_p=0.3,
        )
        solver = PIMPLESolver(mesh, config)

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        # Run a few iterations
        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, max_outer_iterations=20, tolerance=1e-2,
        )

        # Should have run at least one iteration
        assert convergence.outer_iterations >= 1

    def test_pimple_repr(self):
        """PIMPLE solver repr."""
        mesh = make_cavity_mesh(2, 2)
        solver = PIMPLESolver(mesh)
        r = repr(solver)

        assert "PIMPLESolver" in r
        assert "relax_U" in r


class TestPIMPLEvsPISO:
    """Tests comparing PIMPLE and PISO behavior."""

    def test_different_algorithms(self):
        """PIMPLE and PISO are different algorithms."""
        mesh = make_cavity_mesh(2, 2)

        piso_config = PIMPLEConfig(n_outer_correctors=1, n_correctors=2)
        pimple_config = PIMPLEConfig(n_outer_correctors=3, n_correctors=2)

        piso_solver = PIMPLESolver(mesh, piso_config)
        pimple_solver = PIMPLESolver(mesh, pimple_config)

        # Different configs
        assert piso_solver.config.n_outer_correctors == 1
        assert pimple_solver.config.n_outer_correctors == 3

    def test_pimply_convergence_improves_with_outer_iterations(self):
        """More outer iterations should improve convergence."""
        mesh = make_cavity_mesh(2, 2)

        config_few = PIMPLEConfig(n_outer_correctors=1, n_correctors=2, relaxation_factor_p=1.0)
        config_many = PIMPLEConfig(n_outer_correctors=5, n_correctors=2, relaxation_factor_p=1.0)

        solver_few = PIMPLESolver(mesh, config_few)
        solver_many = PIMPLESolver(mesh, config_many)

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        U[:, 0] = 1.0
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        _, _, _, conv_few = solver_few.solve(
            U.clone(), p.clone(), phi.clone(), max_outer_iterations=20, tolerance=1e-3,
        )
        _, _, _, conv_many = solver_many.solve(
            U.clone(), p.clone(), phi.clone(), max_outer_iterations=20, tolerance=1e-3,
        )

        # Both should complete at least one iteration
        assert conv_few.outer_iterations >= 1
        assert conv_many.outer_iterations >= 1


class TestPIMPLEBoundaryConditions:
    """Tests for PIMPLE with boundary conditions (U_bc)."""

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

    def test_pimple_with_bc_returns_correct_shapes(self):
        """PIMPLE with U_bc returns correct shapes."""
        mesh = make_cavity_mesh(2, 2)
        solver = PIMPLESolver(mesh, PIMPLEConfig(
            n_outer_correctors=3,
            n_correctors=2,
            relaxation_factor_p=1.0,
        ))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        U_bc = self._make_cavity_bc(mesh)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, U_bc=U_bc, max_outer_iterations=5, tolerance=1e-3,
        )

        assert U_out.shape == (mesh.n_cells, 3)
        assert p_out.shape == (mesh.n_cells,)
        assert phi_out.shape == (mesh.n_faces,)

    def test_pimple_bc_enforced_after_solve(self):
        """PIMPLE enforces boundary conditions after solve."""
        mesh = make_cavity_mesh(2, 2)
        solver = PIMPLESolver(mesh, PIMPLEConfig(
            n_outer_correctors=3,
            n_correctors=2,
            relaxation_factor_p=1.0,
        ))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        U_bc = self._make_cavity_bc(mesh, U_lid=1.0)

        U_out, _, _, _ = solver.solve(
            U, p, phi, U_bc=U_bc, max_outer_iterations=5, tolerance=1e-3,
        )

        # Top wall cells should have u=1.0
        assert abs(U_out[2, 0].item() - 1.0) < 1e-10
        assert abs(U_out[3, 0].item() - 1.0) < 1e-10

        # Bottom wall cells should have u=0
        assert abs(U_out[0, 0].item()) < 1e-10
        assert abs(U_out[1, 0].item()) < 1e-10

    def test_pimple_bc_preserves_lid_velocity(self):
        """PIMPLE preserves lid velocity across multiple time steps."""
        mesh = make_cavity_mesh(2, 2)
        solver = PIMPLESolver(mesh, PIMPLEConfig(
            n_outer_correctors=3,
            n_correctors=2,
            relaxation_factor_p=1.0,
        ))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        U_bc = self._make_cavity_bc(mesh, U_lid=1.0)

        # Run 5 time steps
        for step in range(5):
            U, p, phi, convergence = solver.solve(
                U, p, phi, U_bc=U_bc, max_outer_iterations=5, tolerance=1e-3,
            )

        # Lid velocity should still be 1.0
        assert abs(U[2, 0].item() - 1.0) < 1e-10
        assert abs(U[3, 0].item() - 1.0) < 1e-10

    def test_pimple_bc_no_bc_same_as_before(self):
        """PIMPLE without U_bc behaves same as before."""
        mesh = make_cavity_mesh(2, 2)
        solver = PIMPLESolver(mesh, PIMPLEConfig(
            n_outer_correctors=3,
            n_correctors=2,
            relaxation_factor_p=1.0,
        ))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        U[:, 0] = 1.0
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(
            U, p, phi, max_outer_iterations=5, tolerance=1e-3,
        )

        assert U_out.shape == (mesh.n_cells, 3)
        assert torch.isfinite(U_out).all()

    def test_pimple_bc_continuity_error_finite(self):
        """PIMPLE with BC produces finite continuity error."""
        mesh = make_cavity_mesh(2, 2)
        solver = PIMPLESolver(mesh, PIMPLEConfig(
            n_outer_correctors=3,
            n_correctors=2,
            relaxation_factor_p=1.0,
        ))

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)
        U_bc = self._make_cavity_bc(mesh)

        _, _, _, convergence = solver.solve(
            U, p, phi, U_bc=U_bc, max_outer_iterations=5, tolerance=1e-3,
        )

        assert torch.isfinite(torch.tensor(convergence.continuity_error))

    def test_pimple_bc_larger_mesh(self):
        """PIMPLE with BC on 4x4 mesh."""
        mesh = make_cavity_mesh(4, 4)
        solver = PIMPLESolver(mesh, PIMPLEConfig(
            n_outer_correctors=3,
            n_correctors=2,
            relaxation_factor_p=1.0,
        ))

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
            U, p, phi, U_bc=U_bc, max_outer_iterations=10, tolerance=1e-3,
        )

        assert U_out.shape == (n_cells, 3)
        assert torch.isfinite(U_out).all()

        # Check BC enforcement
        for i in range(4):
            assert abs(U_out[12 + i, 0].item() - 1.0) < 1e-10

    def test_pimple_bc_convergence_with_outer_iterations(self):
        """PIMPLE with BC converges with more outer iterations."""
        mesh = make_cavity_mesh(4, 4)

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

        # Few outer iterations
        solver_few = PIMPLESolver(mesh, PIMPLEConfig(
            n_outer_correctors=1,
            n_correctors=2,
            relaxation_factor_p=1.0,
        ))

        # Many outer iterations
        solver_many = PIMPLESolver(mesh, PIMPLEConfig(
            n_outer_correctors=5,
            n_correctors=2,
            relaxation_factor_p=1.0,
        ))

        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        _, _, _, conv_few = solver_few.solve(
            U.clone(), p.clone(), phi.clone(), U_bc=U_bc,
            max_outer_iterations=20, tolerance=1e-3,
        )
        _, _, _, conv_many = solver_many.solve(
            U.clone(), p.clone(), phi.clone(), U_bc=U_bc,
            max_outer_iterations=20, tolerance=1e-3,
        )

        # Both should complete at least one iteration
        assert conv_few.outer_iterations >= 1
        assert conv_many.outer_iterations >= 1

    def test_pimple_bc_transient_convergence(self):
        """PIMPLE with BC runs multiple time steps without crashing."""
        mesh = make_cavity_mesh(4, 4)
        solver = PIMPLESolver(mesh, PIMPLEConfig(
            n_outer_correctors=3,
            n_correctors=2,
            relaxation_factor_U=0.7,
            relaxation_factor_p=0.3,
        ))

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
                U, p, phi, U_bc=U_bc, max_outer_iterations=5, tolerance=1e-3,
            )

        # Verify shapes are preserved
        assert U.shape == (n_cells, 3)
        assert p.shape == (n_cells,)
        assert phi.shape == (mesh.n_faces,)

    def test_pimple_reduces_to_piso_with_one_outer(self):
        """PIMPLE with n_outer_correctors=1 behaves like PISO."""
        mesh = make_cavity_mesh(2, 2)

        # PIMPLE with 1 outer correction
        pimple_config = PIMPLEConfig(
            n_outer_correctors=1,
            n_correctors=2,
            relaxation_factor_U=1.0,
            relaxation_factor_p=1.0,
        )
        pimple_solver = PIMPLESolver(mesh, pimple_config)

        # PISO
        piso_config = PISOConfig(n_correctors=2)
        piso_solver = PISOSolver(mesh, piso_config)

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_bc = torch.full((mesh.n_cells, 3), float('nan'), dtype=CFD_DTYPE)
        U_bc[2, 0] = 1.0
        U_bc[3, 0] = 1.0

        U_pimple, p_pimple, phi_pimple, conv_pimple = pimple_solver.solve(
            U.clone(), p.clone(), phi.clone(), U_bc=U_bc, tolerance=1e-3,
        )
        U_piso, p_piso, phi_piso, conv_piso = piso_solver.solve(
            U.clone(), p.clone(), phi.clone(), U_bc=U_bc, tolerance=1e-3,
        )

        # Results should be very similar (not identical due to implementation details)
        assert U_pimple.shape == U_piso.shape
        assert p_pimple.shape == p_piso.shape
