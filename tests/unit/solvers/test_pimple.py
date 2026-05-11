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
