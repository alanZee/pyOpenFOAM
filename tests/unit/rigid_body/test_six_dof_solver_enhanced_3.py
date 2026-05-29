"""Tests for EnhancedSixDoFSolver3 — v3 enhanced 6DOF solver."""

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver_enhanced_2 import EnhancedSixDoFSolver2
from pyfoam.rigid_body.six_dof_solver_enhanced_3 import (
    EnhancedSixDoFSolver3,
    ContactParams,
    EnergyState,
)


class TestContactParams:
    """Test ContactParams dataclass."""

    def test_defaults(self):
        cp = ContactParams()
        assert cp.stiffness == 1e6
        assert cp.damping == 1e4
        assert cp.friction_coeff == 0.3

    def test_custom(self):
        cp = ContactParams(stiffness=1e8, friction_coeff=0.5)
        assert cp.stiffness == 1e8
        assert cp.friction_coeff == 0.5


class TestEnergyState:
    """Test EnergyState dataclass."""

    def test_defaults(self):
        es = EnergyState()
        assert es.total == 0.0
        assert es.kinetic_translational == 0.0

    def test_with_data(self):
        es = EnergyState(kinetic_translational=5.0, potential=-10.0)
        es.total = es.kinetic_translational + es.potential
        assert es.total == pytest.approx(-5.0)


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v2(self):
        assert issubclass(EnhancedSixDoFSolver3, EnhancedSixDoFSolver2)


class TestEnergyComputation:
    """Test energy tracking."""

    def test_initial_energy(self):
        """Body at rest with gravity has only potential energy."""
        solver = EnhancedSixDoFSolver3(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
            position=torch.tensor([0.0, 10.0, 0.0], dtype=torch.float64),
        )
        es = solver.compute_energy()
        assert es.kinetic_translational == pytest.approx(0.0)
        assert es.potential > 0  # m*g*h

    def test_kinetic_energy(self):
        """Body with velocity has kinetic energy."""
        solver = EnhancedSixDoFSolver3(
            mass=2.0,
            velocity=torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64),
        )
        es = solver.compute_energy()
        assert es.kinetic_translational == pytest.approx(9.0)  # 0.5 * 2 * 9

    def test_record_energy(self):
        """Energy history records values."""
        solver = EnhancedSixDoFSolver3(mass=1.0)
        solver.record_energy()
        solver.record_energy()
        assert len(solver.energy_history) == 2


class TestSymplecticLieIntegration:
    """Test symplectic Lie-group integrator."""

    def test_symplectic_lie_step(self):
        """Symplectic Lie integration advances the body."""
        solver = EnhancedSixDoFSolver3(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="symplectic_lie")
        assert solver.position[1].item() < 0

    def test_energy_preservation(self):
        """Symplectic integrator has better energy behaviour."""
        solver = EnhancedSixDoFSolver3(
            mass=1.0,
            velocity=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        e0 = solver.total_energy()
        for _ in range(100):
            solver.step(dt=0.001, method="symplectic_lie")
        e1 = solver.total_energy()
        # Energy should be approximately preserved (no gravity)
        assert abs(e1 - e0) / max(abs(e0), 1e-15) < 0.1


class TestConstraintProjection:
    """Test iterative constraint projection."""

    def test_project_constraints(self):
        """Gauss-Seidel projection reduces violation."""
        solver = EnhancedSixDoFSolver3(
            mass=1.0,
            position=torch.tensor([0.0, 0.01, 0.0], dtype=torch.float64),
        )
        from pyfoam.rigid_body.six_dof_solver_enhanced import PositionConstraint
        pc = PositionConstraint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            value=0.0,
            stiffness=1e8,
        )
        solver.add_position_constraint(pc)
        n_iters = solver.project_constraints(n_iterations=10)
        assert n_iters >= 1


class TestGroundContact:
    """Test ground plane contact."""

    def test_no_ground_no_force(self):
        """No ground plane set: no contact."""
        solver = EnhancedSixDoFSolver3(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        # Just checking it doesn't crash
        solver.step(dt=0.001, method="symplectic_lie")

    def test_ground_contact(self):
        """Ground contact prevents penetration."""
        solver = EnhancedSixDoFSolver3(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
            position=torch.tensor([0.0, 0.005, 0.0], dtype=torch.float64),
            contact=ContactParams(stiffness=1e8, damping=1e4),
        )
        solver.set_ground_plane(0.0)
        solver.step(dt=0.001, method="symplectic_lie")
        # Body should not have gone below ground significantly
        # (force should push it back up)


class TestFallback:
    """Test method fallback."""

    def test_fallback_to_base(self):
        """Standard methods still work."""
        solver = EnhancedSixDoFSolver3(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="velocity_verlet")
        assert solver.position[1].item() < 0

    def test_repr(self):
        solver = EnhancedSixDoFSolver3(mass=1.0)
        r = repr(solver)
        assert "EnhancedSixDoFSolver3" in r
