"""Tests for EnhancedSixDoFSolver — enhanced 6DOF solver with constraints."""

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver import SixDoFSolver
from pyfoam.rigid_body.six_dof_solver_enhanced import (
    EnhancedSixDoFSolver,
    PositionConstraint,
    VelocityConstraint,
    ConstraintType,
)


# ---------------------------------------------------------------------------
# Constraint tests
# ---------------------------------------------------------------------------


class TestPositionConstraint:
    """Test PositionConstraint."""

    def test_axis_normalized(self):
        """Axis is normalised."""
        pc = PositionConstraint(
            axis=torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64),
            value=1.0,
        )
        assert torch.allclose(pc.axis.norm(), torch.tensor(1.0, dtype=torch.float64))

    def test_equality_correction(self):
        """Equality constraint generates correction force."""
        pc = PositionConstraint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            value=2.0,
            stiffness=100.0,
        )
        pos = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64)
        f = pc.correction(pos)
        # Should push in -y direction (pos > value)
        assert f[1].item() < 0

    def test_inequality_no_correction_when_within(self):
        """Inequality constraint: no correction when within limit."""
        pc = PositionConstraint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            value=5.0,
            constraint_type=ConstraintType.INEQUALITY,
        )
        pos = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64)
        f = pc.correction(pos)
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))

    def test_inequality_correction_when_violated(self):
        """Inequality constraint: correction when violated."""
        pc = PositionConstraint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            value=2.0,
            constraint_type=ConstraintType.INEQUALITY,
            stiffness=100.0,
        )
        pos = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64)
        f = pc.correction(pos)
        assert f[1].item() < 0


class TestVelocityConstraint:
    """Test VelocityConstraint."""

    def test_no_correction_within_limit(self):
        """No correction when velocity within limit."""
        vc = VelocityConstraint(
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            max_velocity=10.0,
        )
        vel = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        f = vc.correction(vel)
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))

    def test_correction_when_exceeded(self):
        """Correction when velocity exceeds limit."""
        vc = VelocityConstraint(
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            max_velocity=5.0,
            damping=100.0,
        )
        vel = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        f = vc.correction(vel)
        assert f[0].item() < 0

    def test_negative_excess(self):
        """Negative velocity excess also corrected."""
        vc = VelocityConstraint(
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            max_velocity=5.0,
            damping=100.0,
        )
        vel = torch.tensor([-10.0, 0.0, 0.0], dtype=torch.float64)
        f = vc.correction(vel)
        assert f[0].item() > 0


# ---------------------------------------------------------------------------
# EnhancedSixDoFSolver tests
# ---------------------------------------------------------------------------


class TestEnhancedSixDoFSolver:
    """Test EnhancedSixDoFSolver."""

    def test_inherits_six_dof(self):
        assert issubclass(EnhancedSixDoFSolver, SixDoFSolver)

    def test_creation(self):
        solver = EnhancedSixDoFSolver(mass=2.0)
        assert solver.mass == 2.0
        assert solver.n_position_constraints == 0
        assert solver.n_velocity_constraints == 0

    def test_add_position_constraint(self):
        solver = EnhancedSixDoFSolver()
        pc = PositionConstraint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            value=0.0,
        )
        solver.add_position_constraint(pc)
        assert solver.n_position_constraints == 1

    def test_add_velocity_constraint(self):
        solver = EnhancedSixDoFSolver()
        vc = VelocityConstraint(
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            max_velocity=10.0,
        )
        solver.add_velocity_constraint(vc)
        assert solver.n_velocity_constraints == 1

    def test_clear_constraints(self):
        solver = EnhancedSixDoFSolver()
        solver.add_position_constraint(PositionConstraint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), value=0.0
        ))
        solver.add_velocity_constraint(VelocityConstraint(
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        ))
        solver.clear_constraints()
        assert solver.n_position_constraints == 0
        assert solver.n_velocity_constraints == 0

    def test_velocity_verlet_integration(self):
        """Velocity-Verlet step moves the body."""
        solver = EnhancedSixDoFSolver(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.01, method="velocity_verlet")
        assert solver.position[1].item() < 0  # Moved downward

    def test_energy_tracking(self):
        """Energy history records values."""
        solver = EnhancedSixDoFSolver(
            mass=1.0,
            velocity=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        solver.record_energy()
        solver.step(dt=0.01, method="velocity_verlet")
        solver.record_energy()
        assert len(solver.energy_history) == 2

    def test_energy_drift_no_drift(self):
        """Energy drift is zero for identical energies."""
        solver = EnhancedSixDoFSolver(mass=1.0)
        solver._energy_history = [
            torch.tensor(100.0, dtype=torch.float64),
            torch.tensor(100.0, dtype=torch.float64),
        ]
        assert solver.energy_drift() == pytest.approx(0.0)

    def test_repr(self):
        solver = EnhancedSixDoFSolver(mass=1.0)
        r = repr(solver)
        assert "EnhancedSixDoFSolver" in r
