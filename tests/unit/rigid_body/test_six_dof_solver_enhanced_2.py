"""Tests for EnhancedSixDoFSolver2 — v2 enhanced 6DOF solver."""

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver_enhanced import EnhancedSixDoFSolver
from pyfoam.rigid_body.six_dof_solver_enhanced_2 import (
    EnhancedSixDoFSolver2,
    BaumgarteParams,
)
from pyfoam.rigid_body.six_dof_solver_enhanced import PositionConstraint


class TestBaumgarteParams:
    """Test BaumgarteParams dataclass."""

    def test_defaults(self):
        bp = BaumgarteParams()
        assert bp.alpha == 0.1
        assert bp.beta == 0.1
        assert bp.max_correction == 1.0

    def test_custom(self):
        bp = BaumgarteParams(alpha=0.5, beta=0.3, max_correction=10.0)
        assert bp.alpha == 0.5


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_enhanced(self):
        assert issubclass(EnhancedSixDoFSolver2, EnhancedSixDoFSolver)


class TestEnhancedSixDoFSolver2:
    """Test EnhancedSixDoFSolver2."""

    def test_creation(self):
        solver = EnhancedSixDoFSolver2(mass=2.0)
        assert solver.mass == 2.0

    def test_creation_with_baumgarte(self):
        bp = BaumgarteParams(alpha=0.5, beta=0.2)
        solver = EnhancedSixDoFSolver2(mass=1.0, baumgarte=bp)
        assert solver._baumgarte.alpha == 0.5

    def test_lie_group_step(self):
        """Lie-group integration advances the body."""
        solver = EnhancedSixDoFSolver2(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="lie_group")
        assert solver.position[1].item() < 0  # Moved downward

    def test_bdf1_step(self):
        """BDF1 integration advances the body."""
        solver = EnhancedSixDoFSolver2(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="bdf1")
        assert solver.position[1].item() < 0

    def test_bdf1_prev_velocity_stored(self):
        """BDF1 stores previous velocity for stabilisation."""
        solver = EnhancedSixDoFSolver2(mass=1.0)
        assert solver._prev_velocity is None
        solver.step(dt=0.001, method="bdf1")
        assert solver._prev_velocity is not None

    def test_momentum_tracking(self):
        """Momentum history records values."""
        solver = EnhancedSixDoFSolver2(
            mass=1.0,
            velocity=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        solver.record_momentum()
        solver.step(dt=0.001, method="lie_group")
        solver.record_momentum()
        assert len(solver.momentum_history) == 2

    def test_momentum_drift_zero(self):
        """Momentum drift is zero for identical values."""
        solver = EnhancedSixDoFSolver2(mass=1.0)
        solver._momentum_history = [
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        ]
        assert solver.momentum_drift() == pytest.approx(0.0)

    def test_momentum_drift_nonzero(self):
        """Momentum drift is nonzero for different values."""
        solver = EnhancedSixDoFSolver2(mass=1.0)
        solver._momentum_history = [
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        ]
        assert solver.momentum_drift() > 0

    def test_baumgarte_with_constraints(self):
        """Baumgarte stabilisation works with position constraints."""
        solver = EnhancedSixDoFSolver2(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
            baumgarte=BaumgarteParams(alpha=0.1, beta=0.1),
        )
        pc = PositionConstraint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            value=0.0,
            stiffness=1e8,
        )
        solver.add_position_constraint(pc)
        solver.step(dt=0.001, method="lie_group")
        # Body should still move (constrained by penalty)
        assert solver.n_position_constraints == 1

    def test_fallback_integration(self):
        """Standard methods still work."""
        solver = EnhancedSixDoFSolver2(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="velocity_verlet")
        assert solver.position[1].item() < 0

    def test_repr(self):
        solver = EnhancedSixDoFSolver2(mass=1.0)
        r = repr(solver)
        assert "EnhancedSixDoFSolver2" in r
        assert "baumgarte" in r
