"""Tests for EnhancedSixDoFSolver4 — v4 enhanced 6DOF solver."""

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver_enhanced_3 import EnhancedSixDoFSolver3
from pyfoam.rigid_body.six_dof_solver_enhanced_4 import (
    EnhancedSixDoFSolver4,
    ForceHistoryEntry,
    StabilityInfo,
)


class TestForceHistoryEntry:
    """Test ForceHistoryEntry dataclass."""

    def test_defaults(self):
        entry = ForceHistoryEntry()
        assert entry.time == 0.0
        assert entry.force.shape == (3,)

    def test_with_data(self):
        entry = ForceHistoryEntry(
            time=0.1,
            force=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        )
        assert entry.time == 0.1


class TestStabilityInfo:
    """Test StabilityInfo dataclass."""

    def test_defaults(self):
        info = StabilityInfo()
        assert info.is_stable is True
        assert info.recommended_dt == float("inf")


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v3(self):
        assert issubclass(EnhancedSixDoFSolver4, EnhancedSixDoFSolver3)


class TestSubstepIntegration:
    """Test substep integration."""

    def test_substep_advances(self):
        solver = EnhancedSixDoFSolver4(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
            n_substeps=4,
        )
        assert solver.n_substeps == 4
        solver.step(dt=0.001, method="substep_lie")
        assert solver.position[1].item() < 0

    def test_default_substeps(self):
        solver = EnhancedSixDoFSolver4(mass=1.0)
        assert solver.n_substeps == 1


class TestForceHistory:
    """Test force history recording."""

    def test_record_force_state(self):
        solver = EnhancedSixDoFSolver4(mass=1.0)
        solver.record_force_state()
        solver.record_force_state()
        assert len(solver.force_history) == 2

    def test_time_tracking(self):
        solver = EnhancedSixDoFSolver4(mass=1.0)
        assert solver.simulation_time == 0.0
        solver.step(dt=0.01)
        assert solver.simulation_time == pytest.approx(0.01)


class TestStabilityAnalysis:
    """Test stability analysis."""

    def test_no_constraints_infinite_dt(self):
        solver = EnhancedSixDoFSolver4(mass=1.0)
        info = solver.analyse_stability()
        assert info.recommended_dt == float("inf")
        assert info.is_stable is True

    def test_with_constraints(self):
        from pyfoam.rigid_body.six_dof_solver_enhanced import PositionConstraint
        solver = EnhancedSixDoFSolver4(mass=1.0)
        pc = PositionConstraint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            value=0.0,
            stiffness=1e6,
        )
        solver.add_position_constraint(pc)
        info = solver.analyse_stability()
        assert info.max_eigenvalue > 0
        assert info.recommended_dt < float("inf")

    def test_recommended_timestep(self):
        from pyfoam.rigid_body.six_dof_solver_enhanced import PositionConstraint
        solver = EnhancedSixDoFSolver4(mass=1.0)
        pc = PositionConstraint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            value=0.0,
            stiffness=1e6,
        )
        solver.add_position_constraint(pc)
        dt = solver.recommended_timestep(safety_factor=0.5)
        assert dt > 0
        assert dt < float("inf")


class TestConstraintDamping:
    """Test constraint damping."""

    def test_set_damping(self):
        solver = EnhancedSixDoFSolver4(mass=1.0)
        solver.set_constraint_damping(0.5)
        assert solver._constraint_damping == 0.5

    def test_damping_clamped(self):
        solver = EnhancedSixDoFSolver4(mass=1.0)
        solver.set_constraint_damping(2.0)
        assert solver._constraint_damping == 1.0


class TestFallback:
    """Test method fallback."""

    def test_fallback_to_base(self):
        solver = EnhancedSixDoFSolver4(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="velocity_verlet")
        assert solver.position[1].item() < 0

    def test_repr(self):
        solver = EnhancedSixDoFSolver4(mass=1.0, n_substeps=4)
        r = repr(solver)
        assert "EnhancedSixDoFSolver4" in r
        assert "substeps=4" in r
