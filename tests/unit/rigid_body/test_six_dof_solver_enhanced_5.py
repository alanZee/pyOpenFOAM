"""Tests for EnhancedSixDoFSolver5 -- v5 enhanced 6DOF solver."""

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver_enhanced_4 import EnhancedSixDoFSolver4
from pyfoam.rigid_body.six_dof_solver_enhanced_5 import (
    EnhancedSixDoFSolver5,
    EnergyTrackingState,
    AdaptiveSubstepConfig,
)


class TestEnergyTrackingState:
    """Test EnergyTrackingState dataclass."""

    def test_defaults(self):
        state = EnergyTrackingState()
        assert state.kinetic == 0.0
        assert state.rotational == 0.0
        assert state.potential == 0.0
        assert state.total == 0.0
        assert state.angular_momentum.shape == (3,)


class TestAdaptiveSubstepConfig:
    """Test AdaptiveSubstepConfig dataclass."""

    def test_defaults(self):
        cfg = AdaptiveSubstepConfig()
        assert cfg.min_substeps == 1
        assert cfg.max_substeps == 32
        assert cfg.error_tolerance == 1e-4

    def test_custom(self):
        cfg = AdaptiveSubstepConfig(min_substeps=2, max_substeps=64)
        assert cfg.min_substeps == 2
        assert cfg.max_substeps == 64


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v4(self):
        assert issubclass(EnhancedSixDoFSolver5, EnhancedSixDoFSolver4)


class TestEnergyComputation:
    """Test energy computation methods."""

    def test_kinetic_energy(self):
        solver = EnhancedSixDoFSolver5(mass=2.0)
        solver._velocity = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        ke = solver.kinetic_energy()
        assert ke == pytest.approx(1.0)  # 0.5 * 2 * 1^2

    def test_rotational_energy(self):
        solver = EnhancedSixDoFSolver5(
            mass=1.0,
            inertia=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        )
        solver._angular_velocity = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        re = solver.rotational_energy()
        assert re == pytest.approx(2.0)  # 0.5 * 1 * 4

    def test_potential_energy_no_gravity(self):
        solver = EnhancedSixDoFSolver5(mass=1.0)
        pe = solver.potential_energy()
        assert pe == 0.0

    def test_potential_energy_with_gravity(self):
        solver = EnhancedSixDoFSolver5(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver._position = torch.tensor([0.0, 10.0, 0.0], dtype=torch.float64)
        pe = solver.potential_energy()
        assert pe == pytest.approx(9.81 * 10.0, rel=1e-3)

    def test_angular_momentum(self):
        solver = EnhancedSixDoFSolver5(
            mass=1.0,
            inertia=torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64),
        )
        solver._angular_velocity = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        L = solver.angular_momentum()
        assert torch.allclose(L, torch.tensor([2.0, 6.0, 12.0], dtype=torch.float64))

    def test_track_energy(self):
        solver = EnhancedSixDoFSolver5(mass=1.0)
        state = solver.track_energy()
        assert isinstance(state, EnergyTrackingState)
        assert len(solver.energy_history) == 1


class TestAdaptiveSubsteps:
    """Test adaptive substep integration."""

    def test_adaptive_substep_advances(self):
        solver = EnhancedSixDoFSolver5(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
            adaptive_substeps=True,
        )
        solver.step(dt=0.001, method="adaptive_substep")
        assert solver.position[1].item() < 0

    def test_default_not_adaptive(self):
        solver = EnhancedSixDoFSolver5(mass=1.0)
        assert solver._adaptive_substeps is False

    def test_custom_adaptive_config(self):
        cfg = AdaptiveSubstepConfig(max_substeps=16)
        solver = EnhancedSixDoFSolver5(
            mass=1.0,
            adaptive_substeps=True,
            adaptive_config=cfg,
        )
        assert solver._adaptive_config.max_substeps == 16


class TestPositionDependentDamping:
    """Test position-dependent damping."""

    def test_damping_at_reference(self):
        solver = EnhancedSixDoFSolver5(mass=1.0)
        solver._position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        solver._velocity = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        ref = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        solver.apply_position_dependent_damping(0.5, ref, 1.0)
        # At reference position, damping should be zero
        assert torch.allclose(
            solver._velocity,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

    def test_damping_at_range(self):
        solver = EnhancedSixDoFSolver5(mass=1.0)
        solver._position = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        solver._velocity = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        ref = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        solver.apply_position_dependent_damping(0.5, ref, 1.0)
        # At full range, damping should be applied
        assert solver._velocity[0].item() < 1.0


class TestRepr:
    """Test string representations."""

    def test_repr(self):
        solver = EnhancedSixDoFSolver5(mass=1.0, adaptive_substeps=True)
        r = repr(solver)
        assert "EnhancedSixDoFSolver5" in r
        assert "adaptive=True" in r
