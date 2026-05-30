"""Tests for EnhancedSixDoFSolver6 -- v6 enhanced 6DOF solver."""

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver_enhanced_5 import EnhancedSixDoFSolver5
from pyfoam.rigid_body.six_dof_solver_enhanced_6 import (
    EnhancedSixDoFSolver6,
    AugmentedLagrangianConfig,
    MultiBodyCoupling,
    EnergyAdaptiveConfig,
)


class TestAugmentedLagrangianConfig:
    """Test AugmentedLagrangianConfig dataclass."""

    def test_defaults(self):
        cfg = AugmentedLagrangianConfig()
        assert cfg.penalty_stiffness == 1e4
        assert cfg.augmentation_factor == 2.0
        assert cfg.tolerance == 1e-6


class TestMultiBodyCoupling:
    """Test MultiBodyCoupling dataclass."""

    def test_defaults(self):
        mc = MultiBodyCoupling()
        assert mc.body_id == 0
        assert mc.coupled_body_id == 1
        assert mc.coupling_point.shape == (3,)


class TestEnergyAdaptiveConfig:
    """Test EnergyAdaptiveConfig dataclass."""

    def test_defaults(self):
        cfg = EnergyAdaptiveConfig()
        assert cfg.max_energy_drift == 1e-3
        assert cfg.min_dt_factor == 0.1


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v5(self):
        assert issubclass(EnhancedSixDoFSolver6, EnhancedSixDoFSolver5)


class TestEnergyDrift:
    """Test energy drift computation."""

    def test_no_drift_initially(self):
        solver = EnhancedSixDoFSolver6(mass=1.0)
        assert solver.energy_drift() == 0.0

    def test_energy_adaptive_flag(self):
        solver = EnhancedSixDoFSolver6(mass=1.0, energy_adaptive=True)
        assert solver._energy_adaptive is True


class TestMultiBodyCoupling:
    """Test multi-body coupling."""

    def test_add_coupling(self):
        solver = EnhancedSixDoFSolver6(mass=1.0)
        coupling = MultiBodyCoupling(
            coupling_stiffness=1e4,
            coupling_damping=1e2,
        )
        solver.add_coupling(coupling)
        assert len(solver._couplings) == 1

    def test_coupling_force(self):
        solver = EnhancedSixDoFSolver6(mass=1.0)
        solver._position = torch.zeros(3, dtype=torch.float64)
        solver._velocity = torch.zeros(3, dtype=torch.float64)
        coupling = MultiBodyCoupling(
            coupling_stiffness=1e4,
            coupling_damping=1e2,
        )
        solver.add_coupling(coupling)
        coupled_pos = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        coupled_vel = torch.zeros(3, dtype=torch.float64)
        force = solver.compute_coupling_force(coupled_pos, coupled_vel)
        assert force.shape == (3,)
        # 弹簧力指向耦合体位置
        assert force[0].item() > 0


class TestVelocitySmoothing:
    """Test velocity smoothing."""

    def test_smoothing(self):
        from pyfoam.rigid_body.six_dof_solver_enhanced_6 import EnhancedSixDoFSolver6
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        history = [
            torch.tensor([0.8, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([0.9, 0.0, 0.0], dtype=torch.float64),
        ]
        smoothed = EnhancedSixDoFSolver6.smooth_velocity(vel, history, window=3)
        # 平均: (0.8 + 0.9 + 1.0) / 3 = 0.9
        assert smoothed[0].item() == pytest.approx(0.9)

    def test_no_history(self):
        from pyfoam.rigid_body.six_dof_solver_enhanced_6 import EnhancedSixDoFSolver6
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        smoothed = EnhancedSixDoFSolver6.smooth_velocity(vel, [], window=3)
        assert torch.allclose(smoothed, vel)


class TestEnergyAdaptiveStep:
    """Test energy-adaptive step method."""

    def test_energy_adaptive_advances(self):
        solver = EnhancedSixDoFSolver6(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
            energy_adaptive=True,
        )
        solver.step(dt=0.001, method="energy_adaptive")
        # Should have moved due to gravity
        assert solver.position[1].item() < 0


class TestRepr:
    """Test string representations."""

    def test_repr(self):
        solver = EnhancedSixDoFSolver6(mass=1.0, energy_adaptive=True)
        r = repr(solver)
        assert "EnhancedSixDoFSolver6" in r
        assert "energy_adaptive=True" in r
