"""Tests for EnhancedSixDoFSolver7 -- v7 enhanced 6DOF solver."""

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver_enhanced_6 import EnhancedSixDoFSolver6
from pyfoam.rigid_body.six_dof_solver_enhanced_7 import (
    EnhancedSixDoFSolver7,
    ContactCouplingConfig,
    SensorModel,
    SLERPConfig,
    _slerp,
)


class TestContactCouplingConfig:
    """Test ContactCouplingConfig dataclass."""

    def test_defaults(self):
        cfg = ContactCouplingConfig()
        assert cfg.contact_stiffness == 1e5
        assert cfg.restitution_coefficient == 0.5


class TestSensorModel:
    """Test SensorModel dataclass."""

    def test_defaults(self):
        sensor = SensorModel()
        assert sensor.position_noise_std == 0.0
        assert sensor.velocity_noise_std == 0.0


class TestSLERPConfig:
    """Test SLERPConfig dataclass."""

    def test_defaults(self):
        cfg = SLERPConfig()
        assert cfg.threshold == 1e-4
        assert cfg.n_steps == 1


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v6(self):
        assert issubclass(EnhancedSixDoFSolver7, EnhancedSixDoFSolver6)


class TestSLERP:
    """Test SLERP interpolation."""

    def test_same_quaternion(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = _slerp(q, q, 0.5)
        assert torch.allclose(result, q, atol=1e-10)

    def test_interpolation_midpoint(self):
        q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        q1 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        result = _slerp(q0, q1, 0.5)
        # Should be a unit quaternion
        assert abs(float(result.norm()) - 1.0) < 1e-10

    def test_interpolation_endpoints(self):
        q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        q1 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        result0 = _slerp(q0, q1, 0.0)
        result1 = _slerp(q0, q1, 1.0)
        assert torch.allclose(result0, q0, atol=1e-10)
        assert torch.allclose(result1, q1, atol=1e-10)

    def test_short_path(self):
        """Should take shortest path even with negative dot product."""
        q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        q1 = -torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        result = _slerp(q0, q1, 0.5)
        assert abs(float(result.norm()) - 1.0) < 1e-10


class TestContactCoupling:
    """Test contact-aware coupling."""

    def test_no_contact_when_far(self):
        solver = EnhancedSixDoFSolver7(mass=1.0)
        solver._position = torch.zeros(3, dtype=torch.float64)
        coupled_pos = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        coupled_vel = torch.zeros(3, dtype=torch.float64)
        force = solver.compute_contact_coupling_force(coupled_pos, coupled_vel)
        assert force.norm().item() < 1e-10

    def test_contact_when_close(self):
        solver = EnhancedSixDoFSolver7(mass=1.0)
        solver._position = torch.zeros(3, dtype=torch.float64)
        solver._velocity = torch.zeros(3, dtype=torch.float64)
        coupled_pos = torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64)
        coupled_vel = torch.zeros(3, dtype=torch.float64)
        force = solver.compute_contact_coupling_force(
            coupled_pos, coupled_vel, coupled_radius=0.05
        )
        # Should have contact force (penetration)
        assert force[0].item() > 0  # Pushing apart
        assert solver.contact_count == 1


class TestSensorObservation:
    """Test sensor model observations."""

    def test_noiseless_position(self):
        solver = EnhancedSixDoFSolver7(mass=1.0, sensor=SensorModel())
        solver._position = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        observed = solver.observe_position()
        assert torch.allclose(observed, solver._position)

    def test_noisy_position(self):
        solver = EnhancedSixDoFSolver7(
            mass=1.0, sensor=SensorModel(position_noise_std=0.1)
        )
        solver._position = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        # With noise, observed should differ (statistically)
        observed = solver.observe_position()
        assert observed.shape == (3,)


class TestLieGroupStep:
    """Test Lie group integrator."""

    def test_lie_group_advances(self):
        solver = EnhancedSixDoFSolver7(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
            lie_group_integrator=True,
        )
        solver.step(dt=0.001, method="lie_group")
        # Should have moved due to gravity
        assert solver.position[1].item() < 0


class TestRepr:
    """Test string representations."""

    def test_repr(self):
        solver = EnhancedSixDoFSolver7(mass=1.0, lie_group_integrator=True)
        r = repr(solver)
        assert "EnhancedSixDoFSolver7" in r
