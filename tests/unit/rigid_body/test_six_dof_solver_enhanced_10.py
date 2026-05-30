"""Tests for EnhancedSixDoFSolver10 -- v10 enhanced 6DOF solver."""

import math

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver_enhanced_9 import EnhancedSixDoFSolver9
from pyfoam.rigid_body.six_dof_solver_enhanced_10 import (
    EnhancedSixDoFSolver10,
    GeometricExactConfig,
    MultiRateConfig10,
    EnergyMomentumConfig,
    _rotation_vector_to_quaternion,
    _exponential_map,
    _EnergyDriftTracker,
)


class TestGeometricExactConfig:
    def test_defaults(self):
        cfg = GeometricExactConfig()
        assert cfg.use_exponential_map is True
        assert cfg.max_exponential_iterations == 20


class TestMultiRateConfig10:
    def test_defaults(self):
        cfg = MultiRateConfig10()
        assert cfg.n_fast_steps_per_slow == 10
        assert cfg.fast_subsystem_ratio == 0.1


class TestEnergyMomentumConfig:
    def test_defaults(self):
        cfg = EnergyMomentumConfig()
        assert cfg.stabilization_parameter == 0.1
        assert cfg.enable_adaptive_stabilization is True


class TestInheritance:
    def test_inherits_v9(self):
        assert issubclass(EnhancedSixDoFSolver10, EnhancedSixDoFSolver9)


class TestRotationVector:
    def test_zero_rotation(self):
        rv = torch.zeros(3, dtype=torch.float64)
        q = _rotation_vector_to_quaternion(rv)
        assert abs(q[0].item() - 1.0) < 1e-10

    def test_pi_rotation(self):
        rv = torch.tensor([math.pi, 0.0, 0.0], dtype=torch.float64)
        q = _rotation_vector_to_quaternion(rv)
        assert abs(q[0].item()) < 1e-10  # w ≈ 0


class TestExponentialMap:
    def test_returns_quaternion(self):
        omega = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        q = _exponential_map(omega, dt=0.01)
        assert q.shape == (4,)
        assert abs(q.norm().item() - 1.0) < 1e-10

    def test_zero_angular_velocity(self):
        omega = torch.zeros(3, dtype=torch.float64)
        q = _exponential_map(omega, dt=0.1)
        assert abs(q[0].item() - 1.0) < 1e-10


class TestEnergyDriftTracker:
    def test_initial_drift_zero(self):
        tracker = _EnergyDriftTracker()
        assert tracker.energy_drift == 0.0

    def test_records_drift(self):
        tracker = _EnergyDriftTracker()
        tracker.record(100.0)
        tracker.record(100.5)
        assert tracker.energy_drift > 0

    def test_max_drift(self):
        tracker = _EnergyDriftTracker()
        tracker.record(100.0)
        tracker.record(105.0)
        tracker.record(102.0)
        assert tracker.max_energy_drift == 5.0

    def test_reset(self):
        tracker = _EnergyDriftTracker()
        tracker.record(100.0)
        tracker.reset()
        assert tracker.energy_drift == 0.0


class TestGeometricExactStep:
    def test_advances(self):
        solver = EnhancedSixDoFSolver10(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="geometric_exact")
        assert solver.position[1].item() < 0


class TestMultiRateStep:
    def test_advances(self):
        solver = EnhancedSixDoFSolver10(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="multi_rate")
        assert solver.position[1].item() < 0


class TestEnergyDrift:
    def test_energy_drift_property(self):
        solver = EnhancedSixDoFSolver10(mass=1.0)
        assert solver.energy_drift == 0.0
        assert solver.max_energy_drift >= 0


class TestRepr:
    def test_repr(self):
        solver = EnhancedSixDoFSolver10(mass=1.0)
        r = repr(solver)
        assert "EnhancedSixDoFSolver10" in r
