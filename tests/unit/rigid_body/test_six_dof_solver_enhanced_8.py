"""Tests for EnhancedSixDoFSolver8 -- v8 enhanced 6DOF solver."""

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver_enhanced_7 import EnhancedSixDoFSolver7
from pyfoam.rigid_body.six_dof_solver_enhanced_8 import (
    EnhancedSixDoFSolver8,
    MultiRateConfig,
    EnergyDriftConfig,
    ConstraintRelaxationConfig,
    _EnergyTracker,
    _AdaptiveRelaxation,
)


class TestMultiRateConfig:
    """Test MultiRateConfig dataclass."""

    def test_defaults(self):
        cfg = MultiRateConfig()
        assert cfg.translation_substeps == 1
        assert cfg.rotation_substeps == 2
        assert cfg.coupling_order == 2


class TestEnergyDriftConfig:
    """Test EnergyDriftConfig dataclass."""

    def test_defaults(self):
        cfg = EnergyDriftConfig()
        assert cfg.enable_correction is True
        assert cfg.drift_threshold == 1e-6


class TestConstraintRelaxationConfig:
    """Test ConstraintRelaxationConfig dataclass."""

    def test_defaults(self):
        cfg = ConstraintRelaxationConfig()
        assert cfg.enable_adaptive_relaxation is True
        assert cfg.min_relaxation == 0.1


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v7(self):
        assert issubclass(EnhancedSixDoFSolver8, EnhancedSixDoFSolver7)


class TestEnergyTracker:
    """Test energy tracker."""

    def test_initial_drift_zero(self):
        cfg = EnergyDriftConfig()
        tracker = _EnergyTracker(cfg)
        assert tracker.energy_drift == 0.0

    def test_records_energy(self):
        cfg = EnergyDriftConfig()
        tracker = _EnergyTracker(cfg)
        tracker.record(100.0)
        tracker.record(100.0)
        assert tracker.energy_drift == 0.0

    def test_detects_drift(self):
        cfg = EnergyDriftConfig(drift_threshold=0.01)
        tracker = _EnergyTracker(cfg)
        tracker.record(100.0)
        tracker.record(90.0)
        assert tracker.energy_drift > 0.05

    def test_correction(self):
        cfg = EnergyDriftConfig(drift_threshold=0.01, correction_factor=0.1)
        tracker = _EnergyTracker(cfg)
        tracker.record(100.0)
        tracker.record(50.0)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        corrected = tracker.compute_correction(vel)
        # Should have applied correction (energy drifted)
        assert tracker.drift_corrections >= 1


class TestAdaptiveRelaxation:
    """Test adaptive constraint relaxation."""

    def test_initial_relaxation(self):
        cfg = ConstraintRelaxationConfig()
        relax = _AdaptiveRelaxation(cfg)
        assert relax.current_relaxation == 1.0

    def test_suggest_relaxation(self):
        cfg = ConstraintRelaxationConfig()
        relax = _AdaptiveRelaxation(cfg)
        val = relax.suggest_relaxation()
        assert cfg.min_relaxation <= val <= cfg.max_relaxation


class TestSymplecticSE3Step:
    """Test symplectic SE(3) integrator."""

    def test_symplectic_se3_advances(self):
        solver = EnhancedSixDoFSolver8(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="symplectic_se3")
        # Should have moved due to gravity
        assert solver.position[1].item() < 0


class TestMultiRateStep:
    """Test multi-rate integrator."""

    def test_multi_rate_advances(self):
        solver = EnhancedSixDoFSolver8(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
            multi_rate=True,
        )
        solver.step(dt=0.001, method="multi_rate")
        assert solver.position[1].item() < 0


class TestEnergyComputation:
    """Test total energy computation."""

    def test_kinetic_energy(self):
        solver = EnhancedSixDoFSolver8(
            mass=2.0,
            inertia=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
            gravity=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        )
        solver._velocity = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        energy = solver.compute_total_energy()
        # KE = 0.5 * m * v^2 = 0.5 * 2 * 1 = 1.0
        assert abs(energy - 1.0) < 1e-10


class TestRepr:
    """Test string representations."""

    def test_repr(self):
        solver = EnhancedSixDoFSolver8(mass=1.0, multi_rate=True)
        r = repr(solver)
        assert "EnhancedSixDoFSolver8" in r
