"""Tests for EnhancedSixDoFSolver9 -- v9 enhanced 6DOF solver."""

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver_enhanced_8 import EnhancedSixDoFSolver8
from pyfoam.rigid_body.six_dof_solver_enhanced_9 import (
    EnhancedSixDoFSolver9,
    ContactRestitutionConfig,
    AdaptiveSubstepConfig,
    ChartSwitchConfig,
    _ChartManager,
    _ContactModel,
)


class TestContactRestitutionConfig:
    def test_defaults(self):
        cfg = ContactRestitutionConfig()
        assert cfg.restitution_coefficient == 0.8
        assert cfg.friction_coefficient == 0.3


class TestAdaptiveSubstepConfig:
    def test_defaults(self):
        cfg = AdaptiveSubstepConfig()
        assert cfg.max_substeps == 10
        assert cfg.min_dt_fraction == 0.01


class TestChartSwitchConfig:
    def test_defaults(self):
        cfg = ChartSwitchConfig()
        assert cfg.singularity_threshold == 0.999
        assert cfg.enable_auto_switch is True


class TestInheritance:
    def test_inherits_v8(self):
        assert issubclass(EnhancedSixDoFSolver9, EnhancedSixDoFSolver8)


class TestChartManager:
    def test_no_switch_initially(self):
        mgr = _ChartManager(ChartSwitchConfig())
        assert mgr.chart_switches == 0

    def test_detects_near_singular(self):
        mgr = _ChartManager(ChartSwitchConfig(singularity_threshold=0.9))
        q = torch.tensor([0.95, 0.1, 0.1, 0.1], dtype=torch.float64)
        q = q / q.norm()
        assert mgr.check_singularity(q) is True
        assert mgr.chart_switches == 1

    def test_no_switch_when_safe(self):
        mgr = _ChartManager(ChartSwitchConfig(singularity_threshold=0.999))
        q = torch.tensor([0.7, 0.5, 0.3, 0.1], dtype=torch.float64)
        q = q / q.norm()
        assert mgr.check_singularity(q) is False


class TestContactModel:
    def test_restores_velocity(self):
        cfg = ContactRestitutionConfig(restitution_coefficient=0.8)
        model = _ContactModel(cfg)
        vel = torch.tensor([1.0, -5.0, 0.0], dtype=torch.float64)
        normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        result = model.apply_restitution(vel, normal)
        # Normal component should be reversed and reduced
        assert result[1].item() > 0  # Reversed direction
        assert model.contact_events == 1

    def test_no_effect_on_outgoing(self):
        cfg = ContactRestitutionConfig(restitution_coefficient=0.8)
        model = _ContactModel(cfg)
        vel = torch.tensor([1.0, 5.0, 0.0], dtype=torch.float64)
        normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        result = model.apply_restitution(vel, normal)
        # Should be unchanged (already moving away)
        assert torch.allclose(result, vel)


class TestLieGroupVariationalStep:
    def test_lie_group_advances(self):
        solver = EnhancedSixDoFSolver9(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="lie_group_variational")
        assert solver.position[1].item() < 0


class TestAdaptiveSubstep:
    def test_adaptive_substep_advances(self):
        solver = EnhancedSixDoFSolver9(
            mass=1.0,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        solver.step(dt=0.001, method="adaptive_substep")
        assert solver.position[1].item() < 0


class TestChartSwitches:
    def test_chart_switches_property(self):
        solver = EnhancedSixDoFSolver9(mass=1.0)
        assert solver.chart_switches >= 0

    def test_contact_events_property(self):
        solver = EnhancedSixDoFSolver9(mass=1.0)
        assert solver.contact_events >= 0


class TestRepr:
    def test_repr(self):
        solver = EnhancedSixDoFSolver9(mass=1.0)
        r = repr(solver)
        assert "EnhancedSixDoFSolver9" in r
