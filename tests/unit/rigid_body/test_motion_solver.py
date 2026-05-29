"""Tests for motion solver base class."""

import pytest
import torch

from pyfoam.rigid_body.motion_solver import MotionSolver


class _DummyBody:
    """Minimal body with position and velocity."""

    def __init__(self):
        self.position = torch.zeros(3, dtype=torch.float64)
        self.velocity = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)


class _ConstantDisplacementSolver(MotionSolver):
    """Concrete implementation for testing: returns velocity * dt."""

    def solve_displacement(self, body, dt):
        return body.velocity * dt


class TestMotionSolver:
    """Test the abstract MotionSolver base class."""

    def test_cannot_instantiate_abstract(self):
        """MotionSolver cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MotionSolver()

    def test_solve_displacement_returns_tensor(self):
        """Concrete subclass returns correct displacement."""
        solver = _ConstantDisplacementSolver()
        body = _DummyBody()
        disp = solver.solve_displacement(body, 0.1)
        expected = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(disp, expected)

    def test_step_updates_position(self):
        """step() applies displacement to body.position."""
        solver = _ConstantDisplacementSolver()
        body = _DummyBody()
        disp = solver.step(body, 0.5)
        expected_pos = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(body.position, expected_pos)
        assert torch.allclose(disp, expected_pos)

    def test_step_returns_displacement(self):
        """step() returns the displacement that was applied."""
        solver = _ConstantDisplacementSolver()
        body = _DummyBody()
        disp = solver.step(body, 0.25)
        assert torch.allclose(disp, torch.tensor([0.25, 0.0, 0.0], dtype=torch.float64))

    def test_multiple_steps_accumulate(self):
        """Multiple step() calls accumulate position."""
        solver = _ConstantDisplacementSolver()
        body = _DummyBody()
        for _ in range(10):
            solver.step(body, 0.1)
        # 10 * 0.1 * [1, 0, 0] = [1.0, 0, 0]
        assert torch.allclose(body.position, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
