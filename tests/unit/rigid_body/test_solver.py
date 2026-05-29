"""Tests for Newton-Euler rigid body solver."""

import pytest
import torch

from pyfoam.rigid_body.solver import RigidBodySolver


class TestRigidBodySolver:
    """Test the Newton-Euler RigidBodySolver."""

    def test_default_init(self):
        """Default solver has correct initial state."""
        solver = RigidBodySolver()
        assert solver.mass == 1.0
        assert torch.allclose(solver.inertia, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(solver.position, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(solver.velocity, torch.zeros(3, dtype=torch.float64))

    def test_force_produces_acceleration(self):
        """F = m*a: 10 N on 2 kg body gives 5 m/s^2."""
        solver = RigidBodySolver(mass=2.0)
        solver.add_force(torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64))
        acc, alpha = solver.solve(dt=0.001)
        assert torch.allclose(acc, torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(alpha, torch.zeros(3, dtype=torch.float64))

    def test_torque_produces_angular_acceleration(self):
        """tau = I*alpha: 5 N*m with I=1 gives 5 rad/s^2."""
        inertia = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        solver = RigidBodySolver(inertia=inertia)
        solver.add_torque(torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64))
        _, alpha = solver.solve(dt=0.001)
        assert torch.allclose(alpha, torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64))

    def test_velocity_integration(self):
        """Velocity integrates correctly over time."""
        solver = RigidBodySolver(mass=1.0)
        for _ in range(1000):
            solver.add_force(torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64))
            solver.solve(dt=0.001)
        # v = a*t = 10 * 1 = 10 m/s
        assert abs(solver.velocity[0].item() - 10.0) < 0.1

    def test_position_integration(self):
        """Position integrates correctly (symplectic Euler)."""
        solver = RigidBodySolver(mass=1.0)
        for _ in range(1000):
            solver.add_force(torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64))
            solver.solve(dt=0.001)
        # x = 0.5 * a * t^2 = 0.5 * 10 * 1 = 5.0
        assert abs(solver.position[0].item() - 5.0) < 0.1

    def test_accumulators_reset_after_solve(self):
        """Force/torque accumulators reset after solve()."""
        solver = RigidBodySolver()
        solver.add_force(torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64))
        solver.solve(dt=0.001)
        # After solve, adding no force and solving again should give zero acc
        acc, _ = solver.solve(dt=0.001)
        assert torch.allclose(acc, torch.zeros(3, dtype=torch.float64))

    def test_linear_momentum(self):
        """p = m * v."""
        vel = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
        solver = RigidBodySolver(mass=2.0, velocity=vel)
        p = solver.linear_momentum()
        assert torch.allclose(p, torch.tensor([6.0, 0.0, 0.0], dtype=torch.float64))

    def test_angular_momentum(self):
        """L = I * omega."""
        omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        inertia = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        solver = RigidBodySolver(inertia=inertia, angular_velocity=omega)
        L = solver.angular_momentum()
        assert torch.allclose(L, torch.tensor([2.0, 6.0, 12.0], dtype=torch.float64))

    def test_kinetic_energy_translational(self):
        """KE_trans = 0.5 * m * v^2."""
        vel = torch.tensor([4.0, 0.0, 0.0], dtype=torch.float64)
        solver = RigidBodySolver(mass=3.0, velocity=vel)
        ke = solver.kinetic_energy()
        expected = 0.5 * 3.0 * 16.0  # 24.0
        assert torch.allclose(ke, torch.tensor(expected, dtype=torch.float64))

    def test_repr(self):
        """__repr__ contains solver info."""
        solver = RigidBodySolver(mass=5.0)
        r = repr(solver)
        assert "RigidBodySolver" in r
        assert "5.0" in r

    def test_position_setter(self):
        """Position can be updated via setter."""
        solver = RigidBodySolver()
        new_pos = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        solver.position = new_pos
        assert torch.allclose(solver.position, new_pos)

    def test_velocity_setter(self):
        """Velocity can be updated via setter."""
        solver = RigidBodySolver()
        new_vel = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        solver.velocity = new_vel
        assert torch.allclose(solver.velocity, new_vel)
