"""Tests for 6DOF rigid body motion solver."""

import math

import pytest
import torch

from pyfoam.rigid_body.six_dof_solver import (
    SixDoFSolver,
    _quat_multiply,
    _quat_conjugate,
    _quat_normalize,
    _quat_rotate_vector,
    _quat_from_angular_velocity,
    _quat_to_rotation_matrix,
)


# ======================================================================
# Quaternion math tests
# ======================================================================


class TestQuaternionMath:
    """Test quaternion helper functions."""

    def test_identity_multiply(self):
        """Multiplying by identity quaternion preserves the original."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = _quat_multiply(q, identity)
        assert torch.allclose(result, q, atol=1e-12)

    def test_conjugate(self):
        """Conjugate negates imaginary parts."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        conj = _quat_conjugate(q)
        expected = torch.tensor([1.0, -2.0, -3.0, -4.0], dtype=torch.float64)
        assert torch.allclose(conj, expected)

    def test_normalize(self):
        """Normalization produces unit quaternion."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        normed = _quat_normalize(q)
        assert torch.allclose(normed.norm(), torch.tensor(1.0, dtype=torch.float64))

    def test_normalize_zero_quaternion(self):
        """Zero quaternion normalizes to identity."""
        q = torch.zeros(4, dtype=torch.float64)
        normed = _quat_normalize(q)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(normed, expected)

    def test_rotate_vector_90_deg_z(self):
        """90-degree rotation about z-axis: (1,0,0) -> (0,1,0)."""
        angle = math.pi / 2
        q = torch.tensor(
            [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)],
            dtype=torch.float64,
        )
        v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        result = _quat_rotate_vector(q, v)
        expected = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        assert torch.allclose(result, expected, atol=1e-12)

    def test_rotate_vector_180_deg_x(self):
        """180-degree rotation about x-axis: (0,1,0) -> (0,-1,0)."""
        angle = math.pi
        q = torch.tensor(
            [math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0],
            dtype=torch.float64,
        )
        v = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        result = _quat_rotate_vector(q, v)
        expected = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)
        assert torch.allclose(result, expected, atol=1e-12)

    def test_from_angular_velocity_zero(self):
        """Zero angular velocity produces identity quaternion."""
        omega = torch.zeros(3, dtype=torch.float64)
        dq = _quat_from_angular_velocity(omega, dt=0.1)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(dq, expected)

    def test_from_angular_velocity_nonzero(self):
        """Nonzero angular velocity produces valid rotation quaternion."""
        omega = torch.tensor([0.0, 0.0, math.pi], dtype=torch.float64)
        dq = _quat_from_angular_velocity(omega, dt=1.0)
        # Should be a unit quaternion
        assert torch.allclose(dq.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)

    def test_rotation_matrix_identity(self):
        """Identity quaternion produces identity rotation matrix."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        R = _quat_to_rotation_matrix(q)
        expected = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(R, expected, atol=1e-12)

    def test_rotation_matrix_90_deg_z(self):
        """90-degree rotation about z-axis produces correct matrix."""
        angle = math.pi / 2
        q = torch.tensor(
            [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)],
            dtype=torch.float64,
        )
        R = _quat_to_rotation_matrix(q)
        # R should rotate (1,0,0) -> (0,1,0)
        v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        result = R @ v
        expected = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        assert torch.allclose(result, expected, atol=1e-12)


# ======================================================================
# SixDoFSolver tests
# ======================================================================


class TestSixDoFSolver:
    """Test the 6DOF rigid body motion solver."""

    def test_default_initialization(self):
        """Default solver has correct initial state."""
        solver = SixDoFSolver()
        assert solver.mass == 1.0
        assert torch.allclose(solver.position, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(solver.velocity, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(
            solver.orientation,
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )
        assert torch.allclose(solver.angular_velocity, torch.zeros(3, dtype=torch.float64))

    def test_custom_initialization(self):
        """Custom initial conditions are stored correctly."""
        pos = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        vel = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        solver = SixDoFSolver(mass=2.0, position=pos, velocity=vel)
        assert solver.mass == 2.0
        assert torch.allclose(solver.position, pos)
        assert torch.allclose(solver.velocity, vel)

    def test_gravity_free_fall(self,):
        """Body under gravity accelerates correctly."""
        g = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64)
        solver = SixDoFSolver(mass=1.0, gravity=g)
        dt = 0.001

        # Step for 1 second (1000 steps)
        for _ in range(1000):
            solver.step(dt, method="symplectic_euler")

        # v = g * t = -9.81 * 1.0 = -9.81 m/s
        assert abs(solver.velocity[1].item() - (-9.81)) < 0.05
        # y = 0.5 * g * t^2 = 0.5 * (-9.81) * 1.0 = -4.905 m
        assert abs(solver.position[1].item() - (-4.905)) < 0.05

    def test_gravity_free_fall_rk4(self):
        """Body under gravity with RK4 is more accurate."""
        g = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64)
        solver = SixDoFSolver(mass=1.0, gravity=g)
        dt = 0.001

        for _ in range(1000):
            solver.step(dt, method="rk4")

        # Should be very close to analytical
        assert abs(solver.velocity[1].item() - (-9.81)) < 0.01
        assert abs(solver.position[1].item() - (-4.905)) < 0.01

    def test_external_force(self):
        """External force accelerates body correctly."""
        solver = SixDoFSolver(mass=2.0)
        dt = 0.001

        # Apply constant 10 N force for 1 second
        for _ in range(1000):
            solver.add_force(torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64))
            solver.step(dt, method="symplectic_euler")

        # a = F/m = 10/2 = 5 m/s^2
        # v = a * t = 5 * 1 = 5 m/s
        assert abs(solver.velocity[0].item() - 5.0) < 0.05
        # x = 0.5 * a * t^2 = 0.5 * 5 * 1 = 2.5 m
        assert abs(solver.position[0].item() - 2.5) < 0.05

    def test_unknown_method_raises(self):
        """Unknown integration method raises ValueError."""
        solver = SixDoFSolver()
        with pytest.raises(ValueError, match="Unknown integration method"):
            solver.step(0.001, method="invalid")

    def test_translate(self):
        """translate() moves the body by the given displacement."""
        solver = SixDoFSolver()
        displacement = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        solver.translate(displacement)
        assert torch.allclose(solver.position, displacement)

    def test_rotate(self):
        """rotate() updates the orientation quaternion."""
        solver = SixDoFSolver()
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        angle = math.pi / 2
        solver.rotate(axis, angle)
        # Verify the quaternion is no longer identity
        assert not torch.allclose(
            solver.orientation,
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )
        # Verify it's still a unit quaternion
        assert torch.allclose(
            solver.orientation.norm(),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-10,
        )

    def test_kinetic_energy_stationary(self):
        """Stationary body has zero kinetic energy."""
        solver = SixDoFSolver(mass=5.0)
        ke = solver.kinetic_energy()
        assert torch.allclose(ke, torch.tensor(0.0, dtype=torch.float64))

    def test_kinetic_energy_translational(self):
        """Translational KE = 0.5 * m * v^2."""
        vel = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)
        solver = SixDoFSolver(mass=2.0, velocity=vel)
        ke = solver.kinetic_energy()
        expected = 0.5 * 2.0 * (3.0 ** 2 + 4.0 ** 2)  # 25.0
        assert torch.allclose(ke, torch.tensor(expected, dtype=torch.float64))

    def test_kinetic_energy_rotational(self):
        """Rotational KE = 0.5 * I * omega^2."""
        omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        inertia = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        solver = SixDoFSolver(inertia=inertia, angular_velocity=omega)
        ke = solver.kinetic_energy()
        # 0.5 * (2*1 + 3*4 + 4*9) = 0.5 * (2 + 12 + 36) = 25.0
        expected = 0.5 * (2.0 * 1.0 + 3.0 * 4.0 + 4.0 * 9.0)
        assert torch.allclose(ke, torch.tensor(expected, dtype=torch.float64))

    def test_rotation_matrix_identity(self):
        """Identity orientation produces identity rotation matrix."""
        solver = SixDoFSolver()
        R = solver.rotation_matrix()
        expected = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(R, expected, atol=1e-12)

    def test_get_state(self):
        """get_state returns a snapshot of the current state."""
        pos = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        vel = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        solver = SixDoFSolver(position=pos, velocity=vel)
        state = solver.get_state()
        assert torch.allclose(state["position"], pos)
        assert torch.allclose(state["velocity"], vel)
        # Verify it's a clone (not a reference)
        state["position"][0] = 999.0
        assert solver.position[0].item() == 1.0

    def test_repr(self):
        """__repr__ includes mass, position, velocity."""
        solver = SixDoFSolver(mass=2.5)
        r = repr(solver)
        assert "2.5" in r
        assert "SixDoFSolver" in r

    def test_moment_induces_rotation(self):
        """External moment induces angular velocity."""
        inertia = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        solver = SixDoFSolver(inertia=inertia)
        dt = 0.001

        # Apply constant moment about z-axis for 1 second
        for _ in range(1000):
            solver.add_moment(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
            solver.step(dt, method="symplectic_euler")

        # alpha = M / I = 1.0 / 1.0 = 1.0 rad/s^2
        # omega = alpha * t = 1.0 rad/s
        assert abs(solver.angular_velocity[2].item() - 1.0) < 0.05

    def test_energy_conservation_no_external(self):
        """Energy is approximately conserved without external forces (symplectic Euler)."""
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        solver = SixDoFSolver(mass=1.0, velocity=vel)
        ke_initial = solver.kinetic_energy().item()

        # Step many times with no external forces
        for _ in range(10000):
            solver.step(0.0001, method="symplectic_euler")

        ke_final = solver.kinetic_energy().item()
        # Symplectic Euler should conserve energy well
        assert abs(ke_final - ke_initial) / ke_initial < 0.01
