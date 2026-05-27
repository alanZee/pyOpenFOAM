"""
Unit tests for particle force models.
"""

from __future__ import annotations

import math
import pytest


class TestGravityForce:
    """Tests for GravityForce."""

    def test_default_gravity(self):
        from pyfoam.lagrangian.forces import GravityForce

        f = GravityForce()
        a = f.acceleration(
            velocity=[0.0, 0.0, 0.0],
            diameter=1e-4,
            density=1000.0,
        )
        assert a == [0.0, 0.0, -9.81]

    def test_custom_gravity(self):
        from pyfoam.lagrangian.forces import GravityForce

        f = GravityForce(g=[1.0, 2.0, 3.0])
        a = f.acceleration(
            velocity=[0.0, 0.0, 0.0],
            diameter=1e-4,
            density=1000.0,
        )
        assert a == [1.0, 2.0, 3.0]

    def test_independent_of_state(self):
        """Gravity does not depend on particle velocity or diameter."""
        from pyfoam.lagrangian.forces import GravityForce

        f = GravityForce(g=[0.0, 0.0, -10.0])
        a1 = f.acceleration(velocity=[0.0, 0.0, 0.0], diameter=1e-3, density=1000.0)
        a2 = f.acceleration(velocity=[100.0, 50.0, -30.0], diameter=1e-5, density=2000.0)
        assert a1 == a2


class TestDragForce:
    """Tests for DragForce."""

    def test_stokes_drag_zero_rel_velocity(self):
        from pyfoam.lagrangian.forces import DragForce

        f = DragForce("stokes")
        a = f.acceleration(
            velocity=[1.0, 0.0, 0.0],
            diameter=1e-4,
            density=1000.0,
            fluid_velocity=[1.0, 0.0, 0.0],
        )
        assert a == [0.0, 0.0, 0.0]

    def test_drag_decelerates_fast_particle(self):
        """A particle faster than fluid should decelerate."""
        from pyfoam.lagrangian.forces import DragForce

        f = DragForce("stokes")
        a = f.acceleration(
            velocity=[10.0, 0.0, 0.0],
            diameter=1e-4,
            density=1000.0,
            fluid_velocity=[0.0, 0.0, 0.0],
        )
        # 阻力加速度方向应与相对速度反向 (指向负 x)
        assert a[0] < 0

    def test_drag_accelerates_slow_particle(self):
        """A particle slower than fluid should accelerate."""
        from pyfoam.lagrangian.forces import DragForce

        f = DragForce("stokes")
        a = f.acceleration(
            velocity=[0.0, 0.0, 0.0],
            diameter=1e-4,
            density=1000.0,
            fluid_velocity=[5.0, 0.0, 0.0],
        )
        assert a[0] > 0

    def test_drag_returns_zero_without_fluid(self):
        from pyfoam.lagrangian.forces import DragForce

        f = DragForce("stokes")
        a = f.acceleration(
            velocity=[10.0, 0.0, 0.0],
            diameter=1e-4,
            density=1000.0,
            fluid_velocity=None,
        )
        assert a == [0.0, 0.0, 0.0]

    def test_schiller_naumann_model(self):
        from pyfoam.lagrangian.forces import DragForce

        f = DragForce("schiller-naumann")
        a = f.acceleration(
            velocity=[5.0, 0.0, 0.0],
            diameter=1e-3,
            density=1000.0,
            fluid_velocity=[0.0, 0.0, 0.0],
        )
        assert a[0] < 0
        assert all(math.isfinite(v) for v in a)

    def test_drag_larger_particle_less_acceleration(self):
        """Larger particles experience less drag acceleration (per unit mass)."""
        from pyfoam.lagrangian.forces import DragForce

        f = DragForce("stokes")
        a_small = f.acceleration(
            velocity=[1.0, 0.0, 0.0],
            diameter=1e-5,
            density=1000.0,
            fluid_velocity=[0.0, 0.0, 0.0],
        )
        a_large = f.acceleration(
            velocity=[1.0, 0.0, 0.0],
            diameter=1e-3,
            density=1000.0,
            fluid_velocity=[0.0, 0.0, 0.0],
        )
        assert abs(a_small[0]) > abs(a_large[0])

    def test_invalid_model_raises(self):
        from pyfoam.lagrangian.forces import DragForce

        with pytest.raises(ValueError, match="Unknown drag model"):
            DragForce("invalid")

    def test_3d_velocity(self):
        from pyfoam.lagrangian.forces import DragForce

        f = DragForce("stokes")
        a = f.acceleration(
            velocity=[3.0, 4.0, 0.0],
            diameter=1e-4,
            density=1000.0,
            fluid_velocity=[0.0, 0.0, 0.0],
        )
        # 阻力加速度方向应与速度方向相反
        assert a[0] < 0
        assert a[1] < 0
        assert abs(a[2]) < 1e-15


class TestLiftForce:
    """Tests for Saffman LiftForce."""

    def test_zero_vorticity_zero_lift(self):
        from pyfoam.lagrangian.forces import LiftForce

        f = LiftForce(vorticity=[0.0, 0.0, 0.0])
        a = f.acceleration(
            velocity=[0.0, 0.0, 0.0],
            diameter=1e-4,
            density=1000.0,
            fluid_velocity=[1.0, 0.0, 0.0],
        )
        assert a == [0.0, 0.0, 0.0]

    def test_no_fluid_velocity_zero_lift(self):
        from pyfoam.lagrangian.forces import LiftForce

        f = LiftForce(vorticity=[0.0, 1.0, 0.0])
        a = f.acceleration(
            velocity=[0.0, 0.0, 0.0],
            diameter=1e-4,
            density=1000.0,
            fluid_velocity=None,
        )
        assert a == [0.0, 0.0, 0.0]

    def test_lift_force_direction(self):
        """Lift force should be perpendicular to both relative velocity and vorticity."""
        from pyfoam.lagrangian.forces import LiftForce

        # 流体沿 +x，涡量沿 +z → 升力沿 +y
        f = LiftForce(vorticity=[0.0, 0.0, 10.0])
        a = f.acceleration(
            velocity=[0.0, 0.0, 0.0],
            diameter=1e-4,
            density=1000.0,
            fluid_velocity=[1.0, 0.0, 0.0],
        )
        # rel = [1, 0, 0], omega_hat = [0, 0, 1]
        # cross = rel x omega_hat = [0*1-0*0, 0*0-1*1, 1*0-0*0] = [0, -1, 0]
        assert a[1] < 0

    def test_lift_finite_values(self):
        from pyfoam.lagrangian.forces import LiftForce

        f = LiftForce(vorticity=[1.0, 2.0, 3.0])
        a = f.acceleration(
            velocity=[5.0, -2.0, 1.0],
            diameter=1e-4,
            density=1000.0,
            fluid_velocity=[3.0, 1.0, -1.0],
        )
        assert all(math.isfinite(v) for v in a)
