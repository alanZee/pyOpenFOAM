"""
Unit tests for Lagrangian wall interaction models.
"""

from __future__ import annotations

import math
import pytest

from pyfoam.lagrangian.wall_interaction import (
    WallInteractionModel,
    ElasticBounce,
    Stick,
    _dot,
    _normalize,
)


# ======================================================================
# WallInteractionModel 抽象基类
# ======================================================================

class TestWallInteractionModelABC:
    """Tests for the WallInteractionModel abstract base."""

    def test_cannot_instantiate(self):
        """WallInteractionModel is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            WallInteractionModel()


# ======================================================================
# 辅助函数
# ======================================================================

class TestHelpers:
    """Tests for module-level helper functions."""

    def test_dot_product(self):
        assert _dot([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
        assert _dot([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
        assert _dot([1, 2, 3], [4, 5, 6]) == pytest.approx(32.0)

    def test_normalize(self):
        n = _normalize([3.0, 4.0, 0.0])
        assert n[0] == pytest.approx(0.6)
        assert n[1] == pytest.approx(0.8)
        assert n[2] == pytest.approx(0.0)

    def test_normalize_zero_vector(self):
        n = _normalize([0.0, 0.0, 0.0])
        assert n == [0.0, 0.0, 0.0]

    def test_normalize_unit_vector_unchanged(self):
        n = _normalize([0.0, 1.0, 0.0])
        assert n[0] == pytest.approx(0.0)
        assert n[1] == pytest.approx(1.0)
        assert n[2] == pytest.approx(0.0)


# ======================================================================
# ElasticBounce
# ======================================================================

class TestElasticBounce:
    """Tests for ElasticBounce."""

    # --- 参数验证 ---

    def test_restitution_out_of_range_low(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ElasticBounce(restitution=-0.1)

    def test_restitution_out_of_range_high(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ElasticBounce(restitution=1.5)

    def test_restitution_boundary_zero(self):
        model = ElasticBounce(restitution=0.0)
        assert model.restitution == 0.0

    def test_restitution_boundary_one(self):
        model = ElasticBounce(restitution=1.0)
        assert model.restitution == 1.0

    # --- 完全弹性碰撞 (e=1) ---

    def test_perfectly_elastic_head_on(self):
        """Perfectly elastic bounce: normal component fully reflected."""
        model = ElasticBounce(restitution=1.0)
        # 粒子沿 -y 方向撞向水平壁面 (法向 +y)
        result = model.interact(
            velocity=[0.0, -5.0, 0.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        v = result["velocity"]
        assert v[0] == pytest.approx(0.0)
        assert v[1] == pytest.approx(5.0)  # 反射
        assert v[2] == pytest.approx(0.0)
        assert result["stuck"] is False

    def test_perfectly_elastic_oblique(self):
        """Oblique elastic bounce: normal reflected, tangential preserved."""
        model = ElasticBounce(restitution=1.0)
        # 速度 (3, -4, 0), 壁面法向 (0, 1, 0)
        # v_n = -4, v_t = (3, 0, 0)
        # v'_n = -e * v_n = 4 (法向反射)
        # v' = v - (1+e)*v_n*n = (3, -4, 0) - 2*(-4)*(0,1,0) = (3, 4, 0)
        result = model.interact(
            velocity=[3.0, -4.0, 0.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        v = result["velocity"]
        assert v[0] == pytest.approx(3.0)   # 切向不变
        assert v[1] == pytest.approx(4.0)   # 法向反射
        assert v[2] == pytest.approx(0.0)

    def test_elastic_speed_preserved(self):
        """Perfectly elastic bounce preserves speed."""
        model = ElasticBounce(restitution=1.0)
        v_in = [3.0, -4.0, 0.0]
        speed_in = math.sqrt(sum(x ** 2 for x in v_in))
        result = model.interact(velocity=v_in, wall_normal=[0.0, 1.0, 0.0])
        speed_out = math.sqrt(sum(x ** 2 for x in result["velocity"]))
        assert speed_out == pytest.approx(speed_in)

    # --- 完全非弹性碰撞 (e=0) ---

    def test_perfectly_inelastic(self):
        """Perfectly inelastic: normal component zeroed, tangential preserved."""
        model = ElasticBounce(restitution=0.0)
        result = model.interact(
            velocity=[3.0, -4.0, 0.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        v = result["velocity"]
        assert v[0] == pytest.approx(3.0)  # 切向不变
        assert v[1] == pytest.approx(0.0)  # 法向分量消失
        assert v[2] == pytest.approx(0.0)

    # --- 部分恢复 ---

    def test_partial_restitution(self):
        """Partial restitution gives intermediate result."""
        model = ElasticBounce(restitution=0.5)
        result = model.interact(
            velocity=[0.0, -10.0, 0.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        v = result["velocity"]
        # v'_n = -(0.5)*(-10) = 5
        assert v[1] == pytest.approx(5.0)

    # --- 粒子远离壁面 ---

    def test_moving_away_unchanged(self):
        """Particle moving away from wall is not modified."""
        model = ElasticBounce(restitution=1.0)
        # 速度 (0, 5, 0) 沿法向 (0, 1, 0) 方向 → 正在远离
        result = model.interact(
            velocity=[1.0, 5.0, 0.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        assert result["velocity"] == [1.0, 5.0, 0.0]
        assert result["stuck"] is False

    # --- 倾斜壁面 ---

    def test_tilted_wall(self):
        """Bounce off a 45-degree tilted wall."""
        model = ElasticBounce(restitution=1.0)
        # 壁面法向 (1/sqrt(2), 1/sqrt(2), 0)
        nx = 1.0 / math.sqrt(2)
        ny = 1.0 / math.sqrt(2)
        # 速度 (-5, -5, 0) 向壁面运动
        result = model.interact(
            velocity=[-5.0, -5.0, 0.0],
            wall_normal=[nx, ny, 0.0],
        )
        v = result["velocity"]
        # v·n = -5*nx + -5*ny = -5*sqrt(2)
        # v' = v - 2*v_n*n = (-5,-5,0) - 2*(-5*sqrt(2))*(nx,ny,0)
        #    = (-5,-5,0) + 10*sqrt(2)*(1/sqrt(2), 1/sqrt(2), 0)
        #    = (-5,-5,0) + (10, 10, 0) = (5, 5, 0)
        assert v[0] == pytest.approx(5.0)
        assert v[1] == pytest.approx(5.0)
        assert v[2] == pytest.approx(0.0)

    # --- 垂直入射 ---

    def test_vertical_wall_horizontal_bounce(self):
        """Particle bouncing off a vertical wall (x-normal)."""
        model = ElasticBounce(restitution=0.9)
        result = model.interact(
            velocity=[-10.0, 2.0, 3.0],
            wall_normal=[1.0, 0.0, 0.0],
        )
        v = result["velocity"]
        # v_n = -10, v'_n = -0.9*(-10) = 9
        # v' = v - (1+0.9)*(-10)*(1,0,0) = (-10+19, 2, 3) = (9, 2, 3)
        assert v[0] == pytest.approx(9.0)
        assert v[1] == pytest.approx(2.0)
        assert v[2] == pytest.approx(3.0)

    # --- repr ---

    def test_repr(self):
        model = ElasticBounce(restitution=0.85)
        r = repr(model)
        assert "ElasticBounce" in r
        assert "0.85" in r


# ======================================================================
# Stick
# ======================================================================

class TestStick:
    """Tests for Stick."""

    def test_stick_zeroes_velocity(self):
        """Particle velocity is zeroed on sticking."""
        model = Stick()
        result = model.interact(
            velocity=[5.0, -3.0, 1.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        assert result["velocity"] == [0.0, 0.0, 0.0]
        assert result["stuck"] is True

    def test_stick_any_approach_direction(self):
        """Particles approaching from any direction stick."""
        model = Stick()
        normals = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        velocities = [
            [0.0, -5.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 0.0, -2.0],
        ]
        for n, v in zip(normals, velocities):
            result = model.interact(velocity=v, wall_normal=n)
            assert result["stuck"] is True
            assert result["velocity"] == [0.0, 0.0, 0.0]

    def test_moving_away_not_stuck(self):
        """Particle moving away from wall is not stuck."""
        model = Stick()
        result = model.interact(
            velocity=[0.0, 5.0, 0.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        assert result["stuck"] is False
        assert result["velocity"] == [0.0, 5.0, 0.0]

    def test_parallel_motion_not_stuck(self):
        """Particle moving parallel to wall (v_n = 0) is not stuck."""
        model = Stick()
        result = model.interact(
            velocity=[5.0, 0.0, 0.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        assert result["stuck"] is False
        assert result["velocity"] == [5.0, 0.0, 0.0]

    # --- 最小接近速度 ---

    def test_min_approach_speed_threshold(self):
        """Particles below min_approach_speed do not stick."""
        model = Stick(min_approach_speed=1.0)
        # v_n = -0.5 (approach speed 0.5 < 1.0)
        result = model.interact(
            velocity=[0.0, -0.5, 0.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        assert result["stuck"] is False
        assert result["velocity"] == [0.0, -0.5, 0.0]

    def test_min_approach_speed_exceeded_sticks(self):
        """Particles above min_approach_speed stick."""
        model = Stick(min_approach_speed=1.0)
        # v_n = -5.0 (approach speed 5.0 > 1.0)
        result = model.interact(
            velocity=[0.0, -5.0, 0.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        assert result["stuck"] is True
        assert result["velocity"] == [0.0, 0.0, 0.0]

    def test_min_approach_speed_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Stick(min_approach_speed=-1.0)

    def test_min_approach_speed_zero_is_valid(self):
        """min_approach_speed=0.0 should be valid (default behavior)."""
        model = Stick(min_approach_speed=0.0)
        result = model.interact(
            velocity=[0.0, -0.001, 0.0],
            wall_normal=[0.0, 1.0, 0.0],
        )
        assert result["stuck"] is True

    # --- 零法向量 ---

    def test_zero_wall_normal_no_interaction(self):
        """Zero wall normal → no interaction."""
        model = Stick()
        result = model.interact(
            velocity=[5.0, -3.0, 0.0],
            wall_normal=[0.0, 0.0, 0.0],
        )
        assert result["stuck"] is False
        assert result["velocity"] == [5.0, -3.0, 0.0]

    def test_zero_wall_normal_elastic_bounce(self):
        """Zero wall normal → no bounce."""
        model = ElasticBounce()
        result = model.interact(
            velocity=[5.0, -3.0, 0.0],
            wall_normal=[0.0, 0.0, 0.0],
        )
        assert result["stuck"] is False
        assert result["velocity"] == [5.0, -3.0, 0.0]

    # --- repr ---

    def test_repr(self):
        model = Stick(min_approach_speed=0.5)
        r = repr(model)
        assert "Stick" in r
        assert "0.5" in r
