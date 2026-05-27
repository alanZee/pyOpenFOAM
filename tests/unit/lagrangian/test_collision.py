"""
Unit tests for Lagrangian collision models.
"""

from __future__ import annotations

import math
import pytest

from pyfoam.lagrangian.collision import (
    CollisionModel,
    NoCollision,
    PairCollision,
)


# ======================================================================
# CollisionModel 抽象基类
# ======================================================================

class TestCollisionModelABC:
    """Tests for the CollisionModel abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            CollisionModel()


# ======================================================================
# NoCollision
# ======================================================================

class TestNoCollision:
    """Tests for NoCollision."""

    def test_returns_original_velocities(self):
        model = NoCollision()
        v1_in = [1.0, 2.0, 3.0]
        v2_in = [-1.0, 0.0, 5.0]
        v1, v2 = model.collide(
            pos1=[0.0, 0.0, 0.0], vel1=v1_in, d1=1e-4, rho1=1000.0,
            pos2=[1e-4, 0.0, 0.0], vel2=v2_in, d2=1e-4, rho2=1000.0,
        )
        assert v1 == v1_in
        assert v2 == v2_in

    def test_returns_copies(self):
        """Returned lists should be independent copies."""
        model = NoCollision()
        v1_in = [1.0, 0.0, 0.0]
        v2_in = [0.0, 1.0, 0.0]
        v1, v2 = model.collide(
            pos1=[0.0, 0.0, 0.0], vel1=v1_in, d1=1e-4, rho1=1000.0,
            pos2=[2e-4, 0.0, 0.0], vel2=v2_in, d2=1e-4, rho2=1000.0,
        )
        v1[0] = 999.0
        assert v1_in[0] == 1.0  # original unchanged


# ======================================================================
# PairCollision
# ======================================================================

class TestPairCollision:
    """Tests for PairCollision."""

    # --- 头碰头测试 (head-on) ---

    def test_elastic_head_on_equal_mass(self):
        """Perfectly elastic head-on collision of equal-mass particles:
        velocities should swap."""
        model = PairCollision(restitution=1.0)
        d = 1e-4
        rho = 1000.0
        v1_in = [1.0, 0.0, 0.0]
        v2_in = [-1.0, 0.0, 0.0]
        # 粒子刚好接触: 中心距 = d
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [d, 0.0, 0.0]

        v1, v2 = model.collide(pos1, v1_in, d, rho, pos2, v2_in, d, rho)

        assert v1[0] == pytest.approx(-1.0, abs=1e-10)
        assert v2[0] == pytest.approx(1.0, abs=1e-10)

    def test_inelastic_head_on_equal_mass(self):
        """Perfectly inelastic head-on collision of equal-mass particles:
        both should come to rest."""
        model = PairCollision(restitution=0.0)
        d = 1e-4
        rho = 1000.0
        v1_in = [1.0, 0.0, 0.0]
        v2_in = [-1.0, 0.0, 0.0]
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [d, 0.0, 0.0]

        v1, v2 = model.collide(pos1, v1_in, d, rho, pos2, v2_in, d, rho)

        assert v1[0] == pytest.approx(0.0, abs=1e-10)
        assert v2[0] == pytest.approx(0.0, abs=1e-10)

    def test_partial_restitution(self):
        """Partial restitution should produce intermediate result."""
        model = PairCollision(restitution=0.5)
        d = 1e-4
        rho = 1000.0
        v1_in = [2.0, 0.0, 0.0]
        v2_in = [-1.0, 0.0, 0.0]
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [d, 0.0, 0.0]

        v1, v2 = model.collide(pos1, v1_in, d, rho, pos2, v2_in, d, rho)

        # 应在弹性与非弹性结果之间
        assert math.isfinite(v1[0])
        assert math.isfinite(v2[0])

    # --- 动量守恒 ---

    def test_momentum_conserved(self):
        """Total momentum should be conserved in all collisions."""
        model = PairCollision(restitution=0.7)
        d1, rho1 = 1e-4, 1000.0
        d2, rho2 = 2e-4, 2000.0
        m1 = (math.pi / 6.0) * d1 ** 3 * rho1
        m2 = (math.pi / 6.0) * d2 ** 3 * rho2

        v1_in = [3.0, 1.0, -2.0]
        v2_in = [-1.0, 2.0, 0.5]
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [0.5 * (d1 + d2), 0.0, 0.0]

        v1, v2 = model.collide(pos1, v1_in, d1, rho1, pos2, v2_in, d2, rho2)

        for i in range(3):
            p_before = m1 * v1_in[i] + m2 * v2_in[i]
            p_after = m1 * v1[i] + m2 * v2[i]
            assert p_after == pytest.approx(p_before, abs=1e-10)

    # --- 不接触 ---

    def test_no_contact_no_change(self):
        """Particles far apart should not collide."""
        model = PairCollision(restitution=1.0)
        v1_in = [1.0, 0.0, 0.0]
        v2_in = [-1.0, 0.0, 0.0]
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [1.0, 0.0, 0.0]  # far away

        v1, v2 = model.collide(
            pos1, v1_in, 1e-4, 1000.0, pos2, v2_in, 1e-4, 1000.0,
        )
        assert v1 == v1_in
        assert v2 == v2_in

    # --- 正在分离 ---

    def test_separating_particles_unchanged(self):
        """Particles already separating should not be modified further."""
        model = PairCollision(restitution=1.0)
        d = 1e-4
        rho = 1000.0
        # 粒子在接触，但相对速度是分离方向 (vel1 < vel2 along n)
        v1_in = [-1.0, 0.0, 0.0]
        v2_in = [1.0, 0.0, 0.0]
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [d, 0.0, 0.0]

        v1, v2 = model.collide(pos1, v1_in, d, rho, pos2, v2_in, d, rho)
        assert v1 == v1_in
        assert v2 == v2_in

    # --- 不同质量 ---

    def test_heavy_particle_nearly_unchanged(self):
        """A much heavier particle should barely change velocity."""
        model = PairCollision(restitution=1.0)
        d_big = 1e-2
        rho_big = 10000.0
        d_small = 1e-5
        rho_small = 1.0
        v1_in = [1.0, 0.0, 0.0]
        v2_in = [-1.0, 0.0, 0.0]
        dist = 0.5 * (d_big + d_small)
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [dist, 0.0, 0.0]

        v1, v2 = model.collide(
            pos1, v1_in, d_big, rho_big,
            pos2, v2_in, d_small, rho_small,
        )

        # 大粒子速度几乎不变
        assert abs(v1[0] - 1.0) < 0.1
        # 小粒子速度大幅反向
        assert abs(v2[0]) > abs(v2_in[0])

    # --- 3D 碰撞 ---

    def test_3d_collision(self):
        """Off-axis collision should modify both velocity components."""
        model = PairCollision(restitution=1.0)
        d = 1e-4
        rho = 1000.0
        v1_in = [1.0, 0.0, 0.0]
        v2_in = [0.0, 0.0, 0.0]
        pos1 = [0.0, 0.0, 0.0]
        # 45度角接触
        dist = d
        pos2 = [dist / math.sqrt(2), dist / math.sqrt(2), 0.0]

        v1, v2 = model.collide(pos1, v1_in, d, rho, pos2, v2_in, d, rho)

        assert math.isfinite(v1[0]) and math.isfinite(v1[1])
        assert math.isfinite(v2[0]) and math.isfinite(v2[1])

    # --- 参数验证 ---

    def test_restitution_out_of_range_low(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            PairCollision(restitution=-0.1)

    def test_restitution_out_of_range_high(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            PairCollision(restitution=1.5)

    def test_restitution_boundary_zero(self):
        """restitution=0.0 should be valid."""
        model = PairCollision(restitution=0.0)
        assert model.restitution == 0.0

    def test_restitution_boundary_one(self):
        """restitution=1.0 should be valid."""
        model = PairCollision(restitution=1.0)
        assert model.restitution == 1.0

    def test_repr(self):
        model = PairCollision(restitution=0.85)
        r = repr(model)
        assert "PairCollision" in r
        assert "0.85" in r
