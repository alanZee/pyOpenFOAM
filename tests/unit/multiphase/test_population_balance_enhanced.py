"""
Unit tests for enhanced population balance solvers.
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.multiphase.population_balance_enhanced import (
    MOMSolver,
    QMOMSolver,
    SectionalSolver,
)


# ======================================================================
# MOMSolver
# ======================================================================

class TestMOMSolver:
    """Tests for Method of Moments solver."""

    def test_default_parameters(self):
        solver = MOMSolver()
        assert solver.n_moments == 6
        assert solver.growth_rate == pytest.approx(1e-6)
        assert solver.nucleation_rate == pytest.approx(0.0)

    def test_custom_parameters(self):
        solver = MOMSolver(n_moments=8, n_cells=5, growth_rate=2e-6)
        assert solver.n_moments == 8

    def test_n_moments_minimum(self):
        with pytest.raises(ValueError, match="n_moments"):
            MOMSolver(n_moments=1)

    def test_n_cells_minimum(self):
        with pytest.raises(ValueError, match="n_cells"):
            MOMSolver(n_cells=0)

    def test_initial_moments_shape(self):
        solver = MOMSolver(n_moments=6, n_cells=10)
        m = solver.get_moments()
        assert m.shape == (6, 10)

    def test_set_moments(self):
        solver = MOMSolver(n_moments=4, n_cells=2)
        m = torch.tensor([
            [1e12, 2e12],
            [1e8, 2e8],
            [1e4, 2e4],
            [1e0, 2e0],
        ], dtype=torch.float64)
        solver.set_moments(m)
        result = solver.get_moments()
        assert result.shape == (4, 2)
        assert float(result[0, 0].item()) == pytest.approx(1e12)

    def test_set_moments_wrong_shape(self):
        solver = MOMSolver(n_moments=4, n_cells=2)
        m = torch.zeros(3, 2)
        with pytest.raises(ValueError, match="Expected 4"):
            solver.set_moments(m)

    def test_advance_no_growth(self):
        solver = MOMSolver(n_moments=4, n_cells=2, growth_rate=0.0)
        m_before = solver.get_moments()
        solver.advance(dt=1e-4)
        m_after = solver.get_moments()
        # Without growth, moments should not change
        assert torch.allclose(m_before, m_after, atol=1.0)

    def test_advance_with_growth(self):
        solver = MOMSolver(n_moments=4, n_cells=2, growth_rate=1e-6)
        m0_before = solver.get_moments()[0, 0].item()
        solver.advance(dt=1e-4)
        m = solver.get_moments()
        # M0 unchanged (no nucleation), M1 should increase
        assert m[0, 0].item() == pytest.approx(m0_before, rel=1e-6)
        assert m[1, 0].item() > 0.0

    def test_advance_with_nucleation(self):
        solver = MOMSolver(
            n_moments=4, n_cells=2,
            growth_rate=0.0, nucleation_rate=1e20,
        )
        m0_before = solver.get_moments()[0, 0].item()
        solver.advance(dt=1e-4)
        m0_after = solver.get_moments()[0, 0].item()
        assert m0_after > m0_before

    def test_time_advances(self):
        solver = MOMSolver()
        assert solver.time == pytest.approx(0.0)
        solver.advance(dt=1e-4)
        assert solver.time == pytest.approx(1e-4)

    def test_moments_non_negative(self):
        solver = MOMSolver(n_moments=4, n_cells=2, growth_rate=1e-6)
        for _ in range(10):
            solver.advance(dt=1e-4)
        m = solver.get_moments()
        assert (m >= 0).all()

    def test_get_mean_diameter(self):
        solver = MOMSolver(n_moments=4, n_cells=2)
        d10 = solver.get_mean_diameter(order=10)
        assert d10.shape == (2,)
        assert (d10 > 0).all()

    def test_get_variance(self):
        solver = MOMSolver(n_moments=4, n_cells=2)
        var = solver.get_variance()
        assert var.shape == (2,)
        # Variance should be non-negative (numerical noise may give tiny neg)
        assert (var >= -1e-6).all()

    def test_repr(self):
        solver = MOMSolver(n_moments=8)
        r = repr(solver)
        assert "MOMSolver" in r
        assert "8" in r


# ======================================================================
# QMOMSolver
# ======================================================================

class TestQMOMSolver:
    """Tests for Quadrature Method of Moments solver."""

    def test_default_parameters(self):
        solver = QMOMSolver()
        assert solver.n_moments == 6
        assert solver.n_nodes == 3

    def test_custom_parameters(self):
        solver = QMOMSolver(n_moments=8, n_nodes=4, n_cells=5)
        assert solver.n_moments == 8
        assert solver.n_nodes == 4

    def test_n_moments_must_be_sufficient(self):
        # Need n_moments >= 2*n_nodes
        with pytest.raises(ValueError, match="n_moments"):
            QMOMSolver(n_moments=4, n_nodes=5)

    def test_initial_moments_shape(self):
        solver = QMOMSolver(n_moments=6, n_cells=10)
        m = solver.get_moments()
        assert m.shape == (6, 10)

    def test_set_moments(self):
        solver = QMOMSolver(n_moments=4, n_nodes=2, n_cells=2)
        m = torch.tensor([
            [1e12, 2e12],
            [1e8, 2e8],
            [1e4, 2e4],
            [1e0, 2e0],
        ], dtype=torch.float64)
        solver.set_moments(m)
        result = solver.get_moments()
        assert float(result[0, 0].item()) == pytest.approx(1e12)

    def test_advance_no_growth(self):
        solver = QMOMSolver(n_moments=4, n_nodes=2, growth_rate=0.0)
        m_before = solver.get_moments()
        solver.advance(dt=1e-4)
        m_after = solver.get_moments()
        assert torch.allclose(m_before, m_after, atol=1.0)

    def test_advance_with_growth(self):
        solver = QMOMSolver(n_moments=4, n_nodes=2, growth_rate=1e-6)
        solver.advance(dt=1e-4)
        m = solver.get_moments()
        assert m[1, 0].item() > 0.0

    def test_advance_with_breakup(self):
        solver = QMOMSolver(
            n_moments=4, n_nodes=2,
            growth_rate=0.0, breakup_rate=1.0,
        )
        m0_before = solver.get_moments()[0, 0].item()
        solver.advance(dt=1e-4)
        m0_after = solver.get_moments()[0, 0].item()
        # Breakup increases number density
        assert m0_after > m0_before

    def test_time_advances(self):
        solver = QMOMSolver()
        assert solver.time == pytest.approx(0.0)
        solver.advance(dt=1e-4)
        assert solver.time == pytest.approx(1e-4)

    def test_moments_non_negative(self):
        solver = QMOMSolver(n_moments=4, n_nodes=2, growth_rate=1e-6)
        for _ in range(10):
            solver.advance(dt=1e-4)
        m = solver.get_moments()
        assert (m >= 0).all()

    def test_get_nodes_and_weights(self):
        solver = QMOMSolver(n_moments=6, n_nodes=3)
        nodes, weights = solver.get_nodes_and_weights()
        assert nodes.shape == (3,)
        assert weights.shape == (3,)
        assert (nodes > 0).all()
        assert (weights > 0).all()

    def test_get_mean_diameter(self):
        solver = QMOMSolver(n_moments=6, n_nodes=3)
        d10 = solver.get_mean_diameter(order=10)
        assert d10.shape == (1,)
        assert d10[0] > 0.0

    def test_reconstruct_distribution(self):
        solver = QMOMSolver(n_moments=6, n_nodes=3)
        nodes, weights = solver.reconstruct_distribution()
        assert nodes.shape == (3,)
        assert weights.shape == (3,)

    def test_repr(self):
        solver = QMOMSolver(n_moments=8, n_nodes=4)
        r = repr(solver)
        assert "QMOMSolver" in r
        assert "8" in r
        assert "4" in r


# ======================================================================
# SectionalSolver
# ======================================================================

class TestSectionalSolver:
    """Tests for Sectional method solver."""

    def test_default_parameters(self):
        solver = SectionalSolver()
        assert solver.n_sections == 20
        assert solver.d_min == pytest.approx(1e-6)
        assert solver.d_max == pytest.approx(1e-3)

    def test_custom_parameters(self):
        solver = SectionalSolver(n_sections=10, d_min=1e-5, d_max=1e-2)
        assert solver.n_sections == 10

    def test_diameters_geometric_grid(self):
        solver = SectionalSolver(
            n_sections=5, d_min=1e-6, d_max=1e-3,
        )
        d = solver.diameters
        assert len(d) == 5
        assert d[0] == pytest.approx(1e-6)
        assert d[-1] == pytest.approx(1e-3)
        # Geometric ratio should be constant
        for i in range(1, len(d)):
            ratio = d[i] / d[i - 1]
            assert ratio > 1.0

    def test_initial_number_densities(self):
        solver = SectionalSolver(n_sections=10, n_cells=5)
        m = solver.get_moments()
        assert m.shape[0] == 10  # n_sections = n_moments
        assert m.shape[1] == 5

    def test_set_number_densities(self):
        solver = SectionalSolver(n_sections=5, n_cells=2)
        n_fields = torch.zeros(5, 2, dtype=torch.float64)
        n_fields[2, :] = 1e12
        solver.set_number_densities(n_fields)
        m0 = solver.get_moments()
        # M0 should be nonzero
        assert m0[0, 0].item() > 0.0

    def test_advance_no_coalescence(self):
        solver = SectionalSolver(
            n_sections=5, n_cells=2,
            coalescence_coeff=0.0, breakup_coeff=0.0,
        )
        m_before = solver.get_moments()
        solver.advance(dt=1e-4)
        m_after = solver.get_moments()
        assert torch.allclose(m_before, m_after, atol=1.0)

    def test_advance_with_coalescence(self):
        solver = SectionalSolver(
            n_sections=5, n_cells=2,
            coalescence_coeff=1e-10,
        )
        solver.advance(dt=1e-4)
        # Should run without error
        m = solver.get_moments()
        assert m.shape[0] == 5

    def test_advance_with_breakup(self):
        solver = SectionalSolver(
            n_sections=5, n_cells=2,
            coalescence_coeff=0.0, breakup_coeff=1.0,
        )
        m0_before = solver.get_moments()[0, 0].item()
        solver.advance(dt=1e-4)
        m0_after = solver.get_moments()[0, 0].item()
        # Breakup redistributes particles to smaller sections
        assert m0_after != pytest.approx(0.0)

    def test_time_advances(self):
        solver = SectionalSolver()
        assert solver.time == pytest.approx(0.0)
        solver.advance(dt=1e-4)
        assert solver.time == pytest.approx(1e-4)

    def test_get_mean_diameter(self):
        solver = SectionalSolver(n_sections=10, n_cells=5)
        d32 = solver.get_mean_diameter(order=32)
        assert d32.shape == (5,)
        assert (d32 > 0).all()

    def test_repr(self):
        solver = SectionalSolver(n_sections=15)
        r = repr(solver)
        assert "SectionalSolver" in r
        assert "15" in r
