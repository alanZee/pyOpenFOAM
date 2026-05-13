"""Tests for residual monitoring and convergence info (T1.5).

Tests cover:
- ResidualMonitor creation, update, convergence checks (T1.5.1)
- ConvergenceInfo fields and construction (T1.5.2)
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.solvers.residual import ConvergenceInfo, ResidualMonitor


# ===========================================================================
# T1.5.1 — ResidualMonitor
# ===========================================================================


class TestResidualMonitor:
    """Tests for ResidualMonitor class."""

    # ------------------------------------------------------------------
    # Construction and properties
    # ------------------------------------------------------------------

    def test_default_construction(self):
        """ResidualMonitor has correct default parameters."""
        monitor = ResidualMonitor()
        assert monitor.tolerance == 1e-6
        assert monitor.rel_tol == 0.01
        assert monitor.history == []
        assert monitor.initial_residual is None

    def test_custom_construction(self):
        """ResidualMonitor accepts custom parameters."""
        monitor = ResidualMonitor(tolerance=1e-8, rel_tol=0.001, min_iter=5)
        assert monitor.tolerance == 1e-8
        assert monitor.rel_tol == 0.001

    def test_tolerance_property(self):
        """tolerance property is read-only and returns the set value."""
        monitor = ResidualMonitor(tolerance=1e-10)
        assert monitor.tolerance == 1e-10

    def test_rel_tol_property(self):
        """rel_tol property returns the relative tolerance."""
        monitor = ResidualMonitor(rel_tol=0.05)
        assert monitor.rel_tol == 0.05

    def test_history_starts_empty(self):
        """history is initially empty."""
        monitor = ResidualMonitor()
        assert monitor.history == []

    def test_initial_residual_starts_none(self):
        """initial_residual is None before any update."""
        monitor = ResidualMonitor()
        assert monitor.initial_residual is None

    # ------------------------------------------------------------------
    # Update and tracking
    # ------------------------------------------------------------------

    def test_update_records_residual(self):
        """update() records the residual norm in history."""
        monitor = ResidualMonitor(tolerance=1e-6)
        residual = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)

        monitor.update(residual, iteration=0)

        assert len(monitor.history) == 1
        expected = float(torch.norm(residual).item())
        assert monitor.history[0] == pytest.approx(expected, rel=1e-10)

    def test_update_sets_initial_residual(self):
        """update() sets initial_residual on first call."""
        monitor = ResidualMonitor()
        residual = torch.tensor([1.0, 0.0], dtype=CFD_DTYPE)

        monitor.update(residual, iteration=0)

        assert monitor.initial_residual is not None
        expected = float(torch.norm(residual).item())
        assert monitor.initial_residual == pytest.approx(expected, rel=1e-10)

    def test_update_tracks_multiple_residuals(self):
        """update() appends to history for multiple calls."""
        monitor = ResidualMonitor()

        for i in range(5):
            res = torch.tensor([10.0 ** (-i)], dtype=CFD_DTYPE)
            monitor.update(res, iteration=i)

        assert len(monitor.history) == 5
        # History is decreasing
        for i in range(4):
            assert monitor.history[i] >= monitor.history[i + 1]

    def test_update_returns_bool(self):
        """update() returns True/False for convergence."""
        monitor = ResidualMonitor(tolerance=1e-6)
        residual = torch.tensor([1e-10], dtype=CFD_DTYPE)

        converged = monitor.update(residual, iteration=0)
        assert isinstance(converged, bool)

    # ------------------------------------------------------------------
    # Absolute convergence
    # ------------------------------------------------------------------

    def test_absolute_convergence_below_tolerance(self):
        """Converges when residual norm < tolerance."""
        monitor = ResidualMonitor(tolerance=1e-3)
        residual = torch.tensor([1e-5], dtype=CFD_DTYPE)

        converged = monitor.update(residual, iteration=0)
        assert converged is True

    def test_absolute_convergence_above_tolerance(self):
        """Does not converge when residual norm > tolerance."""
        monitor = ResidualMonitor(tolerance=1e-6)
        residual = torch.tensor([1.0], dtype=CFD_DTYPE)

        converged = monitor.update(residual, iteration=0)
        assert converged is False

    def test_absolute_convergence_at_boundary(self):
        """Convergence at exact tolerance boundary."""
        monitor = ResidualMonitor(tolerance=1e-3)
        # Residual exactly at tolerance → res_norm < tolerance is False
        residual = torch.tensor([1e-3], dtype=CFD_DTYPE)

        converged = monitor.update(residual, iteration=0)
        # Strict less-than: exactly at tolerance is NOT converged
        assert converged is False

    # ------------------------------------------------------------------
    # Relative convergence
    # ------------------------------------------------------------------

    def test_relative_convergence(self):
        """Converges when residual / initial_residual < rel_tol."""
        monitor = ResidualMonitor(tolerance=1e-15, rel_tol=0.1)

        # First iteration: large residual
        res_init = torch.tensor([100.0], dtype=CFD_DTYPE)
        monitor.update(res_init, iteration=0)

        # Second iteration: small residual (ratio < 0.1)
        res_small = torch.tensor([5.0], dtype=CFD_DTYPE)
        converged = monitor.update(res_small, iteration=1)

        assert converged is True

    def test_relative_convergence_not_met(self):
        """Does not converge when ratio > rel_tol."""
        monitor = ResidualMonitor(tolerance=1e-15, rel_tol=0.01)

        res_init = torch.tensor([100.0], dtype=CFD_DTYPE)
        monitor.update(res_init, iteration=0)

        # Ratio = 0.5 / 0.01 → not converged
        res_still_large = torch.tensor([50.0], dtype=CFD_DTYPE)
        converged = monitor.update(res_still_large, iteration=1)

        assert converged is False

    def test_relative_convergence_zero_initial(self):
        """Handles zero initial residual gracefully."""
        monitor = ResidualMonitor(tolerance=1e-15, rel_tol=0.01)

        # Zero initial residual
        res_zero = torch.zeros(3, dtype=CFD_DTYPE)
        converged = monitor.update(res_zero, iteration=0)

        # With zero initial, absolute check applies (0 < 1e-15 is True)
        assert converged is True

    # ------------------------------------------------------------------
    # Minimum iterations
    # ------------------------------------------------------------------

    def test_min_iter_delays_convergence(self):
        """Convergence is delayed until min_iter is reached."""
        monitor = ResidualMonitor(tolerance=1.0, min_iter=3)

        # Even with tiny residual, should not converge before iter 3
        res_tiny = torch.tensor([1e-20], dtype=CFD_DTYPE)

        assert monitor.update(res_tiny, iteration=0) is False
        assert monitor.update(res_tiny, iteration=1) is False
        assert monitor.update(res_tiny, iteration=2) is False
        # Now iteration=3 >= min_iter=3
        assert monitor.update(res_tiny, iteration=3) is True

    def test_min_iter_zero_allows_immediate_convergence(self):
        """With min_iter=0, convergence can happen on first iteration."""
        monitor = ResidualMonitor(tolerance=1.0, min_iter=0)

        res_tiny = torch.tensor([1e-20], dtype=CFD_DTYPE)
        converged = monitor.update(res_tiny, iteration=0)
        assert converged is True

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def test_reset_clears_history(self):
        """reset() clears the residual history."""
        monitor = ResidualMonitor()

        for i in range(5):
            monitor.update(torch.tensor([1.0], dtype=CFD_DTYPE), iteration=i)

        assert len(monitor.history) == 5

        monitor.reset()
        assert len(monitor.history) == 0

    def test_reset_clears_initial_residual(self):
        """reset() clears the initial residual."""
        monitor = ResidualMonitor()

        monitor.update(torch.tensor([1.0], dtype=CFD_DTYPE), iteration=0)
        assert monitor.initial_residual is not None

        monitor.reset()
        assert monitor.initial_residual is None

    def test_reset_allows_fresh_solve(self):
        """After reset, monitor can be used for a new solve."""
        monitor = ResidualMonitor(tolerance=1e-6)

        # First solve
        monitor.update(torch.tensor([10.0], dtype=CFD_DTYPE), iteration=0)
        monitor.update(torch.tensor([1.0], dtype=CFD_DTYPE), iteration=1)

        monitor.reset()

        # Second solve — history should be fresh
        assert len(monitor.history) == 0
        assert monitor.initial_residual is None

        monitor.update(torch.tensor([5.0], dtype=CFD_DTYPE), iteration=0)
        assert len(monitor.history) == 1
        assert monitor.initial_residual == pytest.approx(5.0, rel=1e-10)

    # ------------------------------------------------------------------
    # build_info
    # ------------------------------------------------------------------

    def test_build_info_converged(self):
        """build_info with converged=True produces correct info."""
        monitor = ResidualMonitor(tolerance=1e-6)

        monitor.update(torch.tensor([10.0], dtype=CFD_DTYPE), iteration=0)
        monitor.update(torch.tensor([1.0], dtype=CFD_DTYPE), iteration=1)
        monitor.update(torch.tensor([0.01], dtype=CFD_DTYPE), iteration=2)

        info = monitor.build_info(converged=True)

        assert info.converged is True
        assert info.iterations == 3
        assert info.initial_residual == pytest.approx(10.0, rel=1e-10)
        assert info.final_residual == pytest.approx(0.01, rel=1e-10)
        assert info.residual_ratio == pytest.approx(0.001, rel=1e-10)
        assert info.tolerance == 1e-6
        assert len(info.residual_history) == 3

    def test_build_info_not_converged(self):
        """build_info with converged=False."""
        monitor = ResidualMonitor(tolerance=1e-6)

        monitor.update(torch.tensor([1.0], dtype=CFD_DTYPE), iteration=0)

        info = monitor.build_info(converged=False)

        assert info.converged is False
        assert info.iterations == 1

    def test_build_info_empty_history(self):
        """build_info with no updates returns zero values."""
        monitor = ResidualMonitor()
        info = monitor.build_info(converged=False)

        assert info.converged is False
        assert info.iterations == 0
        assert info.final_residual == 0.0
        assert info.initial_residual == 0.0
        assert info.residual_ratio == 0.0
        assert len(info.residual_history) == 0

    def test_build_info_residual_ratio(self):
        """residual_ratio = final / initial."""
        monitor = ResidualMonitor()

        monitor.update(torch.tensor([100.0], dtype=CFD_DTYPE), iteration=0)
        monitor.update(torch.tensor([25.0], dtype=CFD_DTYPE), iteration=1)

        info = monitor.build_info(converged=False)

        assert info.residual_ratio == pytest.approx(0.25, rel=1e-10)


# ===========================================================================
# T1.5.2 — ConvergenceInfo
# ===========================================================================


class TestConvergenceInfo:
    """Tests for ConvergenceInfo dataclass."""

    def test_construction_all_fields(self):
        """ConvergenceInfo can be created with all fields."""
        info = ConvergenceInfo(
            converged=True,
            iterations=10,
            final_residual=1e-7,
            initial_residual=1e-2,
            residual_ratio=1e-5,
            tolerance=1e-6,
        )
        assert info.converged is True
        assert info.iterations == 10
        assert info.final_residual == 1e-7
        assert info.initial_residual == 1e-2
        assert info.residual_ratio == 1e-5
        assert info.tolerance == 1e-6

    def test_residual_history_default_empty(self):
        """residual_history defaults to empty list."""
        info = ConvergenceInfo(
            converged=True,
            iterations=0,
            final_residual=0.0,
            initial_residual=0.0,
            residual_ratio=0.0,
            tolerance=1e-6,
        )
        assert info.residual_history == []

    def test_residual_history_custom(self):
        """residual_history can be populated."""
        history = [1.0, 0.5, 0.1, 0.01, 0.001]
        info = ConvergenceInfo(
            converged=True,
            iterations=5,
            final_residual=0.001,
            initial_residual=1.0,
            residual_ratio=0.001,
            tolerance=1e-6,
            residual_history=history,
        )
        assert len(info.residual_history) == 5
        assert info.residual_history[0] == 1.0
        assert info.residual_history[-1] == 0.001

    def test_converged_is_bool(self):
        """converged field is boolean."""
        info_true = ConvergenceInfo(
            converged=True, iterations=1, final_residual=0.0,
            initial_residual=0.0, residual_ratio=0.0, tolerance=1e-6,
        )
        info_false = ConvergenceInfo(
            converged=False, iterations=100, final_residual=1.0,
            initial_residual=1.0, residual_ratio=1.0, tolerance=1e-6,
        )
        assert info_true.converged is True
        assert info_false.converged is False

    def test_residual_history_independent_instances(self):
        """Each ConvergenceInfo gets its own residual_history."""
        info1 = ConvergenceInfo(
            converged=True, iterations=1, final_residual=0.0,
            initial_residual=0.0, residual_ratio=0.0, tolerance=1e-6,
        )
        info2 = ConvergenceInfo(
            converged=False, iterations=1, final_residual=0.0,
            initial_residual=0.0, residual_ratio=0.0, tolerance=1e-6,
        )
        info1.residual_history.append(0.5)

        assert len(info1.residual_history) == 1
        assert len(info2.residual_history) == 0

    def test_from_residual_monitor_integration(self):
        """ConvergenceInfo built from ResidualMonitor is consistent."""
        monitor = ResidualMonitor(tolerance=1e-6, rel_tol=0.01)

        residuals = [10.0, 5.0, 1.0, 0.1, 0.001]
        for i, r in enumerate(residuals):
            monitor.update(torch.tensor([r], dtype=CFD_DTYPE), iteration=i)

        info = monitor.build_info(converged=True)

        assert info.iterations == 5
        assert info.initial_residual == pytest.approx(10.0)
        assert info.final_residual == pytest.approx(0.001)
        assert len(info.residual_history) == 5
        assert info.tolerance == 1e-6

    def test_fields_are_float(self):
        """Residual fields are Python floats, not tensors."""
        info = ConvergenceInfo(
            converged=True, iterations=1, final_residual=1e-6,
            initial_residual=1.0, residual_ratio=1e-6, tolerance=1e-6,
        )
        assert isinstance(info.final_residual, float)
        assert isinstance(info.initial_residual, float)
        assert isinstance(info.residual_ratio, float)
        assert isinstance(info.tolerance, float)
        assert isinstance(info.iterations, int)
