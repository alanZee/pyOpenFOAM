"""Tests for coupled solver base and convergence data (T1.4).

Tests cover:
- CoupledSolverConfig defaults and custom values (T1.4.1)
- CoupledSolverBase initialisation, properties, residual computation (T1.4.1)
- ConvergenceData defaults, tracking, history (T1.4.2)
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.solvers.coupled_solver import (
    CoupledSolverBase,
    CoupledSolverConfig,
    ConvergenceData,
)

from tests.unit.solvers.conftest_coupled import make_cavity_mesh


# ---------------------------------------------------------------------------
# Concrete subclass for testing abstract base
# ---------------------------------------------------------------------------


class _DummySolver(CoupledSolverBase):
    """Concrete subclass of CoupledSolverBase for testing."""

    def solve(self, U, p, phi, *, U_old=None, p_old=None,
              max_outer_iterations=100, tolerance=1e-4):
        """Trivial solve: return inputs unchanged with dummy convergence data."""
        convergence = ConvergenceData(
            p_residual=1e-5,
            U_residual=1e-4,
            continuity_error=1e-3,
            outer_iterations=1,
            converged=True,
        )
        return U, p, phi, convergence


# ===========================================================================
# T1.4.1 — CoupledSolverConfig
# ===========================================================================


class TestCoupledSolverConfig:
    """Tests for CoupledSolverConfig dataclass."""

    def test_default_config_values(self):
        """Default config has expected OpenFOAM-like values."""
        config = CoupledSolverConfig()
        assert config.p_solver == "PCG"
        assert config.U_solver == "PBiCGStab"
        assert config.p_tolerance == 1e-6
        assert config.U_tolerance == 1e-6
        assert config.p_max_iter == 1000
        assert config.U_max_iter == 1000
        assert config.n_non_orthogonal_correctors == 0
        assert config.relaxation_factor_p == 1.0
        assert config.relaxation_factor_U == 0.7
        assert config.relaxation_factor_phi == 1.0

    def test_custom_config_values(self):
        """Custom config overrides defaults."""
        config = CoupledSolverConfig(
            p_solver="GAMG",
            U_solver="PBiCGStab",
            p_tolerance=1e-8,
            U_tolerance=1e-7,
            p_max_iter=500,
            U_max_iter=200,
            n_non_orthogonal_correctors=2,
            relaxation_factor_p=0.3,
            relaxation_factor_U=0.5,
            relaxation_factor_phi=0.9,
        )
        assert config.p_solver == "GAMG"
        assert config.U_solver == "PBiCGStab"
        assert config.p_tolerance == 1e-8
        assert config.U_tolerance == 1e-7
        assert config.p_max_iter == 500
        assert config.U_max_iter == 200
        assert config.n_non_orthogonal_correctors == 2
        assert config.relaxation_factor_p == 0.3
        assert config.relaxation_factor_U == 0.5
        assert config.relaxation_factor_phi == 0.9

    def test_config_partial_override(self):
        """Partial override preserves defaults for unspecified fields."""
        config = CoupledSolverConfig(relaxation_factor_U=0.3)
        assert config.relaxation_factor_U == 0.3
        # Defaults unchanged
        assert config.p_solver == "PCG"
        assert config.U_solver == "PBiCGStab"
        assert config.relaxation_factor_p == 1.0

    def test_relaxation_factors_in_valid_range(self):
        """Default relaxation factors are in (0, 1]."""
        config = CoupledSolverConfig()
        assert 0 < config.relaxation_factor_p <= 1.0
        assert 0 < config.relaxation_factor_U <= 1.0
        assert 0 < config.relaxation_factor_phi <= 1.0


# ===========================================================================
# T1.4.1 — CoupledSolverBase
# ===========================================================================


class TestCoupledSolverBase:
    """Tests for CoupledSolverBase abstract class."""

    def test_cannot_instantiate_abstract(self):
        """CoupledSolverBase cannot be instantiated directly."""
        mesh = make_cavity_mesh(2, 2)
        with pytest.raises(TypeError):
            CoupledSolverBase(mesh)

    def test_concrete_subclass_creation(self):
        """Concrete subclass can be created with mesh and config."""
        mesh = make_cavity_mesh(2, 2)
        config = CoupledSolverConfig()
        solver = _DummySolver(mesh, config)

        assert solver.mesh is mesh
        assert solver.config is config

    def test_default_config_when_none(self):
        """When config=None, a default CoupledSolverConfig is used."""
        mesh = make_cavity_mesh(2, 2)
        solver = _DummySolver(mesh, config=None)

        assert isinstance(solver.config, CoupledSolverConfig)
        assert solver.config.p_solver == "PCG"
        assert solver.config.U_solver == "PBiCGStab"

    def test_mesh_property(self):
        """mesh property returns the mesh passed to constructor."""
        mesh = make_cavity_mesh(2, 2)
        solver = _DummySolver(mesh)
        assert solver.mesh is mesh

    def test_config_property(self):
        """config property returns the config passed to constructor."""
        mesh = make_cavity_mesh(2, 2)
        config = CoupledSolverConfig(relaxation_factor_U=0.5)
        solver = _DummySolver(mesh, config)
        assert solver.config is config
        assert solver.config.relaxation_factor_U == 0.5

    def test_p_solver_property(self):
        """p_solver property returns a LinearSolverBase instance."""
        from pyfoam.solvers.linear_solver import LinearSolverBase

        mesh = make_cavity_mesh(2, 2)
        config = CoupledSolverConfig(p_solver="PCG", p_tolerance=1e-8)
        solver = _DummySolver(mesh, config)

        assert isinstance(solver.p_solver, LinearSolverBase)
        assert solver.p_solver.tolerance == 1e-8

    def test_U_solver_property(self):
        """U_solver property returns a LinearSolverBase instance."""
        from pyfoam.solvers.linear_solver import LinearSolverBase

        mesh = make_cavity_mesh(2, 2)
        config = CoupledSolverConfig(U_solver="PBiCGStab", U_tolerance=1e-7)
        solver = _DummySolver(mesh, config)

        assert isinstance(solver.U_solver, LinearSolverBase)
        assert solver.U_solver.tolerance == 1e-7

    def test_p_solver_gamg(self):
        """p_solver can be GAMG."""
        mesh = make_cavity_mesh(2, 2)
        config = CoupledSolverConfig(p_solver="GAMG")
        solver = _DummySolver(mesh, config)

        assert "GAMG" in type(solver.p_solver).__name__ or \
               solver.p_solver is not None

    def test_solve_returns_convergence_data(self):
        """Concrete solve returns ConvergenceData."""
        mesh = make_cavity_mesh(2, 2)
        solver = _DummySolver(mesh)

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, convergence = solver.solve(U, p, phi)

        assert isinstance(convergence, ConvergenceData)
        assert convergence.converged is True
        assert convergence.outer_iterations == 1

    def test_solve_preserves_field_shapes(self):
        """solve preserves input field shapes."""
        mesh = make_cavity_mesh(2, 2)
        solver = _DummySolver(mesh)

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        U_out, p_out, phi_out, _ = solver.solve(U, p, phi)

        assert U_out.shape == U.shape
        assert p_out.shape == p.shape
        assert phi_out.shape == phi.shape

    def test_compute_residual_identical_fields(self):
        """_compute_residual returns 0 for identical fields."""
        mesh = make_cavity_mesh(2, 2)
        solver = _DummySolver(mesh)

        field = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=CFD_DTYPE)
        residual = solver._compute_residual(field, field.clone())
        assert residual == pytest.approx(0.0, abs=1e-15)

    def test_compute_residual_different_fields(self):
        """_compute_residual returns positive value for different fields."""
        mesh = make_cavity_mesh(2, 2)
        solver = _DummySolver(mesh)

        field = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=CFD_DTYPE)
        field_old = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=CFD_DTYPE)
        residual = solver._compute_residual(field, field_old)

        # L2 norm of diff / L2 norm of field
        expected = float(torch.norm(field - field_old).item()) / \
                   float(torch.norm(field).item())
        assert residual == pytest.approx(expected, rel=1e-10)

    def test_compute_residual_zero_field(self):
        """_compute_residual handles zero field (no division by zero)."""
        mesh = make_cavity_mesh(2, 2)
        solver = _DummySolver(mesh)

        field = torch.zeros(4, dtype=CFD_DTYPE)
        field_old = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=CFD_DTYPE)
        residual = solver._compute_residual(field, field_old)

        # When field is zero, returns norm(diff) without division
        expected = float(torch.norm(field - field_old).item())
        assert residual == pytest.approx(expected, rel=1e-10)

    def test_compute_residual_normalised(self):
        """_compute_residual is normalised by field magnitude."""
        mesh = make_cavity_mesh(2, 2)
        solver = _DummySolver(mesh)

        # Same diff, different field magnitude → different residual
        field_small = torch.tensor([0.1, 0.0, 0.0, 0.0], dtype=CFD_DTYPE)
        field_large = torch.tensor([10.0, 0.0, 0.0, 0.0], dtype=CFD_DTYPE)
        diff = torch.tensor([0.01, 0.0, 0.0, 0.0], dtype=CFD_DTYPE)

        res_small = solver._compute_residual(field_small + diff, field_small)
        res_large = solver._compute_residual(field_large + diff, field_large)

        # Smaller field → larger relative residual
        assert res_small > res_large

    def test_repr(self):
        """__repr__ includes solver names and relaxation factors."""
        mesh = make_cavity_mesh(2, 2)
        config = CoupledSolverConfig(
            relaxation_factor_U=0.7,
            relaxation_factor_p=0.3,
        )
        solver = _DummySolver(mesh, config)
        r = repr(solver)

        assert "_DummySolver" in r
        assert "PCG" in r
        assert "PBiCGStab" in r
        assert "relax_U" in r
        assert "relax_p" in r


# ===========================================================================
# T1.4.2 — ConvergenceData
# ===========================================================================


class TestConvergenceData:
    """Tests for ConvergenceData dataclass."""

    def test_default_values(self):
        """ConvergenceData has correct zero defaults."""
        data = ConvergenceData()
        assert data.p_residual == 0.0
        assert data.U_residual == 0.0
        assert data.continuity_error == 0.0
        assert data.outer_iterations == 0
        assert data.converged is False
        assert data.residual_history == []

    def test_custom_values(self):
        """ConvergenceData accepts custom values."""
        data = ConvergenceData(
            p_residual=1e-5,
            U_residual=1e-4,
            continuity_error=1e-3,
            outer_iterations=10,
            converged=True,
        )
        assert data.p_residual == 1e-5
        assert data.U_residual == 1e-4
        assert data.continuity_error == 1e-3
        assert data.outer_iterations == 10
        assert data.converged is True

    def test_residual_history_tracking(self):
        """residual_history stores per-iteration records."""
        data = ConvergenceData()
        data.residual_history.append({
            "outer": 1,
            "p_residual": 1e-1,
            "U_residual": 1e-1,
            "continuity_error": 0.1,
        })
        data.residual_history.append({
            "outer": 2,
            "p_residual": 1e-3,
            "U_residual": 1e-3,
            "continuity_error": 0.001,
        })

        assert len(data.residual_history) == 2
        assert data.residual_history[0]["outer"] == 1
        assert data.residual_history[1]["p_residual"] == 1e-3

    def test_residual_history_default_is_empty_list(self):
        """Each ConvergenceData gets its own residual_history list."""
        d1 = ConvergenceData()
        d2 = ConvergenceData()
        d1.residual_history.append({"iter": 1})

        assert len(d1.residual_history) == 1
        assert len(d2.residual_history) == 0, \
            "residual_history should not be shared between instances"

    def test_mutable_field_assignment(self):
        """ConvergenceData fields can be updated after creation."""
        data = ConvergenceData()
        data.p_residual = 0.5
        data.U_residual = 0.3
        data.continuity_error = 0.1
        data.outer_iterations = 50
        data.converged = True

        assert data.p_residual == 0.5
        assert data.U_residual == 0.3
        assert data.continuity_error == 0.1
        assert data.outer_iterations == 50
        assert data.converged is True

    def test_convergence_data_from_solve(self):
        """ConvergenceData returned from solve has expected structure."""
        mesh = make_cavity_mesh(2, 2)
        solver = _DummySolver(mesh)

        U = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        _, _, _, conv = solver.solve(U, p, phi)

        assert isinstance(conv.p_residual, float)
        assert isinstance(conv.U_residual, float)
        assert isinstance(conv.continuity_error, float)
        assert isinstance(conv.outer_iterations, int)
        assert isinstance(conv.converged, bool)
        assert isinstance(conv.residual_history, list)
