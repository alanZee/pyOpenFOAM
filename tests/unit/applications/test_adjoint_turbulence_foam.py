"""
Unit tests for AdjointTurbulenceFoam — adjoint turbulence optimisation solver.

Tests cover:
- Case loading and field initialisation with turbulence fields
- Adjoint turbulence field initialisation (zero default)
- Turbulence model source terms
- Adjoint k and omega solve
- Turbulence sensitivity computation
- Full run completion
- Turbulence sensitivity field shape and finiteness
- Export functionality
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


# Reuse the adjoint case helper
from tests.unit.applications.test_adjoint_foam import _make_adjoint_case


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adjoint_turb_case(tmp_path):
    """Create a 4x4 adjoint turbulence case."""
    case_dir = tmp_path / "adjoint_turb"
    _make_adjoint_case(case_dir, n_cells_x=4, n_cells_y=4, end_time=5, delta_t=1.0)
    return case_dir


@pytest.fixture
def tiny_adjoint_turb_case(tmp_path):
    """Create a minimal 2x2 adjoint turbulence case."""
    case_dir = tmp_path / "tiny_adj_turb"
    _make_adjoint_case(case_dir, n_cells_x=2, n_cells_y=2, end_time=2, delta_t=1.0)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestAdjointTurbulenceFoamInit:
    """Tests for AdjointTurbulenceFoam initialisation."""

    def test_case_loads(self, adjoint_turb_case):
        """Case directory is readable."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        assert solver.mesh is not None
        assert solver.mesh.n_cells == 16

    def test_turbulence_fields_initialise(self, adjoint_turb_case):
        """Turbulence fields k, omega, nut are initialised."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        assert solver.k.shape == (16,)
        assert solver.omega.shape == (16,)
        assert solver.nut.shape == (16,)
        # Default k is 1e-4, omega is 1.0
        assert torch.allclose(solver.k, torch.full((16,), 1e-4, dtype=CFD_DTYPE))

    def test_adjoint_turb_fields_initialise_to_zero(self, adjoint_turb_case):
        """Adjoint turbulence fields ka, omega_a default to zero."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        assert solver.ka.shape == (16,)
        assert solver.omega_a.shape == (16,)
        assert torch.allclose(solver.ka, torch.zeros_like(solver.ka))
        assert torch.allclose(solver.omega_a, torch.zeros_like(solver.omega_a))

    def test_turbulence_sensitivity_initialised(self, adjoint_turb_case):
        """Turbulence sensitivity field starts as zeros."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        assert solver.turbulence_sensitivity.shape == (16,)
        assert torch.allclose(
            solver.turbulence_sensitivity,
            torch.zeros_like(solver.turbulence_sensitivity),
        )

    def test_primal_fields_initialise(self, adjoint_turb_case):
        """Primal fields U, p are initialised from 0/ directory."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)

    def test_objective_default(self, adjoint_turb_case):
        """Default objective is 'drag' and turbulence model is 'kOmegaSST'."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        assert solver.objective == "drag"
        assert solver.turb_model == "kOmegaSST"


class TestAdjointTurbulenceFoamSolve:
    """Tests for adjoint turbulence solve steps."""

    def test_adjoint_k_produces_finite_values(self, adjoint_turb_case):
        """_solve_adjoint_k returns finite values."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        ka_new = solver._solve_adjoint_k()
        assert torch.isfinite(ka_new).all()
        assert ka_new.shape == (16,)

    def test_adjoint_omega_produces_finite_values(self, adjoint_turb_case):
        """_solve_adjoint_omega returns finite values."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        omega_a_new = solver._solve_adjoint_omega()
        assert torch.isfinite(omega_a_new).all()
        assert omega_a_new.shape == (16,)

    def test_turbulence_k_source_nonzero(self, adjoint_turb_case):
        """Turbulence k source is nonzero on boundary cells."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        source = solver._turbulence_k_source()
        assert source.shape == (16,)
        assert source.abs().sum() > 0

    def test_turbulence_omega_source_nonzero(self, adjoint_turb_case):
        """Turbulence omega source is nonzero on boundary cells."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        source = solver._turbulence_omega_source()
        assert source.shape == (16,)
        assert source.abs().sum() > 0

    def test_turbulence_sensitivity_computation(self, adjoint_turb_case):
        """Turbulence sensitivity combines ka and omega_a."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(adjoint_turb_case)
        # Set some non-zero adjoint values
        solver.ka = torch.ones(16, dtype=CFD_DTYPE)
        solver.omega_a = torch.ones(16, dtype=CFD_DTYPE) * 0.5
        sens = solver._compute_turbulence_sensitivity()
        expected = solver.ka * solver.k + solver.omega_a * solver.omega
        assert torch.allclose(sens, expected)


class TestAdjointTurbulenceFoamRun:
    """Tests for the full solver run."""

    def test_run_completes(self, tiny_adjoint_turb_case):
        """AdjointTurbulenceFoam runs to completion."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(tiny_adjoint_turb_case)
        solver.end_time = 1.0
        conv = solver.run()
        assert conv is not None

    def test_fields_finite_after_run(self, tiny_adjoint_turb_case):
        """All field values are finite after run."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(tiny_adjoint_turb_case)
        solver.end_time = 1.0
        solver.run()
        assert torch.isfinite(solver.Ua).all()
        assert torch.isfinite(solver.pa).all()
        assert torch.isfinite(solver.ka).all()
        assert torch.isfinite(solver.omega_a).all()

    def test_turbulence_sensitivity_computed_after_run(self, tiny_adjoint_turb_case):
        """Turbulence sensitivity is computed after run."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(tiny_adjoint_turb_case)
        solver.end_time = 1.0
        solver.run()
        assert solver.turbulence_sensitivity.shape == (4,)
        assert torch.isfinite(solver.turbulence_sensitivity).all()

    def test_shape_sensitivity_computed_after_run(self, tiny_adjoint_turb_case):
        """Shape sensitivity is computed after run."""
        from pyfoam.applications.adjoint_turbulence_foam import AdjointTurbulenceFoam

        solver = AdjointTurbulenceFoam(tiny_adjoint_turb_case)
        solver.end_time = 1.0
        solver.run()
        assert solver.sensitivity.shape == (4,)
        assert torch.isfinite(solver.sensitivity).all()

    def test_export_availability(self):
        """AdjointTurbulenceFoam is importable from applications."""
        from pyfoam.applications import AdjointTurbulenceFoam

        assert AdjointTurbulenceFoam is not None
