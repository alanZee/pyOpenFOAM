"""Tests for time derivative discretisation schemes (ddt).

Covers:
- EulerDdt — first-order implicit Euler
- SteadyStateDdt — zero time derivative
- CrankNicolsonDdt — second-order CN with blending
- DDT_REGISTRY and create_ddt_scheme factory
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.fv_matrix import FvMatrix

from pyfoam.discretisation.ddt import (
    DdtScheme,
    EulerDdt,
    SteadyStateDdt,
    CrankNicolsonDdt,
    DDT_REGISTRY,
    create_ddt_scheme,
)

from tests.unit.discretisation.conftest import make_fv_mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_field(mesh, values=None):
    """Create a scalar field (n_cells,) — default [1.0, 2.0]."""
    n = mesh.n_cells
    if values is None:
        values = [float(i + 1) for i in range(n)]
    return torch.tensor(values, dtype=mesh.dtype, device=mesh.device)


def _vector_field(mesh):
    """Create a vector field (n_cells, 3)."""
    n = mesh.n_cells
    return torch.tensor(
        [[float(i), float(i) + 0.5, float(i) + 0.25] for i in range(n)],
        dtype=mesh.dtype,
        device=mesh.device,
    )


# ---------------------------------------------------------------------------
# EulerDdt tests
# ---------------------------------------------------------------------------


class TestEulerDdt:
    """First-order implicit Euler scheme."""

    def test_diag_coefficient(self, fv_mesh):
        """Diagonal should equal coeff * V / dt."""
        scheme = EulerDdt(fv_mesh)
        phi = _scalar_field(fv_mesh)
        dt = 0.5
        coeff = 2.0

        mat = scheme.ddt(coeff, phi, dt)

        expected_diag = coeff * fv_mesh.cell_volumes / dt
        torch.testing.assert_close(mat.diag, expected_diag)

    def test_source_coefficient(self, fv_mesh):
        """Source should equal coeff * V * phi / dt (when phi_old=None)."""
        scheme = EulerDdt(fv_mesh)
        phi = _scalar_field(fv_mesh)
        dt = 0.5
        coeff = 2.0

        mat = scheme.ddt(coeff, phi, dt)

        expected_source = coeff * fv_mesh.cell_volumes * phi / dt
        torch.testing.assert_close(mat.source, expected_source)

    def test_phi_old_used_for_source(self, fv_mesh):
        """When phi_old is given, source uses phi_old not phi."""
        scheme = EulerDdt(fv_mesh)
        phi = _scalar_field(fv_mesh, [1.0, 2.0])
        phi_old = _scalar_field(fv_mesh, [10.0, 20.0])
        dt = 0.1
        coeff = 1.0

        mat = scheme.ddt(coeff, phi, dt, phi_old=phi_old)

        # Source should use phi_old
        expected_source = coeff * fv_mesh.cell_volumes * phi_old / dt
        torch.testing.assert_close(mat.source, expected_source)

    def test_returns_fvmatrix(self, fv_mesh):
        mat = EulerDdt(fv_mesh).ddt(1.0, _scalar_field(fv_mesh), 0.01)
        assert isinstance(mat, FvMatrix)

    def test_vector_field(self, fv_mesh):
        """Vector fields should use sum over last dim for source."""
        scheme = EulerDdt(fv_mesh)
        phi = _vector_field(fv_mesh)  # (2, 3)
        dt = 0.5
        coeff = 1.0

        mat = scheme.ddt(coeff, phi, dt)

        expected_source = coeff * fv_mesh.cell_volumes * phi.sum(dim=-1) / dt
        torch.testing.assert_close(mat.source, expected_source)

    def test_repr(self, fv_mesh):
        r = repr(EulerDdt(fv_mesh))
        assert "EulerDdt" in r


# ---------------------------------------------------------------------------
# SteadyStateDdt tests
# ---------------------------------------------------------------------------


class TestSteadyStateDdt:
    """Steady-state (zero time derivative) scheme."""

    def test_diag_is_zero(self, fv_mesh):
        mat = SteadyStateDdt(fv_mesh).ddt(1.0, _scalar_field(fv_mesh), 1.0)
        torch.testing.assert_close(
            mat.diag,
            torch.zeros(fv_mesh.n_cells, dtype=fv_mesh.dtype, device=fv_mesh.device),
        )

    def test_source_is_zero(self, fv_mesh):
        mat = SteadyStateDdt(fv_mesh).ddt(1.0, _scalar_field(fv_mesh), 1.0)
        torch.testing.assert_close(
            mat.source,
            torch.zeros(fv_mesh.n_cells, dtype=fv_mesh.dtype, device=fv_mesh.device),
        )

    def test_coeff_and_dt_ignored(self, fv_mesh):
        """Coeff and dt should have no effect (result always zero)."""
        mat = SteadyStateDdt(fv_mesh).ddt(999.0, _scalar_field(fv_mesh), 1e-12)
        torch.testing.assert_close(
            mat.diag,
            torch.zeros(fv_mesh.n_cells, dtype=fv_mesh.dtype, device=fv_mesh.device),
        )

    def test_returns_fvmatrix(self, fv_mesh):
        mat = SteadyStateDdt(fv_mesh).ddt(1.0, _scalar_field(fv_mesh), 1.0)
        assert isinstance(mat, FvMatrix)


# ---------------------------------------------------------------------------
# CrankNicolsonDdt tests
# ---------------------------------------------------------------------------


class TestCrankNicolsonDdt:
    """Second-order Crank-Nicolson scheme."""

    def test_theta_1_diag(self, fv_mesh):
        """theta=1.0: diag = coeff * V * 1.0 / dt."""
        scheme = CrankNicolsonDdt(fv_mesh, theta=1.0)
        phi = _scalar_field(fv_mesh, [1.0, 2.0])
        phi_old = _scalar_field(fv_mesh, [0.5, 1.5])
        dt = 0.25
        coeff = 3.0

        mat = scheme.ddt(coeff, phi, dt, phi_old=phi_old)

        expected_diag = coeff * fv_mesh.cell_volumes * 1.0 / dt
        torch.testing.assert_close(mat.diag, expected_diag)

    def test_theta_0_is_euler(self, fv_mesh):
        """theta=0.0 should degenerate to Euler (source = coeff*V*phi/dt)."""
        scheme = CrankNicolsonDdt(fv_mesh, theta=0.0)
        phi = _scalar_field(fv_mesh, [1.0, 2.0])
        phi_old = _scalar_field(fv_mesh, [0.5, 1.5])
        dt = 0.5
        coeff = 1.0

        mat = scheme.ddt(coeff, phi, dt, phi_old=phi_old)

        # diag = 0 (theta=0)
        torch.testing.assert_close(
            mat.diag,
            torch.zeros(fv_mesh.n_cells, dtype=fv_mesh.dtype, device=fv_mesh.device),
        )
        # source = coeff * V * (0*phi_old + 1*phi) / dt = coeff * V * phi / dt
        expected_source = coeff * fv_mesh.cell_volumes * phi / dt
        torch.testing.assert_close(mat.source, expected_source)

    def test_blended_source(self, fv_mesh):
        """source = coeff * V / dt * [theta*phi_old + (1-theta)*phi]."""
        theta = 0.5
        scheme = CrankNicolsonDdt(fv_mesh, theta=theta)
        phi = _scalar_field(fv_mesh, [2.0, 4.0])
        phi_old = _scalar_field(fv_mesh, [0.0, 0.0])
        dt = 0.5
        coeff = 1.0

        mat = scheme.ddt(coeff, phi, dt, phi_old=phi_old)

        blended = theta * phi_old + (1.0 - theta) * phi
        expected_source = coeff * fv_mesh.cell_volumes * blended / dt
        torch.testing.assert_close(mat.source, expected_source)

    def test_phi_old_required(self, fv_mesh):
        scheme = CrankNicolsonDdt(fv_mesh)
        phi = _scalar_field(fv_mesh)
        with pytest.raises(ValueError, match="requires phi_old"):
            scheme.ddt(1.0, phi, 0.01)

    def test_theta_property(self, fv_mesh):
        scheme = CrankNicolsonDdt(fv_mesh, theta=0.75)
        assert scheme.theta == 0.75

    def test_invalid_theta(self, fv_mesh):
        with pytest.raises(ValueError, match="theta must be in"):
            CrankNicolsonDdt(fv_mesh, theta=1.5)
        with pytest.raises(ValueError, match="theta must be in"):
            CrankNicolsonDdt(fv_mesh, theta=-0.1)

    def test_vector_field(self, fv_mesh):
        """Vector fields should use sum over last dim."""
        theta = 0.5
        scheme = CrankNicolsonDdt(fv_mesh, theta=theta)
        phi = _vector_field(fv_mesh)
        phi_old = torch.zeros_like(phi)
        dt = 1.0
        coeff = 1.0

        mat = scheme.ddt(coeff, phi, dt, phi_old=phi_old)

        blended = theta * phi_old + (1.0 - theta) * phi
        expected_source = coeff * fv_mesh.cell_volumes * blended.sum(dim=-1) / dt
        torch.testing.assert_close(mat.source, expected_source)

    def test_returns_fvmatrix(self, fv_mesh):
        phi = _scalar_field(fv_mesh)
        mat = CrankNicolsonDdt(fv_mesh).ddt(1.0, phi, 0.01, phi_old=phi)
        assert isinstance(mat, FvMatrix)


# ---------------------------------------------------------------------------
# DDT_REGISTRY and create_ddt_scheme tests
# ---------------------------------------------------------------------------


class TestDdtRegistry:
    """Scheme registry and factory function."""

    def test_registry_contains_expected_schemes(self):
        assert "Euler" in DDT_REGISTRY
        assert "steadyState" in DDT_REGISTRY
        assert "CrankNicolson" in DDT_REGISTRY

    def test_registry_values_are_classes(self):
        assert DDT_REGISTRY["Euler"] is EulerDdt
        assert DDT_REGISTRY["steadyState"] is SteadyStateDdt
        assert DDT_REGISTRY["CrankNicolson"] is CrankNicolsonDdt

    def test_create_euler(self, fv_mesh):
        scheme = create_ddt_scheme("Euler", fv_mesh)
        assert isinstance(scheme, EulerDdt)

    def test_create_steady_state(self, fv_mesh):
        scheme = create_ddt_scheme("steadyState", fv_mesh)
        assert isinstance(scheme, SteadyStateDdt)

    def test_create_cn_with_theta(self, fv_mesh):
        scheme = create_ddt_scheme("CrankNicolson", fv_mesh, theta=0.8)
        assert isinstance(scheme, CrankNicolsonDdt)
        assert scheme.theta == 0.8

    def test_unknown_scheme_raises(self, fv_mesh):
        with pytest.raises(ValueError, match="Unknown ddt scheme"):
            create_ddt_scheme("Nonexistent", fv_mesh)


# ---------------------------------------------------------------------------
# Consistency / numerical tests
# ---------------------------------------------------------------------------


class TestDdtConsistency:
    """Cross-scheme consistency checks."""

    def test_euler_matches_current_fvm_ddt(self, fv_mesh):
        """EulerDdt should produce the same result as the old fvm.ddt logic.

        Old logic: diag  = coeff * V / dt
                   source = coeff * V * phi / dt
        """
        from pyfoam.discretisation.operators import _FvmNamespace, _make_fvmatrix

        coeff = 1.0
        phi = _scalar_field(fv_mesh, [3.0, 7.0])
        dt = 0.1

        # Old logic (directly from operators.py)
        device = fv_mesh.device
        dtype = fv_mesh.dtype
        n_cells = fv_mesh.n_cells
        cell_volumes = fv_mesh.cell_volumes
        phi_data = phi.to(device=device, dtype=dtype)
        old_mat = _make_fvmatrix(fv_mesh)
        old_mat.diag = coeff * cell_volumes / dt
        old_mat.source = coeff * cell_volumes * phi_data / dt

        # New EulerDdt
        new_mat = EulerDdt(fv_mesh).ddt(coeff, phi, dt)

        torch.testing.assert_close(new_mat.diag, old_mat.diag)
        torch.testing.assert_close(new_mat.source, old_mat.source)

    def test_steady_state_does_not_contribute(self, fv_mesh):
        """SteadyStateDdt matrices should be all-zero (no equation contribution)."""
        mat = SteadyStateDdt(fv_mesh).ddt(1.0, _scalar_field(fv_mesh), 0.001)
        assert mat.diag.abs().max() == 0.0
        assert mat.source.abs().max() == 0.0

    def test_cn_blending_covers_euler_range(self, fv_mesh):
        """At theta=0 CN source equals Euler source (with phi_old=phi)."""
        phi = _scalar_field(fv_mesh, [5.0, 10.0])
        dt = 0.2
        coeff = 2.0

        euler_source = EulerDdt(fv_mesh).ddt(coeff, phi, dt).source

        cn = CrankNicolsonDdt(fv_mesh, theta=0.0)
        cn_source = cn.ddt(coeff, phi, dt, phi_old=phi).source

        # With phi_old == phi and theta=0: source = coeff*V*phi/dt == Euler
        torch.testing.assert_close(cn_source, euler_source)
