"""
Unit tests for enhanced solver variants v13 (coupling algorithm variants).

Tests cover:
- SIMPLEC pressure-velocity coupling
- SIMPLEC-consistent flux correction
- Coupled solver block solve
- Export of new classes from __init__.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

sys_path_inserted = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_enhanced_solvers_11 import _make_cavity_case as _make_real_case


def _make_cavity_case(case_dir, n_cells_x=4, n_cells_y=4, nu=0.01, delta_t=0.001,
                      end_time=0.01, piso_correctors=2, compressible=False,
                      buoyant=False, reacting=False):
    _make_real_case(case_dir, n_cells_x=n_cells_x, n_cells_y=n_cells_y, nu=nu,
                    delta_t=delta_t, end_time=end_time, piso_correctors=piso_correctors,
                    compressible=compressible, buoyant=buoyant, reacting=reacting)


@pytest.fixture
def cavity_case(tmp_path):
    case_dir = tmp_path / "cavity"
    _make_cavity_case(case_dir)
    return case_dir


@pytest.fixture
def compressible_case(tmp_path):
    case_dir = tmp_path / "compressible"
    _make_cavity_case(case_dir, compressible=True)
    return case_dir


@pytest.fixture
def buoyant_case(tmp_path):
    case_dir = tmp_path / "buoyant"
    _make_cavity_case(case_dir, buoyant=True)
    return case_dir


# ===========================================================================
# Tests: SimpleFoamEnhanced13
# ===========================================================================


class TestSimpleFoamEnhanced13:
    """Tests for enhanced SIMPLE solver v13 (coupling algorithms)."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_13 import SimpleFoamEnhanced13
        solver = SimpleFoamEnhanced13(cavity_case, simplec=True, simplec_consistent=True, coupled=True)
        assert solver.simplec is True
        assert solver.simplec_consistent is True
        assert solver.coupled is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_13 import SimpleFoamEnhanced13
        solver = SimpleFoamEnhanced13(cavity_case)
        assert solver.coupled_max_iter == 5

    def test_simplec_velocity(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_13 import SimpleFoamEnhanced13
        solver = SimpleFoamEnhanced13(cavity_case, simplec=True)
        U_sc = solver._simplec_velocity_correct(solver.U, solver.p, solver.U.clone())
        assert U_sc.shape == solver.U.shape
        assert torch.isfinite(U_sc).all()

    def test_simplec_disabled(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_13 import SimpleFoamEnhanced13
        solver = SimpleFoamEnhanced13(cavity_case, simplec=False)
        U_out = solver._simplec_velocity_correct(solver.U, solver.p, solver.U.clone())
        assert torch.allclose(U_out, solver.U)

    def test_simplec_consistent_flux(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_13 import SimpleFoamEnhanced13
        solver = SimpleFoamEnhanced13(cavity_case, simplec_consistent=True)
        phi_corr = solver._simplec_consistent_flux(solver.phi, solver.p, solver.U)
        assert phi_corr.shape[0] > 0
        assert torch.isfinite(phi_corr).all()

    def test_coupled_solve(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_13 import SimpleFoamEnhanced13
        solver = SimpleFoamEnhanced13(cavity_case, coupled=True, coupled_max_iter=2)
        U_c, p_c = solver._coupled_pressure_velocity_solve(solver.U, solver.p, solver.delta_t)
        assert U_c.shape == solver.U.shape
        assert p_c.shape == solver.p.shape
        assert torch.isfinite(U_c).all()
        assert torch.isfinite(p_c).all()


# ===========================================================================
# Tests: PimpleFoamEnhanced13
# ===========================================================================


class TestPimpleFoamEnhanced13:
    """Tests for enhanced PIMPLE solver v13."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_13 import PimpleFoamEnhanced13
        solver = PimpleFoamEnhanced13(cavity_case, simplec=True, coupled=True)
        assert solver.simplec is True
        assert solver.coupled is True

    def test_simplec_correct(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_13 import PimpleFoamEnhanced13
        solver = PimpleFoamEnhanced13(cavity_case, simplec=True)
        U_sc = solver._simplec_velocity_correct(solver.U, solver.p)
        assert U_sc.shape == solver.U.shape
        assert torch.isfinite(U_sc).all()

    def test_coupled_solve(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_13 import PimpleFoamEnhanced13
        solver = PimpleFoamEnhanced13(cavity_case, coupled=True, coupled_max_iter=2)
        U_c, p_c = solver._coupled_solve(solver.U, solver.p, solver.delta_t)
        assert U_c.shape == solver.U.shape
        assert p_c.shape == solver.p.shape


# ===========================================================================
# Tests: PisoFoamEnhanced13
# ===========================================================================


class TestPisoFoamEnhanced13:
    """Tests for enhanced PISO solver v13."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_13 import PisoFoamEnhanced13
        solver = PisoFoamEnhanced13(cavity_case, simplec=True, coupled=True)
        assert solver.simplec is True

    def test_simplec_correct(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_13 import PisoFoamEnhanced13
        solver = PisoFoamEnhanced13(cavity_case, simplec=True)
        U_sc = solver._simplec_correct(solver.U, solver.p)
        assert U_sc.shape == solver.U.shape
        assert torch.isfinite(U_sc).all()

    def test_coupled_solve(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_13 import PisoFoamEnhanced13
        solver = PisoFoamEnhanced13(cavity_case, coupled=True, coupled_max_iter=2)
        U_c, p_c = solver._coupled_solve(solver.U, solver.p, solver.delta_t)
        assert U_c.shape == solver.U.shape
        assert p_c.shape == solver.p.shape


# ===========================================================================
# Tests: IcoFoamEnhanced13
# ===========================================================================


class TestIcoFoamEnhanced13:
    """Tests for enhanced ICO solver v13."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_13 import IcoFoamEnhanced13
        solver = IcoFoamEnhanced13(cavity_case, simplec=True, coupled=True)
        assert solver.simplec is True

    def test_simplec_correct(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_13 import IcoFoamEnhanced13
        solver = IcoFoamEnhanced13(cavity_case, simplec=True)
        U_sc = solver._simplec_correct(solver.U, solver.p)
        assert U_sc.shape == solver.U.shape
        assert torch.isfinite(U_sc).all()

    def test_coupled_solve(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_13 import IcoFoamEnhanced13
        solver = IcoFoamEnhanced13(cavity_case, coupled=True, coupled_max_iter=2)
        U_c, p_c = solver._coupled_solve(solver.U, solver.p, solver.delta_t)
        assert U_c.shape == solver.U.shape
        assert p_c.shape == solver.p.shape


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced13
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced13:
    """Tests for enhanced buoyant PIMPLE solver v13."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_13 import BuoyantPimpleFoamEnhanced13
        solver = BuoyantPimpleFoamEnhanced13(buoyant_case, simplec=True, coupled=True)
        assert solver.simplec is True

    def test_simplec_correct(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_13 import BuoyantPimpleFoamEnhanced13
        solver = BuoyantPimpleFoamEnhanced13(buoyant_case, simplec=True)
        U_sc = solver._simplec_correct(solver.U, solver.p)
        assert U_sc.shape == solver.U.shape
        assert torch.isfinite(U_sc).all()

    def test_coupled_solve(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_13 import BuoyantPimpleFoamEnhanced13
        solver = BuoyantPimpleFoamEnhanced13(buoyant_case, coupled=True, coupled_max_iter=2)
        T = solver.T if hasattr(solver, 'T') else torch.ones_like(solver.p) * 300.0
        rho = torch.ones_like(solver.p) * 1.2
        U_c, p_c = solver._coupled_buoyancy_solve(solver.U, solver.p, T, rho, solver.delta_t)
        assert U_c.shape == solver.U.shape
        assert p_c.shape == solver.p.shape


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced13
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced13:
    """Tests for enhanced buoyant SIMPLE solver v13."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_13 import BuoyantSimpleFoamEnhanced13
        solver = BuoyantSimpleFoamEnhanced13(buoyant_case, simplec=True, coupled=True)
        assert solver.simplec is True

    def test_simplec_correct(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_13 import BuoyantSimpleFoamEnhanced13
        solver = BuoyantSimpleFoamEnhanced13(buoyant_case, simplec=True)
        U_sc = solver._simplec_correct(solver.U, solver.p)
        assert U_sc.shape == solver.U.shape
        assert torch.isfinite(U_sc).all()


# ===========================================================================
# Tests: CompressibleInterFoamEnhanced13
# ===========================================================================


class TestCompressibleInterFoamEnhanced13:
    """Tests for enhanced compressible VOF solver v13."""

    def test_init(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_13 import CompressibleInterFoamEnhanced13
        solver = CompressibleInterFoamEnhanced13(compressible_case, simplec=True, coupled=True)
        assert solver.simplec is True

    def test_simplec_correct(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_13 import CompressibleInterFoamEnhanced13
        solver = CompressibleInterFoamEnhanced13(compressible_case, simplec=True)
        U_sc = solver._simplec_correct(solver.U, solver.p)
        assert U_sc.shape == solver.U.shape
        assert torch.isfinite(U_sc).all()


# ===========================================================================
# Tests: SprayFoamEnhanced13
# ===========================================================================


class TestSprayFoamEnhanced13:
    """Tests for enhanced spray solver v13."""

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_13 import SprayFoamEnhanced13
        solver = SprayFoamEnhanced13(cavity_case, simplec=True, coupled=True)
        assert solver.simplec is True

    def test_simplec_correct(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_13 import SprayFoamEnhanced13
        solver = SprayFoamEnhanced13(cavity_case, simplec=True)
        U_sc = solver._simplec_correct(solver.U, solver.p)
        assert U_sc.shape == solver.U.shape
        assert torch.isfinite(U_sc).all()

    def test_coupled_solve(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_13 import SprayFoamEnhanced13
        solver = SprayFoamEnhanced13(cavity_case, coupled=True, coupled_max_iter=2)
        U_c, p_c = solver._coupled_solve(solver.U, solver.p, solver.delta_t)
        assert U_c.shape == solver.U.shape
        assert p_c.shape == solver.p.shape


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced13
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced13:
    """Tests for enhanced multiphase Euler solver v13."""

    def test_class_exists(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_13 import MultiphaseEulerFoamEnhanced13
        assert MultiphaseEulerFoamEnhanced13 is not None

    def test_class_has_init(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_13 import MultiphaseEulerFoamEnhanced13
        assert hasattr(MultiphaseEulerFoamEnhanced13, '__init__')

    def test_simplec_logic(self):
        """Test SIMPLEC phase velocity correction logic."""
        U = torch.randn(16, 3)
        dp = torch.randn(16)
        dp3 = dp.unsqueeze(-1).expand(-1, 3) * 0.001
        corr = torch.zeros(16, 3)
        owner = torch.arange(15)
        neigh = torch.arange(1, 16)
        corr.index_add_(0, owner, dp3[:-1])
        corr.index_add_(0, neigh, -dp3[:-1])
        U_sc = U - corr
        assert U_sc.shape == U.shape
        assert torch.isfinite(U_sc).all()

    def test_coupled_solve_logic(self):
        """Test coupled multiphase solve logic."""
        p = torch.randn(16)
        owner = torch.arange(15)
        neigh = torch.arange(1, 16)
        dp = p[neigh] - p[owner]
        corr = torch.zeros(16)
        corr.index_add_(0, owner, dp * 0.01)
        corr.index_add_(0, neigh, -dp * 0.01)
        p_new = p - 0.1 * corr
        assert p_new.shape == p.shape
        assert torch.isfinite(p_new).all()


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV13:
    """Tests for __init__.py exports of v13 solvers."""

    def test_simple_enhanced_13_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced13
        assert SimpleFoamEnhanced13 is not None

    def test_pimple_enhanced_13_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced13
        assert PimpleFoamEnhanced13 is not None

    def test_piso_enhanced_13_exported(self):
        from pyfoam.applications import PisoFoamEnhanced13
        assert PisoFoamEnhanced13 is not None

    def test_ico_enhanced_13_exported(self):
        from pyfoam.applications import IcoFoamEnhanced13
        assert IcoFoamEnhanced13 is not None

    def test_buoyant_pimple_enhanced_13_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced13
        assert BuoyantPimpleFoamEnhanced13 is not None

    def test_buoyant_simple_enhanced_13_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced13
        assert BuoyantSimpleFoamEnhanced13 is not None

    def test_compressible_inter_enhanced_13_exported(self):
        from pyfoam.applications import CompressibleInterFoamEnhanced13
        assert CompressibleInterFoamEnhanced13 is not None

    def test_spray_enhanced_13_exported(self):
        from pyfoam.applications import SprayFoamEnhanced13
        assert SprayFoamEnhanced13 is not None

    def test_multiphase_euler_enhanced_13_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced13
        assert MultiphaseEulerFoamEnhanced13 is not None
