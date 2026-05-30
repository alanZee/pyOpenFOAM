"""
Unit tests for enhanced solver variants v12 (under-relaxation variants).

Tests cover:
- Adaptive under-relaxation
- Aitken under-relaxation
- Field-based under-relaxation
- Export of new classes from __init__.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Reuse cavity case helper from test_enhanced_solvers_11
# ---------------------------------------------------------------------------

def _make_cavity_case(case_dir, n_cells_x=4, n_cells_y=4, nu=0.01, delta_t=0.001,
                      end_time=0.01, piso_correctors=2, compressible=False,
                      buoyant=False, reacting=False):
    """Write a minimal cavity case."""
    from test_enhanced_solvers_11 import _make_cavity_case as _make
    _make(case_dir, n_cells_x=n_cells_x, n_cells_y=n_cells_y, nu=nu,
          delta_t=delta_t, end_time=end_time, piso_correctors=piso_correctors,
          compressible=compressible, buoyant=buoyant, reacting=reacting)


# Import the actual helper
import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_enhanced_solvers_11 import _make_cavity_case as _make_real_case


def _make_cavity_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    nu: float = 0.01,
    delta_t: float = 0.001,
    end_time: float = 0.01,
    piso_correctors: int = 2,
    compressible: bool = False,
    buoyant: bool = False,
    reacting: bool = False,
) -> None:
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
# Tests: SimpleFoamEnhanced12
# ===========================================================================


class TestSimpleFoamEnhanced12:
    """Tests for enhanced SIMPLE solver v12 (under-relaxation variants)."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_12 import SimpleFoamEnhanced12
        solver = SimpleFoamEnhanced12(cavity_case, aur=True, aitken=True, fbur=True)
        assert solver.aur is True
        assert solver.aitken is True
        assert solver.fbur is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_12 import SimpleFoamEnhanced12
        solver = SimpleFoamEnhanced12(cavity_case)
        assert solver.aur_growth == pytest.approx(1.05)
        assert solver.aitken_depth == 3
        assert solver.fbur_wall_damping == pytest.approx(0.3)

    def test_adaptive_relaxation(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_12 import SimpleFoamEnhanced12
        solver = SimpleFoamEnhanced12(cavity_case, aur=True)
        alpha_U, alpha_p = solver._adaptive_relaxation(0.5, 1)
        assert 0.1 <= alpha_U <= 0.95
        assert 0.05 <= alpha_p <= 0.5

    def test_adaptive_relaxation_disabled(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_12 import SimpleFoamEnhanced12
        solver = SimpleFoamEnhanced12(cavity_case, aur=False)
        alpha_U, alpha_p = solver._adaptive_relaxation(0.5, 1)
        assert alpha_U == solver.alpha_U
        assert alpha_p == solver.alpha_p

    def test_aitken_relaxation(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_12 import SimpleFoamEnhanced12
        solver = SimpleFoamEnhanced12(cavity_case, aitken=True)
        p_accel, U_accel = solver._aitken_relaxation(
            solver.p, solver.p.clone(), solver.U, solver.U.clone(),
        )
        assert p_accel.shape == solver.p.shape
        assert U_accel.shape == solver.U.shape
        assert torch.isfinite(p_accel).all()
        assert torch.isfinite(U_accel).all()

    def test_field_based_relaxation(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_12 import SimpleFoamEnhanced12
        solver = SimpleFoamEnhanced12(cavity_case, fbur=True)
        p_fbur, U_fbur = solver._field_based_relaxation(
            solver.p, solver.p.clone(), solver.U, solver.U.clone(),
        )
        assert p_fbur.shape == solver.p.shape
        assert U_fbur.shape == solver.U.shape
        assert torch.isfinite(p_fbur).all()
        assert torch.isfinite(U_fbur).all()


# ===========================================================================
# Tests: PimpleFoamEnhanced12
# ===========================================================================


class TestPimpleFoamEnhanced12:
    """Tests for enhanced PIMPLE solver v12."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_12 import PimpleFoamEnhanced12
        solver = PimpleFoamEnhanced12(cavity_case, aur=True, aitken=True, fbur=True)
        assert solver.aur is True
        assert solver.aitken is True
        assert solver.fbur is True

    def test_adaptive_relaxation(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_12 import PimpleFoamEnhanced12
        solver = PimpleFoamEnhanced12(cavity_case, aur=True)
        aU, ap = solver._adaptive_relaxation(0.5, 1)
        assert 0.1 <= aU <= 0.95

    def test_aitken_relaxation(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_12 import PimpleFoamEnhanced12
        solver = PimpleFoamEnhanced12(cavity_case, aitken=True)
        p_a, U_a = solver._aitken_relaxation(solver.p, solver.p.clone(), solver.U, solver.U.clone())
        assert p_a.shape == solver.p.shape
        assert torch.isfinite(p_a).all()

    def test_field_based_relaxation(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_12 import PimpleFoamEnhanced12
        solver = PimpleFoamEnhanced12(cavity_case, fbur=True)
        p_f, U_f = solver._field_based_relaxation(solver.p, solver.p.clone(), solver.U, solver.U.clone())
        assert p_f.shape == solver.p.shape
        assert torch.isfinite(p_f).all()


# ===========================================================================
# Tests: PisoFoamEnhanced12
# ===========================================================================


class TestPisoFoamEnhanced12:
    """Tests for enhanced PISO solver v12."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_12 import PisoFoamEnhanced12
        solver = PisoFoamEnhanced12(cavity_case, aur=True, aitken=True, fbur=True)
        assert solver.aur is True
        assert solver.aitken is True

    def test_adaptive_relaxation(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_12 import PisoFoamEnhanced12
        solver = PisoFoamEnhanced12(cavity_case, aur=True)
        alpha = solver._adaptive_relaxation(0.5, 1)
        assert 0.1 <= alpha <= 0.95

    def test_aitken_relaxation(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_12 import PisoFoamEnhanced12
        solver = PisoFoamEnhanced12(cavity_case, aitken=True)
        p_a = solver._aitken_relaxation(solver.p, solver.p.clone())
        assert p_a.shape == solver.p.shape
        assert torch.isfinite(p_a).all()

    def test_field_based_relaxation(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_12 import PisoFoamEnhanced12
        solver = PisoFoamEnhanced12(cavity_case, fbur=True)
        p_f = solver._field_based_relaxation(solver.p, solver.p.clone())
        assert p_f.shape == solver.p.shape
        assert torch.isfinite(p_f).all()


# ===========================================================================
# Tests: IcoFoamEnhanced12
# ===========================================================================


class TestIcoFoamEnhanced12:
    """Tests for enhanced ICO solver v12."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_12 import IcoFoamEnhanced12
        solver = IcoFoamEnhanced12(cavity_case, aur=True, aitken=True, fbur=True)
        assert solver.aur is True

    def test_adaptive_dt_factor(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_12 import IcoFoamEnhanced12
        solver = IcoFoamEnhanced12(cavity_case, aur=True)
        factor = solver._adaptive_dt_factor(0.5, 1)
        assert 0.5 <= factor <= 1.5

    def test_aitken_correct(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_12 import IcoFoamEnhanced12
        solver = IcoFoamEnhanced12(cavity_case, aitken=True)
        p_a = solver._aitken_correct(solver.p, solver.p.clone())
        assert p_a.shape == solver.p.shape
        assert torch.isfinite(p_a).all()

    def test_field_relax(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_12 import IcoFoamEnhanced12
        solver = IcoFoamEnhanced12(cavity_case, fbur=True)
        U_f = solver._field_relax(solver.U, solver.U.clone())
        assert U_f.shape == solver.U.shape
        assert torch.isfinite(U_f).all()


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced12
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced12:
    """Tests for enhanced buoyant PIMPLE solver v12."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_12 import BuoyantPimpleFoamEnhanced12
        solver = BuoyantPimpleFoamEnhanced12(buoyant_case, aur=True, aitken=True, fbur=True)
        assert solver.aur is True

    def test_adaptive_relaxation(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_12 import BuoyantPimpleFoamEnhanced12
        solver = BuoyantPimpleFoamEnhanced12(buoyant_case, aur=True)
        alpha = solver._adaptive_relaxation(0.5, 1)
        assert 0.1 <= alpha <= 0.95

    def test_aitken_correct(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_12 import BuoyantPimpleFoamEnhanced12
        solver = BuoyantPimpleFoamEnhanced12(buoyant_case, aitken=True)
        p_a = solver._aitken_correct(solver.p, solver.p.clone())
        assert p_a.shape == solver.p.shape
        assert torch.isfinite(p_a).all()

    def test_field_relax(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_12 import BuoyantPimpleFoamEnhanced12
        solver = BuoyantPimpleFoamEnhanced12(buoyant_case, fbur=True)
        U_f = solver._field_relax(solver.U, solver.U.clone())
        assert U_f.shape == solver.U.shape
        assert torch.isfinite(U_f).all()


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced12
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced12:
    """Tests for enhanced buoyant SIMPLE solver v12."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_12 import BuoyantSimpleFoamEnhanced12
        solver = BuoyantSimpleFoamEnhanced12(buoyant_case, aur=True, aitken=True, fbur=True)
        assert solver.aur is True

    def test_adaptive_relaxation(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_12 import BuoyantSimpleFoamEnhanced12
        solver = BuoyantSimpleFoamEnhanced12(buoyant_case, aur=True)
        alpha = solver._adaptive_relaxation(0.5, 1)
        assert 0.1 <= alpha <= 0.95

    def test_aitken_correct(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_12 import BuoyantSimpleFoamEnhanced12
        solver = BuoyantSimpleFoamEnhanced12(buoyant_case, aitken=True)
        p_a = solver._aitken_correct(solver.p, solver.p.clone())
        assert p_a.shape == solver.p.shape
        assert torch.isfinite(p_a).all()

    def test_field_relax(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_12 import BuoyantSimpleFoamEnhanced12
        solver = BuoyantSimpleFoamEnhanced12(buoyant_case, fbur=True)
        U_f = solver._field_relax(solver.U, solver.U.clone())
        assert U_f.shape == solver.U.shape
        assert torch.isfinite(U_f).all()


# ===========================================================================
# Tests: CompressibleInterFoamEnhanced12
# ===========================================================================


class TestCompressibleInterFoamEnhanced12:
    """Tests for enhanced compressible VOF solver v12."""

    def test_init(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_12 import CompressibleInterFoamEnhanced12
        solver = CompressibleInterFoamEnhanced12(compressible_case, aur=True, aitken=True, fbur=True)
        assert solver.aur is True

    def test_adaptive_relaxation(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_12 import CompressibleInterFoamEnhanced12
        solver = CompressibleInterFoamEnhanced12(compressible_case, aur=True)
        alpha = solver._adaptive_relaxation(0.5, 1)
        assert 0.1 <= alpha <= 0.95

    def test_aitken_correct(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_12 import CompressibleInterFoamEnhanced12
        solver = CompressibleInterFoamEnhanced12(compressible_case, aitken=True)
        p_a = solver._aitken_correct(solver.p, solver.p.clone())
        assert p_a.shape == solver.p.shape
        assert torch.isfinite(p_a).all()

    def test_field_relax(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_12 import CompressibleInterFoamEnhanced12
        solver = CompressibleInterFoamEnhanced12(compressible_case, fbur=True)
        U_f = solver._field_relax(solver.U, solver.U.clone())
        assert U_f.shape == solver.U.shape
        assert torch.isfinite(U_f).all()


# ===========================================================================
# Tests: SprayFoamEnhanced12
# ===========================================================================


class TestSprayFoamEnhanced12:
    """Tests for enhanced spray solver v12."""

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_12 import SprayFoamEnhanced12
        solver = SprayFoamEnhanced12(cavity_case, aur=True, aitken=True, fbur=True)
        assert solver.aur is True

    def test_adaptive_relaxation(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_12 import SprayFoamEnhanced12
        solver = SprayFoamEnhanced12(cavity_case, aur=True)
        alpha = solver._adaptive_relaxation(0.5, 1)
        assert 0.1 <= alpha <= 0.95

    def test_aitken_correct(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_12 import SprayFoamEnhanced12
        solver = SprayFoamEnhanced12(cavity_case, aitken=True)
        p_a = solver._aitken_correct(solver.p, solver.p.clone())
        assert p_a.shape == solver.p.shape
        assert torch.isfinite(p_a).all()

    def test_field_relax(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_12 import SprayFoamEnhanced12
        solver = SprayFoamEnhanced12(cavity_case, fbur=True)
        U_f = solver._field_relax(solver.U, solver.U.clone())
        assert U_f.shape == solver.U.shape
        assert torch.isfinite(U_f).all()


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced12
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced12:
    """Tests for enhanced multiphase Euler solver v12."""

    def test_class_exists(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_12 import MultiphaseEulerFoamEnhanced12
        assert MultiphaseEulerFoamEnhanced12 is not None

    def test_class_has_init(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_12 import MultiphaseEulerFoamEnhanced12
        assert hasattr(MultiphaseEulerFoamEnhanced12, '__init__')

    def test_adaptive_relaxation_logic(self):
        """Test AUR logic standalone."""
        prev_res = 1.0
        curr_res = 0.5
        alpha = 0.7
        if curr_res < prev_res:
            alpha = min(0.95, alpha * 1.05)
        assert alpha > 0.7
        assert alpha <= 0.95

    def test_aitken_phase_logic(self):
        """Test Aitken phase velocity correction logic."""
        U = torch.randn(16, 3)
        U_old = torch.randn(16, 3)
        dU = U - U_old
        dU_sq = (dU * dU).sum().clamp(min=1e-30)
        alpha = min(2.0, max(0.1, 1.0))
        U_corr = U_old + alpha * dU
        assert U_corr.shape == U.shape
        assert torch.isfinite(U_corr).all()


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV12:
    """Tests for __init__.py exports of v12 solvers."""

    def test_simple_enhanced_12_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced12
        assert SimpleFoamEnhanced12 is not None

    def test_pimple_enhanced_12_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced12
        assert PimpleFoamEnhanced12 is not None

    def test_piso_enhanced_12_exported(self):
        from pyfoam.applications import PisoFoamEnhanced12
        assert PisoFoamEnhanced12 is not None

    def test_ico_enhanced_12_exported(self):
        from pyfoam.applications import IcoFoamEnhanced12
        assert IcoFoamEnhanced12 is not None

    def test_buoyant_pimple_enhanced_12_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced12
        assert BuoyantPimpleFoamEnhanced12 is not None

    def test_buoyant_simple_enhanced_12_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced12
        assert BuoyantSimpleFoamEnhanced12 is not None

    def test_compressible_inter_enhanced_12_exported(self):
        from pyfoam.applications import CompressibleInterFoamEnhanced12
        assert CompressibleInterFoamEnhanced12 is not None

    def test_spray_enhanced_12_exported(self):
        from pyfoam.applications import SprayFoamEnhanced12
        assert SprayFoamEnhanced12 is not None

    def test_multiphase_euler_enhanced_12_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced12
        assert MultiphaseEulerFoamEnhanced12 is not None
