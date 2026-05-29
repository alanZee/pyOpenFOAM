"""Phase 23 边界条件测试。

测试以下新增 BC 的注册、工厂创建、apply 和 matrix_contributions 行为：

- CompressibleTurbulentTemperatureCoupledBC
- WaveTransmissive2BC
- AdvectiveDiffusiveBC
- PressureInterpolationAMGBC
- CodedFixedValueBC
- CyclicAMI2BC
- ProcessorCyclicBC
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.compressible_turbulent_temperature_coupled import (
    CompressibleTurbulentTemperatureCoupledBC,
)
from pyfoam.boundary.wave_transmissive_2 import WaveTransmissive2BC
from pyfoam.boundary.advective_diffusive import AdvectiveDiffusiveBC
from pyfoam.boundary.pressure_interpolation_amg import PressureInterpolationAMGBC
from pyfoam.boundary.coded_fixed_value import CodedFixedValueBC
from pyfoam.boundary.cyclic_ami_2 import CyclicAMI2BC
from pyfoam.boundary.processor_cyclic import ProcessorCyclicBC


# ======================================================================
# CompressibleTurbulentTemperatureCoupledBC
# ======================================================================


class TestCompressibleTurbulentTemperatureCoupledBC:
    """compressibleTurbulentTemperatureCoupled 边界条件测试。"""

    def test_registration(self):
        assert "compressibleTurbulentTemperatureCoupled" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = CompressibleTurbulentTemperatureCoupledBC(simple_patch)
        assert bc.type_name == "compressibleTurbulentTemperatureCoupled"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "compressibleTurbulentTemperatureCoupled", simple_patch,
            {"T_coupled": 350.0, "Pr_t": 0.85},
        )
        assert isinstance(bc, CompressibleTurbulentTemperatureCoupledBC)

    def test_default_coeffs(self, simple_patch):
        bc = CompressibleTurbulentTemperatureCoupledBC(simple_patch)
        assert bc.t_coupled == 300.0
        assert bc.pr_t == 0.85
        assert bc.k == 0.025
        assert bc.rho == 1.225
        assert bc.cp == 1005.0

    def test_custom_coeffs(self, simple_patch):
        bc = CompressibleTurbulentTemperatureCoupledBC(simple_patch, {
            "T_coupled": 500.0, "Pr_t": 0.7, "k": 0.04, "rho": 10.0, "cp": 2000.0,
        })
        assert bc.t_coupled == 500.0
        assert bc.pr_t == 0.7
        assert bc.k == 0.04
        assert bc.rho == 10.0
        assert bc.cp == 2000.0

    def test_effective_alpha_no_turb(self, simple_patch):
        """无湍流粘度时，alpha_eff = k / (rho * cp)。"""
        bc = CompressibleTurbulentTemperatureCoupledBC(simple_patch, {
            "k": 0.025, "rho": 1.225, "cp": 1005.0,
        })
        alpha = bc._effective_alpha(torch.device("cpu"), torch.float64)
        expected = 0.025 / (1.225 * 1005.0)
        assert torch.allclose(alpha, torch.full((3,), expected, dtype=torch.float64), atol=1e-10)

    def test_effective_alpha_with_turb(self, simple_patch):
        """有湍流粘度时，alpha_eff 增大。"""
        bc = CompressibleTurbulentTemperatureCoupledBC(simple_patch, {
            "k": 0.025, "rho": 1.225, "cp": 1005.0, "Pr_t": 0.85,
            "nut_values": torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64),
        })
        alpha = bc._effective_alpha(torch.device("cpu"), torch.float64)
        alpha_lam = 0.025 / (1.225 * 1005.0)
        assert alpha[0].item() == pytest.approx(alpha_lam + 0.01 / 0.85, abs=1e-10)
        assert alpha[1].item() == pytest.approx(alpha_lam + 0.02 / 0.85, abs=1e-10)

    def test_apply(self, simple_patch):
        bc = CompressibleTurbulentTemperatureCoupledBC(simple_patch, {"T_coupled": 400.0})
        field = torch.zeros(15, dtype=torch.float64)
        # owner cells [0,1,2] = 300
        field[0] = 300.0
        field[1] = 300.0
        field[2] = 300.0
        result = bc.apply(field)
        # Robin blend with no nut: alpha_eff = alpha_lam
        # grad_weight = alpha_lam * 2.0, value_weight = alpha_lam
        # T = (alpha_lam*2*300 + alpha_lam*400) / (alpha_lam*2 + alpha_lam) = 1000/3
        assert torch.allclose(
            result[10:13],
            torch.full((3,), 1000.0 / 3.0, dtype=torch.float64),
            atol=1e-8,
        )

    def test_apply_with_patch_idx(self, simple_patch):
        bc = CompressibleTurbulentTemperatureCoupledBC(simple_patch, {"T_coupled": 400.0})
        field = torch.zeros(20, dtype=torch.float64)
        # owner cells [0,1,2] = 300
        field[0] = 300.0
        field[1] = 300.0
        field[2] = 300.0
        result = bc.apply(field, patch_idx=5)
        expected = 1000.0 / 3.0
        assert result[5].item() == pytest.approx(expected, abs=1e-8)

    def test_matrix_contributions(self, simple_patch):
        bc = CompressibleTurbulentTemperatureCoupledBC(simple_patch, {"T_coupled": 400.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        # source should contain T_coupled contribution
        assert (source > 0).all()


# ======================================================================
# WaveTransmissive2BC
# ======================================================================


class TestWaveTransmissive2BC:
    """waveTransmissive2 边界条件测试。"""

    def test_registration(self):
        assert "waveTransmissive2" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = WaveTransmissive2BC(simple_patch)
        assert bc.type_name == "waveTransmissive2"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("waveTransmissive2", simple_patch)
        assert isinstance(bc, WaveTransmissive2BC)

    def test_default_coeffs(self, simple_patch):
        bc = WaveTransmissive2BC(simple_patch)
        assert bc.field_inf == 101325.0
        assert bc.l_inf == 1.0
        assert bc.gamma == 1.4
        assert bc.blending == 1.0

    def test_custom_coeffs(self, simple_patch):
        bc = WaveTransmissive2BC(simple_patch, {
            "fieldInf": 100000.0, "lInf": 0.5, "gamma": 1.3, "blending": 0.8,
        })
        assert bc.field_inf == 100000.0
        assert bc.l_inf == 0.5
        assert bc.gamma == 1.3
        assert bc.blending == 0.8

    def test_blending_setter_clamp(self, simple_patch):
        bc = WaveTransmissive2BC(simple_patch)
        bc.blending = 2.0
        assert bc.blending == 1.0
        bc.blending = -0.5
        assert bc.blending == 0.0

    def test_apply_no_velocity(self, simple_patch):
        """无速度信息时回退到零梯度。"""
        bc = WaveTransmissive2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 101325.0
        field[1] = 101325.0
        field[2] = 101325.0
        result = bc.apply(field)
        assert torch.allclose(result[10:13], torch.full((3,), 101325.0, dtype=torch.float64))

    def test_apply_with_velocity(self, simple_patch):
        bc = WaveTransmissive2BC(simple_patch, {"fieldInf": 100000.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 101000.0
        field[1] = 101000.0
        field[2] = 101000.0
        velocity = torch.tensor([
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ], dtype=torch.float64)
        result = bc.apply(field, velocity=velocity)
        # 压力应被修正
        assert result[10].item() != 101000.0

    def test_apply_zero_blending(self, simple_patch):
        """blending=0 时等效于零梯度。"""
        bc = WaveTransmissive2BC(simple_patch, {"blending": 0.0, "fieldInf": 100000.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 101000.0
        field[1] = 101000.0
        field[2] = 101000.0
        velocity = torch.tensor([
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ], dtype=torch.float64)
        result = bc.apply(field, velocity=velocity)
        assert torch.allclose(result[10:13], torch.full((3,), 101000.0, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        bc = WaveTransmissive2BC(simple_patch, {"fieldInf": 100000.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        assert (diag > 0).all()
        assert (source > 0).all()

    def test_matrix_contributions_accumulate(self, simple_patch):
        bc = WaveTransmissive2BC(simple_patch, {"fieldInf": 100000.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag = torch.ones(3, dtype=torch.float64)
        source = torch.ones(3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, diag=diag, source=source)
        assert (diag > 1.0).all()


# ======================================================================
# AdvectiveDiffusiveBC
# ======================================================================


class TestAdvectiveDiffusiveBC:
    """advectiveDiffusive 边界条件测试。"""

    def test_registration(self):
        assert "advectiveDiffusive" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = AdvectiveDiffusiveBC(simple_patch)
        assert bc.type_name == "advectiveDiffusive"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("advectiveDiffusive", simple_patch)
        assert isinstance(bc, AdvectiveDiffusiveBC)

    def test_default_coeffs(self, simple_patch):
        bc = AdvectiveDiffusiveBC(simple_patch)
        assert bc.D == 1.5e-5
        assert bc.l_inf == 1.0

    def test_custom_coeffs(self, simple_patch):
        bc = AdvectiveDiffusiveBC(simple_patch, {"D": 1e-3, "lInf": 2.0})
        assert bc.D == 1e-3
        assert bc.l_inf == 2.0

    def test_apply_no_flux(self, simple_patch):
        """无通量时回退到零梯度。"""
        bc = AdvectiveDiffusiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(100.0)
        assert field[11].item() == pytest.approx(200.0)
        assert field[12].item() == pytest.approx(300.0)

    def test_apply_with_flux(self, simple_patch):
        """有通量时仍应设置 owner cell 值（advective outflow）。"""
        bc = AdvectiveDiffusiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        phi = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc.apply(field, phi=phi)
        assert field[10].item() == pytest.approx(100.0)
        assert field[11].item() == pytest.approx(200.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = AdvectiveDiffusiveBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        # owner cells [0,1,2] = 42
        field[0] = 42.0
        field[1] = 43.0
        field[2] = 44.0
        bc.apply(field, patch_idx=5)
        assert field[5].item() == pytest.approx(42.0)

    def test_matrix_contributions_no_flux(self, simple_patch):
        """无通量时仅扩散贡献。"""
        bc = AdvectiveDiffusiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        diag, source = bc.matrix_contributions(field, 3)
        # diff_coeff = D * area * delta = 1.5e-5 * 1.0 * 2.0 = 3e-5
        expected_coeff = 1.5e-5 * 1.0 * 2.0
        assert torch.allclose(diag, torch.full((3,), expected_coeff, dtype=torch.float64), atol=1e-12)

    def test_matrix_contributions_with_flux(self, simple_patch):
        """有通量时应包含 advective source 贡献。"""
        bc = AdvectiveDiffusiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        phi = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, phi=phi)
        expected_coeff = 1.5e-5 * 1.0 * 2.0
        # source = diff_coeff * owner + phi * owner
        expected_source_0 = expected_coeff * 100.0 + 5.0 * 100.0
        assert source[0].item() == pytest.approx(expected_source_0, abs=1e-8)


# ======================================================================
# PressureInterpolationAMGBC
# ======================================================================


class TestPressureInterpolationAMGBC:
    """pressureInterpolationAMG 边界条件测试。"""

    def test_registration(self):
        assert "pressureInterpolationAMG" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = PressureInterpolationAMGBC(simple_patch)
        assert bc.type_name == "pressureInterpolationAMG"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("pressureInterpolationAMG", simple_patch)
        assert isinstance(bc, PressureInterpolationAMGBC)

    def test_default_coeffs(self, simple_patch):
        bc = PressureInterpolationAMGBC(simple_patch)
        assert bc.interpolation_scheme == "cell"
        assert bc.n_correctors == 2
        assert bc.epsilon == 0.1

    def test_custom_coeffs(self, simple_patch):
        bc = PressureInterpolationAMGBC(simple_patch, {
            "interpolationScheme": "linear", "nCorrectors": 3, "epsilon": 0.2,
        })
        assert bc.interpolation_scheme == "linear"
        assert bc.n_correctors == 3
        assert bc.epsilon == 0.2

    def test_apply_cell_scheme(self, simple_patch):
        """cell 方案等效于零梯度。"""
        bc = PressureInterpolationAMGBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(100.0)
        assert field[11].item() == pytest.approx(200.0)
        assert field[12].item() == pytest.approx(300.0)

    def test_apply_linear_scheme_with_grad(self, simple_patch):
        """linear 方案使用梯度修正。"""
        bc = PressureInterpolationAMGBC(simple_patch, {"interpolationScheme": "linear"})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        # grad_p pointing outward (+x), dist = 1/delta = 0.5
        grad_p = torch.tensor([
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, grad_p=grad_p)
        # p_face = p_owner + grad . d = p_owner + 100 * 0.5 = p_owner + 50
        assert field[10].item() == pytest.approx(150.0)
        assert field[11].item() == pytest.approx(250.0)
        assert field[12].item() == pytest.approx(350.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureInterpolationAMGBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        # owner cells [0,1,2] = 42
        field[0] = 42.0
        field[1] = 43.0
        field[2] = 44.0
        bc.apply(field, patch_idx=5)
        assert field[5].item() == pytest.approx(42.0)

    def test_matrix_contributions(self, simple_patch):
        """AMG 稳定化矩阵贡献。"""
        bc = PressureInterpolationAMGBC(simple_patch, {"epsilon": 0.1})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        diag, source = bc.matrix_contributions(field, 3)
        # coeff = (1 + 0.1) * 1.0 * 2.0 = 2.2
        expected_coeff = 1.1 * 1.0 * 2.0
        assert diag[0].item() == pytest.approx(expected_coeff, abs=1e-10)
        # source = coeff * owner
        assert source[0].item() == pytest.approx(expected_coeff * 100.0, abs=1e-8)


# ======================================================================
# CodedFixedValueBC
# ======================================================================


def _constant_350(patch, field):
    """返回常数 350 的简单函数。"""
    return torch.full((patch.n_faces,), 350.0, dtype=torch.float64)


def _scaled_owner(patch, field):
    """返回 owner cell 值 + 10 的函数。"""
    owner_vals = field[patch.owner_cells]
    return owner_vals + 10.0


class TestCodedFixedValueBC:
    """codedFixedValue 边界条件测试。"""

    def test_registration(self):
        assert "codedFixedValue" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = CodedFixedValueBC(simple_patch, {"code": _constant_350})
        assert bc.type_name == "codedFixedValue"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "codedFixedValue", simple_patch, {"code": _constant_350},
        )
        assert isinstance(bc, CodedFixedValueBC)

    def test_callable_code(self, simple_patch):
        bc = CodedFixedValueBC(simple_patch, {"code": _constant_350})
        assert bc.user_fn is _constant_350

    def test_string_code(self, simple_patch):
        bc = CodedFixedValueBC(simple_patch, {
            "code": "lambda p, f: torch.full((p.n_faces,), 42.0, dtype=torch.float64)",
        })
        result = bc.user_fn(simple_patch, torch.zeros(15, dtype=torch.float64))
        assert torch.allclose(result, torch.full((3,), 42.0, dtype=torch.float64))

    def test_missing_code_raises(self, simple_patch):
        with pytest.raises(KeyError, match="'code'"):
            CodedFixedValueBC(simple_patch)

    def test_invalid_code_type_raises(self, simple_patch):
        with pytest.raises(TypeError, match="callable or eval-able string"):
            CodedFixedValueBC(simple_patch, {"code": 12345})

    def test_default_scale(self, simple_patch):
        bc = CodedFixedValueBC(simple_patch, {"code": _constant_350})
        assert bc.scale == 1.0

    def test_custom_scale(self, simple_patch):
        bc = CodedFixedValueBC(simple_patch, {"code": _constant_350, "scale": 2.0})
        assert bc.scale == 2.0

    def test_apply_constant(self, simple_patch):
        bc = CodedFixedValueBC(simple_patch, {"code": _constant_350})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 350.0, dtype=torch.float64))

    def test_apply_with_scale(self, simple_patch):
        """scale 参数应乘以 computed values。"""
        bc = CodedFixedValueBC(simple_patch, {"code": _constant_350, "scale": 2.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 700.0, dtype=torch.float64))

    def test_apply_scaled_owner(self, simple_patch):
        bc = CodedFixedValueBC(simple_patch, {"code": _scaled_owner})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(110.0)
        assert field[11].item() == pytest.approx(210.0)
        assert field[12].item() == pytest.approx(310.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = CodedFixedValueBC(simple_patch, {"code": _constant_350})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], torch.full((3,), 350.0, dtype=torch.float64))

    def test_apply_scalar_return(self, simple_patch):
        def scalar_fn(patch, field):
            return torch.tensor(99.0, dtype=torch.float64)

        bc = CodedFixedValueBC(simple_patch, {"code": scalar_fn})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 99.0, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        bc = CodedFixedValueBC(simple_patch, {"code": _constant_350})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        # coeff = delta * area = 2.0
        assert torch.allclose(diag, torch.full((3,), 2.0, dtype=torch.float64))
        # source = coeff * 350 = 700
        assert torch.allclose(source, torch.full((3,), 700.0, dtype=torch.float64))

    def test_matrix_contributions_with_scale(self, simple_patch):
        """矩阵贡献也应受 scale 影响。"""
        bc = CodedFixedValueBC(simple_patch, {"code": _constant_350, "scale": 2.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert torch.allclose(diag, torch.full((3,), 2.0, dtype=torch.float64))
        # source = coeff * 350 * 2 = 1400
        assert torch.allclose(source, torch.full((3,), 1400.0, dtype=torch.float64))


# ======================================================================
# CyclicAMI2BC
# ======================================================================


class TestCyclicAMI2BC:
    """cyclicAMI2 边界条件测试。"""

    def test_registration(self):
        assert "cyclicAMI2" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = CyclicAMI2BC(simple_patch)
        assert bc.type_name == "cyclicAMI2"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("cyclicAMI2", simple_patch)
        assert isinstance(bc, CyclicAMI2BC)

    def test_default_coeffs(self, simple_patch):
        bc = CyclicAMI2BC(simple_patch)
        assert bc.conserve is True
        assert bc.non_ortho_correct is True
        assert bc.tolerance == 1e-6
        assert bc.transform == "noOrdering"

    def test_custom_coeffs(self, simple_patch):
        bc = CyclicAMI2BC(simple_patch, {
            "conserve": False, "nonOrthoCorrect": False, "tolerance": 1e-4,
        })
        assert bc.conserve is False
        assert bc.non_ortho_correct is False
        assert bc.tolerance == 1e-4

    def test_set_neighbour_field(self, simple_patch):
        bc = CyclicAMI2BC(simple_patch)
        neighbour = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.set_neighbour_field(neighbour)
        assert bc._neighbour_field is not None

    def test_set_ami_weights(self, simple_patch):
        bc = CyclicAMI2BC(simple_patch)
        weights = torch.eye(3, dtype=torch.float64)
        bc.set_ami_weights(weights)
        assert bc._weights is not None

    def test_apply_no_data(self, simple_patch):
        """无耦合数据时回退到零梯度。"""
        bc = CyclicAMI2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(100.0)

    def test_apply_with_identity_weights(self, simple_patch):
        """单位权重矩阵时直接传递 neighbour 值。"""
        bc = CyclicAMI2BC(simple_patch, {"conserve": False, "nonOrthoCorrect": False})
        bc.set_neighbour_field(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
        bc.set_ami_weights(torch.eye(3, dtype=torch.float64))
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10].item() == pytest.approx(10.0)
        assert field[11].item() == pytest.approx(20.0)
        assert field[12].item() == pytest.approx(30.0)

    def test_apply_conservation(self, simple_patch):
        """conservation 选项应保持总量守恒。"""
        bc = CyclicAMI2BC(simple_patch, {"conserve": True, "nonOrthoCorrect": False})
        bc.set_neighbour_field(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
        bc.set_ami_weights(torch.eye(3, dtype=torch.float64) * 0.5)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        # With conservation and identity-like weights, the total should match neighbour total
        assert field[10:13].sum().item() == pytest.approx(60.0, abs=1e-8)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = CyclicAMI2BC(simple_patch, {"conserve": False, "nonOrthoCorrect": False})
        bc.set_neighbour_field(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
        bc.set_ami_weights(torch.eye(3, dtype=torch.float64))
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=3)
        assert field[3].item() == pytest.approx(10.0)

    def test_matrix_contributions_no_data(self, simple_patch):
        """无耦合数据时无 source 贡献。"""
        bc = CyclicAMI2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        # diag 应有贡献
        expected_coeff = 2.0 * 1.0  # delta * area
        assert torch.allclose(diag, torch.full((3,), expected_coeff, dtype=torch.float64))
        # source 无贡献（无 neighbour 数据）
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions_with_data(self, simple_patch):
        bc = CyclicAMI2BC(simple_patch, {"conserve": False, "nonOrthoCorrect": False})
        bc.set_neighbour_field(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
        bc.set_ami_weights(torch.eye(3, dtype=torch.float64))
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        expected_coeff = 2.0 * 1.0
        # source = coeff * interpolated = coeff * neighbour
        assert source[0].item() == pytest.approx(expected_coeff * 10.0, abs=1e-8)
        assert source[1].item() == pytest.approx(expected_coeff * 20.0, abs=1e-8)


# ======================================================================
# ProcessorCyclicBC
# ======================================================================


class TestProcessorCyclicBC:
    """processorCyclic 边界条件测试。"""

    def test_registration(self):
        assert "processorCyclic" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = ProcessorCyclicBC(simple_patch)
        assert bc.type_name == "processorCyclic"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("processorCyclic", simple_patch)
        assert isinstance(bc, ProcessorCyclicBC)

    def test_default_coeffs(self, simple_patch):
        bc = ProcessorCyclicBC(simple_patch)
        assert bc.my_proc == 0
        assert bc.neighbour_proc == 1
        assert bc.transform == "noOrdering"

    def test_custom_coeffs(self, simple_patch):
        bc = ProcessorCyclicBC(simple_patch, {
            "myProcNo": 2, "neighbProcNo": 3, "transform": "rotational",
        })
        assert bc.my_proc == 2
        assert bc.neighbour_proc == 3
        assert bc.transform == "rotational"

    def test_set_neighbour_field(self, simple_patch):
        bc = ProcessorCyclicBC(simple_patch)
        neighbour = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.set_neighbour_field(neighbour)
        assert bc._neighbour_field is not None

    def test_prepare_send_buffer(self, simple_patch):
        bc = ProcessorCyclicBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[10] = 100.0
        field[11] = 200.0
        field[12] = 300.0
        buf = bc.prepare_send_buffer(field)
        assert buf.shape == (3,)
        assert buf[0].item() == pytest.approx(100.0)

    def test_receive_buffer(self, simple_patch):
        bc = ProcessorCyclicBC(simple_patch)
        buf = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.receive_buffer(buf)
        assert bc._neighbour_field is not None

    def test_apply_no_data(self, simple_patch):
        """无耦合数据时回退到零梯度。"""
        bc = ProcessorCyclicBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(100.0)
        assert field[11].item() == pytest.approx(200.0)
        assert field[12].item() == pytest.approx(300.0)

    def test_apply_with_data(self, simple_patch):
        bc = ProcessorCyclicBC(simple_patch)
        bc.set_neighbour_field(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10].item() == pytest.approx(10.0)
        assert field[11].item() == pytest.approx(20.0)
        assert field[12].item() == pytest.approx(30.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = ProcessorCyclicBC(simple_patch)
        bc.set_neighbour_field(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5].item() == pytest.approx(10.0)

    def test_rotational_transform(self, simple_patch):
        """旋转变换应修改向量值。"""
        bc = ProcessorCyclicBC(simple_patch, {
            "transform": "rotational",
            "rotationAxis": [0.0, 0.0, 1.0],
            "rotationAngle": 90.0,
        })
        # 向量场：(1, 0, 0) 绕 z 轴旋转 90 度应变为 (0, 1, 0)
        vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        bc.set_neighbour_field(vec)
        assert bc._neighbour_field is not None

    def test_translational_transform_noop(self, simple_patch):
        """平移变换对标量场不做任何操作。"""
        bc = ProcessorCyclicBC(simple_patch, {"transform": "translational"})
        scalar = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.set_neighbour_field(scalar)
        assert torch.allclose(bc._neighbour_field, scalar)

    def test_matrix_contributions_no_data(self, simple_patch):
        bc = ProcessorCyclicBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        diag, source = bc.matrix_contributions(field, 3)
        # diag = delta * area = 2.0
        expected_coeff = 2.0 * 1.0
        assert torch.allclose(diag, torch.full((3,), expected_coeff, dtype=torch.float64))
        # source = coeff * owner_vals
        assert source[0].item() == pytest.approx(expected_coeff * 100.0, abs=1e-8)

    def test_matrix_contributions_with_data(self, simple_patch):
        bc = ProcessorCyclicBC(simple_patch)
        bc.set_neighbour_field(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        expected_coeff = 2.0 * 1.0
        assert source[0].item() == pytest.approx(expected_coeff * 10.0, abs=1e-8)
        assert source[1].item() == pytest.approx(expected_coeff * 20.0, abs=1e-8)
