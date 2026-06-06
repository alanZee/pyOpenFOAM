"""
Tests for physical_properties module.
"""
import pytest
import torch

from pyfoam.physical_properties import PhysicalProperties, ConstantViscosity, PolynomialViscosity


class TestPhysicalProperties:
    """物性参数测试。"""

    def test_default_values(self):
        props = PhysicalProperties()
        assert props.nu == pytest.approx(1e-5)
        assert props.rho == pytest.approx(1.0)
        assert props.Cp == pytest.approx(1005.0)

    def test_mu(self):
        props = PhysicalProperties(nu=1e-5, rho=1.2)
        assert props.mu == pytest.approx(1.2e-5)

    def test_prandtl(self):
        props = PhysicalProperties(nu=1.8e-5, rho=1.0, Cp=1005.0, kappa=0.026)
        Pr = props.Pr
        assert Pr > 0
        assert Pr < 10  # 空气 Pr ≈ 0.7

    def test_thermal_diffusivity(self):
        props = PhysicalProperties(rho=1.0, Cp=1000.0, kappa=0.025)
        alpha = props.alpha
        assert alpha > 0
        assert alpha == pytest.approx(2.5e-5)

    def test_from_dict(self):
        d = {"nu": 2e-5, "rho": 1.5}
        props = PhysicalProperties.from_dict(d)
        assert props.nu == pytest.approx(2e-5)
        assert props.rho == pytest.approx(1.5)

    def test_repr(self):
        props = PhysicalProperties()
        assert "PhysicalProperties" in repr(props)


class TestConstantViscosity:
    """常粘度模型测试。"""

    def test_constant_value(self):
        model = ConstantViscosity(nu=1e-5)
        result = model.nu()
        assert result.item() == pytest.approx(1e-5)

    def test_with_temperature(self):
        model = ConstantViscosity(nu=1e-5)
        T = torch.tensor([300.0, 400.0, 500.0])
        result = model.nu(T)
        assert result.shape == (3,)
        assert (result == 1e-5).all()


class TestPolynomialViscosity:
    """多项式粘度模型测试。"""

    def test_constant_polynomial(self):
        model = PolynomialViscosity([1e-5])
        T = torch.tensor([300.0])
        result = model.nu(T)
        assert result.item() == pytest.approx(1e-5)

    def test_linear_polynomial(self):
        # nu(T) = a0 + a1*T
        model = PolynomialViscosity([1e-5, 1e-8])
        T = torch.tensor([300.0])
        expected = 1e-5 + 1e-8 * 300
        result = model.nu(T)
        assert result.item() == pytest.approx(expected)

    def test_no_temperature(self):
        model = PolynomialViscosity([1e-5])
        result = model.nu()
        assert result.item() == pytest.approx(1e-5)

    def test_coefficients(self):
        model = PolynomialViscosity([1, 2, 3])
        assert model.coefficients == [1, 2, 3]
