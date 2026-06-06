"""
Tests for specie_transfer module.
"""
import pytest
import torch

from pyfoam.specie_transfer import SimpleDiffusionModel


class TestSimpleDiffusionModel:
    """简单扩散模型测试。"""

    def test_mass_flux(self):
        model = SimpleDiffusionModel(D_mass=1e-5, D_turb=1e-3)
        Y = torch.tensor([0.01, 0.02, 0.03])
        T = torch.tensor([300.0, 350.0, 400.0])
        alpha = torch.tensor([0.5, 0.5, 0.5])
        flux = model.mass_flux(Y, T, alpha)
        assert flux.shape == (3,)
        assert torch.isfinite(flux).all()

    def test_heat_flux(self):
        model = SimpleDiffusionModel(latent_heat=2.26e6)
        T = torch.tensor([300.0, 350.0])
        mass_flux = torch.tensor([1e-3, 2e-3])
        heat = model.heat_flux(T, mass_flux)
        assert heat.shape == (2,)
        assert (heat > 0).all()

    def test_properties(self):
        model = SimpleDiffusionModel(D_mass=2e-5, latent_heat=3e6)
        assert model.molecular_diffusivity == pytest.approx(2e-5)
        assert model.latent_heat == pytest.approx(3e6)
