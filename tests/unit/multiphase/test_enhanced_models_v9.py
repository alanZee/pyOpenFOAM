"""Tests for enhanced multiphase models (Phase 19)."""

import pytest
import torch


# ---- Incompressible VOF v10 ----

class TestIncompressibleMultiphaseVoFEnhanced9:
    def test_import(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_9 import (
            IncompressibleMultiphaseVoFEnhanced9,
        )
        assert IncompressibleMultiphaseVoFEnhanced9 is not None

    def test_create(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_9 import (
            IncompressibleMultiphaseVoFEnhanced9,
        )
        model = IncompressibleMultiphaseVoFEnhanced9(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert model._n_phases == 2

    def test_sub_cell(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_9 import (
            IncompressibleMultiphaseVoFEnhanced9,
        )
        model = IncompressibleMultiphaseVoFEnhanced9(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            enable_sub_cell=True,
        )
        alphas = torch.tensor([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
        result = model.sub_cell_correct(alphas)
        assert result.shape == alphas.shape

    def test_cfl_compression_blend(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_9 import (
            IncompressibleMultiphaseVoFEnhanced9,
        )
        model = IncompressibleMultiphaseVoFEnhanced9(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            cfl_aware_blend=True,
        )
        c_low = model.cfl_compression_blend(0.1)
        c_high = model.cfl_compression_blend(5.0)
        assert c_low > c_high

    def test_repr(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_9 import (
            IncompressibleMultiphaseVoFEnhanced9,
        )
        model = IncompressibleMultiphaseVoFEnhanced9(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert "Enhanced9" in repr(model)


# ---- Compressible VOF v10 ----

class TestCompressibleMultiphaseVoFEnhanced9:
    def test_import(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_9 import (
            CompressibleMultiphaseVoFEnhanced9,
        )
        assert CompressibleMultiphaseVoFEnhanced9 is not None

    def test_create(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_9 import (
            CompressibleMultiphaseVoFEnhanced9,
        )
        model = CompressibleMultiphaseVoFEnhanced9(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert model._n_phases == 2

    def test_dynamic_eos_weight(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_9 import (
            CompressibleMultiphaseVoFEnhanced9,
        )
        model = CompressibleMultiphaseVoFEnhanced9(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            dynamic_eos_blend=True,
        )
        Ma = torch.tensor([0.1, 0.5, 0.9])
        w = model.dynamic_eos_weight(Ma)
        assert w.shape == (3,)
        assert (w >= 0).all() and (w <= 1).all()


# ---- Multicomponent v10 ----

class TestMulticomponentMixtureEnhanced9:
    def test_import(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_9 import (
            MulticomponentMixtureEnhanced9,
        )
        assert MulticomponentMixtureEnhanced9 is not None

    def test_create(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_9 import (
            MulticomponentMixtureEnhanced9,
        )
        mix = MulticomponentMixtureEnhanced9(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            D=[2e-9, 1e-9],
            cross_diffusion_coeff=0.1,
        )
        assert mix._n_species == 2

    def test_cross_diffusion_flux(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_9 import (
            MulticomponentMixtureEnhanced9,
        )
        mix = MulticomponentMixtureEnhanced9(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            D=[2e-9, 1e-9],
            cross_diffusion_coeff=0.1,
        )
        Y = torch.tensor([[0.5, 0.5]])
        grad_Y = torch.tensor([[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]])
        flux = mix.cross_diffusion_flux(Y, grad_Y)
        assert flux.shape == grad_Y.shape

    def test_cross_diffusion_disabled(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_9 import (
            MulticomponentMixtureEnhanced9,
        )
        mix = MulticomponentMixtureEnhanced9(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
        )
        Y = torch.tensor([[0.5, 0.5]])
        grad_Y = torch.tensor([[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]])
        flux = mix.cross_diffusion_flux(Y, grad_Y)
        assert (flux == 0).all()

    def test_mixture_kappa_correlation(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_9 import (
            MulticomponentMixtureEnhanced9,
        )
        mix = MulticomponentMixtureEnhanced9(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            kappa=[0.6, 0.17],
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([300.0])
        k = mix.mixture_kappa_correlation(Y, T)
        assert k.numel() == 1
        assert float(k.item()) > 0


# ---- Interfacial area v11 ----

class TestInterfacialAreaEnhanced10:
    def test_import(self):
        from pyfoam.multiphase.interfacial_area_enhanced_10 import (
            InterfacialArea10Model,
            MultiScaleAreaModel,
            ContactLineAreaModel,
            MonitoredAreaModel,
        )
        assert InterfacialArea10Model is not None

    def test_multi_scale(self):
        from pyfoam.multiphase.interfacial_area_enhanced_10 import MultiScaleAreaModel
        model = MultiScaleAreaModel(d32_0=3e-3)
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert a_i.shape == (2,)
        assert (a_i > 0).all()

    def test_contact_line(self):
        from pyfoam.multiphase.interfacial_area_enhanced_10 import ContactLineAreaModel
        model = ContactLineAreaModel(C_contact=0.1)
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert (a_i > 0).all()

    def test_monitored(self):
        from pyfoam.multiphase.interfacial_area_enhanced_10 import MonitoredAreaModel
        model = MonitoredAreaModel()
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert (a_i > 0).all()
        assert len(model.alert_history) == 1

    def test_registry(self):
        from pyfoam.multiphase.interfacial_area_enhanced_10 import InterfacialArea10Model
        types = InterfacialArea10Model.available_types()
        assert "multiScale" in types
        assert "contactLine" in types
        assert "monitored" in types


# ---- Turbulence damping v12 ----

class TestTurbulenceDampingEnhanced11:
    def test_import(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_11 import (
            TurbulenceDamping11EnhancedModel,
            GradientAwareDamping,
            ScaleDependentDamping,
            AdaptiveMonitoringDamping,
        )
        assert TurbulenceDamping11EnhancedModel is not None

    def test_gradient_aware(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_11 import GradientAwareDamping
        model = GradientAwareDamping()
        alpha = torch.tensor([0.3, 0.5, 0.8])
        d = model.compute_damping_factor(alpha, grad_alpha=torch.tensor([1.0, 0.5, 0.1]))
        assert d.shape == (3,)
        assert (d >= 0).all()

    def test_scale_dependent(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_11 import ScaleDependentDamping
        model = ScaleDependentDamping()
        alpha = torch.tensor([0.5])
        d = model.compute_damping_factor(alpha, delta=torch.tensor([0.01]), L_integral=torch.tensor([0.1]))
        assert d.numel() == 1

    def test_adaptive_monitoring(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_11 import AdaptiveMonitoringDamping
        model = AdaptiveMonitoringDamping(adaptation_rate=0.1)
        alpha = torch.tensor([0.5])
        d = model.compute_damping_factor(alpha)
        assert d.numel() == 1
        assert len(model.alert_history) == 1

    def test_registry(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_11 import TurbulenceDamping11EnhancedModel
        types = TurbulenceDamping11EnhancedModel.available_types()
        assert "gradientAware" in types
        assert "scaleDependent" in types
        assert "adaptiveMonitoring" in types
