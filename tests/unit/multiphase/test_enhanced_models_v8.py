"""Tests for enhanced multiphase models (Phase 18)."""

import pytest
import torch


# ---- Incompressible VOF v9 ----

class TestIncompressibleMultiphaseVoFEnhanced8:
    def test_import(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_8 import (
            IncompressibleMultiphaseVoFEnhanced8,
        )
        assert IncompressibleMultiphaseVoFEnhanced8 is not None

    def test_create(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_8 import (
            IncompressibleMultiphaseVoFEnhanced8,
        )
        model = IncompressibleMultiphaseVoFEnhanced8(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert model._n_phases == 2

    def test_quality_metrics(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_8 import (
            IncompressibleMultiphaseVoFEnhanced8,
        )
        model = IncompressibleMultiphaseVoFEnhanced8(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            quality_metrics=True,
        )
        alphas = torch.tensor([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
        qm = model.compute_quality_metrics(alphas)
        assert "smearing" in qm
        assert "boundedness" in qm

    def test_plic_flag(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_8 import (
            IncompressibleMultiphaseVoFEnhanced8,
        )
        model = IncompressibleMultiphaseVoFEnhanced8(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            plic_integration=True,
        )
        assert model.plic_integration_enabled

    def test_repr(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_8 import (
            IncompressibleMultiphaseVoFEnhanced8,
        )
        model = IncompressibleMultiphaseVoFEnhanced8(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert "Enhanced8" in repr(model)


# ---- Compressible VOF v9 ----

class TestCompressibleMultiphaseVoFEnhanced8:
    def test_import(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_8 import (
            CompressibleMultiphaseVoFEnhanced8,
        )
        assert CompressibleMultiphaseVoFEnhanced8 is not None

    def test_create(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_8 import (
            CompressibleMultiphaseVoFEnhanced8,
        )
        model = CompressibleMultiphaseVoFEnhanced8(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert model._n_phases == 2

    def test_pressure_wave_damping(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_8 import (
            CompressibleMultiphaseVoFEnhanced8,
        )
        model = CompressibleMultiphaseVoFEnhanced8(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            pressure_wave_damping=True,
            wave_damping_coeff=0.1,
        )
        p = torch.tensor([101325.0, 200000.0])
        alphas = torch.tensor([[0.3], [0.8]])
        p_damped = model.pressure_wave_damping(p, alphas, 0.001)
        assert p_damped.shape == (2,)

    def test_mixture_gamma(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_8 import (
            CompressibleMultiphaseVoFEnhanced8,
        )
        model = CompressibleMultiphaseVoFEnhanced8(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        alphas = torch.tensor([[0.3]])
        gamma_mix = model.mixture_gamma(alphas)
        assert gamma_mix.numel() == 1
        assert 1.0 < float(gamma_mix.item()) < 3.0


# ---- Multicomponent v9 ----

class TestMulticomponentMixtureEnhanced8:
    def test_import(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_8 import (
            MulticomponentMixtureEnhanced8,
        )
        assert MulticomponentMixtureEnhanced8 is not None

    def test_create(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_8 import (
            MulticomponentMixtureEnhanced8,
        )
        mix = MulticomponentMixtureEnhanced8(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            D=[2e-9, 1e-9],
            wilson_lambda=[[1.0, 0.5], [0.8, 1.0]],
            stefan_flow_coeff=0.1,
        )
        assert mix._n_species == 2

    def test_wilson_activity(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_8 import (
            MulticomponentMixtureEnhanced8,
        )
        mix = MulticomponentMixtureEnhanced8(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            D=[2e-9, 1e-9],
            wilson_lambda=[[1.0, 0.5], [0.8, 1.0]],
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([300.0])
        gamma = mix.wilson_activity_coefficients(Y, T)
        assert gamma.shape == (1, 2)
        assert (gamma > 0).all()

    def test_wilson_activity_no_lambda(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_8 import (
            MulticomponentMixtureEnhanced8,
        )
        mix = MulticomponentMixtureEnhanced8(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([300.0])
        gamma = mix.wilson_activity_coefficients(Y, T)
        assert (gamma == 1.0).all()  # No lambda -> ideal

    def test_wassiljewa_conductivity(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_8 import (
            MulticomponentMixtureEnhanced8,
        )
        mix = MulticomponentMixtureEnhanced8(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            kappa=[0.6, 0.17],
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([300.0])
        k = mix.wassiljewa_conductivity(Y, T)
        assert k.numel() == 1
        assert float(k.item()) > 0

    def test_stefan_flow(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_8 import (
            MulticomponentMixtureEnhanced8,
        )
        mix = MulticomponentMixtureEnhanced8(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            stefan_flow_coeff=0.1,
        )
        Y = torch.tensor([[0.5, 0.5]])
        Y_int = torch.tensor([[0.6, 0.4]])
        v = mix.stefan_flow_correction(Y, Y_int)
        assert v.numel() == 1
        assert float(v.item()) >= 0


# ---- Interfacial area v10 ----

class TestInterfacialAreaEnhanced9:
    def test_import(self):
        from pyfoam.multiphase.interfacial_area_enhanced_9 import (
            InterfacialArea9Model,
            FractalInterfaceArea,
            TurbulenceIntensityArea,
            PBECoupledArea,
        )
        assert InterfacialArea9Model is not None

    def test_fractal_interface(self):
        from pyfoam.multiphase.interfacial_area_enhanced_9 import FractalInterfaceArea
        model = FractalInterfaceArea(d32_0=3e-3, D_fractal=2.5)
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert a_i.shape == (2,)
        assert (a_i > 0).all()

    def test_turbulence_intensity(self):
        from pyfoam.multiphase.interfacial_area_enhanced_9 import TurbulenceIntensityArea
        model = TurbulenceIntensityArea()
        alpha = torch.tensor([0.3, 0.5])
        k = torch.tensor([0.01, 0.02])
        U_mean = torch.tensor([1.0, 2.0])
        a_i = model.compute(alpha, 2, k=k, U_mean=U_mean)
        assert (a_i > 0).all()

    def test_pbe_coupled(self):
        from pyfoam.multiphase.interfacial_area_enhanced_9 import PBECoupledArea
        model = PBECoupledArea(tau_relax=0.05)
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert (a_i > 0).all()

    def test_registry(self):
        from pyfoam.multiphase.interfacial_area_enhanced_9 import InterfacialArea9Model
        types = InterfacialArea9Model.available_types()
        assert "fractalInterface" in types
        assert "turbulenceIntensity" in types
        assert "pbeCoupled" in types


# ---- Turbulence damping v10 ----

class TestTurbulenceDampingEnhanced10:
    def test_import(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_10 import (
            TurbulenceDamping10EnhancedModel,
            DistanceWeightedDamping,
            TKEBudgetCorrectionDamping,
            InterfaceTurbulenceFeedbackDamping,
        )
        assert TurbulenceDamping10EnhancedModel is not None

    def test_distance_weighted(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_10 import DistanceWeightedDamping
        model = DistanceWeightedDamping(damping_coeff=10.0, d_ref=0.01)
        alpha = torch.tensor([0.3, 0.5, 0.8])
        d = model.compute_damping_factor(alpha, distance=torch.tensor([0.001, 0.01, 0.1]))
        assert d.shape == (3,)
        assert (d >= 0).all()

    def test_tke_budget(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_10 import TKEBudgetCorrectionDamping
        model = TKEBudgetCorrectionDamping()
        alpha = torch.tensor([0.5])
        d = model.compute_damping_factor(alpha, P_k=torch.tensor([0.1]), epsilon=torch.tensor([1.0]))
        assert d.numel() == 1

    def test_interface_feedback(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_10 import InterfaceTurbulenceFeedbackDamping
        model = InterfaceTurbulenceFeedbackDamping()
        alpha = torch.tensor([0.5])
        d = model.compute_damping_factor(alpha, curvature=torch.tensor([0.5]), anisotropy_ratio=torch.tensor([0.3]))
        assert d.numel() == 1
        assert float(d.item()) >= 0

    def test_registry(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_10 import TurbulenceDamping10EnhancedModel
        types = TurbulenceDamping10EnhancedModel.available_types()
        assert "distanceWeighted" in types
        assert "tkeBudgetCorrection" in types
        assert "interfaceTurbulenceFeedback" in types
