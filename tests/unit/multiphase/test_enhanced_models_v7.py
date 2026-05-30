"""Tests for enhanced multiphase models (Phase 17)."""

import pytest
import torch


# ---- Incompressible VOF v8 ----

class TestIncompressibleMultiphaseVoFEnhanced7:
    def test_import(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_7 import (
            IncompressibleMultiphaseVoFEnhanced7,
        )
        assert IncompressibleMultiphaseVoFEnhanced7 is not None

    def test_create(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_7 import (
            IncompressibleMultiphaseVoFEnhanced7,
        )
        model = IncompressibleMultiphaseVoFEnhanced7(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert model._n_phases == 2

    def test_topology_flag(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_7 import (
            IncompressibleMultiphaseVoFEnhanced7,
        )
        model = IncompressibleMultiphaseVoFEnhanced7(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            topology_analysis=True,
        )
        assert model.topology_analysis_enabled

    def test_repr(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_7 import (
            IncompressibleMultiphaseVoFEnhanced7,
        )
        model = IncompressibleMultiphaseVoFEnhanced7(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert "Enhanced7" in repr(model)


# ---- Compressible VOF v8 ----

class TestCompressibleMultiphaseVoFEnhanced7:
    def test_import(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_7 import (
            CompressibleMultiphaseVoFEnhanced7,
        )
        assert CompressibleMultiphaseVoFEnhanced7 is not None

    def test_create(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_7 import (
            CompressibleMultiphaseVoFEnhanced7,
        )
        model = CompressibleMultiphaseVoFEnhanced7(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert model._n_phases == 2

    def test_impedance(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_7 import (
            CompressibleMultiphaseVoFEnhanced7,
        )
        model = CompressibleMultiphaseVoFEnhanced7(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            impedance_matching=True,
        )
        alphas = torch.tensor([[0.3]])
        p = torch.tensor([101325.0])
        T = torch.tensor([300.0])
        Z = model.acoustic_impedance(alphas, p, T)
        assert Z.numel() == 1
        assert float(Z.item()) > 0


# ---- Multicomponent v8 ----

class TestMulticomponentMixtureEnhanced7:
    def test_import(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_7 import (
            MulticomponentMixtureEnhanced7,
        )
        assert MulticomponentMixtureEnhanced7 is not None

    def test_create(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_7 import (
            MulticomponentMixtureEnhanced7,
        )
        mix = MulticomponentMixtureEnhanced7(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            D=[2e-9, 1e-9],
            margules_A12=1000.0,
            margules_A21=800.0,
        )
        assert mix._n_species == 2

    def test_margules(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_7 import (
            MulticomponentMixtureEnhanced7,
        )
        mix = MulticomponentMixtureEnhanced7(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            D=[2e-9, 1e-9],
            margules_A12=1000.0,
            margules_A21=800.0,
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([300.0])
        gamma = mix.margules_activity_coefficients(Y, T)
        assert gamma.shape == (1, 2)
        assert (gamma > 0).all()

    def test_mixing_enthalpy(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_7 import (
            MulticomponentMixtureEnhanced7,
        )
        mix = MulticomponentMixtureEnhanced7(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            D=[2e-9, 1e-9],
            margules_A12=1000.0,
            margules_A21=800.0,
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([300.0])
        H = mix.mixing_enthalpy(Y, T)
        assert H.numel() == 1


# ---- Interfacial area v9 ----

class TestInterfacialAreaEnhanced8:
    def test_import(self):
        from pyfoam.multiphase.interfacial_area_enhanced_8 import (
            InterfacialArea8Model,
            TurbulentEnhancedArea,
            SizeDependentBreakupArea,
            TimeRelaxationArea,
        )
        assert InterfacialArea8Model is not None

    def test_turbulent_enhanced(self):
        from pyfoam.multiphase.interfacial_area_enhanced_8 import TurbulentEnhancedArea
        model = TurbulentEnhancedArea(d32_0=3e-3)
        alpha = torch.tensor([0.3, 0.5])
        k = torch.tensor([0.01, 0.02])
        a_i = model.compute(alpha, 2, k=k)
        assert a_i.shape == (2,)
        assert (a_i > 0).all()

    def test_size_dependent_breakup(self):
        from pyfoam.multiphase.interfacial_area_enhanced_8 import SizeDependentBreakupArea
        model = SizeDependentBreakupArea()
        alpha = torch.tensor([0.3, 0.5])
        We = torch.tensor([5.0, 10.0])
        a_i = model.compute(alpha, 2, We=We)
        assert (a_i > 0).all()

    def test_time_relaxation(self):
        from pyfoam.multiphase.interfacial_area_enhanced_8 import TimeRelaxationArea
        model = TimeRelaxationArea(tau_relax=0.05)
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert (a_i > 0).all()

    def test_registry(self):
        from pyfoam.multiphase.interfacial_area_enhanced_8 import InterfacialArea8Model
        types = InterfacialArea8Model.available_types()
        assert "turbulentEnhanced" in types
        assert "sizeDependentBreakup" in types
        assert "timeRelaxation" in types


# ---- Turbulence damping v9 ----

class TestTurbulenceDampingEnhanced9:
    def test_import(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_9 import (
            TurbulenceDamping9EnhancedModel,
            AdaptiveCoefficientDamping,
            TKEProductionDamping,
            StratifiedFlowDamping,
        )
        assert TurbulenceDamping9EnhancedModel is not None

    def test_adaptive_coefficient(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_9 import AdaptiveCoefficientDamping
        model = AdaptiveCoefficientDamping(damping_coeff=10.0)
        alpha = torch.tensor([0.3, 0.5, 0.8])
        d = model.compute_damping_factor(alpha, grad_alpha_mag=torch.tensor([0.1, 0.5, 0.01]))
        assert d.shape == (3,)
        assert (d >= 0).all()

    def test_tke_production(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_9 import TKEProductionDamping
        model = TKEProductionDamping()
        alpha = torch.tensor([0.5])
        d = model.compute_damping_factor(alpha, P_k=torch.tensor([0.1]), epsilon=torch.tensor([1.0]))
        assert d.numel() == 1

    def test_stratified_flow(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_9 import StratifiedFlowDamping
        model = StratifiedFlowDamping()
        alpha = torch.tensor([0.5])
        d = model.compute_damping_factor(alpha, S_mag=torch.tensor([10.0]), L_char=torch.tensor([0.01]))
        assert d.numel() == 1
        assert float(d.item()) >= 0

    def test_registry(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_9 import TurbulenceDamping9EnhancedModel
        types = TurbulenceDamping9EnhancedModel.available_types()
        assert "adaptiveCoefficient" in types
        assert "tkeProduction" in types
        assert "stratifiedFlow" in types
