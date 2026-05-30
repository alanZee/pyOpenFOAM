"""Tests for enhanced multiphase models (Phase 20)."""

import pytest
import torch


# ---- Incompressible VOF v11 ----

class TestIncompressibleMultiphaseVoFEnhanced10:
    def test_import(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_10 import (
            IncompressibleMultiphaseVoFEnhanced10,
        )
        assert IncompressibleMultiphaseVoFEnhanced10 is not None

    def test_create(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_10 import (
            IncompressibleMultiphaseVoFEnhanced10,
        )
        model = IncompressibleMultiphaseVoFEnhanced10(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert model._n_phases == 2

    def test_momentum_correction(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_10 import (
            IncompressibleMultiphaseVoFEnhanced10,
        )
        model = IncompressibleMultiphaseVoFEnhanced10(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            enable_momentum_correction=True,
        )
        alphas = torch.tensor([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
        f = model.momentum_correction_factor(alphas)
        assert f.shape == (3,)

    def test_repr(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_10 import (
            IncompressibleMultiphaseVoFEnhanced10,
        )
        model = IncompressibleMultiphaseVoFEnhanced10(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert "Enhanced10" in repr(model)


# ---- Compressible VOF v11 ----

class TestCompressibleMultiphaseVoFEnhanced10:
    def test_import(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_10 import (
            CompressibleMultiphaseVoFEnhanced10,
        )
        assert CompressibleMultiphaseVoFEnhanced10 is not None

    def test_create(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_10 import (
            CompressibleMultiphaseVoFEnhanced10,
        )
        model = CompressibleMultiphaseVoFEnhanced10(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            acoustic_damping=True,
        )
        assert model._n_phases == 2
        assert model._acoustic_damp is True

    def test_acoustic_damping(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_10 import (
            CompressibleMultiphaseVoFEnhanced10,
        )
        model = CompressibleMultiphaseVoFEnhanced10(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            acoustic_damping=True,
        )
        p = torch.tensor([101325.0, 200000.0, 150000.0])
        alphas = torch.tensor([[0.3, 0.7], [0.5, 0.5], [0.9, 0.1]])
        f = model.acoustic_damping_factor(p, 101325.0, alphas)
        assert f.shape == (3,)
        assert (f >= 0).all() and (f <= 1).all()

    def test_thermal_equilibrium(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_10 import (
            CompressibleMultiphaseVoFEnhanced10,
        )
        model = CompressibleMultiphaseVoFEnhanced10(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            thermal_equilibrium=True,
        )
        T_eq = model.thermal_equilibrium_temperature([400.0, 300.0], [0.5, 0.5])
        assert 300.0 <= T_eq <= 400.0


# ---- Multicomponent v11 ----

class TestMulticomponentMixtureEnhanced10:
    def test_import(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_10 import (
            MulticomponentMixtureEnhanced10,
        )
        assert MulticomponentMixtureEnhanced10 is not None

    def test_create(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_10 import (
            MulticomponentMixtureEnhanced10,
        )
        mix = MulticomponentMixtureEnhanced10(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            mixing_rule="ideal",
        )
        assert mix._n_species == 2
        assert mix._mixing_rule == "ideal"

    def test_ideal_mixture_property(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_10 import (
            MulticomponentMixtureEnhanced10,
        )
        mix = MulticomponentMixtureEnhanced10(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
        )
        Y = torch.tensor([[0.5, 0.5]])
        rho_mix = mix.ideal_mixture_property(Y, "rho")
        assert rho_mix.numel() == 1
        assert 789.0 <= float(rho_mix.item()) <= 1000.0

    def test_mixing_sensitivity(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_10 import (
            MulticomponentMixtureEnhanced10,
        )
        mix = MulticomponentMixtureEnhanced10(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([300.0])
        sens = mix.mixing_sensitivity(Y, T)
        assert "drho_dY" in sens
        assert "dmu_dY" in sens


# ---- Interfacial area v12 ----

class TestInterfacialAreaEnhanced11:
    def test_import(self):
        from pyfoam.multiphase.interfacial_area_enhanced_11 import (
            InterfacialArea11Model,
            DynamicSauterDiameterModel,
            PBECoupledTransportModel,
            BudgetTrackingModel,
        )
        assert InterfacialArea11Model is not None

    def test_dynamic_sauter(self):
        from pyfoam.multiphase.interfacial_area_enhanced_11 import DynamicSauterDiameterModel
        model = DynamicSauterDiameterModel(d32_0=3e-3)
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert a_i.shape == (2,)
        assert (a_i > 0).all()

    def test_pbe_transport(self):
        from pyfoam.multiphase.interfacial_area_enhanced_11 import PBECoupledTransportModel
        model = PBECoupledTransportModel(n_bins=5)
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert (a_i > 0).all()

    def test_budget_tracking(self):
        from pyfoam.multiphase.interfacial_area_enhanced_11 import BudgetTrackingModel
        model = BudgetTrackingModel()
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert (a_i > 0).all()
        assert len(model.budget_history) == 1

    def test_registry(self):
        from pyfoam.multiphase.interfacial_area_enhanced_11 import InterfacialArea11Model
        types = InterfacialArea11Model.available_types()
        assert "dynamicSauter" in types
        assert "pbeTransport" in types
        assert "budgetTracking" in types


# ---- Turbulence damping v13 ----

class TestTurbulenceDampingEnhanced12:
    def test_import(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_12 import (
            TurbulenceDamping12EnhancedModel,
            InterfaceNormalDamping,
            TKEBudgetDamping,
            TimeAveragedDamping,
        )
        assert TurbulenceDamping12EnhancedModel is not None

    def test_interface_normal(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_12 import InterfaceNormalDamping
        model = InterfaceNormalDamping()
        alpha = torch.tensor([0.3, 0.5, 0.8])
        d = model.compute_damping_factor(alpha, grad_alpha=torch.tensor([1.0, 0.5, 0.1]))
        assert d.shape == (3,)
        assert (d >= 0).all()

    def test_tke_budget(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_12 import TKEBudgetDamping
        model = TKEBudgetDamping(tau_t_ref=0.01)
        alpha = torch.tensor([0.5])
        d = model.compute_damping_factor(alpha, k=torch.tensor([0.01]))
        assert d.numel() == 1

    def test_time_averaged(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_12 import TimeAveragedDamping
        model = TimeAveragedDamping(averaging_coeff=0.1)
        alpha = torch.tensor([0.5])
        d1 = model.compute_damping_factor(alpha)
        d2 = model.compute_damping_factor(alpha)
        assert d1.numel() == 1
        assert d2.numel() == 1

    def test_registry(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_12 import TurbulenceDamping12EnhancedModel
        types = TurbulenceDamping12EnhancedModel.available_types()
        assert "interfaceNormal" in types
        assert "tkeBudget" in types
        assert "timeAveraged" in types
