"""Tests for enhanced multiphase models (Phase 16)."""

import pytest
import torch


# ---- Incompressible VOF v7 ----

class TestIncompressibleMultiphaseVoFEnhanced6:
    def test_import(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_6 import (
            IncompressibleMultiphaseVoFEnhanced6,
        )
        assert IncompressibleMultiphaseVoFEnhanced6 is not None

    def test_create(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_6 import (
            IncompressibleMultiphaseVoFEnhanced6,
        )
        model = IncompressibleMultiphaseVoFEnhanced6(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert model._n_phases == 2

    def test_geometric_recon_flag(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_6 import (
            IncompressibleMultiphaseVoFEnhanced6,
        )
        model = IncompressibleMultiphaseVoFEnhanced6(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            geometric_reconstruction=True,
        )
        assert model.geometric_reconstruction_enabled

    def test_repr(self):
        from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_6 import (
            IncompressibleMultiphaseVoFEnhanced6,
        )
        model = IncompressibleMultiphaseVoFEnhanced6(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert "Enhanced6" in repr(model)


# ---- Compressible VOF v7 ----

class TestCompressibleMultiphaseVoFEnhanced6:
    def test_import(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_6 import (
            CompressibleMultiphaseVoFEnhanced6,
        )
        assert CompressibleMultiphaseVoFEnhanced6 is not None

    def test_create(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_6 import (
            CompressibleMultiphaseVoFEnhanced6,
        )
        model = CompressibleMultiphaseVoFEnhanced6(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert model._n_phases == 2
        assert model.acoustic_CFL == 0.5

    def test_viscosity_mixing(self):
        from pyfoam.multiphase.compressible_multiphase_vof_enhanced_6 import (
            CompressibleMultiphaseVoFEnhanced6,
        )
        model = CompressibleMultiphaseVoFEnhanced6(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            viscosity_mixing="logarithmic",
        )
        alphas = torch.tensor([[0.3]])
        mu = model.mixture_viscosity(alphas)
        assert mu.numel() == 1
        assert float(mu.item()) > 0


# ---- Multicomponent v7 ----

class TestMulticomponentMixtureEnhanced6:
    def test_import(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_6 import (
            MulticomponentMixtureEnhanced6,
        )
        assert MulticomponentMixtureEnhanced6 is not None

    def test_create(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_6 import (
            MulticomponentMixtureEnhanced6,
        )
        mix = MulticomponentMixtureEnhanced6(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
            D=[2.1e-5, 2.1e-5],
        )
        assert mix._n_species == 2

    def test_uniquac(self):
        from pyfoam.multiphase.multicomponent_mixture_enhanced_6 import (
            MulticomponentMixtureEnhanced6,
        )
        mix = MulticomponentMixtureEnhanced6(
            species=["A", "B"],
            M=[18e-3, 46e-3],
            rho=[1000.0, 789.0],
            mu=[1e-3, 1.2e-3],
            Cp=[4180.0, 2440.0],
            D=[2e-9, 1e-9],
            uniquac_r=[0.92, 2.11],
            uniquac_q=[1.4, 1.97],
        )
        Y = torch.tensor([[0.5, 0.5]])
        T = torch.tensor([300.0])
        gamma = mix.uniquac_activity_coefficients(Y, T)
        assert gamma.shape == (1, 2)
        assert (gamma > 0).all()


# ---- Interfacial area v8 ----

class TestInterfacialAreaEnhanced7:
    def test_import(self):
        from pyfoam.multiphase.interfacial_area_enhanced_7 import (
            InterfacialArea7Model,
            HamakerCorrectedArea,
            CoalescenceBreakupEquilibrium,
            PolydispersePBETArea,
        )
        assert InterfacialArea7Model is not None

    def test_hamaker(self):
        from pyfoam.multiphase.interfacial_area_enhanced_7 import HamakerCorrectedArea
        model = HamakerCorrectedArea(d32_0=3e-3)
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert a_i.shape == (2,)
        assert (a_i > 0).all()

    def test_coalescence_breakup(self):
        from pyfoam.multiphase.interfacial_area_enhanced_7 import CoalescenceBreakupEquilibrium
        model = CoalescenceBreakupEquilibrium()
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert (a_i > 0).all()

    def test_polydisperse(self):
        from pyfoam.multiphase.interfacial_area_enhanced_7 import PolydispersePBETArea
        model = PolydispersePBETArea(n_bins=5)
        alpha = torch.tensor([0.3, 0.5])
        a_i = model.compute(alpha, 2)
        assert (a_i > 0).all()

    def test_registry(self):
        from pyfoam.multiphase.interfacial_area_enhanced_7 import InterfacialArea7Model
        types = InterfacialArea7Model.available_types()
        assert "hamaker" in types
        assert "coalescenceBreakupEq" in types

    def test_source_terms(self):
        from pyfoam.multiphase.interfacial_area_enhanced_7 import HamakerCorrectedArea
        model = HamakerCorrectedArea()
        alpha = torch.tensor([0.5])
        a_i = torch.tensor([100.0])
        st = model.source_terms(alpha, a_i)
        assert "relaxation" in st


# ---- Turbulence damping v8 ----

class TestTurbulenceDampingEnhanced8:
    def test_import(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_8 import (
            TurbulenceDamping8EnhancedModel,
            InterfaceAwareLESDamping,
            TurbulenceInterfaceCoupledDamping,
            MultiScaleCascadeDamping,
        )
        assert TurbulenceDamping8EnhancedModel is not None

    def test_interface_aware_les(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_8 import InterfaceAwareLESDamping
        model = InterfaceAwareLESDamping(damping_coeff=10.0)
        alpha = torch.tensor([0.3, 0.5, 0.8])
        grad_alpha = torch.tensor([0.1, 0.5, 0.01])
        d = model.compute_damping_factor(alpha, grad_alpha_mag=grad_alpha)
        assert d.shape == (3,)
        assert (d >= 0).all()

    def test_coupled(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_8 import TurbulenceInterfaceCoupledDamping
        model = TurbulenceInterfaceCoupledDamping()
        alpha = torch.tensor([0.5])
        d = model.compute_damping_factor(alpha)
        assert d.numel() == 1

    def test_cascade(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_8 import MultiScaleCascadeDamping
        model = MultiScaleCascadeDamping(N_scales=4)
        alpha = torch.tensor([0.5])
        d = model.compute_damping_factor(alpha)
        assert d.numel() == 1
        assert float(d.item()) >= 0

    def test_registry(self):
        from pyfoam.multiphase.turbulence_damping_enhanced_8 import TurbulenceDamping8EnhancedModel
        types = TurbulenceDamping8EnhancedModel.available_types()
        assert "interfaceAwareLES" in types
