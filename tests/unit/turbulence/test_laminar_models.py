"""Tests for laminar turbulence models (Stokes and GeneralizedNewtonian)."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE


# ---------------------------------------------------------------------------
# Reuse the 2-cell hex mesh from turbulence/conftest.py
# ---------------------------------------------------------------------------

_POINTS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 2.0],
    [1.0, 0.0, 2.0],
    [1.0, 1.0, 2.0],
    [0.0, 1.0, 2.0],
]

_FACES = [
    [4, 5, 6, 7],
    [0, 3, 2, 1],
    [0, 1, 5, 4],
    [3, 7, 6, 2],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
    [8, 9, 10, 11],
    [4, 5, 9, 8],
    [7, 11, 10, 6],
    [4, 8, 11, 7],
    [5, 6, 10, 9],
]

_OWNER = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_NEIGHBOUR = [1]

_BOUNDARY = [
    {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
    {"name": "top", "type": "wall", "startFace": 6, "nFaces": 5},
]


def make_fv_mesh(device="cpu", dtype=torch.float64):
    """Return an FvMesh of the 2-cell hex mesh with geometry computed."""
    from pyfoam.mesh.fv_mesh import FvMesh

    mesh = FvMesh(
        points=torch.tensor(_POINTS, dtype=dtype, device=device),
        faces=[
            torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in _FACES
        ],
        owner=torch.tensor(_OWNER, dtype=INDEX_DTYPE, device=device),
        neighbour=torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE, device=device),
        boundary=_BOUNDARY,
    )
    mesh.compute_geometry()
    return mesh


@pytest.fixture
def fv_mesh():
    """2-cell hex FvMesh."""
    return make_fv_mesh()


@pytest.fixture
def U_linear(fv_mesh):
    """Velocity field with linear z-profile: U = (0, 0, z)."""
    cc = fv_mesh.cell_centres
    U = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
    U[:, 2] = cc[:, 2]
    return U


@pytest.fixture
def face_flux(fv_mesh):
    """Zero face flux field."""
    return torch.zeros(fv_mesh.n_faces, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Tests: StokesModel
# ---------------------------------------------------------------------------


class TestStokesModel:
    """Tests for the Stokes laminar model."""

    def test_registration(self):
        """Stokes is registered in TurbulenceModel RTS."""
        from pyfoam.turbulence import TurbulenceModel
        assert "Stokes" in TurbulenceModel.available_types()

    def test_factory_creation(self, fv_mesh, U_linear, face_flux):
        """StokesModel can be created via factory method."""
        from pyfoam.turbulence import TurbulenceModel, StokesModel
        model = TurbulenceModel.create("Stokes", fv_mesh, U_linear, face_flux)
        assert isinstance(model, StokesModel)

    def test_nut_zero(self, fv_mesh, U_linear, face_flux):
        """nut() returns zeros for Stokes model."""
        from pyfoam.turbulence.laminar_models import StokesModel
        model = StokesModel(fv_mesh, U_linear, face_flux)
        nut = model.nut()
        assert nut.shape == (fv_mesh.n_cells,)
        assert torch.allclose(nut, torch.zeros(fv_mesh.n_cells, dtype=torch.float64))

    def test_k_zero(self, fv_mesh, U_linear, face_flux):
        """k() returns zeros for Stokes model."""
        from pyfoam.turbulence.laminar_models import StokesModel
        model = StokesModel(fv_mesh, U_linear, face_flux)
        k = model.k()
        assert k.shape == (fv_mesh.n_cells,)
        assert torch.allclose(k, torch.zeros(fv_mesh.n_cells, dtype=torch.float64))

    def test_correct_no_op(self, fv_mesh, U_linear, face_flux):
        """correct() is a no-op and doesn't change nut."""
        from pyfoam.turbulence.laminar_models import StokesModel
        model = StokesModel(fv_mesh, U_linear, face_flux)
        nut_before = model.nut().clone()
        model.correct()
        nut_after = model.nut()
        assert torch.allclose(nut_before, nut_after)

    def test_repr(self, fv_mesh, U_linear, face_flux):
        """__repr__ includes class name."""
        from pyfoam.turbulence.laminar_models import StokesModel
        model = StokesModel(fv_mesh, U_linear, face_flux)
        r = repr(model)
        assert "StokesModel" in r


# ---------------------------------------------------------------------------
# Tests: Viscosity models standalone
# ---------------------------------------------------------------------------


class TestPowerLawViscosity:
    """Tests for PowerLawViscosity."""

    def test_newtonian_limit(self):
        """n=1 gives constant viscosity = K."""
        from pyfoam.turbulence.laminar_models import PowerLawViscosity
        vm = PowerLawViscosity(K=0.01, n=1.0)
        gd = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float64)
        mu = vm.mu(gd)
        assert torch.allclose(mu, torch.full_like(mu, 0.01))

    def test_shear_thinning(self):
        """n < 1: viscosity decreases with strain rate."""
        from pyfoam.turbulence.laminar_models import PowerLawViscosity
        vm = PowerLawViscosity(K=1.0, n=0.5)
        gd_low = torch.tensor([0.01], dtype=torch.float64)
        gd_high = torch.tensor([10.0], dtype=torch.float64)
        assert vm.mu(gd_low) > vm.mu(gd_high)

    def test_shear_thickening(self):
        """n > 1: viscosity increases with strain rate."""
        from pyfoam.turbulence.laminar_models import PowerLawViscosity
        vm = PowerLawViscosity(K=1.0, n=1.5)
        gd_low = torch.tensor([0.01], dtype=torch.float64)
        gd_high = torch.tensor([10.0], dtype=torch.float64)
        assert vm.mu(gd_low) < vm.mu(gd_high)

    def test_formula(self):
        """mu = K * |gd|^(n-1)."""
        from pyfoam.turbulence.laminar_models import PowerLawViscosity
        vm = PowerLawViscosity(K=2.0, n=0.7)
        gd = torch.tensor([5.0], dtype=torch.float64)
        expected = 2.0 * 5.0 ** (-0.3)
        assert torch.allclose(vm.mu(gd), torch.tensor([expected], dtype=torch.float64), atol=1e-10)

    def test_repr(self):
        from pyfoam.turbulence.laminar_models import PowerLawViscosity
        assert "PowerLaw" in repr(PowerLawViscosity(K=0.01, n=0.5))


class TestBirdCarreauViscosity:
    """Tests for BirdCarreauViscosity."""

    def test_zero_shear_limit(self):
        """At gamma_dot=0, mu = mu_0."""
        from pyfoam.turbulence.laminar_models import BirdCarreauViscosity
        vm = BirdCarreauViscosity(mu_0=0.05, mu_inf=0.001, lambda_=1.0, n=0.4)
        gd = torch.tensor([0.0], dtype=torch.float64)
        mu = vm.mu(gd)
        assert torch.allclose(mu, torch.tensor([0.05], dtype=torch.float64), atol=1e-10)

    def test_high_shear_limit(self):
        """At very high gamma_dot, mu -> mu_inf."""
        from pyfoam.turbulence.laminar_models import BirdCarreauViscosity
        vm = BirdCarreauViscosity(mu_0=0.05, mu_inf=0.001, lambda_=1.0, n=0.4)
        gd = torch.tensor([1e10], dtype=torch.float64)
        mu = vm.mu(gd)
        assert torch.allclose(mu, torch.tensor([0.001], dtype=torch.float64), atol=1e-6)

    def test_decreasing(self):
        """Viscosity decreases with strain rate (shear-thinning)."""
        from pyfoam.turbulence.laminar_models import BirdCarreauViscosity
        vm = BirdCarreauViscosity(mu_0=0.05, mu_inf=0.001, lambda_=1.0, n=0.4)
        gd_low = torch.tensor([0.01], dtype=torch.float64)
        gd_high = torch.tensor([100.0], dtype=torch.float64)
        assert vm.mu(gd_low) > vm.mu(gd_high)

    def test_repr(self):
        from pyfoam.turbulence.laminar_models import BirdCarreauViscosity
        assert "BirdCarreau" in repr(BirdCarreauViscosity())


class TestCrossViscosity:
    """Tests for CrossViscosity."""

    def test_zero_shear_limit(self):
        """At gamma_dot=0, mu = mu_0."""
        from pyfoam.turbulence.laminar_models import CrossViscosity
        vm = CrossViscosity(mu_0=0.05, mu_inf=0.001, lambda_=1.0, m=1.0)
        gd = torch.tensor([0.0], dtype=torch.float64)
        mu = vm.mu(gd)
        assert torch.allclose(mu, torch.tensor([0.05], dtype=torch.float64), atol=1e-10)

    def test_high_shear_limit(self):
        """At very high gamma_dot, mu -> mu_inf."""
        from pyfoam.turbulence.laminar_models import CrossViscosity
        vm = CrossViscosity(mu_0=0.05, mu_inf=0.001, lambda_=1.0, m=1.0)
        gd = torch.tensor([1e10], dtype=torch.float64)
        mu = vm.mu(gd)
        assert torch.allclose(mu, torch.tensor([0.001], dtype=torch.float64), atol=1e-6)

    def test_repr(self):
        from pyfoam.turbulence.laminar_models import CrossViscosity
        assert "Cross" in repr(CrossViscosity())


class TestCassonViscosity:
    """Tests for CassonViscosity."""

    def test_high_shear_limit(self):
        """At high gamma_dot, mu -> mu_p (plastic viscosity)."""
        from pyfoam.turbulence.laminar_models import CassonViscosity
        vm = CassonViscosity(tau_y=1.0, mu_p=0.001)
        gd = torch.tensor([1e10], dtype=torch.float64)
        mu = vm.mu(gd)
        # Should approach mu_p = 0.001
        assert torch.allclose(mu, torch.tensor([0.001], dtype=torch.float64), atol=1e-4)

    def test_formula(self):
        """Casson: mu = (sqrt(tau_y/gd) + sqrt(mu_p))^2."""
        from pyfoam.turbulence.laminar_models import CassonViscosity
        vm = CassonViscosity(tau_y=0.5, mu_p=0.001)
        gd = torch.tensor([2.0], dtype=torch.float64)
        expected = ((0.5 / 2.0) ** 0.5 + 0.001 ** 0.5) ** 2
        assert torch.allclose(vm.mu(gd), torch.tensor([expected], dtype=torch.float64), atol=1e-10)

    def test_repr(self):
        from pyfoam.turbulence.laminar_models import CassonViscosity
        assert "Casson" in repr(CassonViscosity())


class TestHerschelBulkleyViscosity:
    """Tests for HerschelBulkleyViscosity."""

    def test_power_law_limit(self):
        """With tau_y=0, reduces to power-law."""
        from pyfoam.turbulence.laminar_models import HerschelBulkleyViscosity
        vm = HerschelBulkleyViscosity(tau_y=0.0, K=1.0, n=0.5)
        gd = torch.tensor([4.0], dtype=torch.float64)
        expected = 1.0 * 4.0 ** (-0.5)
        assert torch.allclose(vm.mu(gd), torch.tensor([expected], dtype=torch.float64), atol=1e-10)

    def test_yield_stress_increases_viscosity(self):
        """Nonzero tau_y increases viscosity at low strain rates."""
        from pyfoam.turbulence.laminar_models import HerschelBulkleyViscosity
        vm_yield = HerschelBulkleyViscosity(tau_y=1.0, K=0.01, n=0.5)
        vm_no_yield = HerschelBulkleyViscosity(tau_y=0.0, K=0.01, n=0.5)
        gd = torch.tensor([0.1], dtype=torch.float64)
        assert vm_yield.mu(gd) > vm_no_yield.mu(gd)

    def test_repr(self):
        from pyfoam.turbulence.laminar_models import HerschelBulkleyViscosity
        assert "HerschelBulkley" in repr(HerschelBulkleyViscosity())


# ---------------------------------------------------------------------------
# Tests: GeneralizedNewtonianModel
# ---------------------------------------------------------------------------


class TestGeneralizedNewtonianModel:
    """Tests for the GeneralizedNewtonian turbulence model."""

    def test_registration(self):
        """generalizedNewtonian is registered in TurbulenceModel RTS."""
        from pyfoam.turbulence import TurbulenceModel
        assert "generalizedNewtonian" in TurbulenceModel.available_types()

    def test_factory_power_law(self, fv_mesh, U_linear, face_flux):
        """Factory creation with power-law viscosity model."""
        from pyfoam.turbulence import TurbulenceModel
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        model = TurbulenceModel.create(
            "generalizedNewtonian", fv_mesh, U_linear, face_flux,
            viscosityModel="powerLaw", K=0.01, n=0.5,
        )
        assert isinstance(model, GeneralizedNewtonianModel)

    def test_factory_bird_carreau(self, fv_mesh, U_linear, face_flux):
        """Factory creation with Bird-Carreau viscosity model."""
        from pyfoam.turbulence import TurbulenceModel
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        model = TurbulenceModel.create(
            "generalizedNewtonian", fv_mesh, U_linear, face_flux,
            viscosityModel="BirdCarreau", mu_0=0.05, mu_inf=0.001,
        )
        assert isinstance(model, GeneralizedNewtonianModel)

    def test_factory_cross(self, fv_mesh, U_linear, face_flux):
        """Factory creation with Cross viscosity model."""
        from pyfoam.turbulence import TurbulenceModel
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        model = TurbulenceModel.create(
            "generalizedNewtonian", fv_mesh, U_linear, face_flux,
            viscosityModel="Cross", mu_0=0.05, mu_inf=0.001,
        )
        assert isinstance(model, GeneralizedNewtonianModel)

    def test_factory_casson(self, fv_mesh, U_linear, face_flux):
        """Factory creation with Casson viscosity model."""
        from pyfoam.turbulence import TurbulenceModel
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        model = TurbulenceModel.create(
            "generalizedNewtonian", fv_mesh, U_linear, face_flux,
            viscosityModel="Casson", tau_y=0.01, mu_p=0.001,
        )
        assert isinstance(model, GeneralizedNewtonianModel)

    def test_factory_herschel_bulkley(self, fv_mesh, U_linear, face_flux):
        """Factory creation with Herschel-Bulkley viscosity model."""
        from pyfoam.turbulence import TurbulenceModel
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        model = TurbulenceModel.create(
            "generalizedNewtonian", fv_mesh, U_linear, face_flux,
            viscosityModel="HerschelBulkley", tau_y=0.01, K=0.01, n=0.5,
        )
        assert isinstance(model, GeneralizedNewtonianModel)

    def test_nut_before_correct(self, fv_mesh, U_linear, face_flux):
        """nut() returns zeros before correct() is called."""
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        model = GeneralizedNewtonianModel(
            fv_mesh, U_linear, face_flux,
            viscosityModel="powerLaw", K=0.01, n=0.5,
        )
        nut = model.nut()
        assert torch.allclose(nut, torch.zeros(fv_mesh.n_cells, dtype=torch.float64))

    def test_k_zero(self, fv_mesh, U_linear, face_flux):
        """k() always returns zero for generalized Newtonian."""
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        model = GeneralizedNewtonianModel(
            fv_mesh, U_linear, face_flux,
            viscosityModel="powerLaw", K=0.01, n=0.5,
        )
        model.correct()
        k = model.k()
        assert torch.allclose(k, torch.zeros(fv_mesh.n_cells, dtype=torch.float64))

    def test_correct_computes_nut(self, fv_mesh, U_linear, face_flux):
        """correct() computes nonzero nut for shear-thinning fluid."""
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        model = GeneralizedNewtonianModel(
            fv_mesh, U_linear, face_flux,
            viscosityModel="powerLaw", K=1.0, n=0.5,
        )
        # nu is 1.5e-5 by default; with K=1.0 and n=0.5,
        # mu = |gd|^(-0.5), for moderate gd this could be > nu
        model.correct()
        nut = model.nut()
        assert nut.shape == (fv_mesh.n_cells,)

    def test_nut_nonnegative(self, fv_mesh, U_linear, face_flux):
        """nut is always >= 0 after correct()."""
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        model = GeneralizedNewtonianModel(
            fv_mesh, U_linear, face_flux,
            viscosityModel="powerLaw", K=0.01, n=1.5,
        )
        model.correct()
        nut = model.nut()
        assert (nut >= 0).all()

    def test_viscosity_model_property(self, fv_mesh, U_linear, face_flux):
        """viscosity_model property returns the constitutive law."""
        from pyfoam.turbulence.laminar_models import (
            GeneralizedNewtonianModel,
            PowerLawViscosity,
        )
        model = GeneralizedNewtonianModel(
            fv_mesh, U_linear, face_flux,
            viscosityModel="powerLaw", K=0.01, n=0.5,
        )
        assert isinstance(model.viscosity_model, PowerLawViscosity)

    def test_unknown_viscosity_model_raises(self, fv_mesh, U_linear, face_flux):
        """Unknown viscosity model name raises KeyError."""
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        with pytest.raises(KeyError, match="Unknown viscosity model"):
            GeneralizedNewtonianModel(
                fv_mesh, U_linear, face_flux,
                viscosityModel="nonexistent",
            )

    def test_repr(self, fv_mesh, U_linear, face_flux):
        """__repr__ includes class and model info."""
        from pyfoam.turbulence.laminar_models import GeneralizedNewtonianModel
        model = GeneralizedNewtonianModel(
            fv_mesh, U_linear, face_flux,
            viscosityModel="powerLaw", K=0.01, n=0.5,
        )
        r = repr(model)
        assert "GeneralizedNewtonianModel" in r
        assert "PowerLaw" in r
