"""Tests for tabulated transport model.

Tests cover:
- TabulatedTransport initialisation and validation
- Linear interpolation of mu and kappa
- Boundary clamping (out-of-range temperatures)
- Properties access
"""

import pytest
import torch

from pyfoam.thermophysical.tabulated_transport import TabulatedTransport


class TestTabulatedTransport:
    """Tests for TabulatedTransport."""

    def test_init_basic(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400, 500],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5, 2.80e-5],
        )
        assert len(transport.T_data) == 4
        assert transport.T_data[0] == 200
        assert transport.T_data[-1] == 500

    def test_init_with_kappa(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
            kappa_data=[0.0181, 0.0263, 0.0338],
        )
        assert transport.kappa_data is not None

    def test_init_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            TabulatedTransport(T_data=[300], mu_data=[1.8e-5])

    def test_init_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="mu_data length"):
            TabulatedTransport(T_data=[200, 300], mu_data=[1.8e-5])

    def test_init_not_increasing_raises(self):
        with pytest.raises(ValueError, match="increasing"):
            TabulatedTransport(T_data=[300, 200], mu_data=[1.8e-5, 2.0e-5])

    def test_mu_at_data_point(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
        )
        mu = transport.mu(T=300.0)
        assert float(mu.item()) == pytest.approx(1.85e-5, rel=1e-6)

    def test_mu_interpolation(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
        )
        # At T=250 (midpoint), should be average of endpoints
        mu = transport.mu(T=250.0)
        expected = 0.5 * (1.33e-5 + 1.85e-5)
        assert float(mu.item()) == pytest.approx(expected, rel=1e-6)

    def test_mu_clamp_below(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
        )
        mu = transport.mu(T=100.0)
        assert float(mu.item()) == pytest.approx(1.33e-5, rel=1e-6)

    def test_mu_clamp_above(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
        )
        mu = transport.mu(T=500.0)
        assert float(mu.item()) == pytest.approx(2.34e-5, rel=1e-6)

    def test_mu_tensor(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
        )
        T = torch.tensor([200.0, 250.0, 300.0, 350.0, 400.0])
        mu = transport.mu(T)
        assert mu.shape == (5,)
        assert (mu > 0).all()

    def test_kappa_tabulated(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
            kappa_data=[0.0181, 0.0263, 0.0338],
        )
        kappa = transport.kappa(T=300.0)
        assert float(kappa.item()) == pytest.approx(0.0263, rel=1e-4)

    def test_kappa_from_prandtl(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
        )
        Cp, Pr = 1005.0, 0.7
        kappa = transport.kappa(T=300.0, Cp=Cp, Pr=Pr)
        mu = transport.mu(T=300.0)
        expected = mu * Cp / Pr
        assert float(kappa.item()) == pytest.approx(float(expected.item()), rel=1e-6)

    def test_properties(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
        )
        assert transport.T_data == [200, 300, 400]
        assert transport.mu_data == [1.33e-5, 1.85e-5, 2.34e-5]
        assert transport.kappa_data is None

    def test_properties_copy(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
        )
        td = transport.T_data
        td.append(999)
        assert transport.T_data == [200, 300, 400]

    def test_n_points(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400, 500, 600],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5, 2.80e-5, 3.22e-5],
        )
        assert len(transport.T_data) == 5

    def test_repr(self):
        transport = TabulatedTransport(
            T_data=[200, 300, 400],
            mu_data=[1.33e-5, 1.85e-5, 2.34e-5],
        )
        r = repr(transport)
        assert "TabulatedTransport" in r
        assert "200" in r

    def test_inherits_from_transport_model(self):
        from pyfoam.thermophysical.transport_model import TransportModel
        transport = TabulatedTransport(
            T_data=[200, 300],
            mu_data=[1.33e-5, 1.85e-5],
        )
        assert isinstance(transport, TransportModel)
        assert hasattr(transport, 'mu')
        assert hasattr(transport, 'kappa')
        assert hasattr(transport, 'nu')
