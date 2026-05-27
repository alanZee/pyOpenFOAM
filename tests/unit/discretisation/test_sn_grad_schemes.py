"""Tests for surface-normal gradient schemes — snGrad.

Tests the ``SnGradScheme`` hierarchy:
- ``UncorrectedSnGrad`` — exact for orthogonal meshes
- ``CorrectedSnGrad`` — full non-orthogonal correction
- ``LimitedSnGrad`` — limited correction with coefficient

Both orthogonal and non-orthogonal 2-cell meshes are used.
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.discretisation.sn_grad import (
    SnGradScheme,
    UncorrectedSnGrad,
    CorrectedSnGrad,
    LimitedSnGrad,
    sn_grad_from_name,
    _SN_GRAD_REGISTRY,
    _compute_correction_vectors,
)


# ---------------------------------------------------------------------------
# Orthogonal 2-cell hex mesh (same as conftest)
# ---------------------------------------------------------------------------

_POINTS_ORTHO = [
    [0.0, 0.0, 0.0],  # 0
    [1.0, 0.0, 0.0],  # 1
    [1.0, 1.0, 0.0],  # 2
    [0.0, 1.0, 0.0],  # 3
    [0.0, 0.0, 1.0],  # 4
    [1.0, 0.0, 1.0],  # 5
    [1.0, 1.0, 1.0],  # 6
    [0.0, 1.0, 1.0],  # 7
    [0.0, 0.0, 2.0],  # 8
    [1.0, 0.0, 2.0],  # 9
    [1.0, 1.0, 2.0],  # 10
    [0.0, 1.0, 2.0],  # 11
]

_FACES_ORTHO = [
    [4, 5, 6, 7],     # 0: internal
    [0, 3, 2, 1],     # 1
    [0, 1, 5, 4],     # 2
    [3, 7, 6, 2],     # 3
    [0, 4, 7, 3],     # 4
    [1, 2, 6, 5],     # 5
    [8, 9, 10, 11],   # 6
    [4, 5, 9, 8],     # 7
    [7, 11, 10, 6],   # 8
    [4, 8, 11, 7],    # 9
    [5, 6, 10, 9],    # 10
]

_OWNER_ORTHO = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_NEIGHBOUR_ORTHO = [1]
_BOUNDARY_ORTHO = [
    {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
    {"name": "top", "type": "wall", "startFace": 6, "nFaces": 5},
]


def _make_ortho_mesh(device="cpu", dtype=torch.float64):
    mesh = FvMesh(
        points=torch.tensor(_POINTS_ORTHO, dtype=dtype, device=device),
        faces=[
            torch.tensor(f, dtype=INDEX_DTYPE, device=device)
            for f in _FACES_ORTHO
        ],
        owner=torch.tensor(_OWNER_ORTHO, dtype=INDEX_DTYPE, device=device),
        neighbour=torch.tensor(
            _NEIGHBOUR_ORTHO, dtype=INDEX_DTYPE, device=device
        ),
        boundary=_BOUNDARY_ORTHO,
    )
    mesh.compute_geometry()
    return mesh


# ---------------------------------------------------------------------------
# Non-orthogonal 2-cell mesh (upper cell sheared in x)
# ---------------------------------------------------------------------------
#
# Lower cell: same as orthogonal (z=0..1).
# Upper cell: top vertices shifted by +0.5 in x, creating a non-zero
# angle between the internal face normal (0,0,1) and the cell-centre
# displacement vector d.
#
#   Cell 1 (sheared):  z=1..2, x shifted
#       11'------10'      11'=(0.5,1,2)  10'=(1.5,1,2)
#       /|       /|       9'=(1.5,0,2)   8'=(0.5,0,2)
#      / |      / |
#     8'------9'  |
#     |  7-----|--6
#     | /      | /
#     |/       |/
#     4--------5
#     |        |
#     3--------2
#    /|       /|
#   / |      / |
#  0--------1  |
#
# Internal face (z=1): (4,5,6,7), owner=0, neighbour=1

_POINTS_SKEW = [
    [0.0, 0.0, 0.0],  # 0
    [1.0, 0.0, 0.0],  # 1
    [1.0, 1.0, 0.0],  # 2
    [0.0, 1.0, 0.0],  # 3
    [0.0, 0.0, 1.0],  # 4
    [1.0, 0.0, 1.0],  # 5
    [1.0, 1.0, 1.0],  # 6
    [0.0, 1.0, 1.0],  # 7
    [0.5, 0.0, 2.0],  # 8  (shifted +0.5 in x)
    [1.5, 0.0, 2.0],  # 9  (shifted +0.5 in x)
    [1.5, 1.0, 2.0],  # 10 (shifted +0.5 in x)
    [0.5, 1.0, 2.0],  # 11 (shifted +0.5 in x)
]

_FACES_SKEW = [
    [4, 5, 6, 7],     # 0: internal (unchanged)
    [0, 3, 2, 1],     # 1: bottom of cell 0
    [0, 1, 5, 4],     # 2: front of cell 0
    [3, 7, 6, 2],     # 3: back of cell 0
    [0, 4, 7, 3],     # 4: left of cell 0
    [1, 2, 6, 5],     # 5: right of cell 0
    [8, 9, 10, 11],   # 6: top of cell 1
    [4, 5, 9, 8],     # 7: front of cell 1
    [7, 11, 10, 6],   # 8: back of cell 1
    [4, 8, 11, 7],    # 9: left of cell 1
    [5, 6, 10, 9],    # 10: right of cell 1
]

_OWNER_SKEW = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_NEIGHBOUR_SKEW = [1]
_BOUNDARY_SKEW = [
    {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
    {"name": "top", "type": "wall", "startFace": 6, "nFaces": 5},
]


def _make_skew_mesh(device="cpu", dtype=torch.float64):
    mesh = FvMesh(
        points=torch.tensor(_POINTS_SKEW, dtype=dtype, device=device),
        faces=[
            torch.tensor(f, dtype=INDEX_DTYPE, device=device)
            for f in _FACES_SKEW
        ],
        owner=torch.tensor(_OWNER_SKEW, dtype=INDEX_DTYPE, device=device),
        neighbour=torch.tensor(
            _NEIGHBOUR_SKEW, dtype=INDEX_DTYPE, device=device
        ),
        boundary=_BOUNDARY_SKEW,
    )
    mesh.compute_geometry()
    return mesh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ortho_mesh():
    return _make_ortho_mesh()


@pytest.fixture
def skew_mesh():
    return _make_skew_mesh()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestSnGradRegistry:
    def test_registry_populated(self):
        assert "uncorrected" in _SN_GRAD_REGISTRY
        assert "corrected" in _SN_GRAD_REGISTRY
        assert "limited" in _SN_GRAD_REGISTRY

    def test_from_name_uncorrected(self, ortho_mesh):
        scheme = sn_grad_from_name("uncorrected", ortho_mesh)
        assert isinstance(scheme, UncorrectedSnGrad)

    def test_from_name_corrected(self, ortho_mesh):
        scheme = sn_grad_from_name("corrected", ortho_mesh)
        assert isinstance(scheme, CorrectedSnGrad)

    def test_from_name_limited(self, ortho_mesh):
        scheme = sn_grad_from_name("limited", ortho_mesh, k_coeff=0.3)
        assert isinstance(scheme, LimitedSnGrad)
        assert scheme.k_coeff == 0.3

    def test_from_name_unknown_raises(self, ortho_mesh):
        with pytest.raises(ValueError, match="Unknown snGrad scheme"):
            sn_grad_from_name("bogus", ortho_mesh)

    def test_from_name_default_limited(self, ortho_mesh):
        scheme = sn_grad_from_name("limited", ortho_mesh)
        assert isinstance(scheme, LimitedSnGrad)
        assert scheme.k_coeff == 0.5


# ---------------------------------------------------------------------------
# Correction vectors (geometry)
# ---------------------------------------------------------------------------


class TestCorrectionVectors:
    def test_ortho_correction_zero(self, ortho_mesh):
        """For a perfectly orthogonal mesh, correction vectors are zero."""
        k = _compute_correction_vectors(ortho_mesh)
        assert k.shape == (1, 3)
        assert torch.allclose(k, torch.zeros_like(k), atol=1e-12)

    def test_skew_correction_nonzero(self, skew_mesh):
        """For a non-orthogonal mesh, correction vectors are nonzero."""
        k = _compute_correction_vectors(skew_mesh)
        assert k.shape == (1, 3)
        assert k.norm() > 1e-6

    def test_skew_correction_perpendicular_to_d(self, skew_mesh):
        """Correction vector should be perpendicular to d = x_N - x_P.

        k = n_hat - d_hat * (d_hat . n_hat), so k . d_hat = 0.
        """
        k = _compute_correction_vectors(skew_mesh)
        cc = skew_mesh.cell_centres
        d = cc[1] - cc[0]
        d_hat = d / d.norm()
        dot = (k[0] * d_hat).sum()
        assert abs(dot.item()) < 1e-12


# ---------------------------------------------------------------------------
# UncorrectedSnGrad — orthogonal mesh
# ---------------------------------------------------------------------------


class TestUncorrectedSnGrad:
    def test_returns_tensor(self, ortho_mesh):
        scheme = UncorrectedSnGrad(ortho_mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = scheme.sn_grad(phi)
        assert isinstance(result, torch.Tensor)

    def test_shape(self, ortho_mesh):
        scheme = UncorrectedSnGrad(ortho_mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = scheme.sn_grad(phi)
        assert result.shape == (ortho_mesh.n_faces,)

    def test_constant_field_zero(self, ortho_mesh):
        """snGrad of a constant field should be zero."""
        scheme = UncorrectedSnGrad(ortho_mesh)
        phi = torch.tensor([5.0, 5.0], dtype=torch.float64)
        result = scheme.sn_grad(phi)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-12)

    def test_boundary_faces_zero(self, ortho_mesh):
        """Boundary face snGrad should be zero (handled by BCs)."""
        scheme = UncorrectedSnGrad(ortho_mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = scheme.sn_grad(phi)
        n_internal = ortho_mesh.n_internal_faces
        assert torch.allclose(
            result[n_internal:], torch.zeros_like(result[n_internal:])
        )

    def test_linear_field_value(self, ortho_mesh):
        """For phi = z-coordinate, snGrad at internal face should be 1.0.

        Cell 0 centre z=0.5, cell 1 centre z=1.5.
        delta = 1/(d . n_hat) = 1.0 (orthogonal).
        snGrad = 1.0 * (1.5 - 0.5) = 1.0.
        """
        scheme = UncorrectedSnGrad(ortho_mesh)
        cc = ortho_mesh.cell_centres
        phi = cc[:, 2]  # z-coordinate
        result = scheme.sn_grad(phi)
        assert abs(result[0].item() - 1.0) < 1e-10

    def test_callable_interface(self, ortho_mesh):
        scheme = UncorrectedSnGrad(ortho_mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(scheme(phi), scheme.sn_grad(phi))

    def test_repr(self, ortho_mesh):
        scheme = UncorrectedSnGrad(ortho_mesh)
        assert "UncorrectedSnGrad" in repr(scheme)


# ---------------------------------------------------------------------------
# CorrectedSnGrad — orthogonal mesh
# ---------------------------------------------------------------------------


class TestCorrectedSnGradOrtho:
    def test_equals_uncorrected_for_ortho(self, ortho_mesh):
        """For an orthogonal mesh, correction is zero → same as uncorrected."""
        unc = UncorrectedSnGrad(ortho_mesh)
        cor = CorrectedSnGrad(ortho_mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(
            cor.sn_grad(phi), unc.sn_grad(phi), atol=1e-10
        )

    def test_constant_field_zero(self, ortho_mesh):
        scheme = CorrectedSnGrad(ortho_mesh)
        phi = torch.tensor([7.0, 7.0], dtype=torch.float64)
        result = scheme.sn_grad(phi)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-10)

    def test_boundary_faces_zero(self, ortho_mesh):
        scheme = CorrectedSnGrad(ortho_mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = scheme.sn_grad(phi)
        n_internal = ortho_mesh.n_internal_faces
        assert torch.allclose(
            result[n_internal:], torch.zeros_like(result[n_internal:])
        )

    def test_repr(self, ortho_mesh):
        scheme = CorrectedSnGrad(ortho_mesh)
        assert "CorrectedSnGrad" in repr(scheme)


# ---------------------------------------------------------------------------
# CorrectedSnGrad — non-orthogonal mesh
# ---------------------------------------------------------------------------


class TestCorrectedSnGradSkew:
    def test_differs_from_uncorrected(self, skew_mesh):
        """For a non-orthogonal mesh, corrected != uncorrected."""
        unc = UncorrectedSnGrad(skew_mesh)
        cor = CorrectedSnGrad(skew_mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert not torch.allclose(
            cor.sn_grad(phi), unc.sn_grad(phi), atol=1e-6
        )

    def test_constant_field_zero(self, skew_mesh):
        """Correction term should vanish for constant fields."""
        scheme = CorrectedSnGrad(skew_mesh)
        phi = torch.tensor([4.0, 4.0], dtype=torch.float64)
        result = scheme.sn_grad(phi)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-10)


# ---------------------------------------------------------------------------
# LimitedSnGrad
# ---------------------------------------------------------------------------


class TestLimitedSnGrad:
    def test_k_coeff_zero_equals_uncorrected(self, ortho_mesh):
        """k_coeff=0 should reproduce uncorrected scheme."""
        unc = UncorrectedSnGrad(ortho_mesh)
        lim = LimitedSnGrad(ortho_mesh, k_coeff=0.0)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(
            lim.sn_grad(phi), unc.sn_grad(phi), atol=1e-12
        )

    def test_k_coeff_one_equals_corrected_ortho(self, ortho_mesh):
        """k_coeff=1 should reproduce corrected scheme (ortho)."""
        cor = CorrectedSnGrad(ortho_mesh)
        lim = LimitedSnGrad(ortho_mesh, k_coeff=1.0)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(
            lim.sn_grad(phi), cor.sn_grad(phi), atol=1e-10
        )

    def test_k_coeff_one_equals_corrected_skew(self, skew_mesh):
        """k_coeff=1 should reproduce corrected scheme (skew)."""
        cor = CorrectedSnGrad(skew_mesh)
        lim = LimitedSnGrad(skew_mesh, k_coeff=1.0)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(
            lim.sn_grad(phi), cor.sn_grad(phi), atol=1e-10
        )

    def test_k_coeff_intermediate(self, skew_mesh):
        """Intermediate k_coeff should interpolate between uncorrected and corrected."""
        unc = UncorrectedSnGrad(skew_mesh)
        cor = CorrectedSnGrad(skew_mesh)
        lim = LimitedSnGrad(skew_mesh, k_coeff=0.5)
        phi = torch.tensor([1.0, 3.0], dtype=torch.float64)

        result_unc = unc.sn_grad(phi)
        result_cor = cor.sn_grad(phi)
        result_lim = lim.sn_grad(phi)

        expected = result_unc + 0.5 * (result_cor - result_unc)
        assert torch.allclose(result_lim, expected, atol=1e-10)

    def test_k_coeff_out_of_range_raises(self, ortho_mesh):
        with pytest.raises(ValueError, match="k_coeff must be in"):
            LimitedSnGrad(ortho_mesh, k_coeff=-0.1)
        with pytest.raises(ValueError, match="k_coeff must be in"):
            LimitedSnGrad(ortho_mesh, k_coeff=1.1)

    def test_default_k_coeff(self, ortho_mesh):
        scheme = LimitedSnGrad(ortho_mesh)
        assert scheme.k_coeff == 0.5

    def test_repr(self, ortho_mesh):
        scheme = LimitedSnGrad(ortho_mesh, k_coeff=0.3)
        assert "LimitedSnGrad" in repr(scheme)
        assert "0.3" in repr(scheme)


# ---------------------------------------------------------------------------
# Scaling and consistency
# ---------------------------------------------------------------------------


class TestScalingAndConsistency:
    def test_sn_grad_scales_with_phi(self, ortho_mesh):
        """Doubling phi should double snGrad."""
        scheme = UncorrectedSnGrad(ortho_mesh)
        phi1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        phi2 = 2.0 * phi1
        assert torch.allclose(
            scheme.sn_grad(phi2), 2.0 * scheme.sn_grad(phi1), atol=1e-10
        )

    def test_sn_grad_antisymmetric(self, ortho_mesh):
        """Swapping phi_P and phi_N should negate snGrad."""
        scheme = UncorrectedSnGrad(ortho_mesh)
        phi1 = torch.tensor([1.0, 3.0], dtype=torch.float64)
        phi2 = torch.tensor([3.0, 1.0], dtype=torch.float64)
        r1 = scheme.sn_grad(phi1)
        r2 = scheme.sn_grad(phi2)
        # Internal face: r1[0] = delta * (3-1) = 2*delta, r2[0] = delta * (1-3) = -2*delta
        assert torch.allclose(r1[:1], -r2[:1], atol=1e-12)

    def test_finite_values_large_gradient(self, ortho_mesh):
        """Large gradient should still produce finite snGrad."""
        scheme = CorrectedSnGrad(ortho_mesh)
        phi = torch.tensor([0.0, 1e6], dtype=torch.float64)
        result = scheme.sn_grad(phi)
        assert torch.isfinite(result).all()

    def test_zero_phi_zero_sngrad(self, ortho_mesh):
        for SchemeCls in [UncorrectedSnGrad, CorrectedSnGrad, LimitedSnGrad]:
            scheme = SchemeCls(ortho_mesh)
            phi = torch.tensor([0.0, 0.0], dtype=torch.float64)
            result = scheme.sn_grad(phi)
            assert torch.allclose(result, torch.zeros_like(result), atol=1e-12)
