"""Tests for interpolation schemes — linear, upwind, linearUpwind, limitedLinear."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.backend import gather
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.discretisation.interpolation import InterpolationScheme, LinearInterpolation
from pyfoam.discretisation.schemes.upwind import UpwindInterpolation
from pyfoam.discretisation.schemes.linear_upwind import LinearUpwindInterpolation
from pyfoam.discretisation.schemes.limited_linear import LimitedLinearInterpolation
from pyfoam.discretisation.schemes.quick import QuickInterpolation
from pyfoam.discretisation.weights import compute_centre_weights, compute_upwind_weights


# ---------------------------------------------------------------------------
# Mesh fixture
# ---------------------------------------------------------------------------

_POINTS = [
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

_FACES = [
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

_OWNER = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_NEIGHBOUR = [1]
_BOUNDARY = [
    {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
    {"name": "top", "type": "wall", "startFace": 6, "nFaces": 5},
]


@pytest.fixture
def mesh():
    m = FvMesh(
        points=torch.tensor(_POINTS, dtype=torch.float64),
        faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in _FACES],
        owner=torch.tensor(_OWNER, dtype=INDEX_DTYPE),
        neighbour=torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE),
        boundary=_BOUNDARY,
    )
    m.compute_geometry()
    return m


@pytest.fixture
def face_flux(mesh):
    """Face flux with positive flow through internal face."""
    U = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    return (mesh.face_areas * U).sum(dim=1)


# ---------------------------------------------------------------------------
# Weights tests
# ---------------------------------------------------------------------------


class TestWeights:
    def test_centre_weights_shape(self, mesh):
        w = compute_centre_weights(
            mesh.cell_centres, mesh.face_centres,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces, mesh.n_faces,
        )
        assert w.shape == (mesh.n_faces,)

    def test_centre_weights_range(self, mesh):
        w = compute_centre_weights(
            mesh.cell_centres, mesh.face_centres,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces, mesh.n_faces,
        )
        assert (w[:mesh.n_internal_faces] >= 0).all()
        assert (w[:mesh.n_internal_faces] <= 1).all()

    def test_centre_weights_boundary_one(self, mesh):
        w = compute_centre_weights(
            mesh.cell_centres, mesh.face_centres,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces, mesh.n_faces,
        )
        assert torch.allclose(
            w[mesh.n_internal_faces:],
            torch.ones(mesh.n_faces - mesh.n_internal_faces, dtype=torch.float64),
        )

    def test_centre_weights_equal_cells(self, mesh):
        w = compute_centre_weights(
            mesh.cell_centres, mesh.face_centres,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces, mesh.n_faces,
        )
        assert abs(w[0].item() - 0.5) < 0.1

    def test_upwind_weights_shape(self, face_flux, mesh):
        w_own, w_neigh = compute_upwind_weights(
            face_flux, mesh.n_internal_faces, mesh.n_faces,
        )
        assert w_own.shape == (mesh.n_faces,)
        assert w_neigh.shape == (mesh.n_faces,)

    def test_upwind_weights_binary(self, face_flux, mesh):
        w_own, w_neigh = compute_upwind_weights(
            face_flux, mesh.n_internal_faces, mesh.n_faces,
        )
        assert ((w_own == 0) | (w_own == 1)).all()
        assert ((w_neigh == 0) | (w_neigh == 1)).all()

    def test_upwind_weights_complement(self, face_flux, mesh):
        w_own, w_neigh = compute_upwind_weights(
            face_flux, mesh.n_internal_faces, mesh.n_faces,
        )
        assert torch.allclose(
            w_own[:mesh.n_internal_faces] + w_neigh[:mesh.n_internal_faces],
            torch.ones(mesh.n_internal_faces, dtype=torch.float64),
        )


# ---------------------------------------------------------------------------
# LinearInterpolation tests
# ---------------------------------------------------------------------------


class TestLinearInterpolation:
    def test_is_scheme_subclass(self, mesh):
        assert issubclass(LinearInterpolation, InterpolationScheme)

    def test_constant_field(self, mesh):
        scheme = LinearInterpolation(mesh)
        phi = torch.tensor([5.0, 5.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert torch.allclose(
            face_vals[:mesh.n_internal_faces],
            torch.tensor([5.0], dtype=torch.float64),
        )

    def test_weighted_average(self, mesh):
        scheme = LinearInterpolation(mesh)
        phi = torch.tensor([0.0, 10.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        # For symmetric mesh, w=0.5, so face value = 5.0
        assert abs(face_vals[0].item() - 5.0) < 1e-10

    def test_boundary_uses_owner(self, mesh):
        scheme = LinearInterpolation(mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            assert abs(face_vals[i].item() - 3.0) < 1e-10
        for i in range(6, 11):
            assert abs(face_vals[i].item() - 7.0) < 1e-10

    def test_requires_1d(self, mesh):
        scheme = LinearInterpolation(mesh)
        phi = torch.zeros((mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_face_value_between_cell_values(self, mesh):
        scheme = LinearInterpolation(mesh)
        phi = torch.tensor([2.0, 8.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert 2.0 <= fv <= 8.0

    def test_callable(self, mesh):
        scheme = LinearInterpolation(mesh)
        phi = torch.full((mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme(phi)
        assert face_vals.shape == (mesh.n_faces,)

    def test_mesh_property(self, mesh):
        scheme = LinearInterpolation(mesh)
        assert scheme.mesh is mesh

    def test_repr(self, mesh):
        scheme = LinearInterpolation(mesh)
        assert "LinearInterpolation" in repr(scheme)


# ---------------------------------------------------------------------------
# UpwindInterpolation tests
# ---------------------------------------------------------------------------


class TestUpwindInterpolation:
    def test_is_scheme_subclass(self, mesh):
        assert issubclass(UpwindInterpolation, InterpolationScheme)

    def test_positive_flux_uses_owner(self, mesh, face_flux):
        scheme = UpwindInterpolation(mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux)
        if face_flux[0] >= 0:
            assert abs(face_vals[0].item() - 3.0) < 1e-10
        else:
            assert abs(face_vals[0].item() - 7.0) < 1e-10

    def test_negative_flux_uses_neighbour(self, mesh):
        scheme = UpwindInterpolation(mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_flux = torch.zeros(mesh.n_faces, dtype=torch.float64)
        face_flux[0] = -1.0
        face_vals = scheme.interpolate(phi, face_flux)
        assert abs(face_vals[0].item() - 7.0) < 1e-10

    def test_no_flux_defaults_to_owner(self, mesh):
        scheme = UpwindInterpolation(mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux=None)
        assert face_vals[0] == 10.0
        for i in range(1, 6):
            assert face_vals[i] == 10.0
        for i in range(6, 11):
            assert face_vals[i] == 20.0

    def test_boundary_uses_owner(self, mesh, face_flux):
        scheme = UpwindInterpolation(mesh)
        phi = torch.tensor([4.0, 9.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux)
        for i in range(1, 6):
            assert abs(face_vals[i].item() - 4.0) < 1e-10

    def test_upwind_bounded(self, mesh, face_flux):
        scheme = UpwindInterpolation(mesh)
        phi = torch.tensor([2.0, 8.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux)
        fv = face_vals[0].item()
        assert fv in [2.0, 8.0]

    def test_requires_1d(self, mesh):
        scheme = UpwindInterpolation(mesh)
        phi = torch.zeros((mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)


# ---------------------------------------------------------------------------
# LinearUpwindInterpolation tests
# ---------------------------------------------------------------------------


class TestLinearUpwindInterpolation:
    def test_is_scheme_subclass(self, mesh):
        assert issubclass(LinearUpwindInterpolation, InterpolationScheme)

    def test_constant_field(self, mesh):
        scheme = LinearUpwindInterpolation(mesh)
        phi = torch.full((mesh.n_cells,), 7.0, dtype=torch.float64)
        flux = torch.ones(mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((mesh.n_faces,), 7.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, mesh):
        scheme = LinearUpwindInterpolation(mesh)
        phi = torch.zeros(mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_requires_1d(self, mesh):
        scheme = LinearUpwindInterpolation(mesh)
        phi = torch.zeros((mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, flux)

    def test_output_shape(self, mesh):
        scheme = LinearUpwindInterpolation(mesh)
        phi = torch.zeros(mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert face_vals.shape == (mesh.n_faces,)

    def test_boundary_faces_finite(self, mesh):
        scheme = LinearUpwindInterpolation(mesh)
        phi = torch.tensor([0.5, 1.5], dtype=torch.float64)
        flux = torch.ones(mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()

    def test_output_is_finite(self, mesh):
        scheme = LinearUpwindInterpolation(mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.tensor(
            [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6, -0.3, 0.1, 0.7, -0.2],
            dtype=torch.float64,
        )
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()


# ---------------------------------------------------------------------------
# LimitedLinearInterpolation tests
# ---------------------------------------------------------------------------


class TestLimitedLinearInterpolation:
    def test_is_scheme_subclass(self, mesh):
        assert issubclass(LimitedLinearInterpolation, InterpolationScheme)

    def test_default_limiter(self, mesh):
        scheme = LimitedLinearInterpolation(mesh)
        assert scheme.limiter_name == "vanLeer"

    def test_custom_limiter(self, mesh):
        scheme = LimitedLinearInterpolation(mesh, limiter="minmod")
        assert scheme.limiter_name == "minmod"

    def test_invalid_limiter(self, mesh):
        with pytest.raises(ValueError, match="Unknown limiter"):
            LimitedLinearInterpolation(mesh, limiter="invalid")

    def test_basic_interpolation(self, mesh, face_flux):
        scheme = LimitedLinearInterpolation(mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux)
        assert face_vals.shape == (mesh.n_faces,)
        assert torch.isfinite(face_vals).all()

    def test_limiter_bounds(self, mesh, face_flux):
        scheme = LimitedLinearInterpolation(mesh, limiter="vanLeer")
        phi = torch.tensor([0.0, 10.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux)
        fv = face_vals[0].item()
        assert 0.0 <= fv <= 10.0 or abs(fv) < 1e-10

    def test_all_limiters(self, mesh, face_flux):
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        for limiter_name in ["vanLeer", "minmod", "superbee"]:
            scheme = LimitedLinearInterpolation(mesh, limiter=limiter_name)
            face_vals = scheme.interpolate(phi, face_flux)
            assert torch.isfinite(face_vals).all(), f"Failed for limiter {limiter_name}"


# ---------------------------------------------------------------------------
# QuickInterpolation tests
# ---------------------------------------------------------------------------


class TestQuickInterpolation:
    def test_is_scheme_subclass(self, mesh):
        assert issubclass(QuickInterpolation, InterpolationScheme)

    def test_constant_field(self, mesh):
        scheme = QuickInterpolation(mesh)
        phi = torch.full((mesh.n_cells,), 3.0, dtype=torch.float64)
        flux = torch.ones(mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((mesh.n_faces,), 3.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, mesh):
        scheme = QuickInterpolation(mesh)
        phi = torch.zeros(mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_requires_1d(self, mesh):
        scheme = QuickInterpolation(mesh)
        phi = torch.zeros((mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, flux)

    def test_output_shape(self, mesh):
        scheme = QuickInterpolation(mesh)
        phi = torch.zeros(mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert face_vals.shape == (mesh.n_faces,)

    def test_fallback_to_linear_for_2cell(self, mesh):
        """For 2-cell mesh, QUICK should fall back to linear."""
        scheme = QuickInterpolation(mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        linear_scheme = LinearInterpolation(mesh)
        linear_face = linear_scheme.interpolate(phi)
        torch.testing.assert_close(face_vals, linear_face)

    def test_boundary_faces_are_linear(self, mesh):
        scheme = QuickInterpolation(mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        for i in range(1, 6):
            torch.testing.assert_close(
                face_vals[i],
                torch.tensor(10.0, dtype=torch.float64),
            )
        for i in range(6, 11):
            torch.testing.assert_close(
                face_vals[i],
                torch.tensor(20.0, dtype=torch.float64),
            )


# ---------------------------------------------------------------------------
# Scheme resolution tests
# ---------------------------------------------------------------------------


class TestSchemeResolution:
    def test_resolve_linear(self, mesh):
        from pyfoam.discretisation.operators import _resolve_scheme
        scheme = _resolve_scheme("Gauss linear", mesh=mesh)
        assert isinstance(scheme, LinearInterpolation)

    def test_resolve_upwind(self, mesh):
        from pyfoam.discretisation.operators import _resolve_scheme
        scheme = _resolve_scheme("Gauss upwind", mesh=mesh)
        assert isinstance(scheme, UpwindInterpolation)

    def test_resolve_linearUpwind(self, mesh):
        from pyfoam.discretisation.operators import _resolve_scheme
        scheme = _resolve_scheme("Gauss linearUpwind", mesh=mesh)
        assert isinstance(scheme, LinearUpwindInterpolation)

    def test_resolve_limitedLinear(self, mesh):
        from pyfoam.discretisation.operators import _resolve_scheme
        scheme = _resolve_scheme("Gauss limitedLinear", mesh=mesh, limiter="minmod")
        assert isinstance(scheme, LimitedLinearInterpolation)
        assert scheme.limiter_name == "minmod"

    def test_resolve_without_gauss_prefix(self, mesh):
        from pyfoam.discretisation.operators import _resolve_scheme
        scheme = _resolve_scheme("linear", mesh=mesh)
        assert isinstance(scheme, LinearInterpolation)

    def test_resolve_unknown_raises(self, mesh):
        from pyfoam.discretisation.operators import _resolve_scheme
        with pytest.raises(ValueError, match="Unknown scheme"):
            _resolve_scheme("Gauss unknownScheme", mesh=mesh)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestSchemeIntegration:
    def test_all_schemes_same_for_constant_field(self, mesh):
        phi = torch.full((mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(mesh.n_faces, dtype=torch.float64)

        linear = LinearInterpolation(mesh)(phi)
        upwind = UpwindInterpolation(mesh)(phi, flux)
        lu = LinearUpwindInterpolation(mesh)(phi, flux)
        quick = QuickInterpolation(mesh)(phi, flux)

        expected = torch.full((mesh.n_faces,), 5.0, dtype=torch.float64)
        torch.testing.assert_close(linear, expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(upwind, expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(lu, expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(quick, expected, atol=1e-10, rtol=1e-10)

    def test_schemes_produce_different_results(self, mesh):
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(mesh.n_faces, dtype=torch.float64)

        linear = LinearInterpolation(mesh)(phi)[0]
        upwind = UpwindInterpolation(mesh)(phi, flux)[0]
        assert linear != upwind

    def test_module_level_imports(self):
        from pyfoam.discretisation import (
            InterpolationScheme,
            LinearInterpolation,
            UpwindInterpolation,
            LinearUpwindInterpolation,
            QuickInterpolation,
            compute_centre_weights,
            compute_upwind_weights,
        )
        assert issubclass(LinearInterpolation, InterpolationScheme)
        assert issubclass(UpwindInterpolation, InterpolationScheme)
        assert issubclass(LinearUpwindInterpolation, InterpolationScheme)
        assert issubclass(QuickInterpolation, InterpolationScheme)
