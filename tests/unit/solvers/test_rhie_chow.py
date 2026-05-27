"""Tests for Rhie-Chow interpolation module.

Tests cover:
- compute_HbyA: H / A_p with safe division, shapes, value correctness
- compute_face_flux_HbyA: face flux interpolation, boundary vs internal
- rhie_chow_correction: pressure gradient correction, edge cases
- compute_face_flux: combined HbyA flux + Rhie-Chow correction
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.solvers.rhie_chow import (
    compute_HbyA,
    compute_face_flux_HbyA,
    rhie_chow_correction,
    compute_face_flux,
)

from tests.unit.solvers.conftest_coupled import make_cavity_mesh


# ===========================================================================
# compute_HbyA
# ===========================================================================


class TestComputeHbyA:
    """Tests for HbyA = H / A_p computation."""

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Output shape is (n_cells, 3)."""
        n_cells = 6
        H = torch.randn(n_cells, 3, dtype=CFD_DTYPE)
        A_p = torch.rand(n_cells, dtype=CFD_DTYPE) + 0.1

        HbyA = compute_HbyA(H, A_p)
        assert HbyA.shape == (n_cells, 3)

    def test_single_cell(self):
        """Works for a single cell."""
        H = torch.tensor([[1.0, 2.0, 3.0]], dtype=CFD_DTYPE)
        A_p = torch.tensor([2.0], dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        assert HbyA.shape == (1, 3)

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def test_values_exact(self):
        """HbyA = H / A_p with known values."""
        H = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=CFD_DTYPE)
        A_p = torch.tensor([2.0, 3.0], dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        expected = torch.tensor(
            [[0.5, 1.0, 1.5], [4.0 / 3, 5.0 / 3, 2.0]], dtype=CFD_DTYPE
        )
        assert torch.allclose(HbyA, expected, atol=1e-10)

    def test_uniform_Ap(self):
        """Uniform A_p scales all cells equally."""
        H = torch.randn(4, 3, dtype=CFD_DTYPE)
        A_p = torch.full((4,), 5.0, dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        assert torch.allclose(HbyA, H / 5.0, atol=1e-10)

    def test_identity_Ap(self):
        """A_p = 1 gives HbyA = H."""
        H = torch.randn(3, 3, dtype=CFD_DTYPE)
        A_p = torch.ones(3, dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        assert torch.allclose(HbyA, H, atol=1e-10)

    # ------------------------------------------------------------------
    # Safe division
    # ------------------------------------------------------------------

    def test_near_zero_Ap(self):
        """Near-zero A_p does not produce inf or nan."""
        H = torch.tensor([[1.0, 2.0, 3.0]], dtype=CFD_DTYPE)
        A_p = torch.tensor([1e-40], dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        assert torch.isfinite(HbyA).all()

    def test_zero_Ap(self):
        """Zero A_p does not produce inf or nan."""
        H = torch.tensor([[1.0, 2.0, 3.0]], dtype=CFD_DTYPE)
        A_p = torch.tensor([0.0], dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        assert torch.isfinite(HbyA).all()

    def test_negative_Ap_safe(self):
        """Negative A_p uses abs before clamping."""
        H = torch.tensor([[2.0, 4.0, 6.0]], dtype=CFD_DTYPE)
        A_p = torch.tensor([-2.0], dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        # abs(-2.0) = 2.0, so result = H / 2.0
        expected = torch.tensor([[1.0, 2.0, 3.0]], dtype=CFD_DTYPE)
        assert torch.allclose(HbyA, expected, atol=1e-10)

    def test_mixed_Ap_values(self):
        """Mixed positive and near-zero A_p produces finite results."""
        H = torch.randn(3, 3, dtype=CFD_DTYPE)
        A_p = torch.tensor([1.0, 1e-40, 0.0], dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        assert torch.isfinite(HbyA).all()

    # ------------------------------------------------------------------
    # Dtype / device
    # ------------------------------------------------------------------

    def test_output_dtype_matches_input(self):
        """Output dtype matches input H dtype."""
        H = torch.randn(4, 3, dtype=CFD_DTYPE)
        A_p = torch.rand(4, dtype=CFD_DTYPE) + 0.1

        HbyA = compute_HbyA(H, A_p)
        assert HbyA.dtype == CFD_DTYPE

    def test_zero_H(self):
        """Zero H gives zero HbyA."""
        H = torch.zeros(3, 3, dtype=CFD_DTYPE)
        A_p = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)

        HbyA = compute_HbyA(H, A_p)
        assert torch.allclose(HbyA, torch.zeros_like(HbyA), atol=1e-10)


# ===========================================================================
# compute_face_flux_HbyA
# ===========================================================================


class TestComputeFaceFluxHbyA:
    """Tests for face flux from HbyA via linear interpolation."""

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Output has shape (n_faces,)."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)

        phi = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, mesh.face_weights,
        )
        assert phi.shape == (mesh.n_faces,)

    # ------------------------------------------------------------------
    # Uniform HbyA
    # ------------------------------------------------------------------

    def test_uniform_HbyA_internal_faces(self):
        """Uniform HbyA: internal flux = HbyA · S_f."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        HbyA[:, 0] = 1.0

        phi = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, mesh.face_weights,
        )

        S = mesh.face_areas[:mesh.n_internal_faces]
        expected = (HbyA[0] * S).sum(dim=1)
        assert torch.allclose(phi[:mesh.n_internal_faces], expected, atol=1e-10)

    def test_zero_HbyA_gives_zero_flux(self):
        """Zero HbyA produces zero flux everywhere."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)

        phi = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, mesh.face_weights,
        )
        assert torch.allclose(phi, torch.zeros_like(phi), atol=1e-10)

    # ------------------------------------------------------------------
    # Boundary faces
    # ------------------------------------------------------------------

    def test_boundary_faces_owner_based(self):
        """Boundary flux = HbyA_owner · S_f (no interpolation)."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        # Give cell 0 a known velocity
        HbyA[0] = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)

        phi = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, mesh.face_weights,
        )

        # Check boundary faces owned by cell 0
        n_int = mesh.n_internal_faces
        for f in range(n_int, mesh.n_faces):
            P = mesh.owner[f].item()
            S = mesh.face_areas[f]
            expected = (HbyA[P] * S).sum()
            assert torch.allclose(phi[f], expected, atol=1e-10), (
                f"Boundary face {f}: expected {expected}, got {phi[f]}"
            )

    # ------------------------------------------------------------------
    # Non-uniform HbyA
    # ------------------------------------------------------------------

    def test_nonuniform_HbyA_interpolation(self):
        """Non-uniform HbyA is linearly interpolated on internal faces."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)

        phi = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, mesh.face_weights,
        )

        # Verify first internal face manually
        f = 0
        P = mesh.owner[f].item()
        N = mesh.neighbour[f].item()
        w = mesh.face_weights[f].item() if mesh.face_weights is not None else 0.5
        HbyA_f = w * HbyA[P] + (1.0 - w) * HbyA[N]
        expected = (HbyA_f * mesh.face_areas[f]).sum()
        assert torch.allclose(phi[f], expected, atol=1e-8)

    # ------------------------------------------------------------------
    # Custom weights
    # ------------------------------------------------------------------

    def test_custom_weights_affect_result(self):
        """Non-default face weights change the interpolated flux."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)

        phi_default = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces,
        )

        weights = torch.full((mesh.n_faces,), 0.75, dtype=CFD_DTYPE)
        phi_weighted = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, weights,
        )

        # Internal faces should differ; boundary faces should be the same
        n_int = mesh.n_internal_faces
        assert not torch.allclose(
            phi_default[:n_int], phi_weighted[:n_int], atol=1e-10
        )
        if mesh.n_faces > n_int:
            assert torch.allclose(
                phi_default[n_int:], phi_weighted[n_int:], atol=1e-10
            )

    def test_default_weights_are_half(self):
        """Default weights (None) produce w=0.5 interpolation."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)

        phi_none = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, None,
        )
        phi_half = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces,
            torch.full((mesh.n_faces,), 0.5, dtype=CFD_DTYPE),
        )
        assert torch.allclose(phi_none, phi_half, atol=1e-10)

    # ------------------------------------------------------------------
    # Mesh sizes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("nx,ny", [(2, 2), (3, 3), (4, 4)])
    def test_various_mesh_sizes(self, nx, ny):
        """Output shape is correct for various mesh sizes."""
        mesh = make_cavity_mesh(nx, ny)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)

        phi = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, mesh.face_weights,
        )
        assert phi.shape == (mesh.n_faces,)


# ===========================================================================
# rhie_chow_correction
# ===========================================================================


class TestRhieChowCorrection:
    """Tests for Rhie-Chow pressure correction on faces."""

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Output shape is (n_faces,)."""
        mesh = make_cavity_mesh(2, 2)
        p = torch.randn(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.rand(mesh.n_cells, dtype=CFD_DTYPE) + 0.5

        correction = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        assert correction.shape == (mesh.n_faces,)

    # ------------------------------------------------------------------
    # Zero correction cases
    # ------------------------------------------------------------------

    def test_uniform_pressure_zero_correction(self):
        """Uniform pressure yields zero correction on all internal faces."""
        mesh = make_cavity_mesh(2, 2)
        p = torch.full((mesh.n_cells,), 101325.0, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        correction = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )

        n_int = mesh.n_internal_faces
        assert torch.allclose(
            correction[:n_int], torch.zeros(n_int, dtype=CFD_DTYPE), atol=1e-10
        )

    def test_boundary_correction_is_zero(self):
        """Boundary face correction is always zero (no neighbour cell)."""
        mesh = make_cavity_mesh(2, 2)
        p = mesh.cell_centres[:, 0].clone()  # linear pressure
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        correction = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )

        n_int = mesh.n_internal_faces
        n_faces = mesh.n_faces
        if n_faces > n_int:
            assert torch.allclose(
                correction[n_int:],
                torch.zeros(n_faces - n_int, dtype=CFD_DTYPE),
                atol=1e-14,
            )

    # ------------------------------------------------------------------
    # Non-zero correction
    # ------------------------------------------------------------------

    def test_linear_pressure_nonzero_correction(self):
        """Linear pressure field produces non-zero correction."""
        mesh = make_cavity_mesh(2, 2)
        p = mesh.cell_centres[:, 0].clone()
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        correction = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            cell_volumes=mesh.cell_volumes, face_weights=mesh.face_weights,
        )

        n_int = mesh.n_internal_faces
        assert correction[:n_int].abs().sum() > 0

    def test_correction_sign_with_pressure_gradient(self):
        """Correction sign is consistent with pressure gradient direction.

        For an internal face with owner P and neighbour N:
            correction = (1/A_p)_f * (p_P - p_N) * |S_f| * delta_f
        When p_P > p_N, the correction is positive.
        """
        mesh = make_cavity_mesh(2, 2)
        # Set pressure so cell 0 has higher pressure than cell 1
        p = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        correction = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )

        # First internal face: owner=0, neighbour=1, p_P - p_N = 2.0 > 0
        # All terms positive -> correction > 0
        assert correction[0].item() > 0, (
            f"Expected positive correction, got {correction[0].item()}"
        )

    # ------------------------------------------------------------------
    # A_p influence
    # ------------------------------------------------------------------

    def test_larger_Ap_smaller_correction(self):
        """Larger A_p produces smaller correction magnitude.

        correction ∝ 1/A_p, so doubling A_p halves the correction.
        """
        mesh = make_cavity_mesh(4, 4)
        p = mesh.cell_centres[:, 0].clone()

        A_p_small = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 0.5
        A_p_large = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 2.0

        corr_small = rhie_chow_correction(
            p, A_p_small, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        corr_large = rhie_chow_correction(
            p, A_p_large, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )

        n_int = mesh.n_internal_faces
        # 1/0.5 = 2, 1/2.0 = 0.5 → ratio = 4
        ratio = (corr_small[:n_int].abs().sum() /
                 corr_large[:n_int].abs().sum())
        assert abs(ratio - 4.0) < 0.01, f"Expected ratio 4.0, got {ratio:.3f}"

    def test_Ap_safety_with_near_zero(self):
        """Near-zero A_p does not produce inf or nan in correction."""
        mesh = make_cavity_mesh(2, 2)
        p = mesh.cell_centres[:, 0].clone()
        A_p = torch.tensor([1e-40, 1.0, 1e-40, 1.0], dtype=CFD_DTYPE)

        correction = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        assert torch.isfinite(correction).all()

    # ------------------------------------------------------------------
    # Zero internal faces edge case
    # ------------------------------------------------------------------

    def test_zero_internal_faces(self):
        """Zero internal faces returns zero correction tensor."""
        mesh = make_cavity_mesh(2, 2)
        p = torch.randn(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        correction = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, 0,
            mesh.cell_volumes, mesh.face_weights,
        )
        assert torch.allclose(correction, torch.zeros_like(correction))

    # ------------------------------------------------------------------
    # Scaling with pressure magnitude
    # ------------------------------------------------------------------

    def test_correction_scales_with_pressure(self):
        """Correction scales linearly with pressure magnitude."""
        mesh = make_cavity_mesh(4, 4)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p1 = mesh.cell_centres[:, 0].clone()
        p2 = p1 * 3.0

        corr1 = rhie_chow_correction(
            p1, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        corr2 = rhie_chow_correction(
            p2, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )

        n_int = mesh.n_internal_faces
        assert torch.allclose(corr2[:n_int], corr1[:n_int] * 3.0, atol=1e-8)


# ===========================================================================
# compute_face_flux (combined)
# ===========================================================================


class TestComputeFaceFlux:
    """Tests for the combined face flux with Rhie-Chow interpolation."""

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Output shape is (n_faces,)."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.randn(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.rand(mesh.n_cells, dtype=CFD_DTYPE) + 0.5

        phi = compute_face_flux(
            HbyA, p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        assert phi.shape == (mesh.n_faces,)

    # ------------------------------------------------------------------
    # Decomposition: phi = phi_HbyA + phi_RC
    # ------------------------------------------------------------------

    def test_equals_sum_of_components(self):
        """Combined flux equals HbyA flux plus Rhie-Chow correction."""
        mesh = make_cavity_mesh(4, 4)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = mesh.cell_centres[:, 0].clone()
        A_p = torch.rand(mesh.n_cells, dtype=CFD_DTYPE) + 0.5

        phi_combined = compute_face_flux(
            HbyA, p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )

        phi_hbya = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, mesh.face_weights,
        )
        phi_rc = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )

        assert torch.allclose(phi_combined, phi_hbya + phi_rc, atol=1e-10)

    # ------------------------------------------------------------------
    # Zero fields
    # ------------------------------------------------------------------

    def test_zero_HbyA_zero_pressure_zero_flux(self):
        """Zero HbyA and zero pressure gives zero flux."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        phi = compute_face_flux(
            HbyA, p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        assert torch.allclose(phi, torch.zeros_like(phi), atol=1e-10)

    def test_zero_pressure_equals_hbya_only(self):
        """Zero pressure: combined flux equals HbyA flux."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        phi = compute_face_flux(
            HbyA, p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        phi_hbya = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, mesh.face_weights,
        )
        assert torch.allclose(phi, phi_hbya, atol=1e-10)

    def test_zero_HbyA_equals_correction_only(self):
        """Zero HbyA: combined flux equals Rhie-Chow correction only."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.zeros(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = mesh.cell_centres[:, 0].clone()
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        phi = compute_face_flux(
            HbyA, p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        phi_rc = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        assert torch.allclose(phi, phi_rc, atol=1e-10)

    # ------------------------------------------------------------------
    # Custom weights propagation
    # ------------------------------------------------------------------

    def test_weights_propagated_to_both_components(self):
        """Face weights are consistently used in both components."""
        mesh = make_cavity_mesh(2, 2)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = mesh.cell_centres[:, 0].clone()
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        weights = torch.full((mesh.n_faces,), 0.25, dtype=CFD_DTYPE)

        phi_w = compute_face_flux(
            HbyA, p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, weights,
        )

        # Manually reconstruct with the same weights
        phi_hbya = compute_face_flux_HbyA(
            HbyA, mesh.face_areas, mesh.owner, mesh.neighbour,
            mesh.n_internal_faces, weights,
        )
        phi_rc = rhie_chow_correction(
            p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, weights,
        )
        assert torch.allclose(phi_w, phi_hbya + phi_rc, atol=1e-10)

    # ------------------------------------------------------------------
    # Different A_p magnitudes
    # ------------------------------------------------------------------

    def test_Ap_affects_correction_not_hbya_component(self):
        """Changing A_p affects HbyA flux AND correction (through 1/A_p)."""
        mesh = make_cavity_mesh(4, 4)
        HbyA_base = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = mesh.cell_centres[:, 0].clone()

        # Compute HbyA for different A_p values
        A_p_1 = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)
        A_p_2 = torch.ones(mesh.n_cells, dtype=CFD_DTYPE) * 2.0

        HbyA_1 = compute_HbyA(HbyA_base, A_p_1)
        HbyA_2 = compute_HbyA(HbyA_base, A_p_2)

        phi_1 = compute_face_flux(
            HbyA_1, p, A_p_1, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        phi_2 = compute_face_flux(
            HbyA_2, p, A_p_2, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )

        # Different A_p should produce different fluxes
        assert not torch.allclose(phi_1, phi_2, atol=1e-8)

    # ------------------------------------------------------------------
    # Mesh sizes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("nx,ny", [(2, 2), (3, 3), (4, 4)])
    def test_various_mesh_sizes(self, nx, ny):
        """Combined flux works for various mesh sizes."""
        mesh = make_cavity_mesh(nx, ny)
        HbyA = torch.randn(mesh.n_cells, 3, dtype=CFD_DTYPE)
        p = torch.randn(mesh.n_cells, dtype=CFD_DTYPE)
        A_p = torch.rand(mesh.n_cells, dtype=CFD_DTYPE) + 0.5

        phi = compute_face_flux(
            HbyA, p, A_p, mesh.face_areas, mesh.delta_coefficients,
            mesh.owner, mesh.neighbour, mesh.n_internal_faces,
            mesh.cell_volumes, mesh.face_weights,
        )
        assert phi.shape == (mesh.n_faces,)
        assert torch.isfinite(phi).all()
