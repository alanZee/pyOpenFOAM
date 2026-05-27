"""
Unit tests for enhanced radiation and interface reconstruction models.
"""

from __future__ import annotations

import torch
import pytest

from pyfoam.core.device import get_device, get_default_dtype


# ---------------------------------------------------------------------------
# P1RadiationEnhanced tests
# ---------------------------------------------------------------------------


class _SimpleMesh:
    """Minimal mesh stub for testing radiation and reconstruction."""

    def __init__(self, n_cells: int = 4, n_internal: int = 3):
        self.n_cells = n_cells
        self.n_internal_faces = n_internal
        self.n_faces = n_internal + 2  # +2 boundary faces

        # 1D chain: cell[i] -- face[i] -- cell[i+1]
        owner_list = list(range(n_internal))
        neighbour_list = list(range(1, n_internal + 1))
        # Boundary faces
        owner_list.append(0)
        owner_list.append(n_cells - 1)

        self.owner = torch.tensor(owner_list, dtype=torch.long)
        self.neighbour = torch.tensor(neighbour_list, dtype=torch.long)

        dx = 1.0 / n_cells
        self.cell_volumes = torch.full((n_cells,), dx)
        self.cell_centres = torch.tensor(
            [((i + 0.5) * dx, 0.5 * dx, 0.5 * dx) for i in range(n_cells)],
            dtype=torch.float64,
        )

        # Face areas: internal faces have area vector (1, 0, 0)
        # boundary faces also (1, 0, 0)
        face_areas_list = [(1.0, 0.0, 0.0)] * self.n_faces
        self.face_areas = torch.tensor(face_areas_list, dtype=torch.float64)

        self.face_centres = torch.tensor(
            [((i + 1) * dx, 0.5 * dx, 0.5 * dx) for i in range(n_internal)]
            + [(0.0, 0.5 * dx, 0.5 * dx), (1.0, 0.5 * dx, 0.5 * dx)],
            dtype=torch.float64,
        )


class TestP1RadiationEnhanced:
    """Unit tests for P1RadiationEnhanced."""

    def test_import(self):
        from pyfoam.models.radiation_2 import P1RadiationEnhanced
        assert P1RadiationEnhanced is not None

    def test_construction(self):
        from pyfoam.models.radiation_2 import P1RadiationEnhanced
        mesh = _SimpleMesh()
        rad = P1RadiationEnhanced(mesh, absorption_coeff=0.5)
        assert rad.absorption_coeff == 0.5
        assert rad.scattering_coeff == 0.0

    def test_sh_returns_correct_shape(self):
        from pyfoam.models.radiation_2 import P1RadiationEnhanced
        mesh = _SimpleMesh(n_cells=10, n_internal=9)
        rad = P1RadiationEnhanced(mesh, absorption_coeff=0.1)
        T = torch.full((10,), 500.0, dtype=torch.float64)
        S = rad.Sh(T)
        assert S.shape == (10,)
        assert torch.isfinite(S).all()

    def test_high_T_emits_radiation(self):
        """Non-uniform T field produces non-zero radiation source."""
        from pyfoam.models.radiation_2 import P1RadiationEnhanced
        mesh = _SimpleMesh(n_cells=4)
        rad = P1RadiationEnhanced(mesh, absorption_coeff=0.5, T_ref=300.0)
        # Non-uniform T: creates gradient so G != 4σT⁴ locally
        T = torch.tensor([500.0, 800.0, 1200.0, 1000.0], dtype=torch.float64)
        S = rad.Sh(T)
        # Source should be non-zero for non-uniform temperature
        assert S.abs().sum() > 0, (
            f"Expected non-zero radiation source for non-uniform T, got {S}"
        )

    def test_low_T_uniform_gives_zero_source(self):
        """Uniform T gives S=0 (G = 4σT⁴ equilibrium)."""
        from pyfoam.models.radiation_2 import P1RadiationEnhanced
        mesh = _SimpleMesh(n_cells=4)
        rad = P1RadiationEnhanced(mesh, absorption_coeff=0.5, T_ref=300.0)
        T = torch.full((4,), 500.0, dtype=torch.float64)
        S = rad.Sh(T)
        assert torch.allclose(
            S, torch.zeros(4, dtype=torch.float64), atol=1e-20
        ), f"Uniform T should give S≈0, got {S}"

    def test_correct_resets_G(self):
        from pyfoam.models.radiation_2 import P1RadiationEnhanced
        mesh = _SimpleMesh()
        rad = P1RadiationEnhanced(mesh)
        T = torch.full((4,), 500.0, dtype=torch.float64)
        rad.Sh(T)
        assert rad.G is not None
        rad.correct()
        assert rad.G is None

    def test_repr(self):
        from pyfoam.models.radiation_2 import P1RadiationEnhanced
        mesh = _SimpleMesh()
        rad = P1RadiationEnhanced(mesh, absorption_coeff=0.3)
        r = repr(rad)
        assert "P1RadiationEnhanced" in r
        assert "0.3" in r


# ---------------------------------------------------------------------------
# PLICReconstruction tests
# ---------------------------------------------------------------------------


class TestPLICReconstruction:
    """Unit tests for PLICReconstruction."""

    def test_import(self):
        from pyfoam.multiphase.interface_reconstruction import (
            InterfaceReconstruction,
            PLICReconstruction,
        )
        assert InterfaceReconstruction is not None
        assert PLICReconstruction is not None

    def test_construction(self):
        from pyfoam.multiphase.interface_reconstruction import PLICReconstruction
        mesh = _SimpleMesh(n_cells=10, n_internal=9)
        plic = PLICReconstruction(mesh, method="gauss")
        assert plic._method == "gauss"

    def test_invalid_method_raises(self):
        from pyfoam.multiphase.interface_reconstruction import PLICReconstruction
        mesh = _SimpleMesh()
        with pytest.raises(ValueError, match="Unknown method"):
            PLICReconstruction(mesh, method="invalid")

    def test_reconstruct_returns_correct_shapes(self):
        from pyfoam.multiphase.interface_reconstruction import PLICReconstruction
        mesh = _SimpleMesh(n_cells=10, n_internal=9)
        plic = PLICReconstruction(mesh)
        alpha = torch.tensor(
            [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.3, 0.7, 0.0, 1.0],
            dtype=torch.float64,
        )
        normals, d = plic.reconstruct(alpha)
        assert normals.shape == (10, 3)
        assert d.shape == (10,)

    def test_reconstruct_all_zero_alpha(self):
        """All-zero alpha produces zero normals and d."""
        from pyfoam.multiphase.interface_reconstruction import PLICReconstruction
        mesh = _SimpleMesh(n_cells=4)
        plic = PLICReconstruction(mesh)
        alpha = torch.zeros(4, dtype=torch.float64)
        normals, d = plic.reconstruct(alpha)
        assert (normals == 0).all()
        assert (d == 0).all()

    def test_reconstruct_all_one_alpha(self):
        """All-one alpha produces zero normals and d (no interface)."""
        from pyfoam.multiphase.interface_reconstruction import PLICReconstruction
        mesh = _SimpleMesh(n_cells=4)
        plic = PLICReconstruction(mesh)
        alpha = torch.ones(4, dtype=torch.float64)
        normals, d = plic.reconstruct(alpha)
        assert (normals == 0).all()
        assert (d == 0).all()

    def test_interface_cells_detection(self):
        from pyfoam.multiphase.interface_reconstruction import PLICReconstruction
        mesh = _SimpleMesh(n_cells=5)
        plic = PLICReconstruction(mesh)
        alpha = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
        mask = plic.compute_interface_cells(alpha, alpha_tol=0.01)
        expected = torch.tensor([False, True, True, True, False])
        assert (mask == expected).all()

    def test_reconstruct_myc_method(self):
        from pyfoam.multiphase.interface_reconstruction import PLICReconstruction
        mesh = _SimpleMesh(n_cells=10, n_internal=9)
        plic = PLICReconstruction(mesh, method="myc")
        alpha = torch.tensor(
            [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.3, 0.7, 0.0, 1.0],
            dtype=torch.float64,
        )
        normals, d = plic.reconstruct(alpha)
        assert normals.shape == (10, 3)
        assert d.shape == (10,)
        assert torch.isfinite(normals).all()
        assert torch.isfinite(d).all()

    def test_repr(self):
        from pyfoam.multiphase.interface_reconstruction import PLICReconstruction
        mesh = _SimpleMesh(n_cells=8)
        plic = PLICReconstruction(mesh)
        r = repr(plic)
        assert "PLICReconstruction" in r
