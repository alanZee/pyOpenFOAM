"""Tests for incompressible N-phase VOF model.

Tests cover:
- IncompressibleMultiphaseVoF initialisation and validation
- Mixture density and viscosity computation
- Volume fraction constraint and normalisation
- Phase advection on a simple mesh
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


# ---------------------------------------------------------------------------
# Minimal mesh (reused from test_vof.py)
# ---------------------------------------------------------------------------

@dataclass
class SimpleMesh:
    n_cells: int
    n_internal_faces: int
    n_faces: int
    owner: torch.Tensor
    neighbour: torch.Tensor
    face_areas: torch.Tensor
    cell_volumes: torch.Tensor


def make_2d_mesh(n_x=4, n_y=4, dx=1.0, dy=1.0, dz=0.1):
    n_cells = n_x * n_y
    owners = []
    neighbours = []
    for j in range(n_y):
        for i in range(n_x - 1):
            owners.append(j * n_x + i)
            neighbours.append(j * n_x + i + 1)
    for j in range(n_y - 1):
        for i in range(n_x):
            owners.append(j * n_x + i)
            neighbours.append((j + 1) * n_x + i)
    n_internal = len(neighbours)
    for i in range(n_x):
        owners.append(i)
    for i in range(n_x):
        owners.append((n_y - 1) * n_x + i)
    for j in range(n_y):
        owners.append(j * n_x)
    for j in range(n_y):
        owners.append(j * n_x + n_x - 1)
    n_boundary = 2 * n_x + 2 * n_y
    n_faces = n_internal + n_boundary

    owner = torch.tensor(owners, dtype=torch.long)
    neigh = torch.tensor(neighbours, dtype=torch.long)

    fa_list = []
    for _ in range(n_y):
        for _ in range(n_x - 1):
            fa_list.append([dy * dz, 0.0, 0.0])
    for _ in range(n_y - 1):
        for _ in range(n_x):
            fa_list.append([0.0, dx * dz, 0.0])
    for _ in range(n_boundary):
        fa_list.append([0.0, 0.0, 0.0])

    face_areas = torch.tensor(fa_list, dtype=CFD_DTYPE)
    cell_volumes = torch.full((n_cells,), dx * dy * dz, dtype=CFD_DTYPE)

    return SimpleMesh(
        n_cells=n_cells,
        n_internal_faces=n_internal,
        n_faces=n_faces,
        owner=owner,
        neighbour=neigh,
        face_areas=face_areas,
        cell_volumes=cell_volumes,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIncompressibleMultiphaseVoF:
    """Tests for IncompressibleMultiphaseVoF."""

    def test_init_two_phases(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        model = IncompressibleMultiphaseVoF(
            ["water", "air"], [998.0, 1.225], [1.002e-3, 1.8e-5],
        )
        assert model.n_phases == 2
        assert model.phase_names == ["water", "air"]

    def test_init_three_phases(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        model = IncompressibleMultiphaseVoF(
            ["water", "air", "oil"],
            [998.0, 1.225, 850.0],
            [1.002e-3, 1.8e-5, 0.03],
        )
        assert model.n_phases == 3

    def test_init_single_phase_raises(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        with pytest.raises(ValueError, match="at least 2"):
            IncompressibleMultiphaseVoF(["water"], [998.0], [1.002e-3])

    def test_init_mismatched_rho_raises(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        with pytest.raises(ValueError, match="rho length"):
            IncompressibleMultiphaseVoF(
                ["water", "air"], [998.0], [1.002e-3, 1.8e-5],
            )

    def test_init_negative_rho_raises(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        with pytest.raises(ValueError, match="rho"):
            IncompressibleMultiphaseVoF(
                ["water", "air"], [-1.0, 1.225], [1.002e-3, 1.8e-5],
            )

    def test_init_negative_mu_raises(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        with pytest.raises(ValueError, match="mu"):
            IncompressibleMultiphaseVoF(
                ["water", "air"], [998.0, 1.225], [-1.0, 1.8e-5],
            )

    def test_mixture_density_two_phase(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        model = IncompressibleMultiphaseVoF(
            ["water", "air"], [1000.0, 1.0], [1e-3, 1e-5],
        )
        # alpha_water = 0.3 => rho_m = 0.3*1000 + 0.7*1 = 300.7
        alphas = torch.full((5,), 0.3, dtype=CFD_DTYPE).unsqueeze(-1)
        rho_m = model.mixture_density(alphas)
        assert rho_m.shape == (5,)
        assert torch.allclose(rho_m, torch.full((5,), 300.7, dtype=CFD_DTYPE), atol=1e-3)

    def test_mixture_viscosity_two_phase(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        model = IncompressibleMultiphaseVoF(
            ["water", "air"], [1000.0, 1.0], [1e-3, 1e-5],
        )
        alphas = torch.full((5,), 0.5, dtype=CFD_DTYPE).unsqueeze(-1)
        mu_m = model.mixture_viscosity(alphas)
        # mu_m = 0.5*1e-3 + 0.5*1e-5 = 5.05e-4
        expected = 0.5 * 1e-3 + 0.5 * 1e-5
        assert torch.allclose(mu_m, torch.full((5,), expected, dtype=CFD_DTYPE), atol=1e-10)

    def test_mixture_density_three_phase(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        model = IncompressibleMultiphaseVoF(
            ["water", "air", "oil"],
            [1000.0, 1.0, 800.0],
            [1e-3, 1e-5, 0.01],
        )
        # N=3, so 2 independent alphas: alpha_water=0.3, alpha_air=0.2
        # alpha_oil = 1 - 0.3 - 0.2 = 0.5
        alphas = torch.tensor([[0.3, 0.2]], dtype=CFD_DTYPE)
        rho_m = model.mixture_density(alphas)
        expected = 0.3 * 1000.0 + 0.2 * 1.0 + 0.5 * 800.0
        assert torch.allclose(rho_m, torch.tensor([expected], dtype=CFD_DTYPE), atol=1e-3)

    def test_compute_last_alpha(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        model = IncompressibleMultiphaseVoF(
            ["water", "air", "oil"],
            [1000.0, 1.0, 800.0],
            [1e-3, 1e-5, 0.01],
        )
        alphas = torch.tensor([[0.3, 0.2], [0.0, 0.0], [0.6, 0.4]], dtype=CFD_DTYPE)
        alpha_N = model.compute_last_alpha(alphas)
        assert torch.allclose(alpha_N, torch.tensor([0.5, 1.0, 0.0], dtype=CFD_DTYPE), atol=1e-6)

    def test_validate_alphas_clamping(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        model = IncompressibleMultiphaseVoF(
            ["water", "air", "oil"],
            [1000.0, 1.0, 800.0],
            [1e-3, 1e-5, 0.01],
        )
        # Total > 1, should renormalise
        alphas = torch.tensor([[0.6, 0.6]], dtype=CFD_DTYPE)
        fixed = model.validate_alphas(alphas)
        total = fixed.sum(dim=-1) + model.compute_last_alpha(fixed)
        assert torch.allclose(total, torch.ones(1, dtype=CFD_DTYPE), atol=1e-5)

    def test_mixture_kinematic_viscosity(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        model = IncompressibleMultiphaseVoF(
            ["water", "air"], [1000.0, 1.0], [1e-3, 1e-5],
        )
        alphas = torch.full((3,), 0.4, dtype=CFD_DTYPE).unsqueeze(-1)
        nu_m = model.mixture_kinematic_viscosity(alphas)
        assert nu_m.shape == (3,)
        assert (nu_m > 0).all()

    def test_advance_no_flux(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        model = IncompressibleMultiphaseVoF(
            ["water", "air"], [1000.0, 1.0], [1e-3, 1e-5],
        )
        mesh = make_2d_mesh()
        n_cells = mesh.n_cells
        alpha = torch.full((n_cells, 1), 0.5, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        result = model.advance(alpha, phi, mesh, 0.01)
        assert result.shape == (n_cells, 1)
        # Should be approximately unchanged with zero flux
        assert torch.allclose(result, alpha, atol=1e-5)

    def test_repr(self):
        from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

        model = IncompressibleMultiphaseVoF(
            ["water", "air"], [1000.0, 1.0], [1e-3, 1e-5],
        )
        r = repr(model)
        assert "IncompressibleMultiphaseVoF" in r
        assert "water" in r
        assert "2" in r
