"""Shared fixtures for solver tests."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix


def make_symmetric_matrix(n_cells: int, owner, neighbour, coeff: float = 1.0):
    """Build a symmetric positive-definite LDU matrix (Poisson-like).

    For a diffusion discretisation:
    - lower[f] = upper[f] = -coeff
    - diag[i] = sum of |off-diag| for cell i (ensures SPD)
    """
    mat = LduMatrix(n_cells, owner, neighbour)
    n_internal = int(neighbour.shape[0])

    lower = -coeff * torch.ones(n_internal, dtype=CFD_DTYPE)
    upper = -coeff * torch.ones(n_internal, dtype=CFD_DTYPE)
    mat.lower = lower
    mat.upper = upper

    # Diagonal: sum of absolute off-diagonal for each cell
    diag = torch.zeros(n_cells, dtype=CFD_DTYPE)
    for f in range(n_internal):
        p = int(owner[f])
        n = int(neighbour[f])
        diag[p] += coeff
        diag[n] += coeff
    # Add a small positive value to ensure strict positivity
    diag += 0.1
    mat.diag = diag

    return mat


def make_asymmetric_matrix(n_cells: int, owner, neighbour):
    """Build an asymmetric LDU matrix (convection + diffusion).

    - lower[f] = -1.0 (diffusion)
    - upper[f] = -1.5 (diffusion + convection upwind bias)
    - diag adjusted to keep diagonal dominance
    """
    mat = LduMatrix(n_cells, owner, neighbour)
    n_internal = int(neighbour.shape[0])

    lower = -1.0 * torch.ones(n_internal, dtype=CFD_DTYPE)
    upper = -1.5 * torch.ones(n_internal, dtype=CFD_DTYPE)
    mat.lower = lower
    mat.upper = upper

    diag = torch.zeros(n_cells, dtype=CFD_DTYPE)
    for f in range(n_internal):
        p = int(owner[f])
        n = int(neighbour[f])
        diag[p] += abs(float(lower[f]))
        diag[n] += abs(float(upper[f]))
    diag += 0.5
    mat.diag = diag

    return mat


@pytest.fixture
def chain_3cell():
    """3-cell chain mesh: 0 -- 1 -- 2."""
    n_cells = 3
    owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
    return n_cells, owner, neighbour


@pytest.fixture
def chain_10cell():
    """10-cell chain mesh."""
    n_cells = 10
    owner = torch.arange(9, dtype=INDEX_DTYPE)
    neighbour = torch.arange(1, 10, dtype=INDEX_DTYPE)
    return n_cells, owner, neighbour


@pytest.fixture
def symmetric_poisson_3cell(chain_3cell):
    """3-cell symmetric SPD matrix (Poisson equation)."""
    n_cells, owner, neighbour = chain_3cell
    return make_symmetric_matrix(n_cells, owner, neighbour)


@pytest.fixture
def symmetric_poisson_10cell(chain_10cell):
    """10-cell symmetric SPD matrix (Poisson equation)."""
    n_cells, owner, neighbour = chain_10cell
    return make_symmetric_matrix(n_cells, owner, neighbour)


@pytest.fixture
def asymmetric_matrix_3cell(chain_3cell):
    """3-cell asymmetric matrix."""
    n_cells, owner, neighbour = chain_3cell
    return make_asymmetric_matrix(n_cells, owner, neighbour)


@pytest.fixture
def asymmetric_matrix_10cell(chain_10cell):
    """10-cell asymmetric matrix."""
    n_cells, owner, neighbour = chain_10cell
    return make_asymmetric_matrix(n_cells, owner, neighbour)
