"""
Pressure equation assembly for incompressible flow solvers.

Derives and assembles the pressure Poisson equation from the continuity
equation by substituting the velocity correction from the momentum equation.

Starting from the discrete momentum equation:
    A_p * U = H(U) - grad(p)

Rearranging:
    U = HbyA - (1/A_p) * grad(p)

Substituting into continuity (div(U) = 0):
    div(HbyA) - div((1/A_p) * grad(p)) = 0

This gives the pressure Poisson equation:
    div((1/A_p) * grad(p)) = div(HbyA)

In discrete form:
    laplacian(1/A_p, p) = div(phiHbyA)

The assembly builds the FvMatrix for the Laplacian operator and sets
the source from the divergence of phiHbyA.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.solvers.linear_solver import LinearSolverBase

__all__ = [
    "assemble_pressure_equation",
    "solve_pressure_equation",
    "correct_velocity",
    "correct_face_flux",
]


def assemble_pressure_equation(
    phiHbyA: torch.Tensor,
    A_p: torch.Tensor,
    mesh: Any,
    face_weights: torch.Tensor | None = None,
) -> FvMatrix:
    """Assemble the pressure Poisson equation.

    Builds:
        laplacian(1/A_p, p) = div(phiHbyA)

    The Laplacian matrix has:
        face_coeff = (1/A_p)_f * |S_f| * delta_f
        lower[f] = -face_coeff / V_P
        upper[f] = -face_coeff / V_N
        diag = -sum(off-diag)

    The source is -div(phiHbyA) (negative because we move it to RHS).

    Args:
        phiHbyA: ``(n_faces,)`` — face flux from HbyA interpolation.
        A_p: ``(n_cells,)`` — diagonal coefficients of momentum matrix.
        mesh: The finite volume mesh.
        face_weights: ``(n_faces,)`` — interpolation weights.

    Returns:
        :class:`~pyfoam.core.fv_matrix.FvMatrix` for the pressure equation.
    """
    device = mesh.device
    dtype = mesh.dtype
    n_cells = mesh.n_cells
    n_internal = mesh.n_internal_faces
    n_faces = mesh.n_faces
    cell_volumes = mesh.cell_volumes
    face_areas = mesh.face_areas
    delta_coeffs = mesh.delta_coefficients
    owner = mesh.owner
    neighbour = mesh.neighbour

    if face_weights is None:
        face_weights = mesh.face_weights

    # Build FvMatrix
    mat = FvMatrix(
        n_cells,
        owner[:n_internal],
        neighbour,
        device=device,
        dtype=dtype,
    )

    # Inverse diagonal: 1/A_p (safe division)
    A_p_safe = A_p.abs().clamp(min=1e-30)
    inv_A_p = 1.0 / A_p_safe

    # Face-interpolated 1/A_p
    w = face_weights[:n_internal]
    int_owner = owner[:n_internal]
    int_neigh = neighbour

    inv_A_p_P = gather(inv_A_p, int_owner)
    inv_A_p_N = gather(inv_A_p, int_neigh)
    inv_A_p_f = w * inv_A_p_P + (1.0 - w) * inv_A_p_N

    # Face area magnitude
    S_mag = face_areas[:n_internal].norm(dim=1)

    # Delta coefficient
    delta_f = delta_coeffs[:n_internal]

    # Face coefficient: (1/A_p)_f * |S_f| * delta_f
    face_coeff = inv_A_p_f * S_mag * delta_f

    # Cell volumes
    V_P = gather(cell_volumes, int_owner)
    V_N = gather(cell_volumes, int_neigh)

    # Matrix coefficients (Laplacian discretisation)
    mat.lower = -face_coeff / V_P
    mat.upper = -face_coeff / V_N

    # Diagonal: sum of off-diagonal contributions
    diag = torch.zeros(n_cells, dtype=dtype, device=device)
    diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
    diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)
    mat.diag = diag

    # Source: divergence of phiHbyA (negative because moving to RHS)
    # For each internal face: flux contribution to owner and neighbour
    source = torch.zeros(n_cells, dtype=dtype, device=device)
    source = source + scatter_add(-phiHbyA[:n_internal], int_owner, n_cells)
    source = source + scatter_add(phiHbyA[:n_internal], int_neigh, n_cells)

    # Boundary face contributions to source
    if n_faces > n_internal:
        bnd_owner = owner[n_internal:]
        source = source + scatter_add(-phiHbyA[n_internal:], bnd_owner, n_cells)

    mat.source = source

    return mat


def solve_pressure_equation(
    p_eqn: FvMatrix,
    p: torch.Tensor,
    solver: LinearSolverBase,
    tolerance: float = 1e-6,
    max_iter: int = 1000,
    reference_cell: int = 0,
) -> tuple[torch.Tensor, int, float]:
    """Solve the pressure Poisson equation.

    Args:
        p_eqn: Assembled pressure equation FvMatrix.
        p: ``(n_cells,)`` — current pressure (used as initial guess).
        solver: Linear solver instance.
        tolerance: Convergence tolerance.
        max_iter: Maximum solver iterations.
        reference_cell: Cell index to pin pressure reference.

    Returns:
        Tuple of ``(p_new, iterations, residual)``.
    """
    # Pin reference pressure to remove singularity
    p_eqn.set_reference(reference_cell, value=0.0)

    # Solve
    p_new, iterations, residual = p_eqn.solve(
        solver, p, tolerance=tolerance, max_iter=max_iter,
    )

    return p_new, iterations, residual


def correct_velocity(
    U: torch.Tensor,
    HbyA: torch.Tensor,
    p: torch.Tensor,
    A_p: torch.Tensor,
    mesh: Any,
) -> torch.Tensor:
    """Correct velocity from the pressure gradient.

    From the momentum equation:
        U = HbyA - (1/A_p) * grad(p)

    The gradient is computed using the Gauss theorem:
        grad(p)_P = (1/V_P) * sum_f(p_f * S_f)

    Args:
        U: ``(n_cells, 3)`` — current velocity (to be corrected).
        HbyA: ``(n_cells, 3)`` — HbyA velocity field.
        p: ``(n_cells,)`` — corrected pressure.
        A_p: ``(n_cells,)`` — diagonal coefficients.
        mesh: The finite volume mesh.

    Returns:
        ``(n_cells, 3)`` — corrected velocity.
    """
    device = mesh.device
    dtype = mesh.dtype
    n_cells = mesh.n_cells
    n_internal = mesh.n_internal_faces
    n_faces = mesh.n_faces
    face_areas = mesh.face_areas
    cell_volumes = mesh.cell_volumes
    owner = mesh.owner
    neighbour = mesh.neighbour

    # Inverse diagonal
    A_p_safe = A_p.abs().clamp(min=1e-30)
    inv_A_p = 1.0 / A_p_safe

    # Compute pressure gradient using Gauss theorem
    # Face pressure via linear interpolation
    w = mesh.face_weights
    int_owner = owner[:n_internal]
    int_neigh = neighbour

    p_P = gather(p, int_owner)
    p_N = gather(p, int_neigh)
    p_face = w[:n_internal] * p_P + (1.0 - w[:n_internal]) * p_N

    # Face contribution: p_f * S_f
    face_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

    # Accumulate into cell gradient
    grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
    grad_p.index_add_(0, int_owner, face_contrib)
    grad_p.index_add_(0, int_neigh, -face_contrib)

    # Boundary face contributions (assuming zero gradient / no correction)
    # For boundary faces, the pressure gradient contribution is zero
    # if we assume dp/dn = 0 on boundaries

    # Divide by cell volume
    V = cell_volumes.unsqueeze(-1).clamp(min=1e-30)
    grad_p = grad_p / V

    # Velocity correction: U = HbyA - (1/A_p) * grad(p)
    U_corrected = HbyA - inv_A_p.unsqueeze(-1) * grad_p

    return U_corrected


def correct_face_flux(
    phi: torch.Tensor,
    p: torch.Tensor,
    A_p: torch.Tensor,
    mesh: Any,
    face_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correct face flux from the pressure solution.

    The corrected flux is:
        φ_f = φHbyA_f - (1/A_p)_f * (p_N - p_P) * |S_f| * delta_f

    Args:
        phi: ``(n_faces,)`` — face flux to correct (phiHbyA + correction).
        p: ``(n_cells,)`` — corrected pressure.
        A_p: ``(n_cells,)`` — diagonal coefficients.
        mesh: The finite volume mesh.
        face_weights: ``(n_faces,)`` — interpolation weights.

    Returns:
        ``(n_faces,)`` — corrected face flux.
    """
    device = mesh.device
    dtype = mesh.dtype
    n_internal = mesh.n_internal_faces
    n_faces = mesh.n_faces
    face_areas = mesh.face_areas
    delta_coeffs = mesh.delta_coefficients
    owner = mesh.owner
    neighbour = mesh.neighbour

    if face_weights is None:
        face_weights = mesh.face_weights

    # Inverse diagonal
    A_p_safe = A_p.abs().clamp(min=1e-30)
    inv_A_p = 1.0 / A_p_safe

    # Face-interpolated 1/A_p
    w = face_weights[:n_internal]
    int_owner = owner[:n_internal]
    int_neigh = neighbour

    inv_A_p_f = w * gather(inv_A_p, int_owner) + (1.0 - w) * gather(inv_A_p, int_neigh)

    # Pressure difference across internal faces
    p_P = gather(p, int_owner)
    p_N = gather(p, int_neigh)
    dp = p_P - p_N

    # Face area magnitude and delta coefficient
    S_mag = face_areas[:n_internal].norm(dim=1)
    delta_f = delta_coeffs[:n_internal]

    # Flux correction
    flux_correction = inv_A_p_f * dp * S_mag * delta_f

    # Apply correction to internal faces only
    phi_corrected = phi.clone()
    phi_corrected[:n_internal] = phi[:n_internal] + flux_correction

    return phi_corrected
