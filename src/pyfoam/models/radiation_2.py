"""
Enhanced P1 radiation model with iterative solution and Aitken acceleration.

Extends the basic P1 radiation model by solving the full radiation
transport equation for incident radiation G iteratively, rather than
using the optically thin approximation.

The P1 equation:

    ∇·(Γ ∇G) - a G = -4 a σ T⁴

where:
    Γ = 1 / (3(a + σ_s))     (diffusion coefficient)
    a = absorption coefficient
    σ_s = scattering coefficient
    σ = Stefan-Boltzmann constant

The equation is solved iteratively using a fixed-point iteration with
Aitken delta-squared acceleration for faster convergence.

The radiation heat source in the energy equation:
    S_rad = a(G - 4σT⁴)

Reference:
    OpenFOAM ``radiationModels::P1``
    Aitken delta-squared method (numerical acceleration)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.models.radiation import RadiationModel, STEFAN_BOLTZMANN

__all__ = ["P1RadiationEnhanced"]

logger = logging.getLogger(__name__)


class P1RadiationEnhanced(RadiationModel):
    """Enhanced P1 radiation model with iterative G-equation solve.

    Solves the full P1 radiation transport equation for incident
    radiation G using an iterative fixed-point scheme with optional
    Aitken delta-squared acceleration.

    The algorithm:
    1. Initialise G from the optically thin approximation
    2. Solve ∇·(Γ ∇G) - a G = -4aσT⁴ iteratively:
       - Assemble the diffusion operator ∇·(Γ ∇G)
       - Solve for G_new via Jacobi iteration
       - Apply Aitken acceleration to speed convergence
    3. Compute S_rad = a(G - 4σT⁴)

    Parameters
    ----------
    mesh : Any
        Finite volume mesh (must have owner, neighbour, cell_volumes,
        face_areas, n_cells, n_internal_faces).
    absorption_coeff : float
        Absorption coefficient ``a`` (1/m). Default 0.1.
    scattering_coeff : float
        Scattering coefficient ``σ_s`` (1/m). Default 0.0.
    T_ref : float
        Reference temperature for radiation (K). Default 300.
    sigma : float
        Stefan-Boltzmann constant. Default 5.670374419e-8.
    max_iter : int
        Maximum iterations for G-equation solve. Default 50.
    tolerance : float
        Convergence tolerance for G residual. Default 1e-6.
    use_aitken : bool
        Enable Aitken delta-squared acceleration. Default True.
    under_relaxation : float
        Under-relaxation factor for Jacobi iteration. Default 0.5.

    Examples::

        radiation = P1RadiationEnhanced(
            mesh, absorption_coeff=0.5, T_ref=300
        )
        S_rad = radiation.Sh(T)  # solve G and return source
    """

    def __init__(
        self,
        mesh: Any,
        absorption_coeff: float = 0.1,
        scattering_coeff: float = 0.0,
        T_ref: float = 300.0,
        sigma: float = STEFAN_BOLTZMANN,
        max_iter: int = 50,
        tolerance: float = 1e-6,
        use_aitken: bool = True,
        under_relaxation: float = 0.5,
    ) -> None:
        self._mesh = mesh
        self._a = absorption_coeff
        self._a_s = scattering_coeff
        self._T_ref = T_ref
        self._sigma = sigma
        self._max_iter = max_iter
        self._tolerance = tolerance
        self._use_aitken = use_aitken
        self._ur = under_relaxation

        self._device = get_device()
        self._dtype = get_default_dtype()

        # Diffusion coefficient: Γ = 1 / (3(a + σ_s))
        denom = 3.0 * (self._a + self._a_s)
        self._Gamma = 1.0 / max(denom, 1e-30)

        # Cached incident radiation G (lazy-initialised)
        self._G: Optional[torch.Tensor] = None

        # Precompute face-to-cell connectivity for Laplacian operator
        self._precompute_connectivity()

    def _precompute_connectivity(self) -> None:
        """Precompute mesh connectivity data for the Laplacian operator."""
        mesh = self._mesh
        n_internal = mesh.n_internal_faces

        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Face area magnitudes
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1, keepdim=True).clamp(min=1e-30)
        self._S_mag = S_mag.squeeze(-1)

        # Unit face normals
        self._face_normal = face_areas / S_mag

        # Distance between cell centres projected onto face normal
        cc_P = mesh.cell_centres[int_owner]
        cc_N = mesh.cell_centres[int_neigh]
        d_PN = cc_N - cc_P

        # Projected distance: d = |d_PN · n̂|
        d_proj = (d_PN * self._face_normal).sum(dim=1).abs().clamp(min=1e-30)
        self._d_PN = d_proj

        # Cell volumes
        self._V = mesh.cell_volumes.clamp(min=1e-30)

        # Store connectivity
        self._int_owner = int_owner
        self._int_neigh = int_neigh

        # Diagonal coefficient contribution (from Laplacian)
        # Each internal face contributes Γ * |S| / d to both owner and neighbour
        n_cells = mesh.n_cells
        face_coeff = self._Gamma * self._S_mag / self._d_PN

        diag = torch.zeros(n_cells, dtype=self._dtype, device=self._device)
        diag = diag + scatter_add(face_coeff, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff, int_neigh, n_cells)

        # Total diagonal: face contributions + absorption
        self._diag = diag + self._a * self._V

    def Sh(self, T: torch.Tensor) -> torch.Tensor:
        """Solve G-equation and return radiation heat source.

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` radiation source term (W/m³).
        """
        T_safe = T.clamp(min=1e-10)
        G = self._solve_G(T_safe)
        return self._a * (G - 4.0 * self._sigma * T_safe.pow(4))

    def _solve_G(self, T: torch.Tensor) -> torch.Tensor:
        """Iteratively solve the P1 equation for incident radiation G.

        The equation: ∇·(Γ ∇G) - a G = -4aσT⁴

        Rearranged for Jacobi iteration:
            G_new = (rhs + Γ * Σ_neighbours(G_neigh * |S|/d)) / diag

        where diag = Σ(Γ * |S|/d) + a * V

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K), already clamped positive.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` incident radiation G (W/m²).
        """
        n_cells = self._mesh.n_cells

        # Right-hand side: 4aσT⁴
        rhs = 4.0 * self._a * self._sigma * T.pow(4)

        # Source term (RHS of the equation, with sign convention)
        source = rhs * self._V

        # Initialise G from optically thin approximation:
        # G ≈ 4σT⁴ (when ∇G is small)
        if self._G is None or self._G.shape[0] != n_cells:
            self._G = 4.0 * self._sigma * T.pow(4)

        G = self._G.clone()

        # Face coefficient for off-diagonal
        face_coeff = self._Gamma * self._S_mag / self._d_PN

        # Aitken acceleration state
        G_prev_prev = None
        lambda_aitken = 1.0

        for iteration in range(self._max_iter):
            G_old = G.clone()

            # Compute off-diagonal contribution: Σ(G_neigh * face_coeff)
            # For owner: neighbour contributes; for neighbour: owner contributes
            off_diag = torch.zeros(
                n_cells, dtype=self._dtype, device=self._device
            )
            off_diag = off_diag + scatter_add(
                gather(G, self._int_neigh) * face_coeff,
                self._int_owner,
                n_cells,
            )
            off_diag = off_diag + scatter_add(
                gather(G, self._int_owner) * face_coeff,
                self._int_neigh,
                n_cells,
            )

            # Jacobi update: G_new = (source + off_diag) / diag
            diag_safe = self._diag.clamp(min=1e-30)
            G_new = (source + off_diag) / diag_safe

            # Under-relaxation
            G_relaxed = (1.0 - self._ur) * G + self._ur * G_new

            # Aitken delta-squared acceleration
            if self._use_aitken and G_prev_prev is not None:
                delta_prev = G_old - G_prev_prev
                delta_curr = G_relaxed - G_old

                # Avoid division by zero
                delta_diff = delta_curr - delta_prev
                denom = (delta_diff * delta_diff).sum().clamp(min=1e-30)
                numer = (delta_prev * delta_diff).sum()

                lambda_new = -lambda_aitken * numer / denom

                # Clamp lambda to reasonable range
                lambda_new = max(0.1, min(lambda_new, 2.0))

                # Apply Aitken acceleration
                G_relaxed = G_old + lambda_new * (G_relaxed - G_old)
                lambda_aitken = lambda_new

            # Store for next Aitken step
            G_prev_prev = G_old
            G = G_relaxed

            # Convergence check
            residual = float((G - G_old).abs().max().item())
            if residual < self._tolerance:
                logger.debug(
                    "P1Enhanced converged in %d iterations (res=%.2e)",
                    iteration + 1,
                    residual,
                )
                break

        self._G = G
        return G

    def correct(self) -> None:
        """Reset cached G to trigger re-solve on next Sh call."""
        self._G = None

    @property
    def absorption_coeff(self) -> float:
        """Absorption coefficient (1/m)."""
        return self._a

    @property
    def scattering_coeff(self) -> float:
        """Scattering coefficient (1/m)."""
        return self._a_s

    @property
    def T_ref(self) -> float:
        """Reference temperature (K)."""
        return self._T_ref

    @property
    def G(self) -> Optional[torch.Tensor]:
        """Current incident radiation field (W/m²), or None if not solved."""
        return self._G

    def __repr__(self) -> str:
        return (
            f"P1RadiationEnhanced(a={self._a}, a_s={self._a_s}, "
            f"T_ref={self._T_ref}, max_iter={self._max_iter}, "
            f"aitken={self._use_aitken})"
        )
