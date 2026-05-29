"""
Additional radiation models for pyOpenFOAM.

Implements supplementary radiation models beyond P1:

**Volume radiation models:**
- :class:`RadiationModel` — re-exported abstract base (from ``radiation``).
- :class:`FvDOMModel` — Finite Volume Discrete Ordinates Method.
- :class:`ViewFactorModel` — surface-to-surface view factor radiation.
- :class:`OpaqueSolidModel` — opaque solid region (no internal radiation).

**Absorption/emission sub-models:**
- :class:`AbsorptionEmissionModel` — abstract base for absorption/emission.
- :class:`ConstantAbsorption` — constant absorption coefficient.
- :class:`WSGGM` — Weighted Sum of Gray Gases Model.

Reference:
    OpenFOAM ``radiationModels::fvDOM``,
    OpenFOAM ``radiationModels::viewFactor``,
    OpenFOAM ``radiationModels::opaqueSolid``,
    OpenFOAM ``absorptionEmissionModels::constant``,
    OpenFOAM ``absorptionEmissionModels::wsggm``.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.models.radiation import RadiationModel, STEFAN_BOLTZMANN

__all__ = [
    "AbsorptionEmissionModel",
    "ConstantAbsorption",
    "WSGGM",
    "FvDOMModel",
    "ViewFactorModel",
    "OpaqueSolidModel",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Absorption / emission sub-models
# ---------------------------------------------------------------------------


class AbsorptionEmissionModel(ABC):
    """Abstract base for absorption and emission coefficient models.

    Subclasses provide temperature- (and optionally species-) dependent
    absorption and emission coefficients used by radiation solvers.
    """

    @abstractmethod
    def absorption_coeff(
        self,
        T: torch.Tensor,
        species: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute absorption coefficient kappa (1/m).

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K).
        species : dict, optional
            Species mass/ mole fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` absorption coefficient (1/m).
        """
        ...

    @abstractmethod
    def emission_coeff(
        self,
        T: torch.Tensor,
        species: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute emission coefficient (1/m).

        For gray-gas models this is typically equal to the absorption
        coefficient (Kirchhoff's law).

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K).
        species : dict, optional
            Species mass/ mole fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` emission coefficient (1/m).
        """
        ...


class ConstantAbsorption(AbsorptionEmissionModel):
    """Constant absorption/emission coefficient.

    The simplest model: a single user-specified value that does not
    depend on temperature or species composition.

    Parameters
    ----------
    kappa : float
        Absorption coefficient (1/m). Default 0.1.

    Examples::

        model = ConstantAbsorption(kappa=0.5)
        k = model.absorption_coeff(T)  # all 0.5
    """

    def __init__(self, kappa: float = 0.1) -> None:
        if kappa < 0:
            raise ValueError(f"Absorption coefficient must be >= 0, got {kappa}")
        self._kappa = kappa

    def absorption_coeff(
        self,
        T: torch.Tensor,
        species: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return constant absorption coefficient."""
        return torch.full_like(T, self._kappa)

    def emission_coeff(
        self,
        T: torch.Tensor,
        species: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return constant emission coefficient (equals absorption)."""
        return torch.full_like(T, self._kappa)

    @property
    def kappa(self) -> float:
        """The constant absorption coefficient (1/m)."""
        return self._kappa

    def __repr__(self) -> str:
        return f"ConstantAbsorption(kappa={self._kappa})"


class WSGGM(AbsorptionEmissionModel):
    """Weighted Sum of Gray Gases Model (WSGGM).

    Computes the effective absorption coefficient as a weighted sum of
    gray-gas contributions:

        kappa(T) = sum_i  a_i * p * exp(-b_i * T)

    where ``a_i`` are pre-exponential factors, ``b_i`` are temperature
    exponents (1/K), and ``p`` is the partial pressure of participating
    species (atm).  A clear-gas (``i = 0``) term with ``b_0 = 0`` is
    added automatically if not present.

    Parameters
    ----------
    a_coeffs : sequence of float
        Pre-exponential factors ``a_i`` (1/(m*atm)).
    b_coeffs : sequence of float
        Temperature exponents ``b_i`` (1/K).
    pressure : float
        Participating-gas partial pressure (atm). Default 0.2 (O2 in air).

    Raises
    ------
    ValueError
        If ``a_coeffs`` and ``b_coeffs`` have different lengths or are empty.

    Examples::

        # Standard 3-gray-gas WSGGM (Smith et al.)
        wsggm = WSGGM(
            a_coeffs=[0.0, 0.446, 0.161, 0.088],
            b_coeffs=[0.0, 12.1, 2.7, 0.06],
            pressure=0.2,
        )
        kappa = wsggm.absorption_coeff(T)
    """

    def __init__(
        self,
        a_coeffs: Sequence[float],
        b_coeffs: Sequence[float],
        pressure: float = 0.2,
    ) -> None:
        if len(a_coeffs) == 0:
            raise ValueError("a_coeffs must be non-empty")
        if len(a_coeffs) != len(b_coeffs):
            raise ValueError(
                f"a_coeffs and b_coeffs must have same length, "
                f"got {len(a_coeffs)} and {len(b_coeffs)}"
            )
        self._a = list(a_coeffs)
        self._b = list(b_coeffs)
        self._p = pressure

    def absorption_coeff(
        self,
        T: torch.Tensor,
        species: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute WSGGM absorption coefficient.

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K).
        species : dict, optional
            Not used; partial pressure is set at construction time.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` effective absorption coefficient (1/m).
        """
        T_safe = T.clamp(min=1e-10)
        kappa = torch.zeros_like(T_safe)
        for a_i, b_i in zip(self._a, self._b):
            kappa = kappa + a_i * self._p * torch.exp(-b_i * T_safe)
        return kappa

    def emission_coeff(
        self,
        T: torch.Tensor,
        species: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Emission coefficient equals absorption (gray-gas Kirchhoff)."""
        return self.absorption_coeff(T, species)

    @property
    def a_coeffs(self) -> list[float]:
        """Pre-exponential factors."""
        return list(self._a)

    @property
    def b_coeffs(self) -> list[float]:
        """Temperature exponents."""
        return list(self._b)

    @property
    def pressure(self) -> float:
        """Participating-gas partial pressure (atm)."""
        return self._p

    def __repr__(self) -> str:
        return (
            f"WSGGM(n_gases={len(self._a)}, pressure={self._p})"
        )


# ---------------------------------------------------------------------------
# Volume radiation models
# ---------------------------------------------------------------------------


class FvDOMModel(RadiationModel):
    """Finite Volume Discrete Ordinates Method (FvDOM).

    Solves the radiative transfer equation (RTE) along a set of discrete
    directions defined by angular quadrature.  The full RTE along
    direction :math:`\\hat{s}_d` is:

    .. math::

        \\hat{s}_d \\cdot \\nabla I_d = \\kappa (a T^4 - I_d)

    A simplified single-sweep implementation is used here (no iterative
    scattering solve).  The total radiation source is obtained by
    summing weighted contributions from all discrete directions.

    Parameters
    ----------
    n_theta : int
        Number of polar angle divisions (meridional). Must be >= 2.
    n_phi : int
        Number of azimuthal angle divisions. Must be >= 2.
    absorption_coeff : float
        Absorption coefficient (1/m). Default 0.1.
    sigma : float
        Stefan-Boltzmann constant. Default 5.670374419e-8.
    T_ref : float
        Reference temperature for source normalisation (K). Default 300.

    Raises
    ------
    ValueError
        If ``n_theta`` or ``n_phi`` is less than 2.

    Examples::

        dom = FvDOMModel(n_theta=4, n_phi=8, absorption_coeff=0.5)
        S = dom.Sh(T)
    """

    def __init__(
        self,
        n_theta: int = 4,
        n_phi: int = 8,
        absorption_coeff: float = 0.1,
        sigma: float = STEFAN_BOLTZMANN,
        T_ref: float = 300.0,
    ) -> None:
        if n_theta < 2:
            raise ValueError(f"n_theta must be >= 2, got {n_theta}")
        if n_phi < 2:
            raise ValueError(f"n_phi must be >= 2, got {n_phi}")

        self._n_theta = n_theta
        self._n_phi = n_phi
        self._a = absorption_coeff
        self._sigma = sigma
        self._T_ref = T_ref

        self._device = get_device()
        self._dtype = get_default_dtype()

        # Build quadrature points and weights
        self._directions, self._weights = self._build_quadrature()

    def _build_quadrature(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build discrete ordinates directions and quadrature weights.

        Uses equal-angle subdivision in polar and azimuthal angles.
        The solid-angle weight for each direction is:
            w = sin(theta) * d_theta * d_phi

        Returns
        -------
        directions : torch.Tensor
            ``(n_directions, 3)`` unit direction vectors.
        weights : torch.Tensor
            ``(n_directions,)`` quadrature weights (solid angle, sr).
        """
        d_theta = math.pi / self._n_theta
        d_phi = 2.0 * math.pi / self._n_phi

        dirs = []
        weights = []
        for i in range(self._n_theta):
            theta = (i + 0.5) * d_theta  # cell-centred
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            for j in range(self._n_phi):
                phi = (j + 0.5) * d_phi
                dx = sin_t * math.cos(phi)
                dy = sin_t * math.sin(phi)
                dz = cos_t
                dirs.append([dx, dy, dz])
                weights.append(sin_t * d_theta * d_phi)

        directions = torch.tensor(dirs, dtype=self._dtype, device=self._device)
        w = torch.tensor(weights, dtype=self._dtype, device=self._device)
        return directions, w

    @property
    def n_directions(self) -> int:
        """Total number of discrete ordinates directions."""
        return self._n_theta * self._n_phi

    @property
    def directions(self) -> torch.Tensor:
        """Direction vectors ``(n_directions, 3)``."""
        return self._directions

    @property
    def weights(self) -> torch.Tensor:
        """Quadrature weights ``(n_directions,)`` (sr)."""
        return self._weights

    def Sh(self, T: torch.Tensor) -> torch.Tensor:
        """Compute FvDOM radiation source term.

        Uses the optically-thin single-iteration approximation:
            I_d = a * sigma * T^4   (black-body intensity per direction)
            S_d = w_d * a * (a*sigma*T^4 - I_d)  (simplified source)

        For the simplified model without mesh coupling the net source is:
            S = a * sigma * T^4 * (4*pi - sum(w)) / pi

        which reduces to the P1-like form when all weights are used.

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` radiation source (W/m3).
        """
        T_safe = T.clamp(min=1e-10)
        T4 = T_safe.pow(4)
        black_body = self._sigma * T4  # sigma*T^4 (W/m2)

        # Sum weighted intensity: total incident radiation approximation
        total_weight = self._weights.sum()
        # Source = a * (4*pi*sigma*T4 - sum(w * I_d))
        # where I_d ~ a*sigma*T4 (optically thin)
        # Simplified: S ~ a * 4*sigma*T4 * (1 - total_weight/(4*pi))
        S = self._a * 4.0 * self._sigma * T4 * (1.0 - total_weight / (4.0 * math.pi))
        return S

    def correct(self) -> None:
        """Update the radiation model (no-op for simplified FvDOM)."""
        pass

    def __repr__(self) -> str:
        return (
            f"FvDOMModel(n_theta={self._n_theta}, n_phi={self._n_phi}, "
            f"a={self._a}, n_dirs={self.n_directions})"
        )


class ViewFactorModel(RadiationModel):
    """View factor radiation between surfaces.

    Implements a simplified surface-to-surface radiation exchange model
    using distance-based view-factor approximation.  In a full
    implementation the view factor matrix F_{ij} is computed from
    geometric considerations; here we use an inverse-distance-squared
    weighting normalised by row sum.

    The radiation source per cell is the net radiative exchange between
    that cell's surface and all other surfaces:

    .. math::

        S_i = \\epsilon_i \\sigma \\sum_j F_{ij} (T_j^4 - T_i^4)

    Parameters
    ----------
    cell_centres : torch.Tensor
        ``(n_cells, 3)`` cell-centre coordinates.
    emissivity : float
        Surface emissivity (0..1). Default 0.9.
    sigma : float
        Stefan-Boltzmann constant. Default 5.670374419e-8.
    T_ref : float
        Reference temperature for radiation exchange (K). Default 300.

    Examples::

        vf = ViewFactorModel(cell_centres, emissivity=0.85)
        S = vf.Sh(T)
    """

    def __init__(
        self,
        cell_centres: torch.Tensor,
        emissivity: float = 0.9,
        sigma: float = STEFAN_BOLTZMANN,
        T_ref: float = 300.0,
    ) -> None:
        if not (0.0 < emissivity <= 1.0):
            raise ValueError(f"emissivity must be in (0, 1], got {emissivity}")
        self._eps = emissivity
        self._sigma = sigma
        self._T_ref = T_ref
        self._cc = cell_centres
        self._device = cell_centres.device
        self._dtype = cell_centres.dtype

        # Precompute normalised view-factor matrix (distance-based weighting)
        self._F = self._build_view_factors(cell_centres)

    @staticmethod
    def _build_view_factors(cc: torch.Tensor) -> torch.Tensor:
        """Build approximate view-factor matrix from cell centres.

        Uses inverse-distance-squared weighting, with zero self-term,
        row-normalised so each row sums to 1.

        Parameters
        ----------
        cc : torch.Tensor
            ``(n, 3)`` cell-centre coordinates.

        Returns
        -------
        torch.Tensor
            ``(n, n)`` row-normalised view-factor matrix.
        """
        # Pairwise distances (n, n)
        diff = cc.unsqueeze(1) - cc.unsqueeze(0)  # (n, n, 3)
        dist_sq = diff.pow(2).sum(dim=-1)  # (n, n)
        dist_sq = dist_sq.clamp(min=1e-30)

        # Inverse-distance-squared weighting (self-weight = 0)
        n = cc.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=cc.device)
        F = torch.zeros((n, n), dtype=cc.dtype, device=cc.device)
        F[mask] = 1.0 / dist_sq[mask]

        # Row-normalise
        row_sum = F.sum(dim=1, keepdim=True).clamp(min=1e-30)
        F = F / row_sum
        return F

    @property
    def view_factors(self) -> torch.Tensor:
        """View-factor matrix ``(n_cells, n_cells)``."""
        return self._F

    @property
    def emissivity(self) -> float:
        """Surface emissivity."""
        return self._eps

    def Sh(self, T: torch.Tensor) -> torch.Tensor:
        """Compute view-factor radiation source.

        S_i = eps * sigma * sum_j(F_ij * (T_j^4 - T_i^4))

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` radiation source (W/m3).
        """
        T_safe = T.clamp(min=1e-10)
        T4 = T_safe.pow(4)  # (n,)

        # Net exchange: sum_j F_ij * (T_j^4 - T_i^4)
        # F @ T4 gives sum_j(F_ij * T_j^4); subtract T_i^4 * sum_j(F_ij)
        exchange = self._F @ T4 - T4  # since rows sum to 1
        return self._eps * self._sigma * exchange

    def correct(self) -> None:
        """Update the radiation model (no-op for view-factor)."""
        pass

    def __repr__(self) -> str:
        n = self._F.shape[0]
        return f"ViewFactorModel(n_cells={n}, emissivity={self._eps})"


class OpaqueSolidModel(RadiationModel):
    """Opaque solid radiation model.

    For opaque solid regions where all radiation is absorbed at the
    surface.  The internal radiation source is simply zero — there is
    no volumetric radiative heat transfer inside the solid.

    This model is typically used as a placeholder for solid-region
    coupling in conjugate heat transfer problems.

    Parameters
    ----------
    n_cells : int
        Number of cells in the solid region.
    T_ref : float
        Reference temperature (K). Default 300. Not used for source
        computation but stored for API consistency.

    Examples::

        solid_rad = OpaqueSolidModel(n_cells=100)
        S = solid_rad.Sh(T)  # all zeros
    """

    def __init__(
        self,
        n_cells: int,
        T_ref: float = 300.0,
    ) -> None:
        self._n_cells = n_cells
        self._T_ref = T_ref
        self._device = get_device()
        self._dtype = get_default_dtype()

    @property
    def n_cells(self) -> int:
        """Number of solid cells."""
        return self._n_cells

    @property
    def T_ref(self) -> float:
        """Reference temperature (K)."""
        return self._T_ref

    def Sh(self, T: torch.Tensor) -> torch.Tensor:
        """Return zero source (opaque solid has no volumetric radiation).

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` zeros (W/m3).
        """
        return torch.zeros_like(T)

    def correct(self) -> None:
        """Update the radiation model (no-op)."""
        pass

    def __repr__(self) -> str:
        return f"OpaqueSolidModel(n_cells={self._n_cells})"
