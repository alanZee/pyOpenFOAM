"""
Interface reconstruction methods for multiphase flows.

Provides geometric reconstruction of the liquid-gas interface from the
Volume of Fluid (VOF) scalar field α. These methods are used in
isoAdvector and similar geometric VOF schemes to compute the plane
that cuts each interfacial cell.

Classes:

- :class:`InterfaceReconstruction` — abstract base for interface
  reconstruction methods.
- :class:`PLICReconstruction` — Piecewise Linear Interface Calculation.

PLIC computes, for each cell where 0 < α < 1:
1. The interface normal ``n̂`` (from gradient of α)
2. The plane constant ``d`` such that the plane n̂·x = d divides the
   cell into two sub-volumes matching the given α

Reference:
    Scardovelli, R. & Zaleski, S. (1999). "Direct numerical simulation
    of free-surface and interfacial flow." Annu. Rev. Fluid Mech. 31, 567-603.

    OpenFOAM ``VoF::PLIC`` / ``isoAdvector``
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["InterfaceReconstruction", "PLICReconstruction"]

logger = logging.getLogger(__name__)


class InterfaceReconstruction(ABC):
    """Abstract base class for interface reconstruction methods.

    Subclasses implement :meth:`reconstruct` which computes the
    interface normal and position from a volume fraction field.
    """

    @abstractmethod
    def reconstruct(
        self,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the interface from the volume fraction field.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction field.

        Returns
        -------
        normals : torch.Tensor
            ``(n_cells, 3)`` interface normal vectors (unit).
            Zero where there is no interface.
        plane_constants : torch.Tensor
            ``(n_cells,)`` plane constants ``d`` such that the
            interface is the plane ``n̂·x = d``. Zero where there
            is no interface.
        """
        ...

    @abstractmethod
    def compute_interface_cells(
        self,
        alpha: torch.Tensor,
        alpha_tol: float = 1e-6,
    ) -> torch.Tensor:
        """Identify cells that contain an interface.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction field.
        alpha_tol : float
            Tolerance for identifying interface cells
            (alpha_tol < alpha < 1 - alpha_tol).

        Returns
        -------
        torch.Tensor
            ``(n_interface,)`` boolean mask of interface cells.
        """
        ...


class PLICReconstruction(InterfaceReconstruction):
    """Piecewise Linear Interface Calculation (PLIC).

    Reconstructs a planar interface in each interfacial cell from the
    volume fraction field. The interface is defined by a normal vector
    and a plane constant that ensures the sub-volume fraction matches α.

    Algorithm:
    1. Compute the interface normal from the gradient of α using the
       Mixed Youngs-Centered (MYC) scheme or simple Gauss gradient.
    2. For each interface cell, find the plane constant ``d`` via
       bisection such that the sub-volume fraction equals α.

    Parameters
    ----------
    mesh : Any
        Finite volume mesh (must have owner, neighbour, cell_volumes,
        face_areas, cell_centres, n_cells, n_internal_faces).
    method : str
        Normal computation method: ``"gauss"`` (Gauss gradient) or
        ``"myc"`` (Mixed Youngs-Centered). Default ``"gauss"``.

    Examples::

        plic = PLICReconstruction(mesh)
        normals, d = plic.reconstruct(alpha)

        # Interface cells
        mask = plic.compute_interface_cells(alpha)
    """

    def __init__(
        self,
        mesh: Any,
        method: str = "gauss",
    ) -> None:
        if method not in ("gauss", "myc"):
            raise ValueError(
                f"Unknown method '{method}'. Available: 'gauss', 'myc'."
            )
        self._mesh = mesh
        self._method = method

        self._device = get_device()
        self._dtype = get_default_dtype()

        # Precompute mesh data
        self._n_cells = mesh.n_cells
        self._n_internal = mesh.n_internal_faces
        self._int_owner = mesh.owner[:self._n_internal]
        self._int_neigh = mesh.neighbour

    def reconstruct(
        self,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct PLIC interface from volume fraction.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction field.

        Returns
        -------
        normals : torch.Tensor
            ``(n_cells, 3)`` interface unit normals.
        plane_constants : torch.Tensor
            ``(n_cells,)`` plane constants ``d``.
        """
        # Step 1: Compute interface normal
        if self._method == "myc":
            normals = self._compute_normal_myc(alpha)
        else:
            normals = self._compute_normal_gauss(alpha)

        # Step 2: Compute plane constant via bisection
        plane_constants = self._compute_plane_constant(alpha, normals)

        return normals, plane_constants

    def compute_interface_cells(
        self,
        alpha: torch.Tensor,
        alpha_tol: float = 1e-6,
    ) -> torch.Tensor:
        """Identify cells containing an interface.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction field.
        alpha_tol : float
            Tolerance. Cells with alpha_tol < alpha < 1 - alpha_tol
            are interface cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` boolean mask (True for interface cells).
        """
        return (alpha > alpha_tol) & (alpha < (1.0 - alpha_tol))

    # ------------------------------------------------------------------
    # Normal computation
    # ------------------------------------------------------------------

    def _compute_normal_gauss(
        self,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute interface normal via Gauss gradient of α.

        n̂ = ∇α / |∇α|

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` unit normal vectors.
        """
        mesh = self._mesh
        n_cells = self._n_cells
        n_internal = self._n_internal

        # Gauss gradient using face interpolation
        alpha_P = gather(alpha, self._int_owner)
        alpha_N = gather(alpha, self._int_neigh)
        alpha_face = 0.5 * (alpha_P + alpha_N)

        face_contrib = alpha_face.unsqueeze(-1) * mesh.face_areas[:n_internal]

        grad = torch.zeros(
            n_cells, 3, dtype=self._dtype, device=self._device
        )
        grad.index_add_(0, self._int_owner, face_contrib)
        grad.index_add_(0, self._int_neigh, -face_contrib)

        # Boundary contributions
        if mesh.n_faces > n_internal:
            bnd_owner = mesh.owner[n_internal:]
            alpha_bnd = gather(alpha, bnd_owner)
            bnd_contrib = alpha_bnd.unsqueeze(-1) * mesh.face_areas[n_internal:]
            grad.index_add_(0, bnd_owner, bnd_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad = grad / V

        # Normalise
        grad_mag = grad.norm(dim=1, keepdim=True).clamp(min=1e-30)
        normals = grad / grad_mag

        # Zero normals where no interface
        has_interface = self.compute_interface_cells(alpha)
        normals = normals * has_interface.unsqueeze(-1).float()

        return normals

    def _compute_normal_myc(
        self,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Mixed Youngs-Centered (MYC) normal computation.

        Combines the Youngs method (finite differences on a local
        stencil) with the centered Gauss gradient. The component with
        the largest absolute value uses Youngs; others use centered.

        For unstructured meshes, we approximate MYC by blending
        the Gauss gradient with a cell-local directional estimate.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` unit normal vectors.
        """
        # Start with Gauss gradient
        n_gauss = self._compute_normal_gauss(alpha)

        # Youngs approximation: use neighbour differences
        mesh = self._mesh
        n_cells = self._n_cells

        alpha_P = gather(alpha, self._int_owner)
        alpha_N = gather(alpha, self._int_neigh)
        d_alpha = alpha_N - alpha_P

        # Face normals
        face_areas = mesh.face_areas[:self._n_internal]
        face_mag = face_areas.norm(dim=1, keepdim=True).clamp(min=1e-30)
        face_normal = face_areas / face_mag

        # Weighted gradient estimate per cell: sum d_alpha * n̂_face
        weighted_grad = torch.zeros(
            n_cells, 3, dtype=self._dtype, device=self._device
        )
        contrib = d_alpha.unsqueeze(-1) * face_normal
        weighted_grad.index_add_(0, self._int_owner, contrib)
        weighted_grad.index_add_(0, self._int_neigh, -contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        weighted_grad = weighted_grad / V

        # Normalise Youngs estimate
        wg_mag = weighted_grad.norm(dim=1, keepdim=True).clamp(min=1e-30)
        n_youngs = weighted_grad / wg_mag

        # MYC blending: for each component, choose the method whose
        # normal component has larger absolute value
        # Simplified: use Gauss where |n_gauss| > |n_youngs|, else Youngs
        use_gauss = n_gauss.abs() >= n_youngs.abs()
        normals = torch.where(use_gauss, n_gauss, n_youngs)

        # Renormalise after blending
        norm_mag = normals.norm(dim=1, keepdim=True).clamp(min=1e-30)
        normals = normals / norm_mag

        # Zero where no interface
        has_interface = self.compute_interface_cells(alpha)
        normals = normals * has_interface.unsqueeze(-1).float()

        return normals

    # ------------------------------------------------------------------
    # Plane constant computation
    # ------------------------------------------------------------------

    def _compute_plane_constant(
        self,
        alpha: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the PLIC plane constant for each interface cell.

        Uses bisection to find ``d`` such that the fraction of the cell
        volume on the positive side of the plane n̂·x = d equals α.

        For axis-aligned cells (simplified model):
            d is found such that: volume_fraction(n, d) = alpha

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction field.
        normals : torch.Tensor
            ``(n_cells, 3)`` interface normals.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` plane constants.
        """
        n_cells = self._n_cells
        has_interface = self.compute_interface_cells(alpha)

        plane_d = torch.zeros(
            n_cells, dtype=self._dtype, device=self._device
        )

        if not has_interface.any():
            return plane_d

        # Get interface cell indices
        interface_idx = has_interface.nonzero(as_tuple=True)[0]

        # For each interface cell, compute plane constant via bisection
        # Approximate each cell as a hexahedron with extent from cell centre
        # Use a simplified model: cell is a cube with side = V^(1/3)
        cell_vol = self._mesh.cell_volumes[interface_idx]
        cell_side = cell_vol.pow(1.0 / 3.0).clamp(min=1e-30)
        cell_centres = self._mesh.cell_centres[interface_idx]

        n_intf = normals[interface_idx]  # (n_intf, 3)
        a_intf = alpha[interface_idx]    # (n_intf,)

        # Bisection: find d such that volume_fraction(n, d) = alpha
        # For a unit cube centred at origin, the plane n·x = d divides
        # the cube. We parameterise d ∈ [d_min, d_max].
        #
        # d_min = min(n · corner) for all corners
        # d_max = max(n · corner) for all corners
        #
        # For a cube with side h centred at c, corners are at
        # c + (±h/2, ±h/2, ±h/2).

        h = cell_side
        c = cell_centres

        # Compute dot products with all 8 corners
        signs = torch.tensor(
            [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
             [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
            dtype=self._dtype,
            device=self._device,
        )  # (8, 3)

        # corners[i] = c + 0.5 * h * signs  for each cell
        # d_corners[k, j] = n_intf[k] · corner[k, j]
        h_half = 0.5 * h.unsqueeze(-1)  # (n_intf, 1)
        corners = c.unsqueeze(1) + h_half.unsqueeze(-1) * signs.unsqueeze(0)
        # corners: (n_intf, 8, 3)

        d_corners = (n_intf.unsqueeze(1) * corners).sum(dim=2)
        # d_corners: (n_intf, 8)

        d_min = d_corners.min(dim=1).values
        d_max = d_corners.max(dim=1).values

        # Bisection
        d_lo = d_min.clone()
        d_hi = d_max.clone()
        n_bisect = 30

        for _ in range(n_bisect):
            d_mid = 0.5 * (d_lo + d_hi)
            vol_frac = self._compute_sub_volume_fraction(
                n_intf, d_mid, c, h
            )

            # Update bounds
            too_high = vol_frac > a_intf
            d_lo = torch.where(too_high, d_lo, d_mid)
            d_hi = torch.where(too_high, d_mid, d_hi)

        d_final = 0.5 * (d_lo + d_hi)
        plane_d[interface_idx] = d_final

        return plane_d

    @staticmethod
    def _compute_sub_volume_fraction(
        normals: torch.Tensor,
        d: torch.Tensor,
        centres: torch.Tensor,
        side: torch.Tensor,
    ) -> torch.Tensor:
        """Compute volume fraction for a plane cutting a cube.

        For a cube with side h centred at c, the plane n̂·x = d divides
        it into two regions. This computes the fraction of volume on
        the side where n̂·x < d (i.e., the "negative" side).

        Uses the analytical formula for a plane cutting an axis-aligned
        cube, approximated by projecting onto the dominant normal axis.

        Parameters
        ----------
        normals : (n, 3)
        d : (n,)
        centres : (n, 3)
        side : (n,)

        Returns
        -------
        torch.Tensor
            (n,) volume fraction ∈ [0, 1].
        """
        # Project cell centre onto normal
        d_centre = (normals * centres).sum(dim=1)

        # Signed distance from centre to plane
        dist = (d - d_centre) / (0.5 * side).clamp(min=1e-30)

        # Maximum possible distance (half-diagonal projected onto normal)
        # For a cube, max distance from centre to corner along n̂
        # = (h/2) * (|n_x| + |n_y| + |n_z|)
        n_abs = normals.abs()
        d_max = n_abs.sum(dim=1).clamp(min=1e-30)

        # Normalised distance: -1 to +1 for plane through cell
        x = dist / d_max

        # Volume fraction on the negative side (n·x < d):
        # For x ∈ [-1, 1]: V_frac = 0.5 * (1 + x) (linear approximation)
        # Clamp to [0, 1]
        vol_frac = (0.5 * (1.0 + x)).clamp(0.0, 1.0)

        return vol_frac

    def __repr__(self) -> str:
        return (
            f"PLICReconstruction(n_cells={self._n_cells}, "
            f"method='{self._method}')"
        )
