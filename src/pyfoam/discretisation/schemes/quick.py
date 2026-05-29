"""
QUICK interpolation scheme -- third-order with deferred correction.

The QUICK (Quadratic Upstream Interpolation for Convective Kinematics)
scheme uses a three-point upstream-biased quadratic interpolation.

For a face with upwind cell *up*, downwind cell *dn*, and
upstream-of-upstream cell *2up*:

.. math::

    \\phi_f = \\frac{3}{4}\\phi_{up} + \\frac{3}{8}\\phi_{dn}
    - \\frac{1}{8}\\phi_{2up}

If the *2up* cell cannot be found (boundary-adjacent upwind cell,
or 2-cell mesh), the scheme falls back to linear interpolation for
that face.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather
from pyfoam.core.dtype import INDEX_DTYPE

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["QuickInterpolation"]


class QuickInterpolation(InterpolationScheme):
    """Third-order QUICK interpolation scheme.

    Uses quadratic upstream-biased interpolation.  The face value is:

    .. math::

        \\phi_f = \\frac{3}{4}\\phi_{up} + \\frac{3}{8}\\phi_{dn}
        - \\frac{1}{8}\\phi_{2up}

    where *up* is upwind, *dn* is downwind, and *2up* is two cells upstream.

    For faces where the 2up cell cannot be found (e.g. upwind cell is
    boundary-adjacent with only one internal face), falls back to linear
    interpolation.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    # QUICK coefficients
    _C_UP = 0.75     # 3/4 -- upwind cell weight
    _C_DN = 0.375    # 3/8 -- downwind cell weight
    _C_2UP = -0.125  # -1/8 -- two-upstream cell weight

    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        self._weights = compute_centre_weights(
            mesh.cell_centres,
            mesh.face_centres,
            mesh.owner,
            mesh.neighbour,
            mesh.n_internal_faces,
            mesh.n_faces,
            device=mesh.device,
            dtype=mesh.dtype,
        )
        # Precompute cell-to-face connectivity and 2up indices
        self._build_connectivity()

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _build_connectivity(self) -> None:
        """Build cell-to-face mapping and per-face 2up cell indices.

        For each internal face *f* with owner *P* and neighbour *N*:

        - ``_2up_cell_pos[f]`` = 2up cell index when flux >= 0 (upwind = P)
        - ``_2up_cell_neg[f]`` = 2up cell index when flux < 0  (upwind = N)

        A value of -1 means the 2up cell could not be determined and
        linear fallback should be used for that face.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        device = mesh.device

        # 1. Build cell -> list of face indices (all faces, internal + boundary)
        cell_faces: list[list[int]] = [[] for _ in range(n_cells)]
        for f in range(n_faces):
            cell_faces[owner[f].item()].append(f)
        for f in range(n_internal):
            cell_faces[neighbour[f].item()].append(f)
        self._cell_faces = cell_faces

        # 2. Compute 2up cell index for each internal face (both directions)
        two_up_pos = torch.full(
            (n_internal,), -1, dtype=INDEX_DTYPE, device=device
        )
        two_up_neg = torch.full(
            (n_internal,), -1, dtype=INDEX_DTYPE, device=device
        )

        for f in range(n_internal):
            P = owner[f].item()
            N = neighbour[f].item()

            # Flux >= 0: upwind = P, downwind = N -> find cell beyond P
            two_up_pos[f] = self._find_2up_cell(P, f, exclude=N)

            # Flux < 0: upwind = N, downwind = P -> find cell beyond N
            two_up_neg[f] = self._find_2up_cell(N, f, exclude=P)

        self._2up_cell_pos = two_up_pos  # (n_internal,)
        self._2up_cell_neg = two_up_neg  # (n_internal,)

    def _find_2up_cell(self, cell: int, face: int, exclude: int) -> int:
        """Find the cell on the other side of *cell*, not through *face*.

        Searches the faces of *cell* (excluding *face*) for a face that
        has *cell* as owner or neighbour, and returns the cell on the
        opposite side -- provided it is not *exclude*.

        Returns -1 if no suitable cell is found (cell has only one
        internal face, or the only other neighbour is *exclude*).
        """
        mesh = self._mesh
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        for g in self._cell_faces[cell]:
            if g == face:
                continue
            # Only internal faces have a cell on the other side
            if g < n_internal:
                P_g = owner[g].item()
                N_g = neighbour[g].item()
                if P_g == cell:
                    candidate = N_g
                else:
                    candidate = P_g
                if candidate != exclude:
                    return candidate
        return -1

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """QUICK interpolation of cell values to faces.

        For each internal face the scheme determines upwind/downwind from
        the sign of *face_flux*, then locates the two-cells-upstream
        value.  Faces where the 2up cell cannot be found fall back to
        linear interpolation.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Required.

        Returns:
            ``(n_faces,)`` face values.

        Raises:
            ValueError: If *phi* is not 1-D or *face_flux* is None.
        """
        if phi.dim() != 1:
            raise ValueError(
                f"Expected 1-D input tensor, got {phi.dim()}-D. "
                f"Interpolation operates on scalar fields only."
            )
        if face_flux is None:
            raise ValueError(
                "QuickInterpolation requires 'face_flux'."
            )

        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        phi = phi.to(device=device, dtype=dtype)
        face_values = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            face_values = gather(phi, owner)
            return face_values

        # --- Owner and neighbour values for all internal faces ---
        phi_P = gather(phi, owner[:n_internal])      # (n_int,)
        phi_N = gather(phi, neighbour[:n_internal])   # (n_int,)

        # --- Determine upwind / downwind from flux sign ---
        int_flux = face_flux[:n_internal].to(device=device, dtype=dtype)
        is_pos = int_flux >= 0.0  # True -> owner is upwind

        phi_up = torch.where(is_pos, phi_P, phi_N)   # upwind value
        phi_dn = torch.where(is_pos, phi_N, phi_P)   # downwind value

        # --- Gather 2up cell values (both directions) ---
        has_2up_pos = self._2up_cell_pos >= 0
        has_2up_neg = self._2up_cell_neg >= 0

        phi_2up_pos = torch.zeros(n_internal, dtype=dtype, device=device)
        phi_2up_neg = torch.zeros(n_internal, dtype=dtype, device=device)

        if has_2up_pos.any():
            valid_idx = self._2up_cell_pos[has_2up_pos]
            phi_2up_pos[has_2up_pos] = gather(phi, valid_idx)
        if has_2up_neg.any():
            valid_idx = self._2up_cell_neg[has_2up_neg]
            phi_2up_neg[has_2up_neg] = gather(phi, valid_idx)

        # Select the 2up value based on flux direction
        has_2up = torch.where(is_pos, has_2up_pos, has_2up_neg)
        phi_2up = torch.where(is_pos, phi_2up_pos, phi_2up_neg)

        # --- Apply QUICK formula where 2up is available ---
        phi_quick = (
            self._C_UP * phi_up
            + self._C_DN * phi_dn
            + self._C_2UP * phi_2up
        )

        # --- Fallback to linear where 2up is not available ---
        w = self._weights[:n_internal]
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        face_values[:n_internal] = torch.where(has_2up, phi_quick, phi_linear)

        # --- Boundary faces: use owner values ---
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
