"""
Wall-function boundary conditions.

Implements turbulence wall-function BCs following the OpenFOAM approach.
In OpenFOAM syntax::

    type   nutkWallFunction;
    value  uniform 0;

Wall functions bridge the viscous sublayer and log-law region,
providing effective turbulent viscosity ``ν_t`` at wall faces.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = [
    "NutkWallFunctionBC",
    "NutLowReWallFunctionBC",
    "NutUWallFunctionBC",
    "NutURoughWallFunctionBC",
    "NutUSpaldingWallFunctionBC",
    "KqRWallFunctionBC",
    "EpsilonWallFunctionBC",
    "OmegaWallFunctionBC",
]

# Von Karman constant
_KAPPA: float = 0.41
# Empirical constant for log-law
_E: float = 9.8


@BoundaryCondition.register("nutkWallFunction")
class NutkWallFunctionBC(BoundaryCondition):
    """k-equation-based wall function for turbulent viscosity (ν_t).

    Computes ν_t at the wall from the log-law:

        u⁺ = (1/κ) ln(E y⁺)

    where y⁺ = u_τ y / ν and u_τ = √(C_μ^{1/2} k).

    Coefficients:
        - ``value``: initial/existing ν_t value (default 0)
        - ``Cmu``: k-ε model constant (default 0.09)
        - ``kappa``: von Karman constant (default 0.41)
        - ``E``: wall roughness parameter (default 9.8)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._cmu: float = float(self._coeffs.get("Cmu", 0.09))
        self._kappa: float = float(self._coeffs.get("kappa", _KAPPA))
        self._E: float = float(self._coeffs.get("E", _E))

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        nu: float,
    ) -> torch.Tensor:
        """Compute turbulent viscosity at wall faces.

        Args:
            k: Turbulent kinetic energy at wall-adjacent cells
                shape ``(n_faces,)``.
            y: Wall-normal distance from cell centre to face
                shape ``(n_faces,)``.
            nu: Molecular kinematic viscosity.

        Returns:
            ν_t at each wall face, shape ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        # Friction velocity: u_τ = C_μ^{1/4} * sqrt(k)
        u_tau = self._cmu**0.25 * torch.sqrt(k.clamp(min=1e-16))

        # y⁺ = u_τ * y / ν
        y_plus = u_tau * y / max(nu, 1e-30)
        y_plus = y_plus.clamp(min=1e-4)

        # Effective ν_t from log-law
        # ν_t = κ u_τ y / ln(E y⁺)
        nut = self._kappa * u_tau * y / torch.log(self._E * y_plus)

        # Ensure non-negative
        nut = nut.clamp(min=0.0)
        return nut

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """If a ``nut`` coefficient is provided, set face values.

        Otherwise, the field is left unchanged (nut must be computed
        externally via :meth:`compute_nut`).
        """
        if "value" in self._coeffs:
            val = self._coeffs["value"]
            if isinstance(val, torch.Tensor):
                val_tensor = val.to(device=field.device, dtype=field.dtype)
            else:
                val_tensor = torch.full(
                    (self._patch.n_faces,),
                    float(val),
                    device=field.device,
                    dtype=field.dtype,
                )
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = val_tensor
            else:
                field[self._patch.face_indices] = val_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall functions: zero matrix contribution (explicit treatment).

        Wall-function BCs modify the effective viscosity field rather
        than contributing to the matrix directly.
        """
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


@BoundaryCondition.register("kqRWallFunction")
class KqRWallFunctionBC(BoundaryCondition):
    """Wall function for k, q (TKE), and R (Reynolds stress).

    Prescribes turbulence quantities at wall faces based on
    the local equilibrium assumption.

    Coefficients:
        - ``value``: initial/existing value (default 0)
        - ``Cmu``: model constant (default 0.09)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._cmu: float = float(self._coeffs.get("Cmu", 0.09))

    def compute_k_wall(
        self,
        u_tau: torch.Tensor,
    ) -> torch.Tensor:
        """Compute k at wall faces from friction velocity.

        Under local equilibrium:
            k = u_τ² / √C_μ

        Args:
            u_tau: Friction velocity at wall faces.

        Returns:
            k at wall faces.
        """
        device = get_device()
        dtype = get_default_dtype()
        u_tau = u_tau.to(device=device, dtype=dtype)
        return u_tau**2 / math.sqrt(self._cmu)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set wall-face values from coefficients if available."""
        if "value" in self._coeffs:
            val = self._coeffs["value"]
            if isinstance(val, torch.Tensor):
                val_tensor = val.to(device=field.device, dtype=field.dtype)
            else:
                val_tensor = torch.full(
                    (self._patch.n_faces,),
                    float(val),
                    device=field.device,
                    dtype=field.dtype,
                )
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = val_tensor
            else:
                field[self._patch.face_indices] = val_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall functions: zero matrix contribution (explicit treatment)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


@BoundaryCondition.register("nutLowReWallFunction")
class NutLowReWallFunctionBC(BoundaryCondition):
    """Low-Re wall function for turbulent viscosity (ν_t).

    For low-Reynolds-number models, this wall function sets ν_t = 0
    at the wall boundary.  The wall-adjacent cell resolves the viscous
    sublayer directly.

    This is the OpenFOAM ``nutLowReWallFunction`` which simply sets
    the turbulent viscosity to zero at wall faces.

    Coefficients:
        - ``value``: initial ν_t value (default 0)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set ν_t = 0 at wall faces.

        Args:
            field: Turbulent viscosity field.
            patch_idx: Optional start index into field.
        """
        device = field.device
        dtype = field.dtype

        # Set wall face values to zero
        zero_vals = torch.zeros(self._patch.n_faces, device=device, dtype=dtype)

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = zero_vals
        else:
            field[self._patch.face_indices] = zero_vals
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero matrix contribution (explicit treatment)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


@BoundaryCondition.register("epsilonWallFunction")
class EpsilonWallFunctionBC(BoundaryCondition):
    """Wall function for turbulent dissipation rate (ε).

    Prescribes ε at wall faces based on local equilibrium:

        ε = C_μ^{3/4} k^{3/2} / (κ y)

    where y is the wall-normal distance from the cell centre to the face.

    Coefficients:
        - ``value``: initial ε value (default 0)
        - ``Cmu``: k-ε model constant (default 0.09)
        - ``kappa``: von Karman constant (default 0.41)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._cmu: float = float(self._coeffs.get("Cmu", 0.09))
        self._kappa: float = float(self._coeffs.get("kappa", _KAPPA))

    def compute_epsilon(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ε at wall faces.

        ε = C_μ^{3/4} k^{3/2} / (κ y)

        Args:
            k: Turbulent kinetic energy at wall-adjacent cells
                shape ``(n_faces,)``.
            y: Wall-normal distance from cell centre to face
                shape ``(n_faces,)``.

        Returns:
            ε at each wall face, shape ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        epsilon = (
            self._cmu**0.75
            * k.clamp(min=1e-16) ** 1.5
            / (self._kappa * y.clamp(min=1e-10))
        )

        return epsilon.clamp(min=1e-10)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """If a value coefficient is provided, set face values.

        Otherwise, the field is left unchanged (ε must be computed
        externally via :meth:`compute_epsilon`).
        """
        if "value" in self._coeffs:
            val = self._coeffs["value"]
            if isinstance(val, torch.Tensor):
                val_tensor = val.to(device=field.device, dtype=field.dtype)
            else:
                val_tensor = torch.full(
                    (self._patch.n_faces,),
                    float(val),
                    device=field.device,
                    dtype=field.dtype,
                )
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = val_tensor
            else:
                field[self._patch.face_indices] = val_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall functions: zero matrix contribution (explicit treatment)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


@BoundaryCondition.register("omegaWallFunction")
class OmegaWallFunctionBC(BoundaryCondition):
    """Wall function for specific dissipation rate (ω).

    Prescribes ω at wall faces based on Wilcox (2006):

        ω = √k / (C_μ^{1/4} κ y)

    This provides the correct asymptotic behaviour ω → ∞ as y → 0.

    Coefficients:
        - ``value``: initial ω value (default 0)
        - ``Cmu``: k-ω model constant (default 0.09)
        - ``kappa``: von Karman constant (default 0.41)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._cmu: float = float(self._coeffs.get("Cmu", 0.09))
        self._kappa: float = float(self._coeffs.get("kappa", _KAPPA))

    def compute_omega(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ω at wall faces.

        ω = √k / (C_μ^{1/4} κ y)

        Args:
            k: Turbulent kinetic energy at wall-adjacent cells
                shape ``(n_faces,)``.
            y: Wall-normal distance from cell centre to face
                shape ``(n_faces,)``.

        Returns:
            ω at each wall face, shape ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        omega = torch.sqrt(k.clamp(min=1e-16)) / (
            self._cmu**0.25 * self._kappa * y.clamp(min=1e-10)
        )

        return omega.clamp(min=1e-10)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """If a value coefficient is provided, set face values.

        Otherwise, the field is left unchanged (ω must be computed
        externally via :meth:`compute_omega`).
        """
        if "value" in self._coeffs:
            val = self._coeffs["value"]
            if isinstance(val, torch.Tensor):
                val_tensor = val.to(device=field.device, dtype=field.dtype)
            else:
                val_tensor = torch.full(
                    (self._patch.n_faces,),
                    float(val),
                    device=field.device,
                    dtype=field.dtype,
                )
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = val_tensor
            else:
                field[self._patch.face_indices] = val_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall functions: zero matrix contribution (explicit treatment)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


@BoundaryCondition.register("nutUWallFunction")
class NutUWallFunctionBC(BoundaryCondition):
    """Velocity-based wall function for turbulent viscosity (ν_t).

    Computes ν_t from the velocity magnitude at the wall-adjacent cell
    using an iterative Newton-Raphson approach to solve:

        |U_parallel| = (u_tau / kappa) * ln(E * y_plus)

    This is the OpenFOAM ``nutUWallFunction`` which derives u_tau
    directly from the velocity rather than from k.

    Coefficients:
        - ``value``: initial/existing ν_t value (default 0)
        - ``kappa``: von Karman constant (default 0.41)
        - ``E``: wall roughness parameter (default 9.8)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._kappa: float = float(self._coeffs.get("kappa", _KAPPA))
        self._E: float = float(self._coeffs.get("E", _E))

    def compute_nut(
        self,
        U: torch.Tensor,
        y: torch.Tensor,
        nu: float,
    ) -> torch.Tensor:
        """Compute ν_t from velocity using log-law.

        Args:
            U: Velocity vector at wall-adjacent cells, shape ``(n_faces, 3)``.
            y: Wall-normal distance from cell centre to face, shape ``(n_faces,)``.
            nu: Molecular kinematic viscosity.

        Returns:
            ν_t at each wall face, shape ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        U = U.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        U_mag = torch.sqrt((U * U).sum(dim=-1)).clamp(min=1e-10)
        y = y.clamp(min=1e-10)
        nu_safe = max(nu, 1e-30)

        # 初始猜测：u_tau = sqrt(nu * U_mag / y)
        u_tau = torch.sqrt(nu_safe * U_mag / y)

        # Newton-Raphson 迭代
        for _ in range(20):
            y_plus = (u_tau * y / nu_safe).clamp(min=1e-4)
            ln_Ey = torch.log(self._E * y_plus).clamp(min=1e-4)

            f_val = u_tau / self._kappa * ln_Ey - U_mag
            df_val = (ln_Ey + 1.0) / self._kappa

            delta = f_val / df_val.clamp(min=1e-10)
            u_tau = (u_tau - delta).clamp(min=1e-10)

            if torch.max(torch.abs(delta)) < 1e-8 * torch.max(u_tau):
                break

        y_plus = (u_tau * y / nu_safe).clamp(min=1e-4)
        nut = self._kappa * u_tau * y / torch.log(self._E * y_plus)

        return nut.clamp(min=0.0)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set wall-face values from coefficients if available."""
        if "value" in self._coeffs:
            val = self._coeffs["value"]
            if isinstance(val, torch.Tensor):
                val_tensor = val.to(device=field.device, dtype=field.dtype)
            else:
                val_tensor = torch.full(
                    (self._patch.n_faces,),
                    float(val),
                    device=field.device,
                    dtype=field.dtype,
                )
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = val_tensor
            else:
                field[self._patch.face_indices] = val_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall functions: zero matrix contribution (explicit treatment)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


@BoundaryCondition.register("nutURoughWallFunction")
class NutURoughWallFunctionBC(BoundaryCondition):
    """Rough-wall velocity-based wall function for turbulent viscosity.

    Modifies the log-law for wall roughness using the equivalent
    sand-grain model:

        |U_parallel| = (u_tau / kappa) * ln(y / (Ks * Cs + y_0))

    where Ks is the equivalent sand-grain roughness height and Cs is
    the roughness constant.  When Ks = 0, reduces to smooth-wall
    behaviour.

    Coefficients:
        - ``value``: initial/existing ν_t value (default 0)
        - ``Ks``: sand-grain roughness height (default 0)
        - ``Cs``: roughness constant (default 0.5)
        - ``kappa``: von Karman constant (default 0.41)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._Ks: float = float(self._coeffs.get("Ks", 0.0))
        self._Cs: float = float(self._coeffs.get("Cs", 0.5))
        self._kappa: float = float(self._coeffs.get("kappa", _KAPPA))

    def compute_nut(
        self,
        U: torch.Tensor,
        y: torch.Tensor,
        nu: float,
    ) -> torch.Tensor:
        """Compute ν_t from velocity with rough-wall correction.

        Args:
            U: Velocity vector at wall-adjacent cells, shape ``(n_faces, 3)``.
            y: Wall-normal distance from cell centre to face, shape ``(n_faces,)``.
            nu: Molecular kinematic viscosity.

        Returns:
            ν_t at each wall face, shape ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        U = U.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        U_mag = torch.sqrt((U * U).sum(dim=-1)).clamp(min=1e-10)
        y = y.clamp(min=1e-10)
        nu_safe = max(nu, 1e-30)

        rough_length = self._Ks * self._Cs + nu_safe * 0.01
        u_tau = torch.sqrt(nu_safe * U_mag / y)

        for _ in range(20):
            y_over_y0 = (y / rough_length).clamp(min=1.0 + 1e-6)
            ln_ratio = torch.log(y_over_y0).clamp(min=1e-4)

            f_val = u_tau / self._kappa * ln_ratio - U_mag
            df_val = ln_ratio / self._kappa

            delta = f_val / df_val.clamp(min=1e-10)
            u_tau = (u_tau - delta).clamp(min=1e-10)

            if torch.max(torch.abs(delta)) < 1e-8 * torch.max(u_tau):
                break

        nut = self._kappa * u_tau * y
        return nut.clamp(min=0.0)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set wall-face values from coefficients if available."""
        if "value" in self._coeffs:
            val = self._coeffs["value"]
            if isinstance(val, torch.Tensor):
                val_tensor = val.to(device=field.device, dtype=field.dtype)
            else:
                val_tensor = torch.full(
                    (self._patch.n_faces,),
                    float(val),
                    device=field.device,
                    dtype=field.dtype,
                )
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = val_tensor
            else:
                field[self._patch.face_indices] = val_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall functions: zero matrix contribution (explicit treatment)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


@BoundaryCondition.register("nutUSpaldingWallFunction")
class NutUSpaldingWallFunctionBC(BoundaryCondition):
    """Spalding unified wall function for turbulent viscosity.

    Uses Spalding's law-of-the-wall which provides a single continuous
    expression from the viscous sublayer through the log-law region:

        y⁺ = u⁺ + exp(-κB) * [exp(κu⁺) - 1 - κu⁺ - (κu⁺)²/2 - (κu⁺)³/6]

    where B = ln(E) / κ.

    This eliminates the discontinuity between viscous and log-law
    sublayers that occurs with piecewise wall functions.

    Coefficients:
        - ``value``: initial/existing ν_t value (default 0)
        - ``kappa``: von Karman constant (default 0.41)
        - ``E``: wall roughness parameter (default 9.8)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._kappa: float = float(self._coeffs.get("kappa", _KAPPA))
        self._E: float = float(self._coeffs.get("E", _E))

    def compute_nut(
        self,
        U: torch.Tensor,
        y: torch.Tensor,
        nu: float,
    ) -> torch.Tensor:
        """Compute ν_t using Spalding's unified wall function.

        Args:
            U: Velocity vector at wall-adjacent cells, shape ``(n_faces, 3)``.
            y: Wall-normal distance from cell centre to face, shape ``(n_faces,)``.
            nu: Molecular kinematic viscosity.

        Returns:
            ν_t at each wall face, shape ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        U = U.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        U_mag = torch.sqrt((U * U).sum(dim=-1)).clamp(min=1e-10)
        y = y.clamp(min=1e-10)
        nu_safe = max(nu, 1e-30)
        B = math.log(self._E) / self._kappa
        exp_neg_kB = math.exp(-self._kappa * B)

        u_tau = torch.sqrt(nu_safe * U_mag / y)

        for _ in range(30):
            u_plus = (U_mag / u_tau).clamp(min=1e-4, max=200.0)
            y_plus = u_tau * y / nu_safe

            ku = self._kappa * u_plus
            ku = ku.clamp(max=50.0)
            exp_ku = torch.exp(ku)
            taylor = 1.0 + ku + ku**2 / 2.0 + ku**3 / 6.0
            spalding = u_plus + exp_neg_kB * (exp_ku - taylor)

            residual = y_plus - spalding

            dy_plus_dutau = y / nu_safe
            du_plus_dutau = -U_mag / (u_tau * u_tau)

            dspalding_du_plus = (
                1.0
                + exp_neg_kB
                * (
                    self._kappa * exp_ku
                    - self._kappa
                    - self._kappa**2 * u_plus
                    - self._kappa**3 * u_plus**2 / 2.0
                )
            )
            dspalding_dutau = dspalding_du_plus * du_plus_dutau

            df = dy_plus_dutau - dspalding_dutau
            df = df.clamp(min=1e-20)

            delta = residual / df
            u_tau = (u_tau - delta).clamp(min=1e-10)

            if torch.max(torch.abs(delta)) < 1e-8 * torch.max(u_tau):
                break

        y_plus = (u_tau * y / nu_safe).clamp(min=1e-4)
        nut = self._kappa * u_tau * y / torch.log(self._E * y_plus)

        return nut.clamp(min=0.0)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set wall-face values from coefficients if available."""
        if "value" in self._coeffs:
            val = self._coeffs["value"]
            if isinstance(val, torch.Tensor):
                val_tensor = val.to(device=field.device, dtype=field.dtype)
            else:
                val_tensor = torch.full(
                    (self._patch.n_faces,),
                    float(val),
                    device=field.device,
                    dtype=field.dtype,
                )
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = val_tensor
            else:
                field[self._patch.face_indices] = val_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall functions: zero matrix contribution (explicit treatment)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source
