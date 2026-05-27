"""
Laminar turbulence models for Newtonian and generalised-Newtonian fluids.

Provides:

- :class:`StokesModel` — Newtonian (Stokes) laminar model.  The turbulent
  viscosity is identically zero; the effective viscosity equals the
  molecular viscosity alone.
- :class:`GeneralizedNewtonianModel` — Generalised-Newtonian model where
  the apparent viscosity depends on the local shear-strain rate.  Supports
  several constitutive laws:

  - **powerLaw**:  mu = k * |gamma_dot|^(n - 1)
  - **BirdCarreau**:  mu = mu_inf + (mu_0 - mu_inf) * (1 + (lambda * |gamma_dot|)^2)^((n-1)/2)
  - **Cross**:  mu = mu_inf + (mu_0 - mu_inf) / (1 + (lambda * |gamma_dot|)^m)
  - **Casson**:  mu = (sqrt(tau_y / |gamma_dot|) + sqrt(mu_p))^2
  - **HerschelBulkley**:  mu = tau_y / |gamma_dot| + k * |gamma_dot|^(n-1)

  Registered as ``"generalizedNewtonian"`` in the ``TurbulenceModel`` RTS
  table.  The viscosity model is selected via the ``viscosityModel``
  keyword in kwargs.

Usage::

    from pyfoam.turbulence.laminar_models import TurbulenceModel

    # Stokes (Newtonian laminar)
    model = TurbulenceModel.create("Stokes", mesh, U, phi)

    # Power-law
    model = TurbulenceModel.create(
        "generalizedNewtonian", mesh, U, phi,
        viscosityModel="powerLaw", K=0.01, n=0.6,
    )

    # Bird-Carreau
    model = TurbulenceModel.create(
        "generalizedNewtonian", mesh, U, phi,
        viscosityModel="BirdCarreau",
        mu_0=0.05, mu_inf=0.001, lambda_=1.0, n=0.4,
    )
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .turbulence_model import TurbulenceModel

__all__ = [
    "StokesModel",
    "GeneralizedNewtonianModel",
    "PowerLawViscosity",
    "BirdCarreauViscosity",
    "CrossViscosity",
    "CassonViscosity",
    "HerschelBulkleyViscosity",
]

logger = logging.getLogger(__name__)

# Small value to prevent division by zero in strain-rate calculations
_EPS = 1e-30


# ---------------------------------------------------------------------------
# Viscosity model hierarchy
# ---------------------------------------------------------------------------


class ViscosityModelBase:
    """Abstract base for non-Newtonian viscosity constitutive laws.

    Subclasses implement :meth:`mu` to compute the apparent viscosity
    from the shear-strain rate magnitude.
    """

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        """Compute apparent viscosity from shear-strain rate magnitude.

        Args:
            gamma_dot: |gamma_dot| = sqrt(2 * S_ij * S_ij) ``(n_cells,)``.

        Returns:
            ``(n_cells,)`` apparent dynamic viscosity.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PowerLawViscosity(ViscosityModelBase):
    """Power-law viscosity model.

    mu = K * |gamma_dot|^(n - 1)

    Parameters
    ----------
    K : float
        Consistency index (Pa*s^n).  Default 0.01.
    n : float
        Power-law index.  n < 1: shear-thinning; n > 1: shear-thickening.
        Default 1.0 (Newtonian).
    """

    def __init__(self, K: float = 0.01, n: float = 1.0) -> None:
        self.K = K
        self.n = n

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        gd = gamma_dot.clamp(min=_EPS)
        return self.K * gd.pow(self.n - 1.0)

    def __repr__(self) -> str:
        return f"PowerLawViscosity(K={self.K}, n={self.n})"


class BirdCarreauViscosity(ViscosityModelBase):
    """Bird-Carreau viscosity model.

    mu = mu_inf + (mu_0 - mu_inf) * (1 + (lambda * |gamma_dot|)^2)^((n-1)/2)

    Parameters
    ----------
    mu_0 : float
        Zero-shear viscosity.  Default 0.05.
    mu_inf : float
        Infinite-shear viscosity.  Default 0.001.
    lambda_ : float
        Time constant.  Default 1.0.
    n : float
        Power-law index.  Default 0.4.
    """

    def __init__(
        self,
        mu_0: float = 0.05,
        mu_inf: float = 0.001,
        lambda_: float = 1.0,
        n: float = 0.4,
    ) -> None:
        self.mu_0 = mu_0
        self.mu_inf = mu_inf
        self.lambda_ = lambda_
        self.n = n

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        gd = gamma_dot.clamp(min=0.0)
        factor = (1.0 + (self.lambda_ * gd).pow(2)).pow((self.n - 1.0) / 2.0)
        return self.mu_inf + (self.mu_0 - self.mu_inf) * factor

    def __repr__(self) -> str:
        return (
            f"BirdCarreauViscosity(mu_0={self.mu_0}, mu_inf={self.mu_inf}, "
            f"lambda_={self.lambda_}, n={self.n})"
        )


class CrossViscosity(ViscosityModelBase):
    """Cross viscosity model.

    mu = mu_inf + (mu_0 - mu_inf) / (1 + (lambda * |gamma_dot|)^m)

    Parameters
    ----------
    mu_0 : float
        Zero-shear viscosity.  Default 0.05.
    mu_inf : float
        Infinite-shear viscosity.  Default 0.001.
    lambda_ : float
        Time constant.  Default 1.0.
    m : float
        Exponent.  Default 1.0.
    """

    def __init__(
        self,
        mu_0: float = 0.05,
        mu_inf: float = 0.001,
        lambda_: float = 1.0,
        m: float = 1.0,
    ) -> None:
        self.mu_0 = mu_0
        self.mu_inf = mu_inf
        self.lambda_ = lambda_
        self.m = m

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        gd = gamma_dot.clamp(min=0.0)
        denom = 1.0 + (self.lambda_ * gd).pow(self.m)
        return self.mu_inf + (self.mu_0 - self.mu_inf) / denom

    def __repr__(self) -> str:
        return (
            f"CrossViscosity(mu_0={self.mu_0}, mu_inf={self.mu_inf}, "
            f"lambda_={self.lambda_}, m={self.m})"
        )


class CassonViscosity(ViscosityModelBase):
    """Casson viscosity model.

    mu = (sqrt(tau_y / |gamma_dot|) + sqrt(mu_p))^2

    Regularised for |gamma_dot| -> 0 by clamping.

    Parameters
    ----------
    tau_y : float
        Yield stress (Pa).  Default 0.01.
    mu_p : float
        Plastic viscosity (Pa*s).  Default 0.001.
    """

    def __init__(self, tau_y: float = 0.01, mu_p: float = 0.001) -> None:
        self.tau_y = tau_y
        self.mu_p = mu_p

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        gd = gamma_dot.clamp(min=_EPS)
        term = torch.sqrt(self.tau_y / gd) + self.mu_p ** 0.5
        return term.pow(2)

    def __repr__(self) -> str:
        return f"CassonViscosity(tau_y={self.tau_y}, mu_p={self.mu_p})"


class HerschelBulkleyViscosity(ViscosityModelBase):
    """Herschel-Bulkley viscosity model.

    mu = tau_y / |gamma_dot| + k * |gamma_dot|^(n - 1)

    Regularised for |gamma_dot| -> 0 by clamping.

    Parameters
    ----------
    tau_y : float
        Yield stress (Pa).  Default 0.01.
    K : float
        Consistency index (Pa*s^n).  Default 0.01.
    n : float
        Power-law index.  Default 0.5.
    """

    def __init__(
        self, tau_y: float = 0.01, K: float = 0.01, n: float = 0.5,
    ) -> None:
        self.tau_y = tau_y
        self.K = K
        self.n = n

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        gd = gamma_dot.clamp(min=_EPS)
        return self.tau_y / gd + self.K * gd.pow(self.n - 1.0)

    def __repr__(self) -> str:
        return (
            f"HerschelBulkleyViscosity(tau_y={self.tau_y}, "
            f"K={self.K}, n={self.n})"
        )


# ---------------------------------------------------------------------------
# Viscosity model registry
# ---------------------------------------------------------------------------

_VISCOSITY_MODELS: dict[str, type] = {
    "powerLaw": PowerLawViscosity,
    "BirdCarreau": BirdCarreauViscosity,
    "Cross": CrossViscosity,
    "Casson": CassonViscosity,
    "HerschelBulkley": HerschelBulkleyViscosity,
}


def _create_viscosity_model(name: str, **kwargs: Any) -> ViscosityModelBase:
    """Factory for viscosity constitutive laws.

    Args:
        name: Model name (``"powerLaw"``, ``"BirdCarreau"``, etc.).
        **kwargs: Model parameters.

    Returns:
        Instantiated viscosity model.

    Raises:
        KeyError: If *name* is not recognised.
    """
    if name not in _VISCOSITY_MODELS:
        available = sorted(_VISCOSITY_MODELS.keys())
        raise KeyError(
            f"Unknown viscosity model '{name}'. Available: {available}"
        )
    cls = _VISCOSITY_MODELS[name]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Stokes laminar model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("Stokes")
class StokesModel(TurbulenceModel):
    """Stokes (Newtonian) laminar model.

    The simplest possible "turbulence" model: the turbulent viscosity
    is zero everywhere, so the effective viscosity is just the
    molecular viscosity mu = rho * nu.

    This model is useful for low-Re laminar Stokes flows where no
    turbulence modelling is needed.

    Usage::

        model = TurbulenceModel.create("Stokes", mesh, U, phi)
        model.correct()          # no-op
        nut = model.nut()        # zeros
    """

    def __init__(self, mesh: Any, U: Any, phi: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(mesh, U, phi)
        self._nut = torch.zeros(
            mesh.n_cells, device=self._device, dtype=self._dtype,
        )

    def nut(self) -> torch.Tensor:
        """Return zero turbulent viscosity."""
        return self._nut

    def k(self) -> torch.Tensor:
        """Return zero turbulent kinetic energy."""
        return self._nut.clone()

    def correct(self) -> None:
        """No-op for Stokes model."""
        pass

    def __repr__(self) -> str:
        return f"StokesModel(n_cells={self._mesh.n_cells})"


# ---------------------------------------------------------------------------
# Generalised-Newtonian model
# ---------------------------------------------------------------------------


@TurbulenceModel.register("generalizedNewtonian")
class GeneralizedNewtonianModel(TurbulenceModel):
    """Generalised-Newtonian laminar model.

    The apparent viscosity depends on the local shear-strain rate
    magnitude.  The "turbulent viscosity" returned by :meth:`nut` is
    the non-Newtonian contribution:

        nut_effective = mu_apparent(|gamma_dot|) / rho - nu

    where nu is the molecular kinematic viscosity.  If the apparent
    viscosity is less than the molecular viscosity, nut is clamped to
    zero.

    The viscosity constitutive law is selected via the ``viscosityModel``
    keyword:

    - ``"powerLaw"`` — K, n
    - ``"BirdCarreau"`` — mu_0, mu_inf, lambda_, n
    - ``"Cross"`` — mu_0, mu_inf, lambda_, m
    - ``"Casson"`` — tau_y, mu_p
    - ``"HerschelBulkley"`` — tau_y, K, n

    Usage::

        model = TurbulenceModel.create(
            "generalizedNewtonian", mesh, U, phi,
            viscosityModel="powerLaw", K=0.01, n=0.6,
        )
        model.correct()
        nut = model.nut()  # non-Newtonian contribution
    """

    def __init__(self, mesh: Any, U: Any, phi: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(mesh, U, phi)

        # Parse viscosity model name; default to powerLaw
        visco_name = kwargs.pop("viscosityModel", "powerLaw")
        self._visco_model = _create_viscosity_model(visco_name, **kwargs)

        # Cache for strain rate magnitude and effective nut
        self._mag_S: torch.Tensor | None = None
        self._nut: torch.Tensor | None = None

    @property
    def viscosity_model(self) -> ViscosityModelBase:
        """The underlying constitutive viscosity model."""
        return self._visco_model

    def nut(self) -> torch.Tensor:
        """Return the non-Newtonian turbulent viscosity contribution.

        nut = max(mu_apparent / rho - nu, 0)

        Returns:
            ``(n_cells,)`` effective non-Newtonian kinematic viscosity.
        """
        if self._nut is None:
            return torch.zeros(
                self._mesh.n_cells,
                device=self._device, dtype=self._dtype,
            )
        return self._nut

    def k(self) -> torch.Tensor:
        """Return zero turbulent kinetic energy (laminar model)."""
        return torch.zeros(
            self._mesh.n_cells,
            device=self._device, dtype=self._dtype,
        )

    def correct(self) -> None:
        """Update the non-Newtonian viscosity from the current velocity.

        Computes the strain-rate magnitude |gamma_dot| = sqrt(2 S_ij S_ij)
        and evaluates the constitutive law.
        """
        self._compute_strain_rate_magnitude()
        # Apparent kinematic viscosity = mu(gamma_dot) / rho
        # Here we assume rho = 1 (incompressible, kinematic formulation)
        mu_app = self._visco_model.mu(self._mag_S)
        # nut = max(mu_app - nu, 0)
        self._nut = (mu_app - self._nu).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Strain rate computation
    # ------------------------------------------------------------------

    def _compute_strain_rate_magnitude(self) -> None:
        """Compute |gamma_dot| = sqrt(2 S_ij S_ij).

        Uses a simplified gradient computation based on finite differences
        of the velocity field stored in self._U.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        dtype = self._dtype
        device = self._device

        U = self._U.to(device=device, dtype=dtype)

        if n_internal == 0:
            self._mag_S = torch.zeros(n_cells, dtype=dtype, device=device)
            return

        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Face weights
        if hasattr(mesh, 'face_weights'):
            w = mesh.face_weights[:n_internal].to(dtype=dtype)
        else:
            w = torch.full((n_internal,), 0.5, dtype=dtype, device=device)

        # Velocity at face: interpolated
        U_P = U[int_owner]  # (n_internal, 3)
        U_N = U[int_neigh]  # (n_internal, 3)
        U_face = w.unsqueeze(-1) * U_P + (1.0 - w).unsqueeze(-1) * U_N

        # Face area vectors (not just magnitudes)
        face_areas = mesh.face_areas[:n_internal].to(dtype=dtype)  # (n_internal, 3)

        # Gradient via Gauss: grad(U)_i = (1/V) * sum_f (U_face * A_f)
        # grad_U[c, i, j] = du_i / dx_j
        # Contribution: U_face[:, i] * face_areas[:, j]
        grad_U = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        for i in range(3):
            for j in range(3):
                contrib = U_face[:, i] * face_areas[:, j]
                grad_U[:, i, j] = grad_U[:, i, j].index_add(0, int_owner, contrib)
                grad_U[:, i, j] = grad_U[:, i, j].index_add(0, int_neigh, -contrib)

        V = mesh.cell_volumes.to(dtype=dtype).clamp(min=_EPS)
        grad_U = grad_U / V.unsqueeze(-1).unsqueeze(-1)

        # Strain rate: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        S = 0.5 * (grad_U + grad_U.transpose(-1, -2))

        # |gamma_dot| = sqrt(2 * S_ij * S_ij)
        S_sq = (S * S).sum(dim=(-2, -1))
        self._mag_S = (2.0 * S_sq).clamp(min=0.0).sqrt()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GeneralizedNewtonianModel("
            f"viscosity_model={self._visco_model!r}, "
            f"n_cells={self._mesh.n_cells})"
        )
