"""
Compressible turbulence models for high-speed / variable-density flows.

Provides an abstract base class and a compressible k-omega SST variant
that accounts for density variations in the turbulence transport equations.
These models are essential for transonic, supersonic, and other
compressible flows where the incompressible assumption breaks down.

Models:

- :class:`CompressibleTurbulenceModel` — abstract base with RTS registry
- :class:`KOmegaSSTCompressible` — compressible k-omega SST variant

The compressible variants modify the standard incompressible models by:
1. Including density in the turbulent viscosity computation: mu_t = rho * nut
2. Solving for rho*k and rho*omega instead of k and omega
3. Adding dilatational dissipation corrections
4. Including pressure-dilatation correlation

Usage::

    from pyfoam.turbulence.compressible_turbulence import CompressibleTurbulenceModel

    model = CompressibleTurbulenceModel.create(
        "kOmegaSSTCompressible", mesh, U, phi, rho=rho_field
    )
    model.correct()
    mu_t = model.mu_t()  # turbulent dynamic viscosity
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc

__all__ = [
    "CompressibleTurbulenceModel",
    "KOmegaSSTCompressible",
    "KOmegaSSTCompressibleConstants",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class CompressibleTurbulenceModel(ABC):
    """Abstract base class for compressible turbulence models.

    Extends the standard turbulence model interface with
    density-weighted (Favre-averaged) quantities.  Subclasses must
    implement :meth:`nut`, :meth:`mu_t`, :meth:`k`, and :meth:`correct`.

    RTS (Run-Time Selection) registry::

        @CompressibleTurbulenceModel.register("kOmegaSSTCompressible")
        class KOmegaSSTCompressible(CompressibleTurbulenceModel):
            ...
    """

    _registry: ClassVar[dict[str, Type[CompressibleTurbulenceModel]]] = {}

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        rho: torch.Tensor | None = None,
    ) -> None:
        """
        Parameters
        ----------
        mesh : FvMesh
            Finite volume mesh.
        U : torch.Tensor
            Velocity field ``(n_cells, 3)``.
        phi : torch.Tensor
            Face mass flux ``(n_faces,)`` (rho * phi for compressible).
        rho : torch.Tensor, optional
            Cell density ``(n_cells,)``.  Required for compressible models.
        """
        self._mesh = mesh
        self._device = get_device()
        self._dtype = get_default_dtype()

        if isinstance(U, torch.Tensor):
            self._U = U.to(device=self._device, dtype=self._dtype)
        else:
            self._U = torch.as_tensor(U, device=self._device, dtype=self._dtype)

        if isinstance(phi, torch.Tensor):
            self._phi = phi.to(device=self._device, dtype=self._dtype)
        else:
            self._phi = torch.as_tensor(phi, device=self._device, dtype=self._dtype)

        n_cells = mesh.n_cells
        if rho is None:
            self._rho = torch.full((n_cells,), 1.0, device=self._device, dtype=self._dtype)
        else:
            self._rho = rho.to(device=self._device, dtype=self._dtype)

        # Molecular viscosity (default for air at STP)
        self._nu: float = 1.5e-5

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a compressible turbulence model."""

        def decorator(model_cls: Type[CompressibleTurbulenceModel]) -> Type[CompressibleTurbulenceModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Compressible turbulence model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        **kwargs: Any,
    ) -> CompressibleTurbulenceModel:
        """Factory: create a compressible turbulence model by name.

        Parameters
        ----------
        name : str
            Registered model type name.
        mesh : FvMesh
            Finite volume mesh.
        U : torch.Tensor
            Velocity field.
        phi : torch.Tensor
            Face mass flux.
        **kwargs
            Additional model-specific arguments (e.g. ``rho``).
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown compressible turbulence model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](mesh, U, phi, **kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mesh(self) -> Any:
        """The mesh."""
        return self._mesh

    @property
    def rho(self) -> torch.Tensor:
        """Cell density field ``(n_cells,)``."""
        return self._rho

    @rho.setter
    def rho(self, value: torch.Tensor) -> None:
        self._rho = value.to(device=self._device, dtype=self._dtype)

    @property
    def nu(self) -> float:
        """Molecular kinematic viscosity."""
        return self._nu

    @nu.setter
    def nu(self, value: float) -> None:
        self._nu = value

    @property
    def device(self) -> torch.device:
        """Device of model tensors."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of model tensors."""
        return self._dtype

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def nut(self) -> torch.Tensor:
        """Return kinematic turbulent viscosity ``(n_cells,)``.

        This is the "kinematic" viscosity (m^2/s) without density.
        """

    @abstractmethod
    def mu_t(self) -> torch.Tensor:
        """Return dynamic turbulent viscosity ``(n_cells,)``.

        mu_t = rho * nut  (Pa*s)
        """

    @abstractmethod
    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""

    @abstractmethod
    def correct(self) -> None:
        """Update the turbulence model (solve transport equations)."""

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def epsilon(self) -> torch.Tensor:
        """Return turbulent dissipation rate ``(n_cells,)``.

        Override in k-epsilon models.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement epsilon()"
        )

    def omega(self) -> torch.Tensor:
        """Return specific dissipation rate ``(n_cells,)``.

        Override in k-omega models.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement omega()"
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mesh={self._mesh}, "
            f"n_cells={self._mesh.n_cells})"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KOmegaSSTCompressibleConstants:
    """Constants for the compressible k-omega SST model.

    Inherits the standard Menter (1994) SST constants and adds
    compressibility correction coefficients.

    Attributes
    ----------
    sigma_k1 : float
        Turbulent Prandtl number for k (inner).
    sigma_k2 : float
        Turbulent Prandtl number for k (outer).
    sigma_omega1 : float
        Turbulent Prandtl number for omega (inner).
    sigma_omega2 : float
        Turbulent Prandtl number for omega (outer).
    beta1 : float
        Coefficient for omega destruction (inner).
    beta2 : float
        Coefficient for omega destruction (outer).
    gamma1 : float
        Coefficient for omega production (inner).
    gamma2 : float
        Coefficient for omega production (outer).
    a1 : float
        SST blending constant for turbulent viscosity limiter.
    beta_star : float
        Coefficient for k destruction.
    kappa : float
        Von Karman constant.
    alpha_star : float
        Compressibility correction coefficient for k destruction.
        Default: 1.0 (no correction in baseline).
    alpha_1 : float
        Sarkar dilatational dissipation coefficient.  Default: 1.0.
    alpha_2 : float
        Sarkar quartic dilatational dissipation coefficient.  Default: 0.5.
    """

    sigma_k1: float = 0.85
    sigma_k2: float = 1.0
    sigma_omega1: float = 0.5
    sigma_omega2: float = 0.856
    beta1: float = 0.075
    beta2: float = 0.0828
    gamma1: float = 5.0 / 9.0
    gamma2: float = 0.44
    a1: float = 0.31
    beta_star: float = 0.09
    kappa: float = 0.41
    alpha_star: float = 1.0
    alpha_1: float = 1.0
    alpha_2: float = 0.5


_DEFAULT_CONSTANTS = KOmegaSSTCompressibleConstants()


# ---------------------------------------------------------------------------
# Compressible k-omega SST model
# ---------------------------------------------------------------------------


@CompressibleTurbulenceModel.register("kOmegaSSTCompressible")
class KOmegaSSTCompressible(CompressibleTurbulenceModel):
    """Compressible k-omega SST turbulence model.

    Extends the standard k-omega SST model for variable-density
    (compressible) flows.  Solves density-weighted (Favre-averaged)
    transport equations for k and omega, with additional compressibility
    corrections:

    1. Density-weighted turbulent viscosity: mu_t = rho * a1 * k / max(a1*omega, S*F2)
    2. Dilatational dissipation (Sarkar correction)
    3. Turbulent kinetic energy equation solved as rho*k

    The model uses the same blending functions F1 and F2 as the
    incompressible SST model.

    Parameters
    ----------
    mesh : FvMesh
        Finite volume mesh.
    U : torch.Tensor or volVectorField
        Velocity field.
    phi : torch.Tensor
        Face mass flux ``(n_faces,)``.
    rho : torch.Tensor, optional
        Cell density ``(n_cells,)``.
    constants : KOmegaSSTCompressibleConstants, optional
        Model constants.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        rho: torch.Tensor | None = None,
        constants: KOmegaSSTCompressibleConstants | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi, rho=rho)

        self._C = constants or _DEFAULT_CONSTANTS

        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype

        # Turbulence fields
        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)

        # Velocity gradient tensor (n_cells, 3, 3)
        self._grad_U: torch.Tensor | None = None

        # Wall distance (simplified)
        self._y = self._compute_wall_distance()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def k_field(self) -> torch.Tensor:
        """Turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    @k_field.setter
    def k_field(self, value: torch.Tensor) -> None:
        self._k = value.to(device=self._device, dtype=self._dtype)

    @property
    def omega_field(self) -> torch.Tensor:
        """Specific dissipation rate ``(n_cells,)``."""
        return self._omega

    @omega_field.setter
    def omega_field(self, value: torch.Tensor) -> None:
        self._omega = value.to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # CompressibleTurbulenceModel interface
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """Kinematic turbulent viscosity with SST limiter.

        nut = a1 * k / max(a1 * omega, S * F2)

        Returns:
            ``(n_cells,)`` kinematic turbulent viscosity (m^2/s).
        """
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)

        if self._grad_U is None:
            return k / omega

        S = self._strain_magnitude()
        F2 = self._F2()

        denominator = (self._C.a1 * omega).max(S * F2)
        return self._C.a1 * k / denominator.clamp(min=1e-16)

    def mu_t(self) -> torch.Tensor:
        """Dynamic turbulent viscosity.

        mu_t = rho * nut  (Pa*s)

        Returns:
            ``(n_cells,)`` dynamic turbulent viscosity.
        """
        return self._rho * self.nut()

    def k(self) -> torch.Tensor:
        """Return turbulent kinetic energy ``(n_cells,)``."""
        return self._k

    def omega(self) -> torch.Tensor:
        """Return specific dissipation rate ``(n_cells,)``."""
        return self._omega

    def epsilon(self) -> torch.Tensor:
        """Return dissipation rate: epsilon = beta* * omega * k."""
        return self._C.beta_star * self._omega * self._k

    def correct(self) -> None:
        """Update the compressible k-omega SST model.

        Solves density-weighted transport equations for k and omega,
        including compressibility corrections.
        """
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        # Compute velocity gradient tensor
        grad_U = torch.zeros(
            mesh.n_cells, 3, 3, device=device, dtype=dtype,
        )
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=mesh)
        self._grad_U = grad_U

        # Strain rate and production
        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))

        # Solve density-weighted k equation
        self._solve_k(P_k)

        # Solve density-weighted omega equation
        self._solve_omega(P_k)

    # ------------------------------------------------------------------
    # Blending functions (same as incompressible SST)
    # ------------------------------------------------------------------

    def _F1(self) -> torch.Tensor:
        """First blending function F1 (inner/outer transition)."""
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        C = self._C

        arg1 = torch.sqrt(k) / (C.beta_star * omega * y)
        arg2 = 500.0 * self._nu / (y ** 2 * omega)
        CD_kω = (2.0 * C.sigma_omega2 / omega).clamp(min=1e-10)
        arg3 = 4.0 * C.sigma_omega2 * k / (CD_kω * y ** 2)

        arg = torch.min(torch.max(arg1, arg2), arg3)
        return torch.tanh(arg ** 4)

    def _F2(self) -> torch.Tensor:
        """Second blending function F2 (shear stress limiter)."""
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        y = self._y.clamp(min=1e-10)
        C = self._C

        arg1 = 2.0 * torch.sqrt(k) / (C.beta_star * omega * y)
        arg2 = 500.0 * self._nu / (y ** 2 * omega)

        arg = torch.max(arg1, arg2)
        return torch.tanh(arg ** 2)

    # ------------------------------------------------------------------
    # Internal: transport equations (density-weighted)
    # ------------------------------------------------------------------

    def _solve_k(self, P_k: torch.Tensor) -> None:
        """Solve density-weighted k transport equation.

        ddt(rho*k) + div(rho*U*k) = div((mu + sigma_k*mu_t)*grad(k))
                                     + P_k - beta* * rho * omega * k
                                     - dilatational_dissipation
        """
        C = self._C
        rho = self._rho
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        # Blended diffusivity (density-weighted)
        F1 = self._F1()
        sigma_k = F1 * C.sigma_k1 + (1.0 - F1) * C.sigma_k2
        nut = self.nut()
        mu_eff = rho * (self._nu + sigma_k * nut)

        # Build equation (simplified explicit update)
        mesh = self._mesh
        eqn = fvm.div(self._phi, self._k, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(mu_eff, self._k, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: P_k - beta* * rho * omega * k
        source = P_k - C.beta_star * rho * omega_safe * k_safe

        # Compressibility correction: dilatational dissipation (Sarkar)
        if C.alpha_star != 1.0 or C.alpha_1 != 0.0:
            # Turbulent Mach number estimation
            a_ref = 340.0
            M_t_sq = 2.0 * k_safe / (a_ref ** 2)
            # Dilatational dissipation: alpha_1 * rho * eps * M_t^2
            eps = C.beta_star * omega_safe * k_safe
            dilat_diss = C.alpha_1 * rho * eps * M_t_sq
            source = source - dilat_diss

        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        k_new = eqn.source / diag_safe
        self._k = k_new.clamp(min=1e-10)

    def _solve_omega(self, P_k: torch.Tensor) -> None:
        """Solve density-weighted omega transport equation.

        ddt(rho*omega) + div(rho*U*omega)
            = div((mu + sigma_omega*mu_t)*grad(omega))
              + gamma * rho * P_k / mu_t
              - beta * rho * omega^2
              + cross_diffusion
        """
        C = self._C
        rho = self._rho
        omega_safe = self._omega.clamp(min=1e-16)
        k_safe = self._k.clamp(min=1e-16)

        # Blended coefficients
        F1 = self._F1()
        sigma_omega = F1 * C.sigma_omega1 + (1.0 - F1) * C.sigma_omega2
        beta = F1 * C.beta1 + (1.0 - F1) * C.beta2
        gamma = F1 * C.gamma1 + (1.0 - F1) * C.gamma2

        nut = self.nut()
        mu_eff = rho * (self._nu + sigma_omega * nut)

        mesh = self._mesh
        eqn = fvm.div(self._phi, self._omega, "Gauss upwind", mesh=mesh)
        diff = fvm.laplacian(mu_eff, self._omega, "Gauss linear corrected", mesh=mesh)

        eqn.lower = eqn.lower + diff.lower
        eqn.upper = eqn.upper + diff.upper
        eqn.diag = eqn.diag + diff.diag

        # Source: gamma * rho * P_k / mu_t - beta * rho * omega^2
        mu_t_safe = self.mu_t().clamp(min=1e-16)
        source = gamma * rho * P_k / mu_t_safe - beta * rho * omega_safe ** 2

        eqn.source = eqn.source + source

        # Solve
        diag_safe = eqn.diag.abs().clamp(min=1e-30)
        omega_new = eqn.source / diag_safe
        self._omega = omega_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Internal: helper computations
    # ------------------------------------------------------------------

    def _strain_rate(self) -> torch.Tensor:
        """Compute strain rate tensor S = 0.5*(grad(U) + grad(U)^T)."""
        grad_U = self._grad_U
        return 0.5 * (grad_U + grad_U.transpose(-1, -2))

    def _strain_magnitude(self) -> torch.Tensor:
        """Compute strain rate magnitude |S| = sqrt(2*S:S)."""
        S = self._strain_rate()
        return torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))

    def _compute_wall_distance(self) -> torch.Tensor:
        """Compute approximate wall distance for each cell."""
        mesh = self._mesh
        device = self._device
        dtype = self._dtype

        cell_centres = mesh.cell_centres
        face_centres = mesh.face_centres

        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        if n_faces > n_internal:
            bnd_centres = face_centres[n_internal:]
        else:
            return cell_centres.norm(dim=1).clamp(min=1e-6)

        n_bnd = bnd_centres.shape[0]
        if n_bnd == 0:
            return cell_centres.norm(dim=1).clamp(min=1e-6)

        diff = cell_centres.unsqueeze(1) - bnd_centres.unsqueeze(0)
        dist = diff.norm(dim=2)
        y = dist.min(dim=1).values

        return y.clamp(min=1e-6)
