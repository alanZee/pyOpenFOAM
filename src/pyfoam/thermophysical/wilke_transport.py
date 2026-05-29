"""
Wilke mixing rule for gas mixture transport properties.

Implements the Wilke (1950) semi-empirical mixing rule for computing
the dynamic viscosity and thermal conductivity of a gas mixture from
the pure-component properties and mole fractions.

References:
    Wilke, C.R. (1950). "A Viscosity Equation for Gas Mixtures."
    *J. Chem. Phys.*, 18(4), 517-519.

Usage::

    from pyfoam.thermophysical.wilke_transport import WilkeTransport
    from pyfoam.thermophysical.transport_model import ConstantViscosity

    # Binary mixture: N2 (79%) + O2 (21%)
    wilke = WilkeTransport(
        transport_models=[ConstantViscosity(mu=1.76e-5), ConstantViscosity(mu=2.05e-5)],
        Mw=[28.0, 32.0],
    )
    mu_mix = wilke.mu(T=300.0, x=[0.79, 0.21])
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel

__all__ = ["WilkeTransport"]

logger = logging.getLogger(__name__)


class WilkeTransport:
    """Wilke mixing rule for gas mixture viscosity and conductivity.

    Computes mixture viscosity using:

    .. math::

        \\mu_{\\text{mix}} = \\sum_i \\frac{x_i \\mu_i}{\\sum_j x_j \\Phi_{ij}}

    where the interaction parameter is:

    .. math::

        \\Phi_{ij} = \\frac{1}{\\sqrt{8}}
        \\left(1 + \\frac{M_i}{M_j}\\right)^{-1/2}
        \\left[1 + \\left(\\frac{\\mu_i}{\\mu_j}\\right)^{1/2}
        \\left(\\frac{M_j}{M_i}\\right)^{1/4}\\right]^2

    Parameters
    ----------
    transport_models : sequence of TransportModel
        One transport model per species, providing pure-component viscosity.
    Mw : sequence of float
        Molecular weights (g/mol) for each species.

    Examples::

        from pyfoam.thermophysical.transport_model import Sutherland

        wilke = WilkeTransport(
            transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5, T_ref=273.15, S=139.0)],
            Mw=[28.014, 31.998],
        )
        mu_mix = wilke.mu(T=300.0, x=[0.79, 0.21])
    """

    def __init__(
        self,
        transport_models: Sequence[TransportModel],
        Mw: Sequence[float],
    ) -> None:
        n = len(transport_models)
        if n == 0:
            raise ValueError("transport_models must not be empty")
        if len(Mw) != n:
            raise ValueError(
                f"Mw length ({len(Mw)}) must match transport_models length ({n})"
            )
        for i, mw in enumerate(Mw):
            if mw <= 0:
                raise ValueError(f"Mw[{i}] must be positive, got {mw}")

        self._models = list(transport_models)
        self._Mw = list(Mw)
        self._n_species = n

    @property
    def n_species(self) -> int:
        """Number of species in the mixture."""
        return self._n_species

    @property
    def Mw(self) -> list[float]:
        """Molecular weights (g/mol) for each species."""
        return self._Mw.copy()

    def _wilke_phi(
        self,
        mu_i: torch.Tensor,
        mu_j: torch.Tensor,
        Mw_i: float,
        Mw_j: float,
    ) -> torch.Tensor:
        """Compute Wilke interaction parameter Phi_ij.

        Args:
            mu_i: Viscosity of species i (scalar or batch).
            mu_j: Viscosity of species j (scalar or batch).
            Mw_i: Molecular weight of species i.
            Mw_j: Molecular weight of species j.

        Returns:
            Phi_ij (same shape as mu_i, mu_j).
        """
        Mw_ratio = Mw_i / Mw_j
        mu_ratio = mu_i / mu_j.clamp(min=1e-30)

        phi = (1.0 / (8.0 ** 0.5)) * (1.0 + Mw_ratio) ** (-0.5) * \
              (1.0 + mu_ratio.sqrt() * (Mw_j / Mw_i) ** 0.25) ** 2

        return phi

    def mu(
        self,
        T: torch.Tensor | float,
        x: Sequence[float],
    ) -> torch.Tensor:
        """Compute mixture dynamic viscosity using Wilke's rule.

        Args:
            T: Temperature (K) — scalar or ``(n_cells,)`` tensor.
            x: Mole fractions for each species (must sum to ~1).

        Returns:
            Mixture dynamic viscosity (Pa·s).
        """
        if len(x) != self._n_species:
            raise ValueError(
                f"x length ({len(x)}) must equal n_species ({self._n_species})"
            )

        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        # 计算纯组分粘度
        mu_pure = [model.mu(T) for model in self._models]

        # 组装 mole fraction 张量
        x_t = torch.tensor(x, dtype=dtype, device=device)  # (n_species,)

        # 计算 Phi_ij —— 形状 (n_species, n_species) 或 (n_species, n_species, n_cells)
        n = self._n_species
        is_batch = mu_pure[0].dim() > 0
        if is_batch:
            n_cells = mu_pure[0].shape[0]
            Phi = torch.zeros(n, n, n_cells, dtype=dtype, device=device)
        else:
            Phi = torch.zeros(n, n, dtype=dtype, device=device)

        for i in range(n):
            for j in range(n):
                Phi[i, j] = self._wilke_phi(
                    mu_pure[i],
                    mu_pure[j],
                    self._Mw[i],
                    self._Mw[j],
                )

        # μ_mix = Σ_i (x_i * μ_i) / Σ_j (x_j * Φ_ij)
        # denom_i = Σ_j x_j * Phi[i,j]
        if is_batch:
            # Phi: (n, n, n_cells), x_t: (n,) -> denom: (n, n_cells)
            denom = torch.einsum('ijc,j->ic', Phi, x_t)
            # mu_stack: (n, n_cells)
            mu_stack = torch.stack(mu_pure, dim=0)
            numerator = (x_t.unsqueeze(-1) * mu_stack).sum(dim=0)
            denominator = (x_t.unsqueeze(-1) * denom).sum(dim=0)
        else:
            denom = torch.matmul(Phi, x_t)  # (n,)
            mu_stack = torch.stack(mu_pure, dim=0)  # (n,)
            numerator = (x_t * mu_stack).sum()
            denominator = (x_t * denom).sum()

        mu_mix = numerator / denominator.clamp(min=1e-30)
        return mu_mix

    def kappa(
        self,
        T: torch.Tensor | float,
        x: Sequence[float],
        Cp: float | Sequence[float] = 1005.0,
        Pr: float | Sequence[float] = 0.7,
    ) -> torch.Tensor:
        """Compute mixture thermal conductivity.

        Uses :math:`\\kappa = \\mu_{\\text{mix}} C_{p,\\text{mix}} / Pr_{\\text{mix}}`
        with mass-weighted Cp and mole-weighted Pr.

        Args:
            T: Temperature (K).
            x: Mole fractions for each species.
            Cp: Specific heat (J/(kg·K)). Scalar or per-species sequence.
            Pr: Prandtl number. Scalar or per-species sequence.

        Returns:
            Mixture thermal conductivity (W/(m·K)).
        """
        mu_mix = self.mu(T, x)

        if isinstance(Cp, (list, tuple)):
            x_t = torch.tensor(x, dtype=get_default_dtype(), device=get_device())
            Cp_arr = torch.tensor(Cp, dtype=get_default_dtype(), device=get_device())
            Cp_mix = float((x_t * Cp_arr).sum().item())
        else:
            Cp_mix = Cp

        if isinstance(Pr, (list, tuple)):
            x_t = torch.tensor(x, dtype=get_default_dtype(), device=get_device())
            Pr_arr = torch.tensor(Pr, dtype=get_default_dtype(), device=get_device())
            Pr_mix = float((x_t * Pr_arr).sum().item())
        else:
            Pr_mix = Pr

        return mu_mix * Cp_mix / Pr_mix

    def __repr__(self) -> str:
        model_names = [type(m).__name__ for m in self._models]
        return f"WilkeTransport(n_species={self._n_species}, models={model_names})"
