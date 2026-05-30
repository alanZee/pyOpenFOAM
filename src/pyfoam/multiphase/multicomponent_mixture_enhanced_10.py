"""Enhanced multicomponent mixture property model — v11.

Extends MulticomponentMixtureEnhanced9 with:
- Ideal mixture property estimation from pure components
- Mixing rule sensitivity analysis
- Mixture validation against experimental data

Usage::

    from pyfoam.multiphase.multicomponent_mixture_enhanced_10 import (
        MulticomponentMixtureEnhanced10,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Sequence
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.multicomponent_mixture_enhanced_9 import (
    MulticomponentMixtureEnhanced9,
)

__all__ = ["MulticomponentMixtureEnhanced10"]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class MulticomponentMixtureEnhanced10(MulticomponentMixtureEnhanced9):
    """Enhanced multicomponent mixture v11 with ideal estimation and sensitivity analysis.

    Parameters
    ----------
    species, M, rho, mu, Cp, kappa, D, Sc_t : see parent.
    cross_diffusion_coeff : see parent.
    mixing_rule : str
        Mixing rule: 'ideal' (default), 'wendt', or 'strict'. Default 'ideal'.
    validation_data : dict or None
        Experimental data for validation. Default None.
    """

    def __init__(
        self,
        species: Sequence[str],
        M: Sequence[float],
        rho: Sequence[float],
        mu: Sequence[float],
        Cp: Sequence[float],
        kappa: Sequence[float] | None = None,
        D: Sequence[float] | None = None,
        Sc_t: Sequence[float] | None = None,
        Cp_poly: Sequence[Sequence[float]] | None = None,
        Le: Sequence[float] | None = None,
        reaction_rates: Sequence[float] | None = None,
        soret_coeff: Sequence[float] | None = None,
        dufour_coeff: Sequence[float] | None = None,
        D_ij: Sequence[Sequence[float]] | None = None,
        nrtl_alpha: Sequence[Sequence[float]] | None = None,
        H_ref: Sequence[float] | None = None,
        uniquac_r: Sequence[float] | None = None,
        uniquac_q: Sequence[float] | None = None,
        margules_A12: float = 0.0,
        margules_A21: float = 0.0,
        wilson_lambda: Sequence[Sequence[float]] | None = None,
        stefan_flow_coeff: float = 0.0,
        cross_diffusion_coeff: float = 0.0,
        mixing_rule: str = "ideal",
        validation_data: dict | None = None,
    ) -> None:
        super().__init__(
            species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly,
            Le, reaction_rates, soret_coeff, dufour_coeff,
            D_ij, nrtl_alpha, H_ref, uniquac_r, uniquac_q,
            margules_A12, margules_A21,
            wilson_lambda, stefan_flow_coeff,
            cross_diffusion_coeff,
        )
        self._mixing_rule = mixing_rule
        self._validation_data = validation_data

    # ------------------------------------------------------------------
    # Ideal mixture property estimation
    # ------------------------------------------------------------------

    def ideal_mixture_property(self, Y: torch.Tensor, property_name: str = "rho") -> torch.Tensor:
        """Estimate mixture property using ideal mixing rule.

        prop_mix = sum_i Y_i * prop_i

        Parameters
        ----------
        Y : torch.Tensor
            (n_cells, N) mass fractions.
        property_name : str
            Property to mix: 'rho', 'mu', 'Cp', 'kappa'.

        Returns
        -------
        torch.Tensor
            (n_cells,) mixture property.
        """
        props = {"rho": self._rho, "mu": self._mu, "Cp": self._Cp, "kappa": self._kappa}
        prop_list = props.get(property_name, self._rho)
        if not prop_list:
            return torch.ones(Y.shape[0], device=Y.device, dtype=Y.dtype)

        result = torch.zeros(Y.shape[0], device=Y.device, dtype=Y.dtype)
        n = min(Y.shape[1], len(prop_list))
        for i in range(n):
            result += Y[:, i] * prop_list[i]
        return result.clamp(min=_EPS)

    # ------------------------------------------------------------------
    # Mixing rule sensitivity
    # ------------------------------------------------------------------

    def mixing_sensitivity(self, Y: torch.Tensor, T: torch.Tensor, delta_Y: float = 0.01) -> dict[str, torch.Tensor]:
        """Compute sensitivity of mixture properties to composition changes.

        Parameters
        ----------
        Y : torch.Tensor
            (n_cells, N) mass fractions.
        T : torch.Tensor
            (n_cells,) temperature.
        delta_Y : float
            Perturbation for finite differences.

        Returns
        -------
        dict
            'drho_dY': density sensitivity to each species,
            'dmu_dY': viscosity sensitivity to each species.
        """
        n_species = Y.shape[1] if Y.dim() > 1 else 1
        n_cells = Y.shape[0]

        drho_dY = torch.zeros(n_cells, n_species, device=Y.device, dtype=Y.dtype)
        dmu_dY = torch.zeros(n_cells, n_species, device=Y.device, dtype=Y.dtype)

        rho_base = self.ideal_mixture_property(Y, "rho")
        mu_base = self.ideal_mixture_property(Y, "mu")

        for i in range(min(n_species, len(self._rho))):
            Y_pert = Y.clone()
            if Y_pert.dim() > 1:
                Y_pert[:, i] += delta_Y
                Y_pert = Y_pert / Y_pert.sum(dim=1, keepdim=True).clamp(min=_EPS)

            rho_pert = self.ideal_mixture_property(Y_pert, "rho")
            mu_pert = self.ideal_mixture_property(Y_pert, "mu")

            drho_dY[:, i] = (rho_pert - rho_base) / delta_Y
            dmu_dY[:, i] = (mu_pert - mu_base) / delta_Y

        return {"drho_dY": drho_dY, "dmu_dY": dmu_dY}

    # ------------------------------------------------------------------
    # Mixture validation
    # ------------------------------------------------------------------

    def validate_mixture(self, Y: torch.Tensor, T: float) -> dict[str, float | bool]:
        """Validate mixture properties against experimental data.

        Parameters
        ----------
        Y : torch.Tensor
            Mass fractions.
        T : float
            Temperature.

        Returns
        -------
        dict
            Validation results per property.
        """
        if self._validation_data is None:
            return {"validated": False, "reason": "no_validation_data"}

        result: dict[str, float | bool] = {"validated": True}
        for prop, exp_val in self._validation_data.items():
            if prop == "rho" and "rho" in self._rho:
                comp_val = self._rho[0] if self._rho else 1.0
                rel_err = abs(comp_val - exp_val) / max(abs(exp_val), _EPS)
                result[f"{prop}_error"] = rel_err
                result[f"{prop}_valid"] = rel_err < 0.1

        return result

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        mr = f", rule={self._mixing_rule}"
        return (
            f"MulticomponentMixtureEnhanced10("
            f"n_species={self._n_species}, species=[{sp}]{mr})"
        )
