"""
广义牛顿粘度模型补全。

对应 OpenFOAM-13 的 MomentumTransportModels/viscosityModels/generalisedNewtonian/。
补全 BirdCarreau、HerschelBulkley、CrossPowerLaw、Casson 模型。
"""
from __future__ import annotations

from typing import Optional

import torch

from pyfoam.core.dtype import CFD_DTYPE


class BirdCarreauModel:
    """Bird-Carreau 粘度模型。

    mu = mu_inf + (mu_0 - mu_inf) * (1 + (lambda * gamma_dot)^2)^((n-1)/2)
    """

    def __init__(
        self,
        mu_0: float = 0.1,
        mu_inf: float = 0.001,
        lambda_p: float = 1.0,
        n: float = 0.5,
    ):
        self._mu_0 = mu_0
        self._mu_inf = mu_inf
        self._lambda = lambda_p
        self._n = n

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        """计算粘度。

        Args:
            gamma_dot: 剪切速率 ``(n_cells,)``。

        Returns:
            粘度 ``(n_cells,)``。
        """
        lam_gd = self._lambda * gamma_dot.abs()
        return self._mu_inf + (self._mu_0 - self._mu_inf) * (
            1 + lam_gd.pow(2)
        ).pow((self._n - 1) / 2)


class HerschelBulkleyModel:
    """Herschel-Bulkley 粘度模型。

    mu = tau_y / gamma_dot + k * gamma_dot^(n-1)  (gamma_dot > 0)
    mu = mu_max                                        (gamma_dot = 0)
    """

    def __init__(
        self,
        tau_y: float = 1.0,
        k: float = 0.1,
        n: float = 0.5,
        mu_max: float = 1e6,
    ):
        self._tau_y = tau_y
        self._k = k
        self._n = n
        self._mu_max = mu_max

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        """计算粘度。"""
        gd_abs = gamma_dot.abs().clamp(min=1e-30)
        mu = self._tau_y / gd_abs + self._k * gd_abs.pow(self._n - 1)
        return mu.clamp(max=self._mu_max)


class CrossPowerLawModel:
    """Cross 幂律粘度模型。

    mu = mu_inf + (mu_0 - mu_inf) / (1 + (lambda * gamma_dot)^m)
    """

    def __init__(
        self,
        mu_0: float = 0.1,
        mu_inf: float = 0.001,
        lambda_p: float = 1.0,
        m: float = 0.8,
    ):
        self._mu_0 = mu_0
        self._mu_inf = mu_inf
        self._lambda = lambda_p
        self._m = m

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        """计算粘度。"""
        lam_gd = self._lambda * gamma_dot.abs()
        return self._mu_inf + (self._mu_0 - self._mu_inf) / (1 + lam_gd.pow(self._m))


class CassonModel:
    """Casson 粘度模型。

    mu = (sqrt(tau_y / gamma_dot) + sqrt(k))^2  (gamma_dot > 0)
    """

    def __init__(self, tau_y: float = 1.0, k: float = 0.01, mu_max: float = 1e6):
        self._tau_y = tau_y
        self._k = k
        self._mu_max = mu_max

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        """计算粘度。"""
        gd_abs = gamma_dot.abs().clamp(min=1e-30)
        mu = (torch.sqrt(self._tau_y / gd_abs) + torch.sqrt(torch.tensor(self._k))).pow(2)
        return mu.clamp(max=self._mu_max)
