"""
Enhanced wave models v2 — irregular, directional, and solitary wave theories.

Extends :class:`~pyfoam.waves.wave_model.WaveModel` with:

- :class:`IrregularWave` — JONSWAP / Pierson-Moskowitz spectrum based irregular waves
- :class:`DirectionalWave` — multi-directional spreading (cos-2s distribution)
- :class:`SolitaryWave` — Boussinesq solitary wave (non-dispersive, shallow water)

References:
    Hasselmann et al. (1973). "Measurements of wind-wave growth and swell decay."
    Pierson & Moskowitz (1964). "A proposed spectral form for fully developed wind seas."
    Boussinesq (1872). "Théorie des ondes et des remous."

Usage::

    from pyfoam.waves.enhanced_2 import IrregularWave, DirectionalWave, SolitaryWave

    wave = IrregularWave(amplitude=1.0, depth=20.0, period=8.0, spectrum="jonswap")
    eta = wave.wave_elevation(x, t=0.0)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from pyfoam.waves.wave_model import GRAVITY, WaveModel

__all__ = ["IrregularWave", "DirectionalWave", "SolitaryWave"]


# ---------------------------------------------------------------------------
# Spectral helper functions
# ---------------------------------------------------------------------------

def _jonswap_spectrum(
    omega: float,
    omega_p: float,
    alpha: float = 0.0081,
    gamma: float = 3.3,
    sigma_a: float = 0.07,
    sigma_b: float = 0.09,
) -> float:
    """JONSWAP spectral density S(omega).

    S(omega) = alpha * g^2 / omega^5
               * exp(-5/4 * (omega_p/omega)^4)
               * gamma^exp(-(omega-omega_p)^2 / (2*sigma^2*omega_p^2))

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    omega_p : float
        Peak angular frequency (rad/s).
    alpha : float
        Phillips constant (default 0.0081).
    gamma : float
        Peak enhancement factor (default 3.3, gamma=1 gives PM spectrum).
    sigma_a, sigma_b : float
        Spectral width parameters for omega < omega_p and omega > omega_p.
    """
    g = GRAVITY
    sigma = sigma_a if omega <= omega_p else sigma_b

    S_pm = alpha * g**2 / omega**5 * math.exp(-1.25 * (omega_p / omega) ** 4)
    r = math.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
    return S_pm * gamma**r


def _pierson_moskowitz_spectrum(
    omega: float,
    omega_p: float,
    alpha: float = 0.0081,
    gamma: float = 1.0,
) -> float:
    """Pierson-Moskowitz spectral density S(omega).

    PM spectrum is the JONSWAP spectrum with gamma=1 (fully developed sea).
    The gamma parameter is accepted for interface compatibility but ignored.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    omega_p : float
        Peak angular frequency (rad/s).
    alpha : float
        Phillips constant.
    gamma : float
        Ignored (PM always uses gamma=1).
    """
    return _jonswap_spectrum(omega, omega_p, alpha=alpha, gamma=1.0)


# ---------------------------------------------------------------------------
# IrregularWave
# ---------------------------------------------------------------------------

@WaveModel.register("irregular")
class IrregularWave(WaveModel):
    """Irregular wave model using spectral superposition (JONSWAP / PM).

    Generates irregular sea states by superposing Airy wave components
    whose amplitudes are drawn from a target spectral density.

    Parameters
    ----------
    amplitude : float
        Significant wave height Hs/2 (m).
    depth : float
        Water depth d (m).
    period : float
        Peak period Tp (s).
    spectrum : str
        Spectrum type: ``"jonswap"`` (default) or ``"pm"`` (Pierson-Moskowitz).
    n_components : int
        Number of spectral components (default 50).
    gamma : float
        JONSWAP peak enhancement factor (default 3.3).
    seed : int, optional
        Random seed for reproducible phase angles.
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        spectrum: str = "jonswap",
        n_components: int = 50,
        gamma: float = 3.3,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._spectrum = spectrum
        self._n_comp = n_components
        self._gamma = gamma

        # 生成分量频率与振幅
        omega_p = 2.0 * math.pi / period
        # 频率范围：0.5*omega_p ~ 3.0*omega_p
        omega_lo = 0.5 * omega_p
        omega_hi = 3.0 * omega_p
        domegas = (omega_hi - omega_lo) / n_components

        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.manual_seed(42)

        # 等间距频率 + 随机相位
        omegas = torch.linspace(
            omega_lo + 0.5 * domegas,
            omega_hi - 0.5 * domegas,
            n_components,
        )
        phases = torch.rand(n_components, generator=rng) * 2.0 * math.pi

        # 根据谱密度计算各分量振幅: A_i = sqrt(2 * S(omega_i) * d_omega)
        spec_fn = _jonswap_spectrum if spectrum == "jonswap" else _pierson_moskowitz_spectrum
        amps = torch.tensor([
            math.sqrt(2.0 * spec_fn(w.item(), omega_p, gamma=self._gamma) * domegas)
            for w in omegas
        ])

        self._omegas = omegas
        self._amps = amps
        self._phases = phases

    @property
    def spectrum_type(self) -> str:
        """返回谱类型名称。"""
        return self._spectrum

    @property
    def n_components(self) -> int:
        """返回分量数量。"""
        return self._n_comp

    def _k_from_omega(self, omega: float) -> float:
        """通过线性弥散关系从 omega 求 k。"""
        g = GRAVITY
        d = self._depth
        k = omega**2 / g  # 深水近似初值
        for _ in range(50):
            tanh_kd = math.tanh(k * d)
            f = g * k * tanh_kd - omega**2
            df = g * (tanh_kd + k * d / math.cosh(k * d) ** 2)
            dk = f / df
            k -= dk
            if abs(dk) < 1e-12:
                break
        return k

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute irregular wave elevation via spectral superposition.

        eta(x, t) = sum_i A_i * cos(k_i*x - omega_i*t + phi_i)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        eta = torch.zeros_like(x, dtype=x.dtype)
        for i in range(self._n_comp):
            omega = self._omegas[i].item()
            k = self._k_from_omega(omega)
            phase = k * x - omega * t + self._phases[i].item()
            eta = eta + self._amps[i].item() * torch.cos(phase)
        return eta

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute irregular wave velocity via spectral superposition.

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m).

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        u = torch.zeros_like(x, dtype=x.dtype)
        w = torch.zeros_like(x, dtype=x.dtype)
        d = self._depth

        for i in range(self._n_comp):
            omega = self._omegas[i].item()
            k = self._k_from_omega(omega)
            phase = k * x - omega * t + self._phases[i].item()
            sinh_kd = math.sinh(k * d)
            coeff = self._amps[i].item() * omega / sinh_kd
            u = u + coeff * torch.cosh(k * z) * torch.cos(phase)
            w = w + coeff * torch.sinh(k * z) * torch.sin(phase)
        return u, w

    def __repr__(self) -> str:
        return (
            f"IrregularWave(spectrum={self._spectrum!r}, "
            f"n_components={self._n_comp}, Hs/2={self._amplitude})"
        )


# ---------------------------------------------------------------------------
# DirectionalWave
# ---------------------------------------------------------------------------

@WaveModel.register("directional")
class DirectionalWave(WaveModel):
    """Multi-directional wave model with cosine-2s spreading.

    Spreads wave energy across directions using a cosine power distribution:
        D(theta) = C(s) * cos^2s((theta - theta_0) / 2)

    Combines multiple directional components with linear superposition.

    Parameters
    ----------
    amplitude : float
        Wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    mean_direction : float
        Mean wave direction theta_0 (rad, default 0 = +x).
    spreading_exponent : float
        Cosine power s (default 10; s=inf gives unidirectional).
    n_directions : int
        Number of directional components (default 16).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        mean_direction: float = 0.0,
        spreading_exponent: float = 10.0,
        n_directions: int = 16,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._theta_0 = mean_direction
        self._s = spreading_exponent
        self._n_dir = n_directions

        # 均匀分布方向角
        dtheta = 2.0 * math.pi / n_directions
        thetas = torch.linspace(
            mean_direction - math.pi + 0.5 * dtheta,
            mean_direction + math.pi - 0.5 * dtheta,
            n_directions,
        )

        # 计算各方向的权重（cos-2s 分布归一化）
        s = spreading_exponent
        weights = torch.tensor([
            math.cos(0.5 * (th.item() - mean_direction)) ** (2 * s)
            if abs(0.5 * (th.item() - mean_direction)) < math.pi / 2
            else 0.0
            for th in thetas
        ])
        # 归一化
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = torch.ones(n_directions) / n_directions

        self._thetas = thetas
        self._weights = weights

    @property
    def mean_direction(self) -> float:
        """返回平均波向角 (rad)。"""
        return self._theta_0

    @property
    def spreading_exponent(self) -> float:
        """返回扩展指数 s。"""
        return self._s

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute directional wave elevation.

        eta(x, t) = sum_i w_i * A * cos(k*x*cos(theta_i) - omega*t)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        k = self.wavenumber
        omega = self.angular_frequency
        eta = torch.zeros_like(x, dtype=x.dtype)

        for i in range(self._n_dir):
            theta = self._thetas[i].item()
            w_i = self._weights[i].item()
            phase = k * x * math.cos(theta) - omega * t
            eta = eta + w_i * self._amplitude * torch.cos(phase)
        return eta

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute directional wave velocity.

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m).

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        k = self.wavenumber
        omega = self.angular_frequency
        d = self._depth
        sinh_kd = math.sinh(k * d)

        u = torch.zeros_like(x, dtype=x.dtype)
        w = torch.zeros_like(x, dtype=x.dtype)

        for i in range(self._n_dir):
            theta = self._thetas[i].item()
            w_i = self._weights[i].item()
            phase = k * x * math.cos(theta) - omega * t
            coeff = w_i * self._amplitude * omega / sinh_kd
            u = u + coeff * math.cos(theta) * torch.cosh(k * z) * torch.cos(phase)
            w = w + coeff * torch.sinh(k * z) * torch.sin(phase)
        return u, w

    def __repr__(self) -> str:
        return (
            f"DirectionalWave(theta_0={self._theta_0:.2f}, "
            f"s={self._s}, n_dir={self._n_dir})"
        )


# ---------------------------------------------------------------------------
# SolitaryWave
# ---------------------------------------------------------------------------

@WaveModel.register("solitary")
class SolitaryWave(WaveModel):
    """Boussinesq solitary wave model.

    A solitary wave is a single hump of water that propagates without change
    of form. Valid for shallow water (d/L << 1) with finite amplitude.

    Free-surface elevation:
        eta = H * sech^2(kappa * (x - c*t))

    where:
        kappa = sqrt(3*H / (4*d^3))
        c = sqrt(g*(d + H))

    Horizontal velocity (Boussinesq approximation):
        u = c * eta / (d + eta)

    Parameters
    ----------
    amplitude : float
        Solitary wave amplitude H (m).  Note: amplitude equals wave height.
    depth : float
        Water depth d (m).
    period : float
        Effective period T (s) — not used in elevation formula but required
        by the base class interface.
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
    ) -> None:
        super().__init__(amplitude, depth, period)
        H = amplitude
        d = depth
        # Boussinesq solitary wave shape parameter
        self._kappa = math.sqrt(3.0 * H / (4.0 * d**3))
        # Celerity
        self._celerity = math.sqrt(GRAVITY * (d + H))

    @property
    def celerity(self) -> float:
        """Solitary wave celerity c (m/s)."""
        return self._celerity

    @property
    def kappa(self) -> float:
        """Shape parameter kappa (1/m)."""
        return self._kappa

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute solitary wave elevation.

        eta = H * sech^2(kappa * (x - c*t))

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        c = self._celerity
        kappa = self._kappa
        xi = kappa * (x - c * t)
        # sech^2 = 1 / cosh^2
        return self._amplitude / torch.cosh(xi) ** 2

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute solitary wave velocity (Boussinesq approximation).

        u = c * eta / (d + eta)    (horizontally uniform)
        w ~ linear in z, proportional to d(eta)/dx

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m).

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        c = self._celerity
        d = self._depth
        eta = self.wave_elevation(x, t)

        # 水平速度（Boussinesq 浅水近似，垂向均匀）
        u = c * eta / (d + eta)

        # 垂向速度（线性垂向分布，满足海底不可穿透 + 运动学边界条件）
        # w = (z / (d + eta)) * du/dx * (d + eta) 的简化
        # 更精确: w ~ -z * d(eta)/dx * c / (d + eta)
        kappa = self._kappa
        xi = kappa * (x - c * 0.0)  # t=0 简化用于形状导数
        sech2 = 1.0 / torch.cosh(kappa * (x - c * t)) ** 2
        tanh_val = torch.tanh(kappa * (x - c * t))
        deta_dx = -2.0 * self._amplitude * kappa * sech2 * tanh_val
        w = -z * c * deta_dx / (d + eta).clamp(min=1e-6)

        return u, w

    def __repr__(self) -> str:
        return (
            f"SolitaryWave(H={self._amplitude}, d={self._depth}, "
            f"c={self._celerity:.2f})"
        )
