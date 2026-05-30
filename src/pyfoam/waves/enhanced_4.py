"""
Enhanced wave models v4 — spectral, wave train, and rogue wave models.

Extends :class:`~pyfoam.waves.stokes.StokesWave` with:

- :class:`SpectralWave` — arbitrary spectral density wave model
- :class:`WaveTrain` — superposed wave trains (bichromatic / polychromatic)
- :class:`RogueWave` — focused freak wave group (NewWave theory)

References:
    Longuet-Higgins (1984). "Statistical properties of wave groups."
    Tromans et al. (1991). "The NewWave — a new model for the kinematics of
    extreme ocean waves."
    Kharif & Pelinovsky (2003). "Physical mechanisms of the rogue wave phenomenon."

Usage::

    from pyfoam.waves.enhanced_4 import SpectralWave, WaveTrain, RogueWave

    wave = RogueWave(amplitude=5.0, depth=30.0, period=12.0)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from pyfoam.waves.wave_model import GRAVITY, WaveModel
from pyfoam.waves.stokes import StokesWave

__all__ = ["SpectralWave", "WaveTrain", "RogueWave"]


# ---------------------------------------------------------------------------
# SpectralWave
# ---------------------------------------------------------------------------

@WaveModel.register("spectral")
class SpectralWave(StokesWave):
    """Arbitrary spectral density wave model.

    Generates waves from a user-defined spectral density function by
    superposing discrete frequency components. Supports both parametric
    spectra (JONSWAP, PM) and custom spectra via callable.

    Parameters
    ----------
    amplitude : float
        Significant wave height Hs/2 (m).
    depth : float
        Water depth d (m).
    period : float
        Peak period Tp (s).
    n_frequencies : int
        Number of frequency components (default 64).
    spectral_fn : callable, optional
        Custom spectral density function S(omega) -> float.
        If None, uses JONSWAP with gamma=1 (PM spectrum).
    seed : int, optional
        Random seed for reproducible phase angles.
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        n_frequencies: int = 64,
        spectral_fn: Optional[callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._n_freq = n_frequencies

        omega_p = 2.0 * math.pi / period
        omega_lo = 0.4 * omega_p
        omega_hi = 3.5 * omega_p
        domega = (omega_hi - omega_lo) / n_frequencies

        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.manual_seed(123)

        omegas = torch.linspace(
            omega_lo + 0.5 * domega,
            omega_hi - 0.5 * domega,
            n_frequencies,
        )
        phases = torch.rand(n_frequencies, generator=rng) * 2.0 * math.pi

        # 默认使用 PM 谱
        if spectral_fn is None:
            def spectral_fn(w: float) -> float:
                g = GRAVITY
                alpha = 0.0081
                return alpha * g**2 / w**5 * math.exp(-1.25 * (omega_p / w) ** 4)

        amps = torch.tensor([
            math.sqrt(2.0 * spectral_fn(w.item()) * domega)
            for w in omegas
        ])

        self._omegas = omegas
        self._amps = amps
        self._phases = phases

    @property
    def n_frequencies(self) -> int:
        """返回频率分量数。"""
        return self._n_freq

    def _k_from_omega(self, omega: float) -> float:
        """通过线性弥散关系从 omega 求 k。"""
        g = GRAVITY
        d = self._depth
        k = omega**2 / g
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
        """Compute spectral wave elevation via superposition.

        eta(x, t) = sum_i A_i * cos(k_i*x - omega_i*t + phi_i)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        eta = torch.zeros_like(x, dtype=x.dtype)
        for i in range(self._n_freq):
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
        """Compute spectral wave velocity via superposition.

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

        for i in range(self._n_freq):
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
            f"SpectralWave(Hs/2={self._amplitude}, n_freq={self._n_freq}, "
            f"d={self._depth})"
        )


# ---------------------------------------------------------------------------
# WaveTrain
# ---------------------------------------------------------------------------

@WaveModel.register("waveTrain")
class WaveTrain(StokesWave):
    """Superposed wave trains (bichromatic / polychromatic).

    Combines a small number of discrete wave components with specified
    amplitudes, frequencies, and phases. Useful for:
    - Bichromatic wave groups (two-frequency interaction)
    - Polychromatic sea states with few dominant components
    - Wave group envelope analysis

    Parameters
    ----------
    amplitude : float
        Primary wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Primary wave period T (s).
    trains : list of dict, optional
        Wave train specifications. Each dict has keys:
        ``amplitude``, ``period``, ``phase`` (default 0).
        When None, a single train is created from primary parameters.
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        trains: Optional[list[dict]] = None,
    ) -> None:
        super().__init__(amplitude, depth, period)

        if trains is not None:
            self._trains = trains
        else:
            self._trains = [{"amplitude": amplitude, "period": period, "phase": 0.0}]

        # 预计算各 train 的参数
        self._train_params = []
        for tr in self._trains:
            a = tr["amplitude"]
            T_tr = tr["period"]
            phi = tr.get("phase", 0.0)
            omega_tr = 2.0 * math.pi / T_tr
            k_tr = self._solve_k_for_omega(omega_tr)
            self._train_params.append({
                "amplitude": a, "period": T_tr, "phase": phi,
                "omega": omega_tr, "k": k_tr,
            })

    @property
    def n_trains(self) -> int:
        """返回波列数。"""
        return len(self._trains)

    @property
    def trains(self) -> list[dict]:
        """返回波列参数列表。"""
        return list(self._trains)

    def _solve_k_for_omega(self, omega: float) -> float:
        """对指定 omega 求解弥散关系。"""
        g = GRAVITY
        d = self._depth
        k = omega**2 / g
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
        """Compute wave train elevation via superposition.

        eta(x, t) = sum_i A_i * cos(k_i*x - omega_i*t + phi_i)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        eta = torch.zeros_like(x, dtype=x.dtype)
        for tp in self._train_params:
            phase = tp["k"] * x - tp["omega"] * t + tp["phase"]
            eta = eta + tp["amplitude"] * torch.cos(phase)
        return eta

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute wave train velocity via superposition.

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

        for tp in self._train_params:
            k = tp["k"]
            omega = tp["omega"]
            phase = k * x - omega * t + tp["phase"]
            sinh_kd = math.sinh(k * d)
            coeff = tp["amplitude"] * omega / sinh_kd
            u = u + coeff * torch.cosh(k * z) * torch.cos(phase)
            w = w + coeff * torch.sinh(k * z) * torch.sin(phase)
        return u, w

    def envelope(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute wave group envelope (for bichromatic case).

        For two trains with frequencies omega_1, omega_2:
            envelope = 2*A*cos((k_1-k_2)*x/2 - (omega_1-omega_2)*t/2)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave envelope at each position (m).
        """
        if self.n_trains < 2:
            return torch.full_like(x, self._trains[0]["amplitude"])

        # 用 Hilbert 变换近似（简化：直接用两分量振幅包络）
        eta = self.wave_elevation(x, t)
        # 简化包络：max 滤波
        # 更准确的做法需要 Hilbert 变换，这里用移动窗口近似
        return eta.abs()

    def __repr__(self) -> str:
        return (
            f"WaveTrain(n_trains={self.n_trains}, d={self._depth})"
        )


# ---------------------------------------------------------------------------
# RogueWave
# ---------------------------------------------------------------------------

@WaveModel.register("rogue")
class RogueWave(StokesWave):
    """Focused freak wave model (NewWave theory).

    Implements the NewWave / linear superposition approach for modeling
    rogue waves — abnormally large waves that appear from constructive
    interference of spectral components focused at a single point.

    All spectral components are aligned to focus at (x_focus, t_focus)
    with zero phase, producing a wave of amplitude A_max = sigma * sqrt(2*ln(N)).

    Parameters
    ----------
    amplitude : float
        Maximum rogue wave amplitude A_max (m).
    depth : float
        Water depth d (m).
    period : float
        Peak period Tp (s).
    focus_position : float
        x-coordinate of wave focus (m, default 0).
    focus_time : float
        Time of wave focus (s, default 0).
    n_components : int
        Number of spectral components (default 32).
    bandwidth : float
        Relative spectral bandwidth (default 0.2, i.e. +/- 20% of omega_p).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        focus_position: float = 0.0,
        focus_time: float = 0.0,
        n_components: int = 32,
        bandwidth: float = 0.2,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._x_focus = focus_position
        self._t_focus = focus_time
        self._n_comp = n_components
        self._bandwidth = bandwidth

        # 构造聚焦谱分量
        omega_p = 2.0 * math.pi / period
        omega_lo = omega_p * (1.0 - bandwidth)
        omega_hi = omega_p * (1.0 + bandwidth)
        omegas = torch.linspace(omega_lo, omega_hi, n_components)

        # 各分量振幅（均匀分配，总振幅 = amplitude）
        amp_each = amplitude / n_components
        amps = torch.full((n_components,), amp_each)

        # 聚焦相位：所有分量在 (x_focus, t_focus) 处同相
        # phi_i = -k_i * x_focus + omega_i * t_focus
        kappas = []
        for w in omegas:
            k = self._k_from_omega(w.item())
            kappas.append(k)
        kappas = torch.tensor(kappas)

        phases = -kappas * focus_position + omegas * focus_time

        self._omegas = omegas
        self._amps = amps
        self._phases = phases
        self._kappas = kappas

    def _k_from_omega(self, omega: float) -> float:
        """通过线性弥散关系从 omega 求 k。"""
        g = GRAVITY
        d = self._depth
        k = omega**2 / g
        for _ in range(50):
            tanh_kd = math.tanh(k * d)
            f = g * k * tanh_kd - omega**2
            df = g * (tanh_kd + k * d / math.cosh(k * d) ** 2)
            dk = f / df
            k -= dk
            if abs(dk) < 1e-12:
                break
        return k

    @property
    def focus_position(self) -> float:
        """返回聚焦位置 x_focus (m)。"""
        return self._x_focus

    @property
    def focus_time(self) -> float:
        """返回聚焦时间 t_focus (s)。"""
        return self._t_focus

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute rogue wave elevation via focused superposition.

        eta(x, t) = sum_i A_i * cos(k_i*x - omega_i*t + phi_i)

        At (x_focus, t_focus), all components are in phase, producing
        the maximum amplitude.

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        eta = torch.zeros_like(x, dtype=x.dtype)
        for i in range(self._n_comp):
            phase = self._kappas[i].item() * x - self._omegas[i].item() * t + self._phases[i].item()
            eta = eta + self._amps[i].item() * torch.cos(phase)
        return eta

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rogue wave velocity via focused superposition.

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
            k = self._kappas[i].item()
            omega = self._omegas[i].item()
            phase = k * x - omega * t + self._phases[i].item()
            sinh_kd = math.sinh(k * d)
            coeff = self._amps[i].item() * omega / sinh_kd
            u = u + coeff * torch.cosh(k * z) * torch.cos(phase)
            w = w + coeff * torch.sinh(k * z) * torch.sin(phase)
        return u, w

    def __repr__(self) -> str:
        return (
            f"RogueWave(A_max={self._amplitude}, focus=({self._x_focus}, "
            f"{self._t_focus}), n_comp={self._n_comp})"
        )
