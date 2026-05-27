"""
Cnoidal wave theory (simplified).

Cnoidal waves are nonlinear periodic shallow-water waves described by
Jacobian elliptic functions. They are solutions of the KdV equation.

Free-surface elevation:
    eta = eta_2 + (eta_1 - eta_2) * cn^2(x - c*t | m)

where:
    cn(z|m) is the Jacobian elliptic cosine function
    m is the elliptic parameter (0 <= m < 1)
    c is the wave celerity
    eta_1, eta_2 are the crest and trough elevations

Simplified formulation using Ursell number (Ur = L^2*H/d^3):
    For m: solved from H/d and Ur via K(m) relation
    celerity: c = sqrt(g*d*(1 + H/d*(1/m - 1/2) + ...))

Reference:
    OpenFOAM ``waveModels::Cnoidal``
    Dingemans (1997), "Water Wave Propagation Over Uneven Bottoms"
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from pyfoam.waves.wave_model import GRAVITY, WaveModel

__all__ = ["CnoidalWave"]

# scipy 用于 Jacobian 椭圆函数；不可用时退回简化近似
try:
    from scipy.special import ellipk, ellipj

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _elliptic_cn(u: torch.Tensor, m: float) -> torch.Tensor:
    """Compute Jacobian elliptic cn(u|m).

    Args:
        u: Argument values.
        m: Elliptic parameter (0 <= m < 1).

    Returns:
        cn(u|m) values.
    """
    if _HAS_SCIPY:
        u_np = u.cpu().numpy()
        sn, cn, dn, _ = ellipj(u_np, m)
        return torch.from_numpy(cn).to(dtype=u.dtype, device=u.device)
    else:
        # 简化近似：m -> 0 时 cn(u) ~ cos(u)
        # 对于浅水波（m -> 1），误差较大
        return torch.cos(u) * (1.0 - 0.25 * m * (1.0 - torch.cos(2.0 * u)))


@WaveModel.register("cnoidal")
class CnoidalWave(WaveModel):
    """Cnoidal wave theory (simplified).

    Nonlinear shallow-water wave model using Jacobian elliptic functions.
    Valid for large Ursell number (Ur = H*L^2/d^3 >> 1), i.e. shallow water
    with finite amplitude.

    Parameters
    ----------
    amplitude : float
        Wave half-height H/2 (m).  Note: amplitude here represents H/2.
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    wavelength : float, optional
        Wavelength L (m). If not given, estimated from period via Airy theory.
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        wavelength: Optional[float] = None,
    ) -> None:
        super().__init__(amplitude, depth, period)
        # 波高 H = 2 * amplitude
        self._H = 2.0 * amplitude
        # 波长：未指定时从线性色散关系估计
        if wavelength is not None:
            self._L = wavelength
        else:
            k_airy = self._solve_dispersion()
            self._L = 2.0 * math.pi / k_airy

        # 预计算椭圆参数 m 和相速度 c
        self._m = self._compute_m()
        self._celerity = self._compute_celerity()

    def _compute_m(self) -> float:
        """Compute elliptic parameter m from wave conditions.

        Uses the relation: L^2 = 16*d^3*K(m) / (3*H)
        where K(m) is the complete elliptic integral of the first kind.
        Iterates until convergence.
        """
        H = self._H
        d = self._depth
        L = self._L
        target = 3.0 * H * L**2 / (16.0 * d**3)  # = K(m)

        if target < 0.5:
            # 浅水效应不显著，退回 Airy
            return 0.0

        if _HAS_SCIPY:
            # 二分法求 m: 使 K(m) == target
            m_lo, m_hi = 0.0, 0.9999
            for _ in range(100):
                m_mid = (m_lo + m_hi) / 2.0
                val = ellipk(m_mid)
                if val < target:
                    m_lo = m_mid
                else:
                    m_hi = m_mid
                if abs(m_hi - m_lo) < 1e-12:
                    break
            return (m_lo + m_hi) / 2.0
        else:
            # 简化估计：K(m) ~ pi/2 * (1 + m/4 + ...) 的近似
            # target = K(m) ~ pi/2 => m ~ 0
            # 对于较大 target，m -> 1
            # 用一阶近似
            m_est = min(0.9999, max(0.0, 1.0 - (math.pi / (2.0 * target)) ** 2))
            return m_est

    def _compute_celerity(self) -> float:
        """Compute wave celerity c.

        c^2 = g*d + g*H/m * (2 - 3*m/2)  (approximate)
        Simplified from KdV solitary wave celerity.
        """
        H = self._H
        d = self._depth
        g = GRAVITY
        m = self._m

        if m < 1e-10:
            return math.sqrt(g * d)

        # 二阶近似 celerity
        c2 = g * d * (1.0 + H / d * (1.0 / m - 0.5))
        return math.sqrt(c2)

    @property
    def celerity(self) -> float:
        """Wave celerity c (m/s)."""
        return self._celerity

    @property
    def elliptic_parameter(self) -> float:
        """Elliptic parameter m."""
        return self._m

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute cnoidal wave elevation.

        eta = H/2 * cn^2(2*K(m)*(x/L - t/T) | m)  (scaled)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        m = self._m
        H = self._H
        d = self._depth
        L = self._L
        T = self._period

        if m < 1e-10:
            # 退化为线性波
            k = 2.0 * math.pi / L
            omega = 2.0 * math.pi / T
            return (H / 2.0) * torch.cos(k * x - omega * t)

        if _HAS_SCIPY:
            K_m = float(ellipk(m))
        else:
            K_m = math.pi / 2.0 * (1.0 + m / 4.0)

        # 归一化相位参数
        phase = 2.0 * K_m * (x / L - t / T)

        cn_val = _elliptic_cn(phase, m)

        # Cnoidal 波高程
        eta = d + (H / m) * cn_val**2 - d
        # 更准确的表达：eta = eta_trough + H * cn^2 / m
        # eta_trough 约为 - H/(2*m) * (1 - m*E/K) 的简化
        # 简化为：eta 在 [0, H] 区间（相对于静水面偏移）
        # 最简形式：eta = (H/m)*cn^2 - H/m + 常数
        # 标准形式：eta = H * (cn^2(u|m) / m - 1) + H/2
        eta = H * (cn_val**2 / m - 1.0) + H / 2.0

        return eta

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cnoidal wave velocity (shallow water approximation).

        Uses shallow water approximation: u ~ c * eta/d, w ~ 0
        (horizontal velocity uniform in depth, vertical velocity negligible).

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m), z in [0, d].

        Returns:
            (u, w) horizontal and vertical velocity (m/s).
        """
        eta = self.wave_elevation(x, t)
        c = self._celerity
        d = self._depth

        # 浅水近似
        u = c * eta / d
        w = torch.zeros_like(u)

        return u, w
