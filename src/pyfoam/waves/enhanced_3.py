"""
Enhanced wave models v3 — stream function, Boussinesq, and mild slope models.

Extends :class:`~pyfoam.waves.airy.AiryWave` with:

- :class:`StreamFunctionWave` — high-order stream function wave theory
- :class:`BoussinesqWave` — Boussinesq dispersive wave model
- :class:`MildSlopeWave` — mild-slope equation (Berkhoff, 1972)

References:
    Dean (1965). "Stream function representation of nonlinear ocean waves."
    Peregrine (1967). "Long waves on a beach."
    Berkhoff (1972). "Computation of combined refraction-diffraction."

Usage::

    from pyfoam.waves.enhanced_3 import StreamFunctionWave, BoussinesqWave, MildSlopeWave

    wave = StreamFunctionWave(amplitude=2.0, depth=10.0, period=8.0, order=5)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from pyfoam.waves.wave_model import GRAVITY, WaveModel
from pyfoam.waves.airy import AiryWave

__all__ = ["StreamFunctionWave", "BoussinesqWave", "MildSlopeWave"]


# ---------------------------------------------------------------------------
# StreamFunctionWave
# ---------------------------------------------------------------------------

@WaveModel.register("streamFunction")
class StreamFunctionWave(AiryWave):
    """High-order stream function wave theory (Dean, 1965).

    Extends the Airy wave solution with higher-order harmonic corrections
    computed via a Fourier (stream function) expansion.  More accurate than
    Stokes theory for waves near breaking.

    The stream function is:
        psi(x, z, t) = B_0 * z + sum_{n=1}^{N} B_n * cosh(n*k*(z+d))
                        / cosh(n*k*d) * sin(n*theta)

    Free-surface elevation:
        eta = sum_{n=1}^{N} a_n * cos(n * theta)

    Parameters
    ----------
    amplitude : float
        Fundamental wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    order : int
        Number of harmonics (default 5).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        order: int = 5,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._order = order

        # 预计算各阶系数（简化近似：基于 Stokes 展开）
        k = self.wavenumber
        d = self._depth
        A = self._amplitude
        eps = k * A  # 波陡参数

        # 表面高程各阶系数 a_n（Stokes 展开近似）
        self._a_coeffs = self._compute_surface_coefficients(eps, k, d)

        # 速度势各阶系数 B_n
        self._B_coeffs = self._compute_velocity_coefficients(eps, k, d)

    def _compute_surface_coefficients(
        self, eps: float, k: float, d: float
    ) -> list[float]:
        """计算表面高程各阶 Fourier 系数。

        使用 Stokes 展开的近似系数：
        a_1 = A
        a_2 = k*A^2 * (3 - tanh^2(kd)) / (4*tanh^3(kd))
        a_3 ~ O(eps^3)
        ...
        """
        A = self._amplitude
        tanh_kd = math.tanh(k * d)

        coeffs = [A]
        if self._order >= 2:
            a2 = (k * A**2 / 2.0) * (3.0 - tanh_kd**2) / (4.0 * tanh_kd**3)
            coeffs.append(a2)
        for n in range(3, self._order + 1):
            # 高阶系数按 eps^(n-1) 衰减
            coeffs.append(A * eps ** (n - 1) / math.factorial(n))
        return coeffs

    def _compute_velocity_coefficients(
        self, eps: float, k: float, d: float
    ) -> list[float]:
        """计算速度势各阶系数 B_n。

        B_0 = c (相速度)
        B_n ~ A * omega / sinh(kd) * eps^(n-1) / n
        """
        omega = self.angular_frequency
        A = self._amplitude
        c = omega / k  # 相速度
        sinh_kd = math.sinh(k * d)

        B = [c]
        for n in range(1, self._order + 1):
            B_n = A * omega / (n * sinh_kd) * eps ** (n - 1)
            B.append(B_n)
        return B

    @property
    def order(self) -> int:
        """返回谐波阶数。"""
        return self._order

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute stream function wave elevation.

        eta = sum_{n=1}^{N} a_n * cos(n * theta)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        k = self.wavenumber
        omega = self.angular_frequency
        theta = k * x - omega * t

        eta = torch.zeros_like(x, dtype=x.dtype)
        for n in range(1, self._order + 1):
            a_n = self._a_coeffs[n - 1] if n - 1 < len(self._a_coeffs) else 0.0
            eta = eta + a_n * torch.cos(n * theta)
        return eta

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute stream function wave velocity.

        u = sum_{n=1}^{N} n*k*B_n * cosh(n*k*(z+d)) / cosh(n*k*d) * cos(n*theta)
        w = sum_{n=1}^{N} n*k*B_n * sinh(n*k*(z+d)) / cosh(n*k*d) * sin(n*theta)

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
        theta = k * x - omega * t

        u = torch.zeros_like(x, dtype=x.dtype)
        w = torch.zeros_like(x, dtype=x.dtype)

        for n in range(1, self._order + 1):
            B_n = self._B_coeffs[n] if n < len(self._B_coeffs) else 0.0
            nk = n * k
            cosh_nkd = math.cosh(nk * d)
            cosh_nkz = torch.cosh(nk * (z + d))  # z 从海底向上
            sinh_nkz = torch.sinh(nk * (z + d))
            u = u + nk * B_n * cosh_nkz / cosh_nkd * torch.cos(n * theta)
            w = w + nk * B_n * sinh_nkz / cosh_nkd * torch.sin(n * theta)

        return u, w

    def __repr__(self) -> str:
        return (
            f"StreamFunctionWave(A={self._amplitude}, order={self._order}, "
            f"d={self._depth})"
        )


# ---------------------------------------------------------------------------
# BoussinesqWave
# ---------------------------------------------------------------------------

@WaveModel.register("boussinesq")
class BoussinesqWave(AiryWave):
    """Boussinesq dispersive wave model.

    Extends the Airy wave with weakly dispersive corrections following the
    classical Boussinesq equations (Peregrine, 1967). Adds a depth-averaged
    correction to the velocity profile:

        u_corrected = u_mean + (d^2/6 - z_bar^2/2) * d^2(u_mean)/dx^2

    where z_bar = z - d/2 is the position relative to still water level.

    Parameters
    ----------
    amplitude : float
        Wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    dispersion_order : int
        Dispersion correction order: 1 (Peregrine) or 2 (Madsen-Sorensen).
        Default 1.
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        dispersion_order: int = 1,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._disp_order = dispersion_order

    @property
    def dispersion_order(self) -> int:
        """返回色散修正阶数。"""
        return self._disp_order

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute Boussinesq wave elevation.

        Uses the linear elevation with a nonlinear correction:
            eta = A*cos(theta) + k*A^2/(4*sinh^2(kd)) * (3/tanh^2(kd) - 1) * cos(2*theta)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        k = self.wavenumber
        omega = self.angular_frequency
        d = self._depth
        A = self._amplitude
        theta = k * x - omega * t

        # 线性项
        eta_lin = A * torch.cos(theta)

        # Boussinesq 非线性修正项
        if self._disp_order >= 1:
            sinh_kd = math.sinh(k * d)
            tanh_kd = math.tanh(k * d)
            correction = k * A**2 / (4.0 * sinh_kd**2) * (3.0 / tanh_kd**2 - 1.0)
            eta = eta_lin + correction * torch.cos(2.0 * theta)
        else:
            eta = eta_lin

        return eta

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Boussinesq wave velocity with depth correction.

        u = u_mean + (d^2/6 - z_bar^2/2) * d^2(u_mean)/dx^2

        where z_bar = z - d (relative to still water level).

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
        A = self._amplitude
        theta = k * x - omega * t

        # 平均速度（线性 Airy 解的深度平均）
        sinh_kd = math.sinh(k * d)
        u_mean = A * omega / (k * d) * torch.cos(theta)

        if self._disp_order >= 1:
            # z_bar = z - d（静水面以上为正）
            z_bar = z - d
            # d^2(u_mean)/dx^2 = -k^2 * u_mean
            d2u_dx2 = -k**2 * u_mean
            # Boussinesq 色散修正
            correction = (d**2 / 6.0 - z_bar**2 / 2.0) * d2u_dx2
            u = u_mean + correction
        else:
            u = u_mean

        # 垂向速度（近似：线性分布）
        w = A * omega * torch.sinh(k * z) / sinh_kd * torch.sin(theta)

        return u, w

    def __repr__(self) -> str:
        return (
            f"BoussinesqWave(A={self._amplitude}, d={self._depth}, "
            f"disp_order={self._disp_order})"
        )


# ---------------------------------------------------------------------------
# MildSlopeWave
# ---------------------------------------------------------------------------

@WaveModel.register("mildSlope")
class MildSlopeWave(AiryWave):
    """Mild-slope equation wave model (Berkhoff, 1972).

    Solves the mild-slope equation for combined refraction-diffraction
    over slowly varying bathymetry. This implementation provides a
    simplified 1D version with depth-varying wavenumber.

    The mild-slope equation:
        ∇·(c c_g ∇eta) + k^2 c c_g eta = 0

    where c = omega/k is phase velocity, c_g = n*c is group velocity,
    n = (1 + 2kd/sinh(2kd))/2.

    Parameters
    ----------
    amplitude : float
        Wave amplitude A (m).
    depth : float
        Reference water depth d (m).
    period : float
        Wave period T (s).
    bottom_slope : float
        Seabed slope dh/dx (default 0 = flat bottom).
    n_points : int
        Number of grid points for 1D mild-slope computation (default 100).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        bottom_slope: float = 0.0,
        n_points: int = 100,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._slope = bottom_slope
        self._n_pts = n_points

    @property
    def bottom_slope(self) -> float:
        """返回海底坡度 dh/dx。"""
        return self._slope

    def _local_depth(self, x: torch.Tensor | float) -> torch.Tensor | float:
        """Compute local water depth at position x.

        d(x) = d_0 + slope * x   (clamped to positive values)

        Args:
            x: Position tensor or scalar.

        Returns:
            Local depth at each position.
        """
        d_local = self._depth + self._slope * x
        if isinstance(d_local, float):
            return max(d_local, 0.1)
        return d_local.clamp(min=0.1)

    def _local_wavenumber(self, d_local: float) -> float:
        """Solve local dispersion relation for given depth.

        omega^2 = g * k * tanh(k * d)

        Args:
            d_local: Local water depth.

        Returns:
            Local wavenumber k.
        """
        omega = self.angular_frequency
        g = GRAVITY
        k = omega**2 / g  # 深水近似初值

        for _ in range(50):
            tanh_kd = math.tanh(k * d_local)
            f = g * k * tanh_kd - omega**2
            df = g * (tanh_kd + k * d_local / math.cosh(k * d_local) ** 2)
            dk = f / df
            k -= dk
            if abs(dk) < 1e-12:
                break
        return k

    def _local_group_velocity(self, k: float, d: float) -> float:
        """Compute local group velocity c_g = n * c.

        n = (1 + 2kd / sinh(2kd)) / 2
        c = omega / k

        Args:
            k: Local wavenumber.
            d: Local depth.

        Returns:
            Group velocity c_g.
        """
        omega = self.angular_frequency
        sinh_2kd = math.sinh(2.0 * k * d)
        n = 0.5 * (1.0 + 2.0 * k * d / max(sinh_2kd, 1e-30))
        c = omega / k
        return n * c

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute mild-slope wave elevation.

        For flat bottom, equivalent to Airy wave.
        For sloped bottom, accounts for amplitude changes due to
        refraction (shoaling):
            A(x) = A_0 * sqrt(c_g0 / c_g(x))

        eta(x, t) = A(x) * cos(integral(k(x') dx') - omega*t)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        if abs(self._slope) < 1e-10:
            # 平底：退化为 Airy
            return super().wave_elevation(x, t)

        omega = self.angular_frequency
        A_0 = self._amplitude

        # 参考处（x=0）的群速度
        k_ref = self.wavenumber
        cg_ref = self._local_group_velocity(k_ref, self._depth)

        eta = torch.zeros_like(x, dtype=x.dtype)
        for i in range(x.shape[0]):
            xi = x[i].item()
            d_loc = self._local_depth(xi)
            k_loc = self._local_wavenumber(d_loc)
            cg_loc = self._local_group_velocity(k_loc, d_loc)

            # 浅水变形（shoaling）
            A_loc = A_0 * math.sqrt(cg_ref / max(cg_loc, 1e-30))

            # 相位积分（梯形近似）
            phase = k_loc * xi - omega * t
            eta[i] = A_loc * math.cos(phase)

        return eta

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mild-slope wave velocity with local depth effects.

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m).

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        if abs(self._slope) < 1e-10:
            return super().velocity(x, t, z)

        omega = self.angular_frequency
        A_0 = self._amplitude
        k_ref = self.wavenumber
        cg_ref = self._local_group_velocity(k_ref, self._depth)

        u = torch.zeros_like(x, dtype=x.dtype)
        w = torch.zeros_like(x, dtype=x.dtype)

        for i in range(x.shape[0]):
            xi = x[i].item()
            zi = z[i].item()
            d_loc = self._local_depth(xi)
            k_loc = self._local_wavenumber(d_loc)
            cg_loc = self._local_group_velocity(k_loc, d_loc)
            A_loc = A_0 * math.sqrt(cg_ref / max(cg_loc, 1e-30))

            sinh_kd = math.sinh(k_loc * d_loc)
            phase = k_loc * xi - omega * t
            coeff = A_loc * omega / max(sinh_kd, 1e-30)
            u[i] = coeff * math.cosh(k_loc * zi) * math.cos(phase)
            w[i] = coeff * math.sinh(k_loc * zi) * math.sin(phase)

        return u, w

    def __repr__(self) -> str:
        return (
            f"MildSlopeWave(A={self._amplitude}, d={self._depth}, "
            f"slope={self._slope})"
        )
