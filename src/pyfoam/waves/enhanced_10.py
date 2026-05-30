"""
Enhanced wave models v10 — absorption-integrated generation and flap diffraction.

Extends :class:`~pyfoam.waves.enhanced_7.WaveGenerationModel` with:

- :class:`AbsorptionGeneration` — wavemaker with built-in active absorption
- :class:`FlapDiffraction` — flap wavemaker with diffraction correction

References:
    OpenFOAM ``waveModels::absorptionGeneration``
    Schaffer & Klopman (2000). "Review of multidirectional active wave absorption methods."
   谷爱軍・栗山幸夫 (2000). "Active absorption of multi-directional waves."

Usage::

    from pyfoam.waves.enhanced_10 import AbsorptionGeneration, FlapDiffraction

    gen = AbsorptionGeneration(amplitude=1.0, depth=10.0, period=8.0)
    eta = gen.generate_elevation(x, t=0.0)
"""

from __future__ import annotations

import math

import torch

from pyfoam.waves.wave_model import GRAVITY, WaveModel
from pyfoam.waves.enhanced_7 import WaveGenerationModel
from pyfoam.waves.enhanced_8 import PistonType, FlapType, _solve_dispersion

__all__ = ["AbsorptionGeneration", "FlapDiffraction"]


# ---------------------------------------------------------------------------
# AbsorptionGeneration
# ---------------------------------------------------------------------------

@WaveGenerationModel.register("absorptionGeneration")
class AbsorptionGeneration(PistonType):
    """Wavemaker with built-in active absorption.

    Combines piston-type wave generation with real-time reflection
    absorption. The wavemaker measures the reflected wave at its face
    and subtracts it from the target motion to prevent re-reflection.

    The effective paddle displacement is:
        X_eff(t) = X_gen(t) - X_abs(t)

    where:
        X_gen(t) = X_0 * sin(omega*t)  (generation signal)
        X_abs(t) = measured reflected wave contribution

    Parameters
    ----------
    amplitude : float
        Target wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    absorption_gain : float
        Absorption gain factor in (0, 1] (default 0.8).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        absorption_gain: float = 0.8,
        paddle_position: float = 0.0,
    ) -> None:
        super().__init__(amplitude, depth, period, paddle_position=paddle_position)
        self._abs_gain = max(0.0, min(1.0, absorption_gain))

    @property
    def absorption_gain(self) -> float:
        """返回吸收增益系数。"""
        return self._abs_gain

    def effective_displacement(self, t: float, eta_reflected: float = 0.0) -> float:
        """Compute effective paddle displacement with absorption.

        X_eff = X_gen - gain * eta_reflected / (transfer_function)

        Parameters
        ----------
        t : float
            Time (s).
        eta_reflected : float
            Measured reflected wave elevation at the paddle face (m).

        Returns
        -------
        float
            Effective paddle displacement (m).
        """
        X_gen = self.paddle_displacement(t)

        # 反射波到 paddle 位移的逆传递函数
        k = self._k
        d = self._depth
        sinh_kd = math.sinh(k * d)
        tanh_kd = math.tanh(k * d)
        F = tanh_kd + k * d / math.cosh(k * d) ** 2
        denom = k * d * F

        if abs(denom) < 1e-30:
            X_abs = 0.0
        else:
            X_abs = self._abs_gain * eta_reflected * denom / sinh_kd

        return X_gen - X_abs

    def generate_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute generated wave elevation (same as piston).

        The absorption correction is applied at the paddle level;
        the far-field wave is the same as a standard piston wavemaker.

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        return super().generate_elevation(x, t)

    def generate_velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute generated wave velocity.

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m).

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        return super().generate_velocity(x, t, z)

    def __repr__(self) -> str:
        return (
            f"AbsorptionGeneration(A={self._amplitude}, d={self._depth}, "
            f"T={self._period}, gain={self._abs_gain})"
        )


# ---------------------------------------------------------------------------
# FlapDiffraction
# ---------------------------------------------------------------------------

@WaveGenerationModel.register("flapDiffraction")
class FlapDiffraction(FlapType):
    """Flap wavemaker with diffraction correction.

    Extends the standard flap wavemaker with a diffraction correction
    factor that accounts for the finite width of the flap and the
    resulting wave diffraction effects.

    The diffraction-corrected amplitude is:
        A_corrected = A * K_diff(k, B)

    where K_diff is the diffraction correction factor:
        K_diff = sin(k*B/2) / (k*B/2)  (sinc function)

    and B is the flap width.

    Parameters
    ----------
    amplitude : float
        Target wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    flap_width : float
        Width of the flap in the lateral direction (m, default 1.0).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        flap_width: float = 1.0,
        hinge_depth: float = 0.0,
        paddle_position: float = 0.0,
    ) -> None:
        super().__init__(
            amplitude, depth, period,
            hinge_depth=hinge_depth,
            paddle_position=paddle_position,
        )
        self._B = flap_width

        # 预计算衍射修正系数
        k = self._k
        half_kb = k * flap_width / 2.0
        if abs(half_kb) < 1e-10:
            self._K_diff = 1.0
        else:
            self._K_diff = math.sin(half_kb) / half_kb

    @property
    def diffraction_coeff(self) -> float:
        """返回衍射修正系数 K_diff。"""
        return self._K_diff

    @property
    def flap_width(self) -> float:
        """返回 flap 宽度 (m)。"""
        return self._B

    def generate_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute flap-generated wave elevation with diffraction correction.

        eta = K_diff * A * cos(k * (x - x_paddle) - omega * t)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        eta_base = super().generate_elevation(x, t)
        return self._K_diff * eta_base

    def generate_velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute flap-generated velocity with diffraction correction.

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m).

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        u_base, w_base = super().generate_velocity(x, t, z)
        return self._K_diff * u_base, self._K_diff * w_base

    def __repr__(self) -> str:
        return (
            f"FlapDiffraction(A={self._amplitude}, d={self._depth}, "
            f"B={self._B}, K_diff={self._K_diff:.4f})"
        )
