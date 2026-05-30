"""
Enhanced wave models v9 — pressure-type generation and irregular generation.

Extends :class:`~pyfoam.waves.enhanced_7.WaveGenerationModel` with:

- :class:`PressureType` — pressure-based wave generation (submerged disturbance)
- :class:`IrregularGeneration` — irregular wave generation via spectral superposition

References:
    OpenFOAM ``waveModels:: pressure`` generation.
    Dean & Dalrymple (1991). "Water Wave Mechanics for Engineers and Scientists."
    Barthel et al. (1983). "Group bounded long waves."

Usage::

    from pyfoam.waves.enhanced_9 import PressureType, IrregularGeneration

    gen = PressureType(amplitude=1.0, depth=10.0, period=8.0, submergence=3.0)
    eta = gen.generate_elevation(x, t=0.0)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from pyfoam.waves.wave_model import GRAVITY, WaveModel
from pyfoam.waves.enhanced_7 import WaveGenerationModel
from pyfoam.waves.enhanced_8 import _solve_dispersion

__all__ = ["PressureType", "IrregularGeneration"]


# ---------------------------------------------------------------------------
# PressureType
# ---------------------------------------------------------------------------

@WaveGenerationModel.register("pressure")
class PressureType(WaveGenerationModel):
    """Pressure-based wave generation (submerged disturbance).

    Generates waves by applying a time-varying pressure disturbance
    at or below the free surface. This models scenarios such as:
    - Submerged object oscillation
    - Pressure plate wavemakers
    - Air-cushion wave generation

    The surface pressure distribution is:
        p(x, t) = p_0 * G(x) * sin(omega * t)

    where G(x) is the spatial distribution (Gaussian or delta function),
    and p_0 is the pressure amplitude related to the target wave amplitude:

        p_0 = rho * g * A * cosh(k*d_s) / cosh(k*d)

    where d_s is the submergence depth of the pressure source.

    Parameters
    ----------
    amplitude : float
        Target wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    submergence : float
        Depth of pressure source below free surface (m, default 0 = surface).
    source_width : float
        Spatial width of pressure distribution (m, default 1.0).
    source_position : float
        x-coordinate of the pressure source center (m, default 0).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        submergence: float = 0.0,
        source_width: float = 1.0,
        source_position: float = 0.0,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._d_s = submergence
        self._width = source_width
        self._x_source = source_position

        omega = self.angular_frequency
        self._k = _solve_dispersion(omega, depth)
        self._p0 = self._compute_pressure_amplitude()

    def _compute_pressure_amplitude(self) -> float:
        """Compute pressure amplitude p_0 from target wave amplitude.

        p_0 = rho * g * A * cosh(k*(d - d_s)) / cosh(k*d)

        Returns:
            Pressure amplitude (Pa).
        """
        rho = 1025.0  # 海水密度 (kg/m^3)
        g = GRAVITY
        A = self._amplitude
        k = self._k
        d = self._depth
        d_s = min(self._d_s, d)  # 不超过水深

        cosh_kd = math.cosh(k * d)
        cosh_kds = math.cosh(k * (d - d_s))

        return rho * g * A * cosh_kds / cosh_kd

    @property
    def pressure_amplitude(self) -> float:
        """返回压力源振幅 p_0 (Pa)。"""
        return self._p0

    @property
    def submergence(self) -> float:
        """返回压力源浸没深度 (m)。"""
        return self._d_s

    def source_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial pressure distribution G(x).

        Gaussian distribution centered at x_source:
            G(x) = exp(-(x - x_source)^2 / (2 * sigma^2))

        where sigma = source_width / 2.

        Parameters
        ----------
        x : torch.Tensor
            Horizontal positions.

        Returns
        -------
        torch.Tensor
            Distribution function G(x).
        """
        sigma = self._width / 2.0
        return torch.exp(-0.5 * ((x - self._x_source) / sigma) ** 2)

    def generate_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute generated wave elevation.

        Uses Airy wave theory with the target amplitude. In the far field
        (far from the source), the generated wave is equivalent to a
        regular Airy wave.

        eta = A * cos(k * (x - x_source) - omega * t)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        k = self._k
        omega = self.angular_frequency
        phase = k * (x - self._x_source) - omega * t

        # 近场：受源分布调制
        G = self.source_distribution(x)
        # 远场：标准 Airy 波
        # 用 tanh 平滑过渡（避免近场人为引入远场效应）
        far_field_weight = torch.tanh((x - self._x_source).abs() / self._width)
        amplitude_mod = G * (1.0 - far_field_weight) + far_field_weight

        return self._amplitude * amplitude_mod * torch.cos(phase)

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
        k = self._k
        omega = self.angular_frequency
        d = self._depth
        phase = k * (x - self._x_source) - omega * t

        # 近场/远场过渡
        G = self.source_distribution(x)
        far_field_weight = torch.tanh((x - self._x_source).abs() / self._width)
        amplitude_mod = G * (1.0 - far_field_weight) + far_field_weight

        A_local = self._amplitude * amplitude_mod
        sinh_kd = math.sinh(k * d)
        coeff = A_local * omega / sinh_kd

        u = coeff * torch.cosh(k * z) * torch.cos(phase)
        w = coeff * torch.sinh(k * z) * torch.sin(phase)
        return u, w

    def __repr__(self) -> str:
        return (
            f"PressureType(A={self._amplitude}, d={self._depth}, "
            f"submergence={self._d_s}, p0={self._p0:.1f})"
        )


# ---------------------------------------------------------------------------
# IrregularGeneration
# ---------------------------------------------------------------------------

@WaveGenerationModel.register("irregularGeneration")
class IrregularGeneration(WaveGenerationModel):
    """Irregular wave generation via spectral superposition.

    Generates irregular sea states at a wavemaker boundary by superposing
    multiple frequency components with phases derived from the JONSWAP
    or Pierson-Moskowitz spectrum. Each component propagates away from
    the generation boundary.

    Parameters
    ----------
    amplitude : float
        Significant wave height Hs/2 (m).
    depth : float
        Water depth d (m).
    period : float
        Peak period Tp (s).
    spectrum : str
        Spectrum type: ``"jonswap"`` (default) or ``"pm"``.
    n_components : int
        Number of spectral components (default 50).
    gamma : float
        JONSWAP peak enhancement factor (default 3.3).
    seed : int, optional
        Random seed for reproducible phases.
    source_position : float
        x-coordinate of generation line (m, default 0).
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
        source_position: float = 0.0,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._n_comp = n_components
        self._x_source = source_position

        omega_p = 2.0 * math.pi / period
        omega_lo = 0.5 * omega_p
        omega_hi = 3.0 * omega_p
        domegas = (omega_hi - omega_lo) / n_components

        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.manual_seed(77)

        omegas = torch.linspace(
            omega_lo + 0.5 * domegas,
            omega_hi - 0.5 * domegas,
            n_components,
        )
        phases = torch.rand(n_components, generator=rng) * 2.0 * math.pi

        # 根据谱密度计算各分量振幅
        from pyfoam.waves.enhanced_2 import _jonswap_spectrum, _pierson_moskowitz_spectrum
        spec_fn = _jonswap_spectrum if spectrum == "jonswap" else _pierson_moskowitz_spectrum

        amps = torch.tensor([
            math.sqrt(2.0 * spec_fn(w.item(), omega_p, gamma=gamma) * domegas)
            for w in omegas
        ])

        # 预计算各分量的波数
        ks = torch.tensor([_solve_dispersion(w.item(), depth) for w in omegas])

        self._omegas = omegas
        self._amps = amps
        self._phases = phases
        self._ks = ks

    @property
    def n_components(self) -> int:
        """返回分量数量。"""
        return self._n_comp

    def generate_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute irregular wave elevation via spectral superposition.

        eta(x, t) = sum_i A_i * cos(k_i * (x - x_source) - omega_i*t + phi_i)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        eta = torch.zeros_like(x, dtype=x.dtype)
        x_rel = x - self._x_source

        for i in range(self._n_comp):
            k = self._ks[i].item()
            omega = self._omegas[i].item()
            phase = k * x_rel - omega * t + self._phases[i].item()
            eta = eta + self._amps[i].item() * torch.cos(phase)
        return eta

    def generate_velocity(
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
        x_rel = x - self._x_source

        for i in range(self._n_comp):
            k = self._ks[i].item()
            omega = self._omegas[i].item()
            phase = k * x_rel - omega * t + self._phases[i].item()
            sinh_kd = math.sinh(k * d)
            coeff = self._amps[i].item() * omega / sinh_kd
            u = u + coeff * torch.cosh(k * z) * torch.cos(phase)
            w = w + coeff * torch.sinh(k * z) * torch.sin(phase)
        return u, w

    def __repr__(self) -> str:
        return (
            f"IrregularGeneration(Hs/2={self._amplitude}, "
            f"n_comp={self._n_comp}, d={self._depth})"
        )
