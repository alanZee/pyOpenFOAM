"""
Enhanced injection models v3.

Adds LagrangianMappingInjector and ManualInjectionRate following OpenFOAM conventions.

- :class:`LagrangianMappingInjector` — inject with Lagrangian-Eulerian field mapping
- :class:`ManualInjectionRate`       — inject with user-specified time-dependent rate
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.injection import Injector
from pyfoam.lagrangian.particle import Particle


__all__ = ["LagrangianMappingInjector", "ManualInjectionRate"]


class LagrangianMappingInjector(Injector):
    """Inject particles with Lagrangian-to-Eulerian field mapping info.

    Each particle carries a ``cell_centre`` attribute indicating which
    Eulerian cell it maps to, facilitating two-way coupling.

    Parameters
    ----------
    cell_centres : list[list[float]]
        List of Eulerian cell centre positions ``[[x,y,z], ...]``.
    particles_per_cell : int
        Number of particles injected per cell.
    velocity : list[float]
        Uniform injection velocity ``[u, v, w]`` (m/s).
    diameter : float
        Particle diameter (m).
    density : float
        Particle material density (kg/m³).
    temperature : float
        Initial particle temperature (K).
    seed : int or None
        Random seed for position jitter within cells.
    """

    def __init__(
        self,
        cell_centres: list[list[float]],
        particles_per_cell: int = 1,
        velocity: list[float] | None = None,
        diameter: float = 1e-4,
        density: float = 1000.0,
        temperature: float = 300.0,
        seed: int | None = None,
    ) -> None:
        if not cell_centres:
            raise ValueError("cell_centres must be non-empty.")
        if particles_per_cell < 1:
            raise ValueError(f"particles_per_cell must be >= 1, got {particles_per_cell}")
        self.cell_centres = [list(c) for c in cell_centres]
        self.particles_per_cell = particles_per_cell
        self.velocity = velocity if velocity is not None else [0.0, 0.0, 0.0]
        self.diameter = diameter
        self.density = density
        self.temperature = temperature
        self.seed = seed

    def inject(self) -> list[Particle]:
        """Inject particles at cell centres with mapping info."""
        rng = random.Random(self.seed)
        particles: list[Particle] = []
        for cell_idx, cc in enumerate(self.cell_centres):
            for _ in range(self.particles_per_cell):
                # 轻微随机偏移，模拟单元内分布
                pos = [cc[j] + rng.gauss(0.0, 1e-6) for j in range(3)]
                p = Particle(
                    position=pos,
                    velocity=list(self.velocity),
                    diameter=self.diameter,
                    density=self.density,
                    temperature=self.temperature,
                )
                # 附加映射信息
                p.cell_id = cell_idx
                particles.append(p)
        return particles


class ManualInjectionRate(Injector):
    """Inject particles at a user-specified time-dependent rate.

    Supports a piecewise-linear injection rate table.  At each call to
    ``inject``, the specified number of particles is produced.

    Parameters
    ----------
    rate_table : list[tuple[float, int]]
        List of ``(time, n_particles)`` entries defining the injection
        schedule.  Particles are injected according to the current entry.
    origin : list[float]
        Injection point ``[x, y, z]`` (m).
    velocity : list[float]
        Particle velocity ``[u, v, w]`` (m/s).
    diameter : float
        Particle diameter (m).
    density : float
        Particle material density (kg/m³).
    temperature : float
        Initial particle temperature (K).
    """

    def __init__(
        self,
        rate_table: list[tuple[float, int]],
        origin: list[float] | None = None,
        velocity: list[float] | None = None,
        diameter: float = 1e-4,
        density: float = 1000.0,
        temperature: float = 300.0,
    ) -> None:
        if not rate_table:
            raise ValueError("rate_table must be non-empty.")
        self.rate_table = sorted(rate_table, key=lambda x: x[0])
        self.origin = origin if origin is not None else [0.0, 0.0, 0.0]
        self.velocity = velocity if velocity is not None else [0.0, 0.0, 0.0]
        self.diameter = diameter
        self.density = density
        self.temperature = temperature
        self._current_index = 0

    def inject(self) -> list[Particle]:
        """Inject particles according to the current rate table entry."""
        if self._current_index >= len(self.rate_table):
            return []

        _, n_particles = self.rate_table[self._current_index]
        self._current_index += 1

        particles: list[Particle] = []
        for _ in range(n_particles):
            particles.append(
                Particle(
                    position=list(self.origin),
                    velocity=list(self.velocity),
                    diameter=self.diameter,
                    density=self.density,
                    temperature=self.temperature,
                )
            )
        return particles

    def reset(self) -> None:
        """重置注入速率表索引。"""
        self._current_index = 0
