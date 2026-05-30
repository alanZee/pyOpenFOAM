"""
Enhanced injection models v6.

- :class:`InjectionFromFile` -- inject from data table
- :class:`ResettableInjector` -- resettable injection wrapper
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.injection import Injector

__all__ = ["InjectionFromFile", "ResettableInjector"]


class InjectionFromFile(Injector):
    """Inject particles from a pre-computed data table.

    Parameters
    ----------
    data_table : list[dict]
        List of particle definitions with 'position', 'velocity', etc.
    """

    def __init__(self, data_table=None, **kw):
        self.data_table = data_table or []

    def inject(self):
        from pyfoam.lagrangian.particle import Particle
        particles = []
        for row in self.data_table:
            particles.append(Particle(
                position=list(row.get('position', [0, 0, 0])),
                velocity=list(row.get('velocity', [0, 0, 0])),
                diameter=row.get('diameter', 1e-4),
                density=row.get('density', 1000.0),
                temperature=row.get('temperature', 300.0),
            ))
        return particles


class ResettableInjector(Injector):
    """An injector wrapper that supports reset and re-injection.

    Parameters
    ----------
    inner : Injector
        The wrapped injector.
    """

    def __init__(self, inner=None, **kw):
        self._inner = inner
        self._injection_count = 0

    def inject(self):
        self._injection_count += 1
        if self._inner:
            return self._inner.inject()
        return []

    def reset(self):
        """重置注入计数。"""
        self._injection_count = 0
