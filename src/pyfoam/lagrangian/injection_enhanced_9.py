"""
Enhanced injection models v9.

- :class:`TemporalInjector` -- time-windowed injection
- :class:`ProbabilisticInjector` -- stochastic activation
"""

from __future__ import annotations

import math
import random as _rng

from pyfoam.lagrangian.injection import Injector

__all__ = ["TemporalInjector", "ProbabilisticInjector"]


class TemporalInjector(Injector):
    """Inject particles only during specified time windows.

    Parameters
    ----------
    inner : Injector
        The wrapped injector.
    time_windows : list[tuple[float, float]]
        List of (t_start, t_end) windows.
    """

    def __init__(self, inner=None, time_windows=None, **kw):
        self._inner = inner
        self.time_windows = time_windows or [(0.0, 1e10)]
        self._current_time = 0.0

    def inject(self):
        for t_start, t_end in self.time_windows:
            if t_start <= self._current_time <= t_end:
                result = self._inner.inject() if self._inner else []
                self._current_time += 1e-4
                return result
        self._current_time += 1e-4
        return []

    def advance_time(self, dt):
        """推进当前时间。"""
        self._current_time += dt


class ProbabilisticInjector(Injector):
    """Stochastic activation of injection.

    Parameters
    ----------
    inner : Injector
        The wrapped injector.
    activation_probability : float
        Probability of injection (0-1). Default ``1.0``.
    seed : int or None
        Random seed.
    """

    def __init__(self, inner=None, activation_probability=1.0, seed=None, **kw):
        self._inner = inner
        self.activation_probability = activation_probability
        self._rng = _rng.Random(seed)

    def inject(self):
        if self._rng.random() < self.activation_probability:
            return self._inner.inject() if self._inner else []
        return []
