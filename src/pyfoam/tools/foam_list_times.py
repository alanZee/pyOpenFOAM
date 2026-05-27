"""
foamListTimes — list time directories in an OpenFOAM case.

Mirrors OpenFOAM's ``foamListTimes`` utility.  Scans a case directory for
numeric time folders (e.g. ``0``, ``0.001``, ``1``), optionally applying
a time selector to return a filtered subset.

Supported selectors:

- ``None`` — return all time directories.
- ``"latestTime"`` — return only the largest time value.
- ``"firstTime"`` — return only the smallest time value.
- A numeric value (``int`` or ``float``) — return all times up to and
  including the given value.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

__all__ = ["foam_list_times"]


# Matches directory names that are plain numbers (possibly with decimals),
# e.g. ``0``, ``0.005``, ``1``, ``100``.
_TIME_DIR_RE = re.compile(r"^\d+(?:\.\d+)?$")


def foam_list_times(
    case_path: Union[str, Path],
    time_selector: Optional[Union[str, float, int]] = None,
) -> List[float]:
    """List time directories in an OpenFOAM case.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    time_selector : str, float, int, or None, optional
        Filter to apply:

        - ``None`` — return all time directories (sorted ascending).
        - ``"latestTime"`` — single-element list with the largest time.
        - ``"firstTime"`` — single-element list with the smallest time.
        - numeric value — return all times *≤* the given value.

    Returns
    -------
    list of float
        Sorted (ascending) list of time values matching the selector.

    Raises
    ------
    FileNotFoundError
        If *case_path* does not exist or is not a directory.
    ValueError
        If *time_selector* is an unrecognised string.
    """
    case_dir = Path(case_path)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    # Collect numeric directory names
    times: list[float] = []
    for entry in case_dir.iterdir():
        if entry.is_dir() and _TIME_DIR_RE.match(entry.name):
            try:
                times.append(float(entry.name))
            except ValueError:
                continue

    times.sort()

    # --- No selector: return all ---
    if time_selector is None:
        return times

    # --- String selectors ---
    if isinstance(time_selector, str):
        if time_selector == "latestTime":
            return [times[-1]] if times else []
        if time_selector == "firstTime":
            return [times[0]] if times else []
        raise ValueError(
            f"Unknown time_selector string: {time_selector!r}. "
            "Expected 'latestTime' or 'firstTime'."
        )

    # --- Numeric selector: return all times ≤ value ---
    max_time = float(time_selector)
    return [t for t in times if t <= max_time + 1e-12]
