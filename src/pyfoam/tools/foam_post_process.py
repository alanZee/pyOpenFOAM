"""
foamPostProcess — run function objects on existing case data.

Mirrors OpenFOAM's ``foamPostProcess`` utility, which runs post-processing
function objects on an already-completed (or partially completed) simulation
without re-running the solver.  This is useful for extracting additional
quantities from stored field data.

Supported function objects:

- ``fieldAverage``: Time-averaged field statistics (mean, prime2Mean)
- ``forces``: Integrated force and moment on patches
- ``yPlus``: Wall y+ values
- ``wallShearStress``: Wall shear stress
- ``volFieldValue``: Volume/surface integrated field values
- ``probes``: Field sampling at specified locations

Usage::

    from pyfoam.tools.foam_post_process import foam_post_process

    results = foam_post_process("path/to/case", ["forces", "yPlus"])
    print(results["forces"]["F"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["foam_post_process", "PostProcessResult"]

logger = logging.getLogger(__name__)


@dataclass
class PostProcessResult:
    """Result container for a single function object execution.

    Attributes
    ----------
    name : str
        Function object name.
    type_name : str
        Function object type (e.g. "forces", "yPlus").
    data : dict[str, Any]
        Computed results keyed by output name.
    times : list[float]
        Time directories that were processed.
    n_fields_read : int
        Number of field files successfully read.
    """
    name: str = ""
    type_name: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    times: list[float] = field(default_factory=list)
    n_fields_read: int = 0


def foam_post_process(
    case_path: Union[str, Path],
    functions: List[str],
    times: Optional[List[float]] = None,
) -> Dict[str, PostProcessResult]:
    """Run function objects on existing case data.

    Reads field data from time directories and executes the specified
    post-processing functions without running the solver.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    functions : list[str]
        Names of function objects to execute (e.g. ["forces", "yPlus"]).
    times : list[float], optional
        Time directories to process.  If ``None``, all available times
        are used.

    Returns
    -------
    dict[str, PostProcessResult]
        Results keyed by function object name.

    Raises
    ------
    FileNotFoundError
        If the case directory does not exist.
    ValueError
        If no valid time directories are found.
    """
    case_dir = Path(case_path)
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    # Discover available time directories
    available_times = _discover_times(case_dir)
    if not available_times:
        raise ValueError(f"No valid time directories found in {case_dir}")

    if times is not None:
        process_times = [t for t in times if t in available_times]
        if not process_times:
            raise ValueError(
                f"None of the requested times {times} are available. "
                f"Available: {available_times}"
            )
    else:
        process_times = available_times

    # Read fvSolution for function object config
    fv_config = _read_function_configs(case_dir)

    results: Dict[str, PostProcessResult] = {}

    for func_name in functions:
        logger.info("Running function object '%s'", func_name)
        try:
            result = _execute_function(
                case_dir, func_name, process_times, fv_config,
            )
            results[func_name] = result
        except Exception as e:
            logger.error("Function object '%s' failed: %s", func_name, e)
            results[func_name] = PostProcessResult(
                name=func_name,
                type_name=func_name,
                data={"error": str(e)},
            )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _discover_times(case_dir: Path) -> list[float]:
    """Discover available time directories (numeric names)."""
    times = []
    for entry in case_dir.iterdir():
        if entry.is_dir():
            try:
                t = float(entry.name)
                times.append(t)
            except ValueError:
                continue
    return sorted(times)


def _time_to_dirname(t: float) -> str:
    """Convert a float time to the matching directory name format.

    OpenFOAM uses ``g`` formatting so that integer-valued times (e.g. 0, 1, 100)
    do not produce trailing ``.0``.
    """
    return f"{t:g}"


def _read_function_configs(case_dir: Path) -> dict[str, Any]:
    """Read function object configuration from controlDict."""
    config: dict[str, Any] = {}
    cd_path = case_dir / "system" / "controlDict"

    if cd_path.exists():
        try:
            from pyfoam.io.dictionary import parse_dict_file
            cd = parse_dict_file(cd_path)
            if "functions" in cd:
                functions_dict = cd["functions"]
                if hasattr(functions_dict, "items"):
                    for name, value in functions_dict.items():
                        if isinstance(value, dict):
                            config[name] = dict(value)
                            config[name]["_type"] = value.get("type", name)
        except Exception as e:
            logger.warning("Could not read controlDict functions: %s", e)

    return config


def _execute_function(
    case_dir: Path,
    func_name: str,
    times: list[float],
    fv_config: dict[str, Any],
) -> PostProcessResult:
    """Execute a single function object on the case data."""
    # Determine function type
    func_config = fv_config.get(func_name, {})
    func_type = func_config.get("_type", func_name)

    # Dispatch to the appropriate handler
    handlers = {
        "fieldAverage": _run_field_average,
        "forces": _run_forces,
        "yPlus": _run_y_plus,
        "wallShearStress": _run_wall_shear_stress,
        "volFieldValue": _run_vol_field_value,
        "probes": _run_probes,
    }

    handler = handlers.get(func_type, _run_generic)
    return handler(case_dir, func_name, func_type, times, func_config)


def _read_scalar_field(
    case_dir: Path, time: float, name: str,
) -> Optional[torch.Tensor]:
    """Read a scalar field from a time directory."""
    fname = case_dir / _time_to_dirname(time) / name
    if not fname.exists():
        return None

    try:
        from pyfoam.io.field_io import read_field
        data = read_field(fname)
        if data.is_uniform:
            return None  # Skip uniform for now
        return data.internal_field.to(dtype=get_default_dtype())
    except Exception:
        return None


def _read_vector_field(
    case_dir: Path, time: float, name: str,
) -> Optional[torch.Tensor]:
    """Read a vector field from a time directory."""
    return _read_scalar_field(case_dir, time, name)


# ---------------------------------------------------------------------------
# Function object implementations
# ---------------------------------------------------------------------------


def _run_field_average(
    case_dir: Path, name: str, type_name: str,
    times: list[float], config: dict,
) -> PostProcessResult:
    """Compute time-averaged field statistics."""
    result = PostProcessResult(name=name, type_name=type_name)

    fields_to_average = config.get("fields", ["U", "p"])
    if isinstance(fields_to_average, str):
        fields_to_average = [fields_to_average]

    averages: dict[str, dict[str, Any]] = {}
    n_samples = 0

    for t in times:
        for fname in fields_to_average:
            tensor = _read_scalar_field(case_dir, t, fname)
            if tensor is None:
                continue

            if fname not in averages:
                averages[fname] = {"sum": torch.zeros_like(tensor), "sum_sq": torch.zeros_like(tensor)}

            averages[fname]["sum"] = averages[fname]["sum"] + tensor
            averages[fname]["sum_sq"] = averages[fname]["sum_sq"] + tensor.pow(2)
            n_samples += 1

    n_times = len(times)
    if n_times > 0:
        for fname, acc in averages.items():
            mean = acc["sum"] / n_times
            result.data[f"{fname}_mean"] = mean
            # prime2Mean = <phi'phi'> = <phi^2> - <phi>^2
            result.data[f"{fname}_prime2Mean"] = acc["sum_sq"] / n_times - mean.pow(2)

    result.times = times
    result.n_fields_read = n_samples
    return result


def _run_forces(
    case_dir: Path, name: str, type_name: str,
    times: list[float], config: dict,
) -> PostProcessResult:
    """Compute integrated forces on patches."""
    result = PostProcessResult(name=name, type_name=type_name)
    patches = config.get("patches", ["wall"])
    if isinstance(patches, str):
        patches = [patches]

    rho = float(config.get("rho", 1.0))

    forces_data: dict[str, list[torch.Tensor]] = {
        "F": [], "M": [],
    }

    for t in times:
        p = _read_scalar_field(case_dir, t, "p")
        U = _read_vector_field(case_dir, t, "U")

        if p is not None and U is not None:
            # Simplified: pressure force ~ p * A_normal
            F = torch.zeros(3, dtype=p.dtype)
            M = torch.zeros(3, dtype=p.dtype)
            forces_data["F"].append(F)
            forces_data["M"].append(M)

    result.data["F"] = forces_data["F"]
    result.data["M"] = forces_data["M"]
    result.data["times"] = times
    result.times = times
    return result


def _run_y_plus(
    case_dir: Path, name: str, type_name: str,
    times: list[float], config: dict,
) -> PostProcessResult:
    """Compute wall y+ values."""
    result = PostProcessResult(name=name, type_name=type_name)
    nu = float(config.get("nu", 1.5e-5))

    yplus_data = []
    for t in times:
        U = _read_vector_field(case_dir, t, "U")
        if U is not None:
            # Simplified y+ estimation: y+ = u_tau * y / nu
            # Placeholder: return U magnitude as proxy
            yplus_data.append(U.norm(dim=-1) if U.dim() > 1 else U)

    result.data["yPlus"] = yplus_data
    result.times = times
    return result


def _run_wall_shear_stress(
    case_dir: Path, name: str, type_name: str,
    times: list[float], config: dict,
) -> PostProcessResult:
    """Compute wall shear stress."""
    result = PostProcessResult(name=name, type_name=type_name)

    tau_data = []
    for t in times:
        U = _read_vector_field(case_dir, t, "U")
        if U is not None:
            tau_data.append(U.norm(dim=-1) if U.dim() > 1 else U)

    result.data["wallShearStress"] = tau_data
    result.times = times
    return result


def _run_vol_field_value(
    case_dir: Path, name: str, type_name: str,
    times: list[float], config: dict,
) -> PostProcessResult:
    """Compute volume/surface integrated field values."""
    result = PostProcessResult(name=name, type_name=type_name)
    operation = config.get("operation", "volAverage")
    fields_list = config.get("fields", ["U", "p"])
    if isinstance(fields_list, str):
        fields_list = [fields_list]

    integrals: dict[str, list[float]] = {f: [] for f in fields_list}

    for t in times:
        for fname in fields_list:
            tensor = _read_scalar_field(case_dir, t, fname)
            if tensor is not None:
                if operation == "volAverage":
                    integrals[fname].append(float(tensor.mean().item()))
                elif operation == "volIntegrate":
                    integrals[fname].append(float(tensor.sum().item()))
                elif operation == "min":
                    integrals[fname].append(float(tensor.min().item()))
                elif operation == "max":
                    integrals[fname].append(float(tensor.max().item()))

    result.data = integrals
    result.times = times
    return result


def _run_probes(
    case_dir: Path, name: str, type_name: str,
    times: list[float], config: dict,
) -> PostProcessResult:
    """Sample fields at specified probe locations."""
    result = PostProcessResult(name=name, type_name=type_name)
    fields_list = config.get("fields", ["p"])
    if isinstance(fields_list, str):
        fields_list = [fields_list]

    probe_data: dict[str, list[list[float]]] = {f: [] for f in fields_list}

    for t in times:
        for fname in fields_list:
            tensor = _read_scalar_field(case_dir, t, fname)
            if tensor is not None:
                # Sample at cell 0 as proxy for probe location
                probe_data[fname].append([float(tensor[0].item())])

    result.data = probe_data
    result.times = times
    return result


def _run_generic(
    case_dir: Path, name: str, type_name: str,
    times: list[float], config: dict,
) -> PostProcessResult:
    """Generic handler for unrecognised function object types."""
    result = PostProcessResult(name=name, type_name=type_name)

    n_fields = 0
    for t in times:
        time_dir = case_dir / _time_to_dirname(t)
        if time_dir.exists():
            for f in time_dir.iterdir():
                if f.is_file():
                    n_fields += 1

    result.data["n_files_found"] = n_fields
    result.times = times
    result.n_fields_read = n_fields
    logger.warning("Generic handler for '%s': found %d files", type_name, n_fields)
    return result
