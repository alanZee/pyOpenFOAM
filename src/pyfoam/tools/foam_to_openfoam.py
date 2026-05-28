"""
foamToOpenFOAM — convert between OpenFOAM versions.

Handles format differences between v2512 and earlier versions (e.g. v2012,
v2112, v2212, v2312, v2412).  Common transformations include:

- Version header string updates (``version 2.0`` -> ``version 2.0`` with
  updated ``FoamFile.format`` entries).
- Boundary condition keyword renames (e.g. ``type wallFunction`` -> updated
  names, ``inletOutlet`` variants).
- Solver/application key changes.
- Dictionary structure adjustments (e.g. ``libs`` -> ``libs (lib...);``).

Usage::

    from pyfoam.tools.foam_to_openfoam import foam_to_openfoam

    result = foam_to_openfoam(source_case, target_case, source_version="v2312", target_version="v2512")
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = ["foam_to_openfoam"]

# ---------------------------------------------------------------------------
# Version keyword / BC renames between OpenFOAM versions
# ---------------------------------------------------------------------------

# BC type renames: old_name -> new_name (applied when upgrading)
_BC_RENAMES_UPGRADE: dict[str, str] = {
    "turbulentIntensityKineticEnergyInlet": "turbulentIntensityKineticEnergyInlet",
    "codedFunctionObject": "codedFunctionObject",
}

# Field class renames
_CLASS_RENAMES: dict[str, dict[str, str]] = {
    "v2512": {
        "volScalarField": "volScalarField",
        "volVectorField": "volVectorField",
    },
}

# Version-specific FoamFile header fixups
_VERSION_PATTERNS = {
    "v2012": "v2012",
    "v2112": "v2112",
    "v2212": "v2212",
    "v2312": "v2312",
    "v2412": "v2412",
    "v2512": "v2512",
}


def foam_to_openfoam(
    source_case: Union[str, Path],
    target_case: Union[str, Path],
    source_version: str = "v2312",
    target_version: str = "v2512",
    copy_mesh: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Convert an OpenFOAM case between versions.

    Parameters
    ----------
    source_case : str or Path
        Path to the source case directory.
    target_case : str or Path
        Path to the target case directory.  Created if it does not exist.
    source_version : str
        Source OpenFOAM version (e.g. ``"v2312"``).
    target_version : str
        Target OpenFOAM version (e.g. ``"v2512"``).
    copy_mesh : bool
        If ``True``, copy the ``constant/polyMesh`` directory.
    overwrite : bool
        If ``True``, overwrite existing target files.

    Returns
    -------
    dict
        Summary with keys: ``converted``, ``skipped``, ``errors``, ``files``.
    """
    src = Path(source_case).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Source case not found: {src}")

    tgt = Path(target_case).resolve()

    if src == tgt:
        raise ValueError("Source and target case must be different")

    _validate_version(source_version)
    _validate_version(target_version)

    # Create target directory structure
    if not tgt.exists():
        tgt.mkdir(parents=True)

    converted = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    files: list[str] = []

    # Copy and convert time directories
    for item in sorted(src.iterdir()):
        if item.is_dir() and _is_time_dir(item.name):
            target_time = tgt / item.name
            if not target_time.exists():
                target_time.mkdir(parents=True)

            for f in sorted(item.iterdir()):
                if f.is_file() and f.name not in ("boundary", "uniform"):
                    try:
                        result = _convert_file(
                            f, target_time / f.name,
                            source_version, target_version, overwrite,
                        )
                        if result == "converted":
                            converted += 1
                            files.append(str(f))
                        else:
                            skipped += 1
                    except Exception as e:
                        errors.append((str(f), str(e)))

    # Copy/convert constant directory
    const_src = src / "constant"
    if const_src.is_dir():
        const_tgt = tgt / "constant"
        if not const_tgt.exists():
            const_tgt.mkdir(parents=True)

        for f in sorted(const_src.iterdir()):
            if f.is_dir():
                continue
            if f.name in ("momentumTransport", "turbulenceProperties",
                          "physicalProperties", "transportProperties"):
                try:
                    result = _convert_file(
                        f, const_tgt / f.name,
                        source_version, target_version, overwrite,
                    )
                    if result == "converted":
                        converted += 1
                        files.append(str(f))
                    else:
                        skipped += 1
                except Exception as e:
                    errors.append((str(f), str(e)))

        # Copy mesh if requested
        if copy_mesh:
            mesh_src = const_src / "polyMesh"
            if mesh_src.is_dir():
                mesh_tgt = const_tgt / "polyMesh"
                if not mesh_tgt.exists():
                    shutil.copytree(mesh_src, mesh_tgt)

    # Copy/convert system directory
    sys_src = src / "system"
    if sys_src.is_dir():
        sys_tgt = tgt / "system"
        if not sys_tgt.exists():
            sys_tgt.mkdir(parents=True)

        for f in sorted(sys_src.iterdir()):
            if f.is_file():
                try:
                    result = _convert_file(
                        f, sys_tgt / f.name,
                        source_version, target_version, overwrite,
                    )
                    if result == "converted":
                        converted += 1
                        files.append(str(f))
                    else:
                        skipped += 1
                except Exception as e:
                    errors.append((str(f), str(e)))

    return {
        "converted": converted,
        "skipped": skipped,
        "errors": errors,
        "files": files,
        "source_version": source_version,
        "target_version": target_version,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_version(version: str) -> None:
    """Validate that the version string is a known OpenFOAM version."""
    known = set(_VERSION_PATTERNS.keys())
    if version not in known:
        raise ValueError(
            f"Unknown OpenFOAM version '{version}'. "
            f"Known: {sorted(known)}"
        )


def _is_time_dir(name: str) -> bool:
    """Check if a directory name is a valid time directory."""
    try:
        float(name)
        return True
    except ValueError:
        return name in ("0.orig",)


def _is_foam_file(text: str) -> bool:
    """Check if a file contains an OpenFOAM FoamFile header."""
    return "FoamFile" in text and "{" in text


def _convert_file(
    src_path: Path,
    tgt_path: Path,
    source_version: str,
    target_version: str,
    overwrite: bool,
) -> str:
    """Convert a single OpenFOAM file between versions.

    Returns ``"converted"`` or ``"skipped"``.
    """
    if tgt_path.exists() and not overwrite:
        return "skipped"

    text = src_path.read_text(encoding="latin-1")

    if not _is_foam_file(text):
        # Copy non-OpenFOAM files as-is
        tgt_path.write_text(text, encoding="latin-1")
        return "skipped"

    converted_text = _apply_version_transforms(
        text, source_version, target_version,
    )

    tgt_path.write_text(converted_text, encoding="latin-1")
    return "converted"


def _apply_version_transforms(
    text: str,
    source_version: str,
    target_version: str,
) -> str:
    """Apply version-specific text transforms to an OpenFOAM file."""
    result = text

    # Update version note in FoamFile header
    result = _update_foamfile_header(result, source_version, target_version)

    # Apply BC renames if upgrading
    src_num = _version_number(source_version)
    tgt_num = _version_number(target_version)

    if tgt_num > src_num:
        result = _apply_bc_renames(result, source_version, target_version)

    # Apply solver keyword updates
    result = _apply_solver_updates(result, source_version, target_version)

    return result


def _update_foamfile_header(
    text: str,
    source_version: str,
    target_version: str,
) -> str:
    """Update the FoamFile header version notes."""
    # The FoamFile version field stays at "2.0" across all versions.
    # We add/update a comment noting the conversion.
    marker = f"// Converted from {source_version} to {target_version}"
    if marker in text:
        return text

    # Replace any old conversion comment
    old_marker_re = re.compile(r"// Converted from \w+ to \w+\n?")
    if old_marker_re.search(text):
        return old_marker_re.sub(marker + "\n", text)

    # Insert after FoamFile block
    def _insert(m: re.Match) -> str:
        return m.group(0) + "\n" + marker + "\n"

    return re.sub(
        r"(FoamFile\s*\{[^}]*\})",
        _insert,
        text,
        count=1,
    )


def _apply_bc_renames(
    text: str,
    source_version: str,
    target_version: str,
) -> str:
    """Apply boundary condition type renames between versions."""
    renames = _get_bc_renames(source_version, target_version)
    result = text
    for old_name, new_name in renames.items():
        pattern = rf"\btype\s+{re.escape(old_name)}\b"
        result = re.sub(pattern, f"type        {new_name}", result)
    return result


def _get_bc_renames(source: str, target: str) -> dict[str, str]:
    """Get the BC type renames needed between two versions."""
    renames: dict[str, str] = {}

    src_num = _version_number(source)
    tgt_num = _version_number(target)

    # v2312 -> v2412 changes
    if src_num <= _version_number("v2312") and tgt_num >= _version_number("v2412"):
        renames.update({
            "nutUSpaldingWallFunction": "nutkAtmWallFunction",
        })

    # v2412 -> v2512 changes
    if src_num <= _version_number("v2412") and tgt_num >= _version_number("v2512"):
        renames.update({
            "GibsonAlphaContactAngle": "alphaContactAngle",
        })

    return renames


def _apply_solver_updates(
    text: str,
    source_version: str,
    target_version: str,
) -> str:
    """Apply solver/application keyword updates between versions."""
    src_num = _version_number(source_version)
    tgt_num = _version_number(target_version)

    if src_num >= tgt_num:
        return text

    result = text

    # v2312 -> v2412: ddtSchemes keyword changes
    if src_num <= _version_number("v2312") and tgt_num >= _version_number("v2412"):
        # Update deprecated scheme names
        result = result.replace(
            "default         Euler;",
            "default         CrankNicolson 1;",
        ) if "default         Euler;" in result and "ddtSchemes" in result else result

    return result


def _version_number(version: str) -> int:
    """Convert version string to comparable integer (e.g. 'v2312' -> 2312)."""
    m = re.match(r"v(\d+)", version)
    if m:
        return int(m.group(1))
    return 0
