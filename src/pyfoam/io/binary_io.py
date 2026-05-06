"""
Binary format handling for OpenFOAM files.

OpenFOAM binary format uses big-endian (network byte order) IEEE 754 doubles
and int32 values. The binary data sits between ASCII headers and ``(`` / `)``
markers.

Binary encoding schemes:
- Scalars: single value (8 bytes double)
- Fields: N values (N × 8 bytes double)
- Points: N × 3 values (N × 3 × 8 bytes double)
- Faces: CompactListList — (N+1) × 4 bytes offsets, then Σ(offsets[i+1] - offsets[i]) × 4 bytes
- Owner/Neighbour: N × 4 bytes int32

All binary data is big-endian regardless of host architecture.
"""

from __future__ import annotations

import struct
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import BinaryIO, Sequence, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "BinaryReader",
    "BinaryWriter",
    "read_binary_scalar",
    "read_binary_label_list",
    "read_binary_scalar_list",
    "read_binary_points",
    "read_binary_faces",
    "read_binary_compact_list_list",
    "write_binary_scalar",
    "write_binary_label_list",
    "write_binary_scalar_list",
    "write_binary_points",
    "write_binary_faces",
    "write_binary_compact_list_list",
]

# OpenFOAM uses big-endian (network byte order)
_ENDIAN = ">"  # big-endian

# Struct format codes
_DOUBLE_FMT = f"{_ENDIAN}d"   # 8-byte IEEE 754 double
_INT32_FMT = f"{_ENDIAN}i"    # 4-byte signed int32
_UINT32_FMT = f"{_ENDIAN}I"   # 4-byte unsigned int32

_DOUBLE_SIZE = struct.calcsize(_DOUBLE_FMT)
_INT32_SIZE = struct.calcsize(_INT32_FMT)


# ---------------------------------------------------------------------------
# BinaryReader
# ---------------------------------------------------------------------------


class BinaryReader:
    """Read big-endian binary data from an OpenFOAM binary file section.

    The reader wraps a binary stream positioned at the start of the binary
    data (after the ASCII header and ``(`` marker).

    Usage::

        reader = BinaryReader(stream)
        n = reader.read_int32()
        values = reader.read_doubles(n)
    """

    def __init__(self, source: Union[str, Path, BinaryIO, bytes]) -> None:
        if isinstance(source, (str, Path)):
            self._stream: BinaryIO = open(source, "rb")
            self._owns_stream = True
        elif isinstance(source, bytes):
            self._stream = BytesIO(source)
            self._owns_stream = False
        else:
            self._stream = source
            self._owns_stream = False

    def close(self) -> None:
        """Close the underlying stream if we opened it."""
        if self._owns_stream:
            self._stream.close()

    def __enter__(self) -> BinaryReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- Primitive reads ----------------------------------------------------

    def read_bytes(self, n: int) -> bytes:
        """Read exactly *n* bytes from the stream."""
        data = self._stream.read(n)
        if len(data) < n:
            raise EOFError(
                f"Expected {n} bytes, got {len(data)}"
            )
        return data

    def read_double(self) -> float:
        """Read a single big-endian IEEE 754 double (8 bytes)."""
        return struct.unpack(_DOUBLE_FMT, self.read_bytes(_DOUBLE_SIZE))[0]

    def read_int32(self) -> int:
        """Read a single big-endian int32 (4 bytes)."""
        return struct.unpack(_INT32_FMT, self.read_bytes(_INT32_SIZE))[0]

    def read_uint32(self) -> int:
        """Read a single big-endian unsigned int32 (4 bytes)."""
        return struct.unpack(_UINT32_FMT, self.read_bytes(_INT32_SIZE))[0]

    # -- Array reads --------------------------------------------------------

    def read_doubles(self, n: int) -> np.ndarray:
        """Read *n* big-endian doubles into a float64 array."""
        data = self.read_bytes(n * _DOUBLE_SIZE)
        return np.frombuffer(data, dtype=">f8", count=n).astype(np.float64)

    def read_int32s(self, n: int) -> np.ndarray:
        """Read *n* big-endian int32 values into an int32 array."""
        data = self.read_bytes(n * _INT32_SIZE)
        return np.frombuffer(data, dtype=">i4", count=n).astype(np.int32)

    # -- Marker scanning ----------------------------------------------------

    def skip_to_open_paren(self) -> None:
        """Skip bytes until ``(`` is found (start of binary data block).

        The ``(`` byte itself is consumed.
        """
        while True:
            b = self._stream.read(1)
            if not b:
                raise EOFError("Reached end of stream before '(' marker")
            if b == b"(":
                return

    def skip_to_close_paren(self) -> None:
        """Skip bytes until ``)`` is found (end of binary data block).

        The ``)`` byte itself is consumed.
        """
        while True:
            b = self._stream.read(1)
            if not b:
                raise EOFError("Reached end of stream before ')' marker")
            if b == b")":
                return

    def read_until_close_paren(self) -> bytes:
        """Read all bytes until ``)`` marker (exclusive of the marker)."""
        chunks: list[bytes] = []
        while True:
            b = self._stream.read(1)
            if not b:
                raise EOFError("Reached end of stream before ')' marker")
            if b == b")":
                return b"".join(chunks)
            chunks.append(b)

    # -- High-level reads ---------------------------------------------------

    def read_binary_scalar_field(self, n: int) -> torch.Tensor:
        """Read *n* scalar values (doubles) and return as a torch tensor."""
        arr = self.read_doubles(n)
        return torch.tensor(arr, dtype=get_default_dtype(), device=get_device())

    def read_binary_label_field(self, n: int) -> torch.Tensor:
        """Read *n* int32 values and return as a torch int64 tensor."""
        arr = self.read_int32s(n)
        return torch.tensor(arr, dtype=torch.int64, device=get_device())

    def read_binary_vector_field(self, n: int) -> torch.Tensor:
        """Read *n* vector values (n×3 doubles) and return as (n, 3) tensor."""
        arr = self.read_doubles(n * 3)
        return torch.tensor(
            arr.reshape(n, 3), dtype=get_default_dtype(), device=get_device()
        )

    def read_binary_compact_list_list(self) -> list[np.ndarray]:
        """Read a CompactListList encoding.

        CompactListList format:
        1. Read N (int32) — number of sub-lists
        2. Read N+1 offsets (int32) — cumulative sizes
        3. Read Σ data values — the flattened data

        Returns:
            List of numpy arrays, one per sub-list.
        """
        n = self.read_int32()
        offsets = self.read_int32s(n + 1)
        total = int(offsets[-1])
        data = self.read_int32s(total)

        result: list[np.ndarray] = []
        for i in range(n):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            result.append(data[start:end])
        return result


# ---------------------------------------------------------------------------
# BinaryWriter
# ---------------------------------------------------------------------------


class BinaryWriter:
    """Write big-endian binary data for OpenFOAM binary files.

    Usage::

        writer = BinaryWriter()
        writer.write_int32(42)
        writer.write_doubles([1.0, 2.0, 3.0])
        data = writer.get_bytes()
    """

    def __init__(self) -> None:
        self._buffer = BytesIO()

    def get_bytes(self) -> bytes:
        """Return the accumulated binary data."""
        return self._buffer.getvalue()

    def write_to(self, stream: BinaryIO) -> None:
        """Write accumulated data to *stream*."""
        stream.write(self.get_bytes())

    # -- Primitive writes ---------------------------------------------------

    def write_bytes(self, data: bytes) -> None:
        """Write raw bytes."""
        self._buffer.write(data)

    def write_double(self, value: float) -> None:
        """Write a single big-endian IEEE 754 double."""
        self._buffer.write(struct.pack(_DOUBLE_FMT, value))

    def write_int32(self, value: int) -> None:
        """Write a single big-endian int32."""
        self._buffer.write(struct.pack(_INT32_FMT, value))

    def write_uint32(self, value: int) -> None:
        """Write a single big-endian unsigned int32."""
        self._buffer.write(struct.pack(_UINT32_FMT, value))

    # -- Array writes -------------------------------------------------------

    def write_doubles(self, values: Sequence[float] | np.ndarray) -> None:
        """Write an array of doubles in big-endian format."""
        arr = np.asarray(values, dtype=">f8")
        self._buffer.write(arr.tobytes())

    def write_int32s(self, values: Sequence[int] | np.ndarray) -> None:
        """Write an array of int32 values in big-endian format."""
        arr = np.asarray(values, dtype=">i4")
        self._buffer.write(arr.tobytes())

    # -- High-level writes --------------------------------------------------

    def write_marker_open(self) -> None:
        """Write the ``(`` marker (start of binary data)."""
        self._buffer.write(b"(")

    def write_marker_close(self) -> None:
        """Write the ``)`` marker (end of binary data)."""
        self._buffer.write(b")")

    def write_binary_scalar(self, value: float) -> None:
        """Write a single scalar value."""
        self.write_double(value)

    def write_binary_scalar_list(self, values: torch.Tensor | np.ndarray | Sequence[float]) -> None:
        """Write a list of scalar values."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy().astype(np.float64)
        self.write_doubles(values)

    def write_binary_label_list(self, values: torch.Tensor | np.ndarray | Sequence[int]) -> None:
        """Write a list of int32 label values."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy().astype(np.int32)
        arr = np.asarray(values, dtype=np.int32)
        self.write_int32s(arr)

    def write_binary_points(self, points: torch.Tensor | np.ndarray) -> None:
        """Write points as flattened N×3 doubles."""
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy().astype(np.float64)
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr.ravel()
        self.write_doubles(arr)

    def write_binary_faces(self, faces: list[np.ndarray] | list[list[int]]) -> None:
        """Write faces using CompactListList encoding.

        Args:
            faces: List of face vertex-index arrays.
        """
        n = len(faces)
        self.write_int32(n)

        # Build offsets (cumulative sizes)
        offsets = np.zeros(n + 1, dtype=np.int32)
        for i, face in enumerate(faces):
            offsets[i + 1] = offsets[i] + len(face)
        self.write_int32s(offsets)

        # Write flattened face data
        for face in faces:
            arr = np.asarray(face, dtype=np.int32)
            self.write_int32s(arr)

    def write_binary_compact_list_list(self, lists: list[np.ndarray]) -> None:
        """Write a CompactListList encoding.

        Args:
            lists: List of int32 arrays.
        """
        n = len(lists)
        self.write_int32(n)

        offsets = np.zeros(n + 1, dtype=np.int32)
        for i, lst in enumerate(lists):
            offsets[i + 1] = offsets[i] + len(lst)
        self.write_int32s(offsets)

        for lst in lists:
            arr = np.asarray(lst, dtype=np.int32)
            self.write_int32s(arr)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def read_binary_scalar(source: Union[str, Path, BinaryIO, bytes]) -> float:
    """Read a single binary scalar from *source*.

    Expects raw double bytes (no markers).
    """
    with BinaryReader(source) as reader:
        value = reader.read_double()
        return value


def read_binary_scalar_list(
    source: Union[str, Path, BinaryIO, bytes],
) -> torch.Tensor:
    """Read a binary scalar list from *source*.

    Expects format: ``N ( val0 val1 ... valN-1 )``
    """
    with BinaryReader(source) as reader:
        n = reader.read_int32()
        reader.skip_to_open_paren()
        arr = reader.read_doubles(n)
        return torch.tensor(arr, dtype=get_default_dtype(), device=get_device())


def read_binary_label_list(
    source: Union[str, Path, BinaryIO, bytes],
) -> torch.Tensor:
    """Read a binary label (int32) list from *source*.

    Expects format: ``N ( val0 val1 ... valN-1 )``
    """
    with BinaryReader(source) as reader:
        n = reader.read_int32()
        reader.skip_to_open_paren()
        arr = reader.read_int32s(n)
        return torch.tensor(arr, dtype=torch.int64, device=get_device())


def read_binary_points(
    source: Union[str, Path, BinaryIO, bytes],
) -> torch.Tensor:
    """Read binary points (N×3 doubles) from *source*.

    Expects format: ``N ( x0 y0 z0 x1 y1 z1 ... )``
    """
    with BinaryReader(source) as reader:
        n = reader.read_int32()
        reader.skip_to_open_paren()
        arr = reader.read_doubles(n * 3)
        return torch.tensor(
            arr.reshape(n, 3), dtype=get_default_dtype(), device=get_device()
        )


def read_binary_faces(
    source: Union[str, Path, BinaryIO, bytes],
) -> list[np.ndarray]:
    """Read binary faces (CompactListList) from *source*."""
    with BinaryReader(source) as reader:
        return reader.read_binary_compact_list_list()


def read_binary_compact_list_list(
    source: Union[str, Path, BinaryIO, bytes],
) -> list[np.ndarray]:
    """Read a CompactListList from *source*."""
    with BinaryReader(source) as reader:
        return reader.read_binary_compact_list_list()


def write_binary_scalar(value: float) -> bytes:
    """Write a single binary scalar and return the bytes."""
    writer = BinaryWriter()
    writer.write_double(value)
    return writer.get_bytes()


def write_binary_scalar_list(values: torch.Tensor | np.ndarray | Sequence[float]) -> bytes:
    """Write a binary scalar list and return the bytes."""
    writer = BinaryWriter()
    if isinstance(values, torch.Tensor):
        n = values.numel()
    else:
        n = len(values)
    writer.write_int32(n)
    writer.write_marker_open()
    writer.write_binary_scalar_list(values)
    writer.write_marker_close()
    return writer.get_bytes()


def write_binary_label_list(values: torch.Tensor | np.ndarray | Sequence[int]) -> bytes:
    """Write a binary label list and return the bytes."""
    writer = BinaryWriter()
    if isinstance(values, torch.Tensor):
        n = values.numel()
    else:
        n = len(values)
    writer.write_int32(n)
    writer.write_marker_open()
    writer.write_binary_label_list(values)
    writer.write_marker_close()
    return writer.get_bytes()


def write_binary_points(points: torch.Tensor | np.ndarray) -> bytes:
    """Write binary points and return the bytes."""
    writer = BinaryWriter()
    if isinstance(points, torch.Tensor):
        n = points.shape[0]
    else:
        n = np.asarray(points).shape[0]
    writer.write_int32(n)
    writer.write_marker_open()
    writer.write_binary_points(points)
    writer.write_marker_close()
    return writer.get_bytes()


def write_binary_faces(faces: list[np.ndarray] | list[list[int]]) -> bytes:
    """Write binary faces (CompactListList) and return the bytes."""
    writer = BinaryWriter()
    writer.write_binary_faces(faces)
    return writer.get_bytes()


def write_binary_compact_list_list(lists: list[np.ndarray]) -> bytes:
    """Write a CompactListList and return the bytes."""
    writer = BinaryWriter()
    writer.write_binary_compact_list_list(lists)
    return writer.get_bytes()
