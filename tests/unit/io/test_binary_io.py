"""Tests for binary I/O — BinaryReader, BinaryWriter, and convenience functions."""

import struct
from io import BytesIO

import numpy as np
import pytest
import torch

from pyfoam.io.binary_io import (
    BinaryReader,
    BinaryWriter,
    read_binary_compact_list_list,
    read_binary_faces,
    read_binary_label_list,
    read_binary_points,
    read_binary_scalar,
    read_binary_scalar_list,
    write_binary_compact_list_list,
    write_binary_faces,
    write_binary_label_list,
    write_binary_points,
    write_binary_scalar,
    write_binary_scalar_list,
)


# ---------------------------------------------------------------------------
# BinaryReader
# ---------------------------------------------------------------------------


class TestBinaryReader:
    def test_read_double(self):
        """Read a single big-endian double."""
        data = struct.pack(">d", 3.14159)
        reader = BinaryReader(data)
        val = reader.read_double()
        assert abs(val - 3.14159) < 1e-10

    def test_read_int32(self):
        """Read a single big-endian int32."""
        data = struct.pack(">i", 42)
        reader = BinaryReader(data)
        val = reader.read_int32()
        assert val == 42

    def test_read_uint32(self):
        """Read a single big-endian unsigned int32."""
        data = struct.pack(">I", 100000)
        reader = BinaryReader(data)
        val = reader.read_uint32()
        assert val == 100000

    def test_read_doubles_array(self):
        """Read an array of doubles."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        data = struct.pack(f">{len(values)}d", *values)
        reader = BinaryReader(data)
        arr = reader.read_doubles(5)
        np.testing.assert_allclose(arr, values)

    def test_read_int32s_array(self):
        """Read an array of int32 values."""
        values = [10, 20, 30, 40]
        data = struct.pack(f">{len(values)}i", *values)
        reader = BinaryReader(data)
        arr = reader.read_int32s(4)
        np.testing.assert_array_equal(arr, values)

    def test_skip_to_open_paren(self):
        """Skip bytes until '(' is found."""
        data = b"some text (binary data)"
        reader = BinaryReader(data)
        reader.skip_to_open_paren()
        # Next byte should be 'b'
        b = reader.read_bytes(1)
        assert b == b"b"

    def test_skip_to_close_paren(self):
        """Skip bytes until ')' is found."""
        data = b"binary data)"
        reader = BinaryReader(data)
        reader.skip_to_close_paren()
        # Should be at end
        assert reader.read_bytes(0) == b""

    def test_eof_error(self):
        """EOFError when reading past end of stream."""
        data = struct.pack(">d", 1.0)
        reader = BinaryReader(data)
        reader.read_double()
        with pytest.raises(EOFError):
            reader.read_double()

    def test_context_manager(self):
        """BinaryReader works as context manager."""
        data = struct.pack(">d", 1.0)
        with BinaryReader(data) as reader:
            val = reader.read_double()
        assert val == 1.0

    def test_read_from_bytes(self):
        """Read from bytes object."""
        data = struct.pack(">d", 42.0)
        reader = BinaryReader(data)
        val = reader.read_double()
        assert val == 42.0

    def test_read_from_bytesio(self):
        """Read from BytesIO stream."""
        data = struct.pack(">d", 42.0)
        stream = BytesIO(data)
        reader = BinaryReader(stream)
        val = reader.read_double()
        assert val == 42.0

    def test_read_binary_scalar_field(self):
        """Read binary scalar field as torch tensor."""
        values = [1.0, 2.0, 3.0]
        data = struct.pack(f">{len(values)}d", *values)
        reader = BinaryReader(data)
        tensor = reader.read_binary_scalar_field(3)
        assert tensor.shape == (3,)
        assert tensor.dtype == torch.float64
        assert torch.allclose(tensor, torch.tensor(values, dtype=torch.float64))

    def test_read_binary_vector_field(self):
        """Read binary vector field as (n, 3) torch tensor."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        data = struct.pack(">6d", *values)
        reader = BinaryReader(data)
        tensor = reader.read_binary_vector_field(2)
        assert tensor.shape == (2, 3)
        assert torch.allclose(tensor, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64))

    def test_read_binary_label_field(self):
        """Read binary label field as int64 torch tensor."""
        values = [10, 20, 30]
        data = struct.pack(">3i", *values)
        reader = BinaryReader(data)
        tensor = reader.read_binary_label_field(3)
        assert tensor.shape == (3,)
        assert tensor.dtype == torch.int64
        assert torch.equal(tensor, torch.tensor(values, dtype=torch.int64))

    def test_read_binary_compact_list_list(self):
        """Read CompactListList encoding."""
        # 3 sub-lists: [0,1], [2], [3,4,5]
        n = 3
        offsets = [0, 2, 3, 6]
        data_list = [0, 1, 2, 3, 4, 5]

        buf = BytesIO()
        buf.write(struct.pack(">i", n))
        buf.write(struct.pack(f">{n+1}i", *offsets))
        buf.write(struct.pack(f">{len(data_list)}i", *data_list))
        buf.seek(0)

        reader = BinaryReader(buf)
        result = reader.read_binary_compact_list_list()
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [0, 1])
        np.testing.assert_array_equal(result[1], [2])
        np.testing.assert_array_equal(result[2], [3, 4, 5])


# ---------------------------------------------------------------------------
# BinaryWriter
# ---------------------------------------------------------------------------


class TestBinaryWriter:
    def test_write_double(self):
        """Write a single double."""
        writer = BinaryWriter()
        writer.write_double(3.14159)
        data = writer.get_bytes()
        val = struct.unpack(">d", data)[0]
        assert abs(val - 3.14159) < 1e-10

    def test_write_int32(self):
        """Write a single int32."""
        writer = BinaryWriter()
        writer.write_int32(42)
        data = writer.get_bytes()
        val = struct.unpack(">i", data)[0]
        assert val == 42

    def test_write_doubles_array(self):
        """Write an array of doubles."""
        values = [1.0, 2.0, 3.0]
        writer = BinaryWriter()
        writer.write_doubles(values)
        data = writer.get_bytes()
        result = struct.unpack(">3d", data)
        assert result == tuple(values)

    def test_write_int32s_array(self):
        """Write an array of int32 values."""
        values = [10, 20, 30]
        writer = BinaryWriter()
        writer.write_int32s(values)
        data = writer.get_bytes()
        result = struct.unpack(">3i", data)
        assert result == tuple(values)

    def test_write_binary_scalar_list(self):
        """Write binary scalar list."""
        values = [1.0, 2.0, 3.0]
        writer = BinaryWriter()
        writer.write_binary_scalar_list(values)
        data = writer.get_bytes()
        result = struct.unpack(">3d", data)
        assert result == tuple(values)

    def test_write_binary_label_list(self):
        """Write binary label list."""
        values = [10, 20, 30]
        writer = BinaryWriter()
        writer.write_binary_label_list(values)
        data = writer.get_bytes()
        result = struct.unpack(">3i", data)
        assert result == tuple(values)

    def test_write_binary_points(self):
        """Write binary points."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        writer = BinaryWriter()
        writer.write_binary_points(points)
        data = writer.get_bytes()
        result = struct.unpack(">6d", data)
        assert result == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    def test_write_binary_faces(self):
        """Write binary faces (CompactListList)."""
        faces = [np.array([0, 1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)]
        writer = BinaryWriter()
        writer.write_binary_faces(faces)
        data = writer.get_bytes()

        # Read back
        reader = BinaryReader(data)
        result = reader.read_binary_compact_list_list()
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [0, 1, 2])
        np.testing.assert_array_equal(result[1], [3, 4])

    def test_write_marker_open_close(self):
        """Write open/close markers."""
        writer = BinaryWriter()
        writer.write_marker_open()
        writer.write_marker_close()
        data = writer.get_bytes()
        assert data == b"()"

    def test_write_to_stream(self):
        """Write to external stream."""
        writer = BinaryWriter()
        writer.write_int32(42)
        stream = BytesIO()
        writer.write_to(stream)
        stream.seek(0)
        val = struct.unpack(">i", stream.read(4))[0]
        assert val == 42

    def test_roundtrip_doubles(self):
        """Write then read back doubles."""
        values = [1.1, 2.2, 3.3, 4.4]
        writer = BinaryWriter()
        writer.write_doubles(values)
        data = writer.get_bytes()

        reader = BinaryReader(data)
        result = reader.read_doubles(4)
        np.testing.assert_allclose(result, values)

    def test_roundtrip_int32s(self):
        """Write then read back int32 values."""
        values = [100, 200, 300]
        writer = BinaryWriter()
        writer.write_int32s(values)
        data = writer.get_bytes()

        reader = BinaryReader(data)
        result = reader.read_int32s(3)
        np.testing.assert_array_equal(result, values)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_read_write_binary_scalar(self):
        """Roundtrip binary scalar."""
        data = write_binary_scalar(42.0)
        val = read_binary_scalar(data)
        assert abs(val - 42.0) < 1e-10

    def test_read_write_binary_scalar_list(self):
        """Roundtrip binary scalar list."""
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        data = write_binary_scalar_list(values)
        result = read_binary_scalar_list(data)
        assert torch.allclose(result, values)

    def test_read_write_binary_label_list(self):
        """Roundtrip binary label list."""
        values = torch.tensor([10, 20, 30], dtype=torch.int64)
        data = write_binary_label_list(values)
        result = read_binary_label_list(data)
        assert torch.equal(result, values)

    def test_read_write_binary_points(self):
        """Roundtrip binary points."""
        points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        data = write_binary_points(points)
        result = read_binary_points(data)
        assert torch.allclose(result, points)

    def test_read_write_binary_faces(self):
        """Roundtrip binary faces."""
        faces = [np.array([0, 1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)]
        data = write_binary_faces(faces)
        result = read_binary_faces(data)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [0, 1, 2])
        np.testing.assert_array_equal(result[1], [3, 4])

    def test_read_write_binary_compact_list_list(self):
        """Roundtrip CompactListList."""
        lists = [np.array([0, 1], dtype=np.int32), np.array([2, 3, 4], dtype=np.int32)]
        data = write_binary_compact_list_list(lists)
        result = read_binary_compact_list_list(data)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [0, 1])
        np.testing.assert_array_equal(result[1], [2, 3, 4])

    def test_scalar_list_with_numpy(self):
        """Scalar list from numpy array."""
        values = np.array([1.0, 2.0, 3.0])
        data = write_binary_scalar_list(values)
        result = read_binary_scalar_list(data)
        assert torch.allclose(result, torch.tensor(values, dtype=torch.float64))

    def test_label_list_with_numpy(self):
        """Label list from numpy array."""
        values = np.array([10, 20, 30], dtype=np.int32)
        data = write_binary_label_list(values)
        result = read_binary_label_list(data)
        assert torch.equal(result, torch.tensor(values, dtype=torch.int64))
