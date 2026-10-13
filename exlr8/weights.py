"""Read individual tensors and byte ranges from safetensors files."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


_DTYPES = {
    "U8": "u1", "I8": "i1", "U16": "<u2", "I16": "<i2",
    "U32": "<u4", "I32": "<i4", "U64": "<u8", "I64": "<i8",
    "F16": "<f2", "BF16": "<u2", "F32": "<f4", "F64": "<f8",
    "BOOL": "?",
}


class SafeTensorFile:
    """Keep the header in memory; read only the requested tensor bytes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            self.header = json.loads(handle.read(size))
        self.data_offset = 8 + size
        self.metadata = self.header.get("__metadata__", {})
        self.tensors = {
            name: info for name, info in self.header.items()
            if name != "__metadata__"
        }

    def raw(
        self, name: str, start_bytes: int = 0, byte_count: int | None = None
    ) -> np.ndarray:
        """Read a shaped tensor, or a flat range. BF16 keep uint16 bit patterns."""
        info = self.tensors[name]
        begin, end = info["data_offsets"]
        dtype = np.dtype(_DTYPES[info["dtype"]])
        whole_tensor = start_bytes == 0 and byte_count is None
        if byte_count is None:
            byte_count = end - begin - start_bytes
        if (start_bytes < 0 or byte_count < 0 or start_bytes + byte_count > end - begin
                or start_bytes % dtype.itemsize or byte_count % dtype.itemsize):
            raise ValueError(f"invalid byte range for {name}: {start_bytes}, {byte_count}")
        result = np.fromfile(
            self.path, dtype=dtype, count=byte_count // dtype.itemsize,
            offset=self.data_offset + begin + start_bytes,
        )
        if result.nbytes != byte_count:
            raise ValueError(f"truncated tensor {name} in {self.path}")
        return result.reshape(info["shape"]) if whole_tensor else result

    def load_slice(self, name: str, start_bytes: int, byte_count: int):
        """Read a byte range as a flat MLX array with the stored scalar type."""
        import mlx.core as mx

        result = mx.array(self.raw(name, start_bytes, byte_count))
        if self.tensors[name]["dtype"] == "BF16":
            result = result.view(mx.bfloat16)
        return result

    def load(self, name: str):
        """Read one complete tensor with its stored shape and scalar type."""
        info = self.tensors[name]
        begin, end = info["data_offsets"]
        return self.load_slice(name, 0, end - begin).reshape(info["shape"])
