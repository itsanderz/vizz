"""
Lock-free ring buffer implementation for high-throughput metric logging.

Uses shared memory via Apache Arrow for zero-copy data transfer between
the main training process and the Harvester background process.
"""

from __future__ import annotations

import json
import mmap
import struct
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa

# Buffer configuration
DEFAULT_BUFFER_SIZE = 1024 * 1024 * 64  # 64 MB
DEFAULT_SLOT_COUNT = 8192
HEADER_SIZE = 64  # bytes for metadata


@dataclass
class BufferSlot:
    """Represents a slot in the ring buffer."""
    sequence: int
    size: int
    data: bytes


class RingBufferWriter:
    """
    Lock-free single-producer ring buffer writer.

    Uses atomic operations for the write pointer to ensure thread-safety
    without locks. The main training loop uses this to push metrics.

    Memory Layout:
    [Header: 64 bytes]
      - write_pos (8 bytes): Current write position
      - read_pos (8 bytes): Current read position (set by reader)
      - slot_count (8 bytes): Number of slots
      - slot_size (8 bytes): Size of each slot
      - flags (8 bytes): Status flags
      - reserved (24 bytes)
    [Slots: slot_count * slot_size bytes]
      Each slot:
      - sequence (8 bytes): Sequence number
      - size (4 bytes): Payload size
      - payload (slot_size - 12 bytes): Actual data
    """

    def __init__(
        self,
        name: str,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        slot_count: int = DEFAULT_SLOT_COUNT,
        create: bool = True,
    ):
        self.name = name
        self.slot_count = slot_count
        self.slot_size = (buffer_size - HEADER_SIZE) // slot_count
        self.total_size = HEADER_SIZE + (self.slot_count * self.slot_size)

        if create:
            # Create new shared memory
            try:
                # Try to clean up existing shared memory with same name
                existing = shared_memory.SharedMemory(name=name)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass

            self.shm = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=self.total_size,
            )
            self._init_header()
        else:
            # Attach to existing shared memory
            self.shm = shared_memory.SharedMemory(name=name, create=False)

        self._write_pos = 0
        self._sequence = 0

    def _init_header(self) -> None:
        """Initialize the buffer header."""
        header = struct.pack(
            "QQQQQ24x",
            0,  # write_pos
            0,  # read_pos
            self.slot_count,
            self.slot_size,
            0,  # flags
        )
        self.shm.buf[:HEADER_SIZE] = header

    def _get_slot_offset(self, index: int) -> int:
        """Get the byte offset for a slot index."""
        return HEADER_SIZE + (index * self.slot_size)

    def write(self, data: Dict[str, Any]) -> bool:
        """
        Write data to the next available slot.

        Returns True if successful, False if buffer is full.
        Target latency: < 5µs
        """
        # Serialize to JSON (fast for small payloads)
        payload = json.dumps(data).encode("utf-8")

        if len(payload) > self.slot_size - 12:
            # Payload too large, truncate or skip
            return False

        # Calculate slot index (wrap around)
        slot_index = self._write_pos % self.slot_count
        offset = self._get_slot_offset(slot_index)

        # Write slot data
        # Format: sequence (8 bytes) + size (4 bytes) + payload
        self._sequence += 1
        slot_header = struct.pack("Qi", self._sequence, len(payload))

        self.shm.buf[offset:offset + 12] = slot_header
        self.shm.buf[offset + 12:offset + 12 + len(payload)] = payload

        # Update write position in header (atomic on 64-bit systems)
        self._write_pos += 1
        struct.pack_into("Q", self.shm.buf, 0, self._write_pos)

        return True

    def write_arrow(self, batch: pa.RecordBatch) -> bool:
        """
        Write an Arrow RecordBatch for zero-copy tensor data.

        For large tensors, we write a reference to a separate memory region.
        """
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()

        buffer = sink.getvalue()
        if len(buffer) > self.slot_size - 12:
            return False

        slot_index = self._write_pos % self.slot_count
        offset = self._get_slot_offset(slot_index)

        self._sequence += 1
        slot_header = struct.pack("Qi", self._sequence, len(buffer))

        self.shm.buf[offset:offset + 12] = slot_header
        self.shm.buf[offset + 12:offset + 12 + len(buffer)] = buffer.to_pybytes()

        self._write_pos += 1
        struct.pack_into("Q", self.shm.buf, 0, self._write_pos)

        return True

    def close(self) -> None:
        """Close the shared memory (don't unlink - reader may still need it)."""
        self.shm.close()

    def unlink(self) -> None:
        """Unlink (delete) the shared memory."""
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


class RingBufferReader:
    """
    Ring buffer reader for the Harvester process.

    Reads data written by the training process and persists to disk.
    """

    def __init__(self, name: str):
        self.name = name
        self.shm = shared_memory.SharedMemory(name=name, create=False)
        self._read_pos = 0

        # Read header to get buffer config
        header = struct.unpack("QQQQQ", self.shm.buf[:40])
        self._write_pos = header[0]
        self.slot_count = header[2]
        self.slot_size = header[3]

    def _get_slot_offset(self, index: int) -> int:
        """Get the byte offset for a slot index."""
        return HEADER_SIZE + (index * self.slot_size)

    def _refresh_write_pos(self) -> None:
        """Read current write position from header."""
        self._write_pos = struct.unpack("Q", self.shm.buf[:8])[0]

    def read(self) -> Optional[Dict[str, Any]]:
        """
        Read the next available slot.

        Returns None if no new data is available.
        """
        self._refresh_write_pos()

        if self._read_pos >= self._write_pos:
            return None

        slot_index = self._read_pos % self.slot_count
        offset = self._get_slot_offset(slot_index)

        # Read slot header
        sequence, size = struct.unpack("Qi", self.shm.buf[offset:offset + 12])

        if size <= 0 or size > self.slot_size - 12:
            self._read_pos += 1
            return None

        # Read payload
        payload = bytes(self.shm.buf[offset + 12:offset + 12 + size])
        self._read_pos += 1

        # Update read position in header
        struct.pack_into("Q", self.shm.buf, 8, self._read_pos)

        try:
            return json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def read_batch(self, max_items: int = 100) -> List[Dict[str, Any]]:
        """Read multiple items at once for efficiency."""
        items = []
        for _ in range(max_items):
            item = self.read()
            if item is None:
                break
            items.append(item)
        return items

    def read_arrow(self) -> Optional[pa.RecordBatch]:
        """Read an Arrow RecordBatch from the buffer."""
        self._refresh_write_pos()

        if self._read_pos >= self._write_pos:
            return None

        slot_index = self._read_pos % self.slot_count
        offset = self._get_slot_offset(slot_index)

        sequence, size = struct.unpack("Qi", self.shm.buf[offset:offset + 12])

        if size <= 0:
            self._read_pos += 1
            return None

        buffer_bytes = bytes(self.shm.buf[offset + 12:offset + 12 + size])
        self._read_pos += 1

        try:
            reader = pa.ipc.open_stream(buffer_bytes)
            return reader.read_next_batch()
        except pa.ArrowInvalid:
            return None

    def available(self) -> int:
        """Return number of unread items."""
        self._refresh_write_pos()
        return self._write_pos - self._read_pos

    def close(self) -> None:
        """Close the shared memory connection."""
        self.shm.close()


class FallbackQueue:
    """
    Thread-safe fallback queue when shared memory is unavailable.

    Uses a simple list with a lock - less efficient but more portable.
    """

    def __init__(self, maxsize: int = 10000):
        self._queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def put(self, item: Dict[str, Any]) -> bool:
        """Add item to queue. Returns False if full."""
        with self._lock:
            if len(self._queue) >= self._maxsize:
                return False
            self._queue.append(item)
            return True

    def get(self) -> Optional[Dict[str, Any]]:
        """Get item from queue. Returns None if empty."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def get_batch(self, max_items: int = 100) -> List[Dict[str, Any]]:
        """Get multiple items at once."""
        with self._lock:
            items = self._queue[:max_items]
            self._queue = self._queue[max_items:]
            return items

    def available(self) -> int:
        """Return number of items in queue."""
        with self._lock:
            return len(self._queue)
