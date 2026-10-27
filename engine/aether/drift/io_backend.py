"""Read-only weight I/O.

Weights are immutable, so this layer only ever reads. Reads bypass the OS page cache
(F_NOCACHE) so that our own residency manager is the single source of truth for what is
in memory, and so memory accounting stays predictable under pressure.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

PAGE = 16 * 1024  # Apple Silicon page size
DIRECT_ALIGN = 4096  # F_NOCACHE bypasses the page cache only for 4 KB-aligned I/O (ADR-004)
DEFAULT_CHUNK = 8 * 1024 * 1024

_local = threading.local()


def _staging(nbytes: int) -> np.ndarray:
    """Thread-local, page-aligned scratch buffer of at least nbytes."""
    buf = getattr(_local, "staging", None)
    if buf is None or len(buf) < nbytes:
        raw = np.empty(nbytes + PAGE, dtype=np.uint8)
        shift = (-raw.ctypes.data) % PAGE
        buf = raw[shift : shift + nbytes]
        _local.staging = buf
        _local.raw = raw  # keep the allocation alive
    return buf


def _all_of(parts: list[Future[Any]]) -> Future[None]:
    done: Future[None] = Future()
    if not parts:
        done.set_result(None)
        return done
    remaining = [len(parts)]
    lock = threading.Lock()

    def finished(f: Future[Any]) -> None:
        if done.done():
            return
        if (exc := f.exception()) is not None:
            done.set_exception(exc)
            return
        with lock:
            remaining[0] -= 1
            last = remaining[0] == 0
        if last:
            done.set_result(None)

    for p in parts:
        p.add_done_callback(finished)
    return done


@dataclass(frozen=True, slots=True)
class ReadRequest:
    path: str
    offset: int
    nbytes: int


@dataclass(slots=True)
class ReadStats:
    bytes_read: int = 0
    reads: int = 0
    busy_seconds: float = 0.0


def open_nocache(path: str) -> int:
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    return fd


class PreadBackend:
    """Parallel pread into caller-provided buffers.

    os.preadv releases the GIL, so a plain thread pool gives real I/O parallelism.
    Large units are split into chunks so a single unit still uses the full queue depth.
    """

    def __init__(self, threads: int = 16, chunk_bytes: int = DEFAULT_CHUNK) -> None:
        self._pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="aether-io")
        self._chunk = chunk_bytes
        self._fds: dict[str, int] = {}
        self._fd_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.stats = ReadStats()
        self.on_read: Callable[[int, float], None] | None = None

    def _fd(self, path: str) -> int:
        with self._fd_lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = open_nocache(path)
                self._fds[path] = fd
            return fd

    def _read_into(self, fd: int, view: memoryview | np.ndarray, offset: int) -> int:
        start = time.perf_counter()
        done = 0
        while done < len(view):
            n = os.preadv(fd, [view[done:]], offset + done)
            if n == 0:
                raise EOFError(f"short read at offset {offset + done}")
            done += n
        elapsed = time.perf_counter() - start
        with self._stats_lock:
            self.stats.bytes_read += done
            self.stats.reads += 1
            self.stats.busy_seconds += elapsed
        if self.on_read is not None:
            self.on_read(done, elapsed)
        return done

    def submit(
        self, req: ReadRequest, buf: bytearray | memoryview | None = None
    ) -> Future[bytearray]:
        """Read one region; resolves to the filled buffer once all chunks are in."""
        out = buf if buf is not None else bytearray(req.nbytes)
        view = memoryview(out)
        if len(view) < req.nbytes:
            raise ValueError("buffer smaller than request")
        fd = self._fd(req.path)
        parts = [
            self._pool.submit(
                self._read_into,
                fd,
                view[pos : min(pos + self._chunk, req.nbytes)],
                req.offset + pos,
            )
            for pos in range(0, req.nbytes, self._chunk)
        ]
        result: Future[bytearray] = Future()
        remaining = [len(parts)]
        lock = threading.Lock()

        def _done(f: Future[int]) -> None:
            if result.done():
                return
            if (exc := f.exception()) is not None:
                result.set_exception(exc)
                return
            with lock:
                remaining[0] -= 1
                finished = remaining[0] == 0
            if finished:
                result.set_result(out)  # type: ignore[arg-type]

        for p in parts:
            p.add_done_callback(_done)
        return result

    def submit_unaligned(self, req: ReadRequest, dest: memoryview) -> Future[None]:
        """Read a region that starts at an unaligned file offset without using the page cache.

        F_NOCACHE only applies to 4 KB-aligned I/O (ADR-004). Each chunk is read aligned into
        a thread-local, page-aligned staging buffer and copied into `dest`.
        """
        if len(dest) < req.nbytes:
            raise ValueError("buffer smaller than request")
        fd = self._fd(req.path)
        parts = [
            self._pool.submit(self._read_staged, fd, req.offset + pos, dest[pos : pos + n])
            for pos in range(0, req.nbytes, self._chunk)
            if (n := min(self._chunk, req.nbytes - pos))
        ]
        return _all_of(parts)

    def _read_staged(self, fd: int, offset: int, dest: memoryview) -> int:
        start = offset // DIRECT_ALIGN * DIRECT_ALIGN
        end = -(-(offset + len(dest)) // DIRECT_ALIGN) * DIRECT_ALIGN
        # At the end of a file the last page is partial: read up to EOF only.
        length = min(end, os.fstat(fd).st_size) - start
        staging = _staging(length)
        got = self._read_into(fd, staging[:length], start)
        lead = offset - start
        if got < lead + len(dest):
            raise EOFError(f"short read at offset {offset}")
        np.copyto(np.frombuffer(dest, dtype=np.uint8), staging[lead : lead + len(dest)])
        return len(dest)

    def read(self, req: ReadRequest) -> bytearray:
        return self.submit(req).result()

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        with self._fd_lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


def invalidate_page_cache(path: str) -> None:
    """Drop a file's pages from the OS page cache (msync MS_INVALIDATE; no sudo needed).

    F_NOCACHE only stops new data from entering the cache; pages that are already cached
    are still served from RAM. Benchmarks call this so they measure the SSD, not RAM.
    """
    import ctypes
    import mmap

    import numpy as np

    size = os.path.getsize(path)
    if size == 0:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    ms_invalidate = 0x2
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        view = np.frombuffer(mm, dtype=np.uint8)
        try:
            if libc.msync(view.ctypes.data, size, ms_invalidate) != 0:
                raise OSError(ctypes.get_errno(), "msync(MS_INVALIDATE) failed")
        finally:
            del view
            mm.close()
    finally:
        os.close(fd)
