"""Hardware facts: chip, RAM, Metal working-set limit, measured SSD speed, memory pressure."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import fcntl
import json
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .drift.io_backend import PreadBackend, ReadRequest

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

PRESSURE_LEVELS = {1: "normal", 2: "warning", 4: "critical"}


def sysctl_int(name: str) -> int | None:
    value = ctypes.c_int64(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if _libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        return None
    return value.value if size.value == 8 else ctypes.c_int32(value.value & 0xFFFFFFFF).value


def sysctl_str(name: str) -> str | None:
    size = ctypes.c_size_t(0)
    if _libc.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0) != 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if _libc.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0) != 0:
        return None
    return buf.value.decode(errors="replace")


def memory_pressure() -> str:
    level = sysctl_int("kern.memorystatus_vm_pressure_level")
    return PRESSURE_LEVELS.get(level or 1, "normal")


@dataclass
class SystemInfo:
    chip: str
    ram_bytes: int
    metal_working_set_bytes: int
    macos: str
    ssd_read_bytes_per_s: float | None = None  # measured; None until first SSD test

    def to_dict(self) -> dict:
        return asdict(self)


def detect(working_set_bytes: int, data_dir: Path) -> SystemInfo:
    """working_set_bytes comes from mx.device_info(), read on the MLX thread by the caller."""
    info = SystemInfo(
        chip=sysctl_str("machdep.cpu.brand_string") or platform.processor(),
        ram_bytes=sysctl_int("hw.memsize") or 0,
        metal_working_set_bytes=working_set_bytes,
        macos=platform.mac_ver()[0],
    )
    cached = data_dir / "system.json"
    with contextlib.suppress(OSError, KeyError, ValueError):
        info.ssd_read_bytes_per_s = json.loads(cached.read_text())["ssd_read_bytes_per_s"]
    return info


def quick_ssd_test(data_dir: Path, size: int = 2 * 1024**3) -> float:
    """≤ 5 s read test on the volume holding the models. Returns bytes/s and caches it."""
    data_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".ssd-test-", dir=data_dir)
    try:
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        block = os.urandom(64 * 1024 * 1024)
        written = 0
        while written < size:
            written += os.write(fd, block)
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        os.close(fd)
        backend = PreadBackend(threads=8)
        try:
            start = time.perf_counter()
            backend.read(ReadRequest(name, 0, written))
            speed = written / (time.perf_counter() - start)
        finally:
            backend.close()
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
        os.unlink(name)
    (data_dir / "system.json").write_text(json.dumps({"ssd_read_bytes_per_s": speed}))
    return speed
