"""Live statistics. Every value comes from a counter that the engine updates itself.

Writers (MLX thread, I/O threads, memory guard) update fields under a lock; the API reads
consistent snapshots. Unknown values stay None and are shown as "—" in the UI.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import psutil

from ..system import memory_pressure


@dataclass
class _Rate:
    """Bytes (or items) per second over a sliding window."""

    window_s: float = 2.0
    samples: deque[tuple[float, float]] = field(default_factory=deque)

    def add(self, amount: float, now: float | None = None) -> None:
        self.samples.append((now or time.monotonic(), amount))

    def value(self, now: float | None = None) -> float:
        now = now or time.monotonic()
        while self.samples and now - self.samples[0][0] > self.window_s:
            self.samples.popleft()
        return sum(a for _, a in self.samples) / self.window_s


class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model: dict[str, Any] = {"id": None, "state": "none", "mode": None, "progress": None}
        self.memory: dict[str, Any] = {
            "budget_bytes": None,
            "weights_bytes": 0,
            "kv_bytes": 0,
            "mlx_active_bytes": 0,
            "mlx_cache_bytes": 0,
            "mlx_peak_bytes": 0,
        }
        self.drift: dict[str, Any] = {
            "layers_total": None,
            "current_layer": None,
            "layer_residency": [],
            "hit_rate": None,
            "pinned_layers": None,
            "stall_seconds": None,
        }
        self.generation: dict[str, Any] = {
            "active": False,
            "decode_tps": None,
            "prefill_tps": None,
            "ttft_ms": None,
            "context_tokens": 0,
            "context_max": None,
            "queue_depth": 0,
        }
        self._read = _Rate()
        self._read_latencies: deque[float] = deque(maxlen=512)

    def update(self, section: str, **values: Any) -> None:
        with self._lock:
            getattr(self, section).update(values)

    def record_read(self, nbytes: int, seconds: float) -> None:
        with self._lock:
            self._read.add(nbytes)
            self._read_latencies.append(seconds)

    def snapshot(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        with self._lock:
            lat = sorted(self._read_latencies)
            return {
                "ts": round(time.time(), 3),
                "model": dict(self.model),
                "memory": {
                    **self.memory,
                    "system_used_bytes": vm.total - vm.available,
                    "system_total_bytes": vm.total,
                    "swap_used_bytes": swap.used,
                    "pressure": memory_pressure(),
                },
                "drift": {
                    **self.drift,
                    "read_bytes_per_s": self._read.value(),
                    "read_p95_ms": round(lat[int(len(lat) * 0.95)] * 1e3, 1) if lat else None,
                    "write_bytes_per_s": 0.0,  # weights are read-only; nothing ever writes
                },
                "generation": dict(self.generation),
            }
