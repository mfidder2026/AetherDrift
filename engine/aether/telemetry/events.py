"""Event bus: ring buffer + log file + asyncio subscribers.

publish() is thread-safe and may be called from the MLX or I/O threads. Subscribers get
events in batches per tick; high-frequency load/evict events are folded into a summary when
there are too many for the UI (start-prompt-v2 §6.10).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

HIGH_FREQUENCY = {"load", "evict"}
MAX_UI_EVENTS_PER_S = 20
RING_SIZE = 10_000

log = logging.getLogger("aether.events")


class EventBus:
    def __init__(self, ring_size: int = RING_SIZE) -> None:
        self._ring: deque[dict[str, Any]] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._seq = 0
        self._subscribers: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict]]] = set()

    def publish(self, type: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "ts": round(time.time(), 3), "type": type, **fields}
            self._ring.append(event)
            subscribers = list(self._subscribers)
        log.info(json.dumps(event, ensure_ascii=False))
        for loop, queue in subscribers:
            with contextlib.suppress(RuntimeError):  # subscriber's loop already closed
                loop.call_soon_threadsafe(queue.put_nowait, event)
        return event

    def since(self, seq: int) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._ring if e["seq"] > seq]

    async def stream(
        self, since: int | None = None, tick: float = 0.25
    ) -> AsyncIterator[list[dict]]:
        """Yields batches of events, throttled for UI consumption."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        entry = (loop, queue)
        with self._lock:
            self._subscribers.add(entry)
        try:
            if since is not None:
                backlog = self.since(since)
                if backlog:
                    yield throttle(backlog[-500:], tick)
            while True:
                first = await queue.get()
                await asyncio.sleep(tick)
                batch = [first]
                while not queue.empty():
                    batch.append(queue.get_nowait())
                yield throttle(batch, tick)
        finally:
            with self._lock:
                self._subscribers.discard(entry)


def throttle(batch: list[dict], window_s: float) -> list[dict]:
    """Fold load/evict events into one summary when they exceed the UI budget."""
    frequent = [e for e in batch if e["type"] in HIGH_FREQUENCY]
    if len(frequent) <= MAX_UI_EVENTS_PER_S * window_s:
        return batch
    other = [e for e in batch if e["type"] not in HIGH_FREQUENCY]
    summary = {
        "seq": frequent[-1]["seq"],
        "ts": frequent[-1]["ts"],
        "type": "summary",
        "loads": sum(1 for e in frequent if e["type"] == "load"),
        "evicts": sum(1 for e in frequent if e["type"] == "evict"),
        "bytes": sum(e.get("bytes", 0) for e in frequent if e["type"] == "load"),
        "window_ms": int(window_s * 1000),
    }
    return sorted([*other, summary], key=lambda e: e["seq"])
