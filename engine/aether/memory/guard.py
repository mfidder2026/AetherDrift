"""Memory guard (start-prompt-v2 §6.9): act before macOS starts swapping heavily.

Warning: Drift gives back pinned layers in steps of 10 % until pressure is normal again.
Critical: the running generation stops and keeps what it has produced so far.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

import psutil

from ..runtime.worker import Worker
from ..system import memory_pressure
from ..telemetry.events import EventBus

log = logging.getLogger("aether.memory")
POLL_S = 0.5
SWAP_GROWTH_BYTES = 1024**3
SWAP_WINDOW_S = 10.0
SHRINK_STEP = 0.10
SHRINK_COOLDOWN_S = 3.0  # give macOS time to react before shrinking again
GB = 1024**3


class MemoryGuard:
    def __init__(self, worker: Worker, events: EventBus) -> None:
        self.worker = worker
        self.events = events
        self.level = "normal"
        self._swap: deque[tuple[float, int]] = deque()
        self._last_shrink = 0.0

    def _swap_growing(self) -> bool:
        now = time.monotonic()
        self._swap.append((now, psutil.swap_memory().used))
        while self._swap and now - self._swap[0][0] > SWAP_WINDOW_S:
            self._swap.popleft()
        return self._swap[-1][1] - self._swap[0][1] > SWAP_GROWTH_BYTES

    def check(self) -> str:
        level = memory_pressure()
        if level == "normal" and self._swap_growing():
            level = "warning"
        action = None
        now = time.monotonic()
        if level in ("warning", "critical") and now - self._last_shrink > SHRINK_COOLDOWN_S:
            freed = self.worker.relieve_memory(SHRINK_STEP)
            self._last_shrink = now
            if freed:
                action = f"Drift-cache verkleind met {freed / GB:.1f} GB"
        if level == "critical" and self.level != "critical" and self.worker.loaded is not None:
            self.worker.cancel_current("memory")
            action = "generatie gestopt om geheugen vrij te maken"
        if level != self.level or action:
            self.events.publish("pressure", level=level, action=action)
        self.level = level
        return level

    async def run(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.check)
            except Exception:
                log.exception("memory guard check failed")
            await asyncio.sleep(POLL_S)
