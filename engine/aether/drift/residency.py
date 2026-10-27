"""Residency manager for dense models (start-prompt-v2 §6.6, Drift-Dense).

Layers 0..P-1 stay pinned in RAM. The other layers stream through a ring of K slots: while
layer i computes, the next `depth` streamed layers are already being read. Reads go straight
into MLX-owned memory (a writable NumPy view of a uint8 MLX buffer), so there is no copy
between the SSD and the GPU.

All methods except the read-completion callbacks run on the MLX thread.
"""

from __future__ import annotations

import bisect
import contextlib
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from ..telemetry.events import EventBus
from ..telemetry.stats import Stats
from .io_backend import PreadBackend, ReadRequest
from .units import SAFETENSORS_DTYPES, DensePlan, LayerUnit

HIT_WINDOW = 512


class UnitBuffer:
    """A flat MLX uint8 buffer plus a NumPy view of the same memory for the I/O threads."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self.array = mx.zeros((max(1, nbytes),), dtype=mx.uint8)
        mx.eval(self.array)
        self.host = np.array(self.array, copy=False)
        assert self.host.flags.writeable

    def tensors(self, unit: LayerUnit) -> dict[str, mx.array]:
        out = {}
        for t in unit.tensors:
            dtype = getattr(mx, SAFETENSORS_DTYPES[t.dtype])
            raw = self.array[t.buffer_offset : t.buffer_offset + t.nbytes]
            out[t.name] = raw.view(dtype).reshape(t.shape)
        return out


def read_unit(backend: PreadBackend, unit: LayerUnit, buf: UnitBuffer) -> Future[None]:
    """Submit the unit's page-aligned span reads; resolves when all of them are in."""
    done: Future[None] = Future()
    if not unit.spans:
        done.set_result(None)
        return done
    remaining = [len(unit.spans)]
    lock = threading.Lock()
    view = memoryview(buf.host)

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

    for s in unit.spans:
        target = view[s.buffer_offset : s.buffer_offset + s.nbytes]
        backend.submit(ReadRequest(s.file, s.file_offset, s.nbytes), target).add_done_callback(
            finished
        )
    return done


@dataclass
class Slot:
    buffer: UnitBuffer
    owner: int | None = None
    future: Future[None] | None = None
    in_use: bool = False


class DenseResidency:
    def __init__(
        self,
        plan: DensePlan,
        pinned_count: int,
        slots: int,
        depth: int,
        backend: PreadBackend,
        events: EventBus,
        stats: Stats,
    ) -> None:
        self.plan = plan
        self.units = {u.layer: u for u in plan.units}
        self.order = [u.layer for u in plan.units]
        self.depth = depth
        self.backend = backend
        self.events = events
        self.stats = stats
        self.pinned: dict[int, dict[str, mx.array]] = {}
        self._pinned_bytes: dict[int, int] = {}
        self._pin_target = self.order[:pinned_count]
        self.streamed = self.order[pinned_count:]
        self._pos = {layer: i for i, layer in enumerate(self.streamed)}
        self.slots = [
            Slot(UnitBuffer(plan.max_unit_bytes)) for _ in range(slots if self.streamed else 0)
        ]
        self._hits: deque[bool] = deque(maxlen=HIT_WINDOW)
        self.stall_seconds = 0.0
        self._shrink_request = 0
        self._lock = threading.Lock()

    # ---- setup ------------------------------------------------------------------------

    @property
    def resident_bytes(self) -> int:
        pinned = sum(self._pinned_bytes.values())
        return self.plan.core_bytes + pinned + sum(s.buffer.nbytes for s in self.slots)

    def load_pinned(self, progress: Any = None) -> None:
        total = sum(self.units[layer].buffer_bytes for layer in self._pin_target) or 1
        done = 0
        for layer in self._pin_target:
            unit = self.units[layer]
            buf = UnitBuffer(unit.buffer_bytes)
            t0 = time.perf_counter()
            read_unit(self.backend, unit, buf).result()
            tensors = buf.tensors(unit)
            # Materialise now: misaligned views become aligned copies, after which the raw
            # buffer can go (otherwise a pinned layer would cost twice its size).
            mx.eval(list(tensors.values()))
            self.pinned[layer] = tensors
            self._pinned_bytes[layer] = unit.buffer_bytes
            del buf
            self.events.publish(
                "load",
                unit=unit.name,
                bytes=unit.nbytes,
                ms=int((time.perf_counter() - t0) * 1000),
                pinned=True,
            )
            done += unit.buffer_bytes
            if progress:
                progress(done / total)
        # Start reading the first streamed layers right away.
        if self.streamed:
            self._prefetch(self._window_after(self.order[-1]))
        self._publish()

    # ---- hot path (MLX thread) ------------------------------------------------------------

    def acquire(self, layer: int) -> dict[str, mx.array]:
        if self._shrink_request:
            self._apply_shrink()
        self.stats.update("drift", current_layer=layer)
        if layer in self.pinned:
            self._hits.append(True)
            self._prefetch(self._window_after(layer))
            return self.pinned[layer]

        slot = self._slot_of(layer)
        if slot is not None and _failed(slot.future):
            slot.owner, slot.future = None, None  # partial data: read it again
            slot = None
        hit = slot is not None and slot.future is not None and slot.future.done()
        if slot is None:
            slot = self._free_slot(exclude=set())
            if slot is None:  # cannot happen with slots >= 1 and nothing in use
                raise RuntimeError("no free Drift slot")
            self._start(slot, layer)
        slot.in_use = True
        self._prefetch(self._window_after(layer))
        if not hit:
            t0 = time.perf_counter()
            assert slot.future is not None
            try:
                slot.future.result()
            except Exception:
                slot.owner, slot.future, slot.in_use = None, None, False
                raise
            finally:
                self.stall_seconds += time.perf_counter() - t0
        self._hits.append(hit)
        self._publish()
        return slot.buffer.tensors(self.units[layer])

    def release(self, layer: int) -> None:
        slot = self._slot_of(layer)
        if slot is not None:
            slot.in_use = False

    # ---- memory pressure ---------------------------------------------------------------

    def pinned_bytes(self) -> int:
        """Thread-safe (the memory guard asks from its own thread)."""
        with self._lock:
            return sum(self._pinned_bytes.values())

    def request_shrink(self, nbytes: int) -> bool:
        """Thread-safe; applied on the MLX thread at the next layer boundary.

        Returns False while an earlier request is still pending: without a generation running
        nothing is applied, and stacking requests would later unpin everything at once.
        """
        with self._lock:
            if self._shrink_request:
                return False
            self._shrink_request = nbytes
            return True

    def _apply_shrink(self) -> None:
        with self._lock:
            want, self._shrink_request = self._shrink_request, 0
            freed = 0
            for layer in sorted(self.pinned, reverse=True):
                if freed >= want:
                    break
                del self.pinned[layer]
                freed += self._pinned_bytes.pop(layer)
                bisect.insort(self.streamed, layer)
        self._pos = {layer: i for i, layer in enumerate(self.streamed)}
        if freed and not self.slots:
            self.slots = [Slot(UnitBuffer(self.plan.max_unit_bytes)) for _ in range(self.depth + 1)]
        mx.clear_cache()
        self.events.publish("unpinned", bytes=freed, pinned_layers=len(self.pinned))
        self._publish()

    # ---- internals ---------------------------------------------------------------------

    def _window_after(self, layer: int) -> list[int]:
        s = self.streamed
        if not s:
            return []
        start = self._pos[layer] + 1 if layer in self._pos else bisect.bisect_right(s, layer)
        return [s[(start + k) % len(s)] for k in range(min(self.depth, len(s)))]

    def _slot_of(self, layer: int) -> Slot | None:
        return next((s for s in self.slots if s.owner == layer), None)

    def _free_slot(self, exclude: set[int]) -> Slot | None:
        free = [s for s in self.slots if not s.in_use and s.owner not in exclude]
        # Prefer empty slots, then slots whose read has finished (not in flight).
        free.sort(key=lambda s: (s.owner is not None, not (s.future is None or s.future.done())))
        return free[0] if free else None

    def _prefetch(self, window: list[int]) -> None:
        keep = set(window)
        for layer in window:
            if self._slot_of(layer) is not None:
                continue
            slot = self._free_slot(exclude=keep)
            if slot is None:
                return
            self._start(slot, layer)

    def _start(self, slot: Slot, layer: int) -> None:
        if slot.future is not None and not slot.future.done():
            # Never overwrite a buffer that is still being filled; its outcome no longer matters.
            with contextlib.suppress(Exception):
                slot.future.result()
        if slot.owner is not None:
            old = self.units[slot.owner]
            self.events.publish("evict", unit=old.name, bytes=old.nbytes)
        unit = self.units[layer]
        slot.owner = layer
        t0 = time.perf_counter()
        fut = read_unit(self.backend, unit, slot.buffer)
        slot.future = fut

        def loaded(f: Future[None]) -> None:
            if f.exception() is None:
                ms = (time.perf_counter() - t0) * 1000
                self.events.publish("load", unit=unit.name, bytes=unit.nbytes, ms=int(ms))

        fut.add_done_callback(loaded)

    def _publish(self) -> None:
        residency = []
        loading = {s.owner for s in self.slots if s.future is not None and not s.future.done()}
        ready = {s.owner for s in self.slots if s.future is not None and s.future.done()}
        for layer in self.order:
            if layer in self.pinned or layer in ready:
                residency.append(1.0)
            elif layer in loading:
                residency.append(0.5)
            else:
                residency.append(0.0)
        hits = self._hits
        self.stats.update(
            "drift",
            layers_total=len(self.order),
            layer_residency=residency,
            hit_rate=round(sum(hits) / len(hits), 3) if hits else None,
            pinned_layers=len(self.pinned),
            stall_seconds=round(self.stall_seconds, 3),
        )

    def close(self) -> None:
        for s in self.slots:
            if s.future is not None and not s.future.done():
                with contextlib.suppress(Exception):
                    s.future.result()
        self.slots.clear()
        self.pinned.clear()
        self._pinned_bytes.clear()


def _failed(fut: Future[Any] | None) -> bool:
    return fut is not None and fut.done() and fut.exception() is not None
