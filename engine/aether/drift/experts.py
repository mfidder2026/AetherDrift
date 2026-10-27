"""Drift-MoE: routed experts live on the SSD; each MoE layer keeps a fixed set of expert
slots in RAM (start-prompt-v2 §6.6).

Per layer, every stacked expert tensor ([n_experts, ...] in the file) gets a slot tensor
[slots, ...] in RAM. A table maps expert id → slot; router indices are translated to slot
indices before `gather_qmm`, so the mlx-lm kernels run unchanged.

Eviction: LFU with exponential ageing, per layer. Only decode uses and fills the slots.

Prefill touches most experts of a layer at once — usually more than there are slots. It runs
on a transient full-size expert buffer instead ([n_experts, ...], one layer at a time): experts
already in a slot are copied in RAM, the rest are read once from the SSD, and the layer runs
with the original router indices. That gives bit-identical results to running fully in RAM
(splitting the batch into groups does not: MLX then picks other kernels), and prefill does not
flush the decode working set.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from ..telemetry.events import EventBus
from ..telemetry.stats import Stats
from .index import WeightIndex
from .io_backend import PreadBackend, ReadRequest, _all_of
from .units import SAFETENSORS_DTYPES

log = logging.getLogger("aether.experts")

HALF_LIFE_TOKENS = 256
HIT_WINDOW = 4096
_SWITCH_RE = re.compile(r"layers\.(\d+)\.(.*?switch_mlp)\.(.+)$")


@dataclass(frozen=True, slots=True)
class ExpertTensor:
    name: str  # relative to the switch module, e.g. "gate_proj.weight"
    file: str
    offset: int  # start of expert 0
    stride: int  # bytes per expert
    dtype: str
    shape: tuple[int, ...]  # per expert (without the expert axis)


@dataclass(slots=True)
class MoELayerPlan:
    layer: int
    module_path: str  # e.g. "mlp.switch_mlp"
    tensors: list[ExpertTensor] = field(default_factory=list)

    @property
    def bytes_per_expert(self) -> int:
        return sum(t.stride for t in self.tensors)


@dataclass(slots=True)
class MoEPlan:
    layers: dict[int, MoELayerPlan]
    n_experts: int
    experts_per_token: int
    resident_bytes: int  # everything that is not a routed expert

    @property
    def bytes_per_expert(self) -> int:
        return max(p.bytes_per_expert for p in self.layers.values())

    def slot_bytes(self, slots: int) -> int:
        return slots * sum(p.bytes_per_expert for p in self.layers.values())

    @property
    def prefill_bytes(self) -> int:
        """The transient full expert buffer for one layer during prefill."""
        return self.n_experts * self.bytes_per_expert


def plan_moe(index: WeightIndex) -> MoEPlan:
    layers: dict[int, MoELayerPlan] = {}
    for t in index.tensors.values():
        if not t.stacked_experts:
            continue
        m = _SWITCH_RE.search(t.name)
        if m is None:
            raise ValueError(f"unexpected expert tensor name: {t.name}")
        layer = int(m.group(1))
        plan = layers.setdefault(layer, MoELayerPlan(layer, m.group(2)))
        plan.tensors.append(
            ExpertTensor(
                m.group(3), t.file, t.offset, t.nbytes // index.n_experts, t.dtype, t.shape[1:]
            )
        )
    return MoEPlan(layers, index.n_experts, index.experts_per_token, index.resident_bytes)


def choose_slots(plan: MoEPlan, weights_budget: int, transient: int = 0) -> int:
    """Expert slots per MoE layer that fit next to the resident part; -1 if fewer than the
    experts one token needs."""
    per_slot = plan.slot_bytes(1)
    room = weights_budget - plan.resident_bytes - transient
    slots = min(plan.n_experts, room // per_slot if per_slot else 0)
    return int(slots) if slots >= plan.experts_per_token else -1


class LayerSlots:
    """Slot tensors of one MoE layer plus the expert↔slot bookkeeping."""

    def __init__(self, plan: MoELayerPlan, slots: int, n_experts: int) -> None:
        self.plan = plan
        self.slots = slots
        self.host: dict[str, np.ndarray] = {}
        self.tensors: dict[str, mx.array] = {}
        for t in plan.tensors:
            raw = mx.zeros((slots * t.stride,), dtype=mx.uint8)
            mx.eval(raw)
            self.host[t.name] = np.array(raw, copy=False)
            dtype = getattr(mx, SAFETENSORS_DTYPES[t.dtype])
            view = raw.view(dtype).reshape((slots, *t.shape))
            mx.eval(view)  # stride is a multiple of the element size: zero-copy view
            self.tensors[t.name] = view
        self.slot_of = np.full(n_experts, -1, dtype=np.int32)
        self.owner = np.full(slots, -1, dtype=np.int32)
        self.count = np.zeros(slots, dtype=np.float64)
        self.last = np.zeros(slots, dtype=np.float64)
        self.pending: dict[int, Future[Any]] = {}  # slot → read in flight
        self.prefetched: set[int] = set()  # experts loaded on a prediction, not yet used

    def score(self, now: float) -> np.ndarray:
        return self.count * np.power(0.5, (now - self.last) / HALF_LIFE_TOKENS)

    @property
    def filled(self) -> int:
        return int((self.owner >= 0).sum())


class ExpertCache:
    def __init__(
        self,
        plan: MoEPlan,
        slots: int,
        backend: PreadBackend,
        events: EventBus,
        stats: Stats,
    ) -> None:
        self.plan = plan
        self.slots = slots
        self.backend = backend
        self.events = events
        self.stats = stats
        self.layers = {
            layer: LayerSlots(p, slots, plan.n_experts) for layer, p in plan.layers.items()
        }
        self.order = sorted(self.layers)
        self.tokens = 0.0  # decode tokens seen, for ageing
        self._hits: deque[bool] = deque(maxlen=HIT_WINDOW)
        self.stall_seconds = 0.0
        # Prefill buffer per tensor name: (host view, [n_experts, ...] MLX view); lazily sized.
        self._full: dict[str, tuple[np.ndarray, mx.array]] = {}
        self.prefetch_enabled: Any = lambda: True  # set by the worker (live setting)
        self.prefetch_issued = 0
        self.prefetch_used = 0

    @property
    def resident_bytes(self) -> int:
        return self.plan.slot_bytes(self.slots)

    def tick(self) -> None:
        """One decode token passed (called once per forward pass with a single token)."""
        self.tokens += 1

    def _start_loads(
        self, layer: int, experts: list[int], protect: set[int], required: bool = True
    ) -> list[Future[Any]]:
        """Assign slots to non-resident `experts` and start reading them (non-blocking).

        required=False (prefetch): load as many as fit, never wait, never fail.
        """
        ls = self.layers[layer]
        now = self.tokens
        missing = [e for e in experts if ls.slot_of[e] < 0]
        if not missing:
            return []
        for slot in [s for s, f in ls.pending.items() if f.done()]:
            self._settle(layer, slot)  # finished reads: slot is evictable again
        if required:
            # Reads still in flight for experts we do not need now: let them finish so
            # their slots can be reused.
            for slot in [s for s in ls.pending if int(ls.owner[s]) not in protect]:
                self._settle(layer, slot)
        free = [s for s in range(ls.slots) if ls.owner[s] < 0]
        if len(free) < len(missing):
            score = ls.score(now)
            candidates = [
                s
                for s in range(ls.slots)
                if ls.owner[s] >= 0 and int(ls.owner[s]) not in protect and s not in ls.pending
            ]
            candidates.sort(key=lambda s: score[s])
            free += candidates[: len(missing) - len(free)]
        if required and len(free) < len(missing):
            raise RuntimeError(f"layer {layer}: no free expert slots")
        evicted = 0
        futures: list[Future[Any]] = []
        for expert, slot in zip(missing, free, strict=False):
            old = int(ls.owner[slot])
            if old >= 0:
                ls.slot_of[old] = -1
                evicted += 1
            ls.owner[slot] = expert
            ls.slot_of[expert] = slot
            ls.count[slot] = 0.0
            ls.last[slot] = now
            reads = []
            for t in ls.plan.tensors:
                dest = memoryview(ls.host[t.name])[slot * t.stride : (slot + 1) * t.stride]
                req = ReadRequest(t.file, t.offset + expert * t.stride, t.stride)
                reads.append(self.backend.submit_unaligned(req, dest))
            fut = _all_of(reads)
            ls.pending[slot] = fut
            futures.append(fut)
            ls.prefetched.discard(old)
            if not required:
                ls.prefetched.add(expert)
                self.prefetch_issued += 1
        loaded = min(len(missing), len(free))
        if loaded:
            self.events.publish(
                "load", unit=f"L{layer}", experts=loaded, bytes=loaded * ls.plan.bytes_per_expert
            )
        if evicted:
            self.events.publish(
                "evict", unit=f"L{layer}", experts=evicted, bytes=evicted * ls.plan.bytes_per_expert
            )
        return futures

    def _settle(self, layer: int, slot: int, raise_errors: bool = False) -> None:
        """Collect a slot's finished (or still running) read.

        A failed read must never leave its expert mapped: the slot holds partial data. The
        expert is unmapped so the next use reads it again.
        """
        ls = self.layers[layer]
        fut = ls.pending.pop(slot, None)
        if fut is None:
            return
        try:
            fut.result()
        except Exception:
            expert = int(ls.owner[slot])
            if expert >= 0:
                ls.slot_of[expert] = -1
                ls.prefetched.discard(expert)
            ls.owner[slot] = -1
            log.warning("layer %d: read of expert %d failed", layer, expert, exc_info=True)
            if raise_errors:
                raise

    def _wait(self, layer: int, experts: np.ndarray) -> None:
        ls = self.layers[layer]
        t0 = time.perf_counter()
        error: Exception | None = None
        for e in experts.tolist():
            slot = int(ls.slot_of[e])
            if slot < 0:
                continue
            try:
                self._settle(layer, slot, raise_errors=True)
            except Exception as exc:  # settle the others too, so none stays mapped
                error = error or exc
        self.stall_seconds += time.perf_counter() - t0
        if error is not None:
            raise error

    def ensure(
        self,
        layer: int,
        experts: np.ndarray,
        decode: bool,
        prefetch: tuple[int, np.ndarray] | None = None,
    ) -> mx.array:
        """Make `experts` resident in `layer`; returns the expert→slot map (int32, [E]).

        `prefetch=(next_layer, predicted)` starts reading the predicted experts of the next
        MoE layer right after this layer's own reads, so they overlap with this layer's
        compute (speculative prefetch, start-prompt-v2 §6.6).
        """
        ls = self.layers[layer]
        now = self.tokens
        needed = experts.tolist()
        # Collect finished reads first: a failed prefetch must count as missing, not present.
        for slot in [s for s, f in ls.pending.items() if f.done()]:
            self._settle(layer, slot)
        present = ls.slot_of[experts] >= 0
        if decode:
            self._hits.extend(present.tolist())
            used = ls.prefetched.intersection(needed)
            self.prefetch_used += len(used)
            ls.prefetched -= used
        if len(needed) - int(present.sum()) > ls.slots:
            raise RuntimeError("more distinct experts than slots")
        self._start_loads(layer, needed, protect=set(needed))
        if prefetch is not None and self.prefetch_enabled():
            next_layer, predicted = prefetch
            self._start_loads(
                next_layer,
                predicted.tolist()[: self.plan.experts_per_token],
                protect=set(predicted.tolist()),
                required=False,  # best effort: never wait, never fail
            )
        self._wait(layer, experts)
        if (ls.slot_of[experts] < 0).any():  # never compute with a wrong expert
            raise RuntimeError(f"layer {layer}: expert not resident after load")
        if decode:
            slots = ls.slot_of[experts]
            ls.count[slots] = ls.score(now)[slots] + 1.0
            ls.last[slots] = now
        mapping = np.where(ls.slot_of >= 0, ls.slot_of, 0).astype(np.int32)
        return mx.array(mapping)

    def prefill_weights(self, layer: int, experts: np.ndarray) -> dict[str, mx.array]:
        """Full-size expert tensors for `layer` holding at least `experts` (prefill path).

        The caller must evaluate everything computed with the previous layer's buffer first.
        """
        ls = self.layers[layer]
        out: dict[str, mx.array] = {}
        for t in ls.plan.tensors:
            host, view = self._full.get(t.name, (None, None))
            if view is None or view.shape != (self.plan.n_experts, *t.shape):
                raw = mx.zeros((self.plan.n_experts * t.stride,), dtype=mx.uint8)
                mx.eval(raw)
                host = np.array(raw, copy=False)
                dtype = getattr(mx, SAFETENSORS_DTYPES[t.dtype])
                view = raw.view(dtype).reshape((self.plan.n_experts, *t.shape))
                mx.eval(view)
                self._full[t.name] = (host, view)
            out[t.name] = view
        reads: list[Future[Any]] = []
        from_ssd = 0
        self._wait(layer, experts[ls.slot_of[experts] >= 0])  # prefetches still in flight
        for expert in experts.tolist():
            slot = int(ls.slot_of[expert])
            for t in ls.plan.tensors:
                host = self._full[t.name][0]
                dest = host[expert * t.stride : (expert + 1) * t.stride]
                if slot >= 0:  # already in RAM: copy from the slot
                    np.copyto(dest, ls.host[t.name][slot * t.stride : (slot + 1) * t.stride])
                else:
                    req = ReadRequest(t.file, t.offset + expert * t.stride, t.stride)
                    reads.append(self.backend.submit_unaligned(req, memoryview(dest)))
            from_ssd += slot < 0
        t0 = time.perf_counter()
        _all_of(reads).result()
        waited = time.perf_counter() - t0
        self.stall_seconds += waited
        if from_ssd:
            self.events.publish(
                "load",
                unit=f"L{layer}",
                experts=from_ssd,
                prefill=True,
                bytes=from_ssd * ls.plan.bytes_per_expert,
                ms=int(waited * 1000),
            )
        return out

    def publish(self, current_layer: int | None, n_layers: int) -> None:
        residency = [
            self.layers[i].filled / self.plan.n_experts if i in self.layers else 1.0
            for i in range(n_layers)
        ]
        hits = self._hits
        self.stats.update(
            "drift",
            layers_total=n_layers,
            current_layer=current_layer,
            layer_residency=residency,
            hit_rate=round(sum(hits) / len(hits), 3) if hits else None,
            pinned_layers=None,
            stall_seconds=round(self.stall_seconds, 3),
            expert_slots=self.slots,
            experts_total=self.plan.n_experts,
            prefetch_accuracy=(
                round(self.prefetch_used / self.prefetch_issued, 3)
                if self.prefetch_issued
                else None
            ),
        )

    def close(self) -> None:
        self.layers.clear()
        self._full.clear()
