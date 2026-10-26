"""Units: the smallest pieces of a model that are loaded and evicted as a whole.

Dense models: one unit per decoder layer. Everything outside the layers (embeddings, final
norm, lm_head) is the always-resident core.

A unit is read as one or more *spans*: page-aligned regions of a safetensors file that cover
the unit's tensors. Alignment matters: macOS only bypasses the page cache (F_NOCACHE) for
aligned reads; unaligned reads silently go through the cache, which then holds a second copy
outside our memory budget. Each span lands in one flat buffer; tensors are views at their
offset inside it, so a unit costs one allocation and no copies.
"""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass, field

from .index import TensorEntry, WeightIndex

PAGE = 16 * 1024
# Tensors of one layer further apart than this in a file are read as separate spans.
MAX_GAP = 1024 * 1024

SAFETENSORS_DTYPES = {
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "U8": "uint8",
    "I8": "int8",
    "U16": "uint16",
    "I16": "int16",
    "U32": "uint32",
    "I32": "int32",
    "U64": "uint64",
    "I64": "int64",
    "BOOL": "bool_",
}
ITEMSIZE = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "BOOL": 1,
}


@dataclass(frozen=True, slots=True)
class Span:
    """A page-aligned file region read in one go."""

    file: str
    file_offset: int
    nbytes: int
    buffer_offset: int


@dataclass(frozen=True, slots=True)
class TensorSlot:
    name: str  # relative to the layer, e.g. "self_attn.q_proj.weight"
    nbytes: int
    buffer_offset: int
    dtype: str  # safetensors dtype
    shape: tuple[int, ...]


@dataclass(slots=True)
class LayerUnit:
    layer: int
    tensors: list[TensorSlot] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    buffer_bytes: int = 0
    realign: bool = False  # some tensors are not aligned to their element size

    @property
    def name(self) -> str:
        return f"L{self.layer}"

    @property
    def nbytes(self) -> int:
        """Bytes read from disk for this unit (incl. alignment)."""
        return sum(s.nbytes for s in self.spans)


@dataclass(slots=True)
class DensePlan:
    core: list[TensorEntry]
    units: list[LayerUnit]
    layer_prefix: str  # e.g. "model.layers."

    @property
    def core_bytes(self) -> int:
        return sum(t.nbytes for t in self.core)

    @property
    def max_unit_bytes(self) -> int:
        return max((u.buffer_bytes for u in self.units), default=0)

    @property
    def transient_bytes(self) -> int:
        """Room for one realigned copy of a layer while it computes."""
        return self.max_unit_bytes if any(u.realign for u in self.units) else 0


def _down(n: int) -> int:
    return n // PAGE * PAGE


def _up(n: int) -> int:
    return (n + PAGE - 1) // PAGE * PAGE


@functools.lru_cache(maxsize=1024)
def _file_size(path: str) -> int:
    return os.path.getsize(path)


def build_unit(layer: int, entries: list[TensorEntry]) -> LayerUnit:
    unit = LayerUnit(layer)
    ordered = sorted(entries, key=lambda e: (e.file, e.offset))
    # Group tensors that sit close together in the same file into one span.
    groups: list[list[TensorEntry]] = []
    for t in ordered:
        last = groups[-1][-1] if groups else None
        if (
            last is not None
            and last.file == t.file
            and t.offset - (last.offset + last.nbytes) <= MAX_GAP
        ):
            groups[-1].append(t)
        else:
            groups.append([t])
    buffer_offset = 0
    for group in groups:
        start = _down(group[0].offset)
        # The last page of a file may be partial; never read past the end.
        end = min(_up(max(t.offset + t.nbytes for t in group)), _file_size(group[0].file))
        unit.spans.append(Span(group[0].file, start, end - start, buffer_offset))
        for t in group:
            rel_offset = buffer_offset + (t.offset - start)
            if rel_offset % ITEMSIZE[t.dtype]:
                # MLX files do not pad their header, so tensors can sit at odd offsets.
                # MLX handles such views correctly (realigning on the GPU when needed).
                unit.realign = True
            rel = t.name.split(f"layers.{layer}.", 1)[1]
            unit.tensors.append(TensorSlot(rel, t.nbytes, rel_offset, t.dtype, t.shape))
        buffer_offset += end - start
    unit.buffer_bytes = buffer_offset
    return unit


def plan_dense(index: WeightIndex) -> DensePlan:
    by_layer: dict[int, list[TensorEntry]] = {}
    core: list[TensorEntry] = []
    prefix: str | None = None
    for t in index.tensors.values():
        if t.layer is None:
            core.append(t)
            continue
        by_layer.setdefault(t.layer, []).append(t)
        m = re.search(rf"^(.*?layers\.){t.layer}\.", t.name)
        if m and prefix is None:
            prefix = m.group(1)
    if prefix is None:
        raise ValueError("no decoder layers found in the weight index")
    units = [build_unit(layer, by_layer[layer]) for layer in sorted(by_layer)]
    return DensePlan(core=core, units=units, layer_prefix=prefix)


def choose_pinned(plan: DensePlan, weights_budget: int, slots: int) -> int:
    """How many leading layers stay resident; -1 if even the minimum does not fit.

    Access is cyclic (0..N-1 every token), so a fixed pinned set beats LRU: with LRU the
    layer needed next is always the one just evicted (start-prompt-v2 §6.6).
    """
    ring = slots * plan.max_unit_bytes + plan.transient_bytes
    room = weights_budget - plan.core_bytes - ring
    if room < 0:
        return -1
    pinned = 0
    for unit in plan.units:
        if unit.buffer_bytes > room:
            break
        room -= unit.buffer_bytes
        pinned += 1
    return pinned
