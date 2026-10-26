"""Weight index built from safetensors headers only (no tensor data is read).

For every tensor we record file, absolute byte offset, size, dtype, shape, layer number and
whether it is a stacked routed-expert tensor ([n_experts, ...]). In MLX format one expert is
then a contiguous slice on axis 0: offset = start + expert_id * (nbytes // n_experts).
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

INDEX_VERSION = 1
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_EXPERT_MARKERS = (".switch_mlp.", ".experts.")
_EXPERT_COUNT_KEYS = ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts")
_LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layers", "num_layers")


@dataclass(frozen=True, slots=True)
class TensorEntry:
    name: str
    file: str
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    layer: int | None
    stacked_experts: bool

    def expert_slice(self, expert: int, n_experts: int) -> tuple[int, int]:
        """(offset, nbytes) of one expert inside a stacked tensor."""
        stride = self.nbytes // n_experts
        return self.offset + expert * stride, stride


@dataclass(slots=True)
class WeightIndex:
    model_dir: str
    tensors: dict[str, TensorEntry]
    n_layers: int
    n_experts: int  # 0 for dense models
    experts_per_token: int
    fingerprint: str

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    @property
    def expert_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values() if t.stacked_experts)

    @property
    def resident_bytes(self) -> int:
        """Everything that is not a routed expert (always in RAM in Drift-MoE)."""
        return self.total_bytes - self.expert_bytes

    @property
    def bytes_per_expert(self) -> int:
        if not self.is_moe:
            return 0
        moe_layers = {t.layer for t in self.tensors.values() if t.stacked_experts}
        return self.expert_bytes // (len(moe_layers) * self.n_experts)

    @property
    def expert_bytes_per_token(self) -> int:
        """Routed-expert bytes touched by one decode token."""
        if not self.is_moe:
            return 0
        moe_layers = {t.layer for t in self.tensors.values() if t.stacked_experts}
        return len(moe_layers) * self.experts_per_token * self.bytes_per_expert

    def layer_bytes(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for t in self.tensors.values():
            if t.layer is not None:
                out[t.layer] = out.get(t.layer, 0) + t.nbytes
        return out

    def to_json(self) -> str:
        data = asdict(self)
        data["version"] = INDEX_VERSION
        data["tensors"] = {k: asdict(v) for k, v in self.tensors.items()}
        return json.dumps(data)

    @classmethod
    def from_json(cls, text: str) -> WeightIndex:
        data = json.loads(text)
        if data.pop("version", None) != INDEX_VERSION:
            raise ValueError("index version mismatch")
        data["tensors"] = {
            k: TensorEntry(**{**v, "shape": tuple(v["shape"])}) for k, v in data["tensors"].items()
        }
        return cls(**data)


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Returns (header, data_start). data_start is where tensor offsets are relative to."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    return header, 8 + length


def _config(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    # Multimodal wrappers keep the language model config nested.
    return {**cfg.get("text_config", {}), **cfg}


def _fingerprint(files: list[Path]) -> str:
    h = hashlib.sha1()
    for f in files:
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def build_index(model_dir: Path) -> WeightIndex:
    model_dir = model_dir.resolve()
    cfg = _config(model_dir)
    n_experts = next((int(cfg[k]) for k in _EXPERT_COUNT_KEYS if cfg.get(k)), 0)
    experts_per_token = int(cfg.get("num_experts_per_tok") or 0)
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")

    tensors: dict[str, TensorEntry] = {}
    for f in files:
        header, data_start = read_safetensors_header(f)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            m = _LAYER_RE.search(name)
            shape = tuple(meta["shape"])
            stacked = (
                n_experts > 0
                and any(marker in name for marker in _EXPERT_MARKERS)
                and len(shape) >= 2
                and shape[0] == n_experts
            )
            tensors[name] = TensorEntry(
                name=name,
                file=str(f.resolve()),  # HF cache snapshots are symlinks to blobs
                offset=data_start + begin,
                nbytes=end - begin,
                dtype=meta["dtype"],
                shape=shape,
                layer=int(m.group(1)) if m else None,
                stacked_experts=stacked,
            )

    layers = {t.layer for t in tensors.values() if t.layer is not None}
    n_layers = next((int(cfg[k]) for k in _LAYER_COUNT_KEYS if cfg.get(k)), len(layers))
    has_stacked = any(t.stacked_experts for t in tensors.values())
    return WeightIndex(
        model_dir=str(model_dir),
        tensors=tensors,
        n_layers=n_layers,
        n_experts=n_experts if has_stacked else 0,
        experts_per_token=experts_per_token if has_stacked else 0,
        fingerprint=_fingerprint(files),
    )


def load_or_build(model_dir: Path, cache_dir: Path) -> WeightIndex:
    """Index cached in the engine data dir (we never write into model folders or the HF cache)."""
    model_dir = model_dir.resolve()
    files = sorted(model_dir.glob("*.safetensors"))
    key = hashlib.sha1(str(model_dir).encode()).hexdigest()[:16]
    path = cache_dir / f"{key}.json"
    try:
        idx = WeightIndex.from_json(path.read_text())
        if idx.fingerprint == _fingerprint(files):
            return idx
    except (OSError, ValueError, KeyError, TypeError):
        pass
    idx = build_index(model_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(idx.to_json())
    return idx
