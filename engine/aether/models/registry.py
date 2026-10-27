"""Model discovery (own models folder + Hugging Face cache), catalog and downloads."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import scan_cache_dir, snapshot_download
from tqdm.auto import tqdm

from ..drift.index import WeightIndex, load_or_build
from ..memory.budget import ModelShape, kv_bytes_per_token

log = logging.getLogger("aether.models")
CATALOG = json.loads((Path(__file__).parent / "catalog.json").read_text())


@dataclass
class LocalModel:
    id: str
    path: str
    source: str  # "models_dir" | "hf_cache"
    size_bytes: int
    model_type: str | None
    is_moe: bool

    def to_dict(self) -> dict:
        return asdict(self)


def is_mlx_model_dir(path: Path) -> bool:
    if not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
        return False
    try:
        cfg = json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        return False
    # MLX-converted models carry their quantization in config.json; unquantized MLX repos
    # are recognised by the mlx-community naming convention.
    return "quantization" in cfg or "mlx" in str(path).lower()


def _describe(model_id: str, path: Path, source: str) -> LocalModel:
    cfg = json.loads((path / "config.json").read_text())
    text_cfg = {**cfg.get("text_config", {}), **cfg}
    moe = any(text_cfg.get(k) for k in ("n_routed_experts", "num_experts", "num_local_experts"))
    size = sum(f.stat().st_size for f in path.glob("*.safetensors"))
    return LocalModel(model_id, str(path), source, size, cfg.get("model_type"), bool(moe))


class Registry:
    def __init__(self, models_dir: Path, index_dir: Path, hf_cache_dir: Path | None = None):
        self.models_dir = models_dir
        self.index_dir = index_dir
        self.hf_cache_dir = hf_cache_dir
        self._downloads: dict[str, threading.Event] = {}

    def scan(self) -> list[LocalModel]:
        found: dict[str, LocalModel] = {}
        if self.models_dir.is_dir():
            for d in sorted(self.models_dir.iterdir()):
                if d.is_dir() and is_mlx_model_dir(d):
                    found[d.name] = _describe(d.name, d, "models_dir")
        try:
            cache = scan_cache_dir(self.hf_cache_dir)
        except Exception:  # no cache yet / unreadable
            cache = None
        if cache is not None:
            for repo in cache.repos:
                if repo.repo_type != "model" or repo.repo_id in found:
                    continue
                for rev in sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True):
                    snap = Path(rev.snapshot_path)
                    if is_mlx_model_dir(snap) and not _incomplete(snap):
                        found[repo.repo_id] = _describe(repo.repo_id, snap, "hf_cache")
                        break
        return list(found.values())

    def resolve(self, model_id: str) -> LocalModel | None:
        p = Path(model_id).expanduser()
        if p.is_absolute() and is_mlx_model_dir(p):
            return _describe(p.name, p, "path")
        return next((m for m in self.scan() if m.id == model_id), None)

    def index(self, model: LocalModel) -> WeightIndex:
        return load_or_build(Path(model.path), self.index_dir)

    def shape(self, model: LocalModel) -> ModelShape:
        idx = self.index(model)
        cfg = json.loads((Path(model.path) / "config.json").read_text())
        text_cfg = {**cfg.get("text_config", {}), **cfg}
        return ModelShape(
            total_bytes=idx.total_bytes,
            resident_bytes=idx.resident_bytes,
            expert_bytes_per_token=idx.expert_bytes_per_token,
            kv_bytes_per_token=kv_bytes_per_token(cfg),
            max_context=int(text_cfg.get("max_position_embeddings") or 0),
        )

    def catalog(self) -> list[dict[str, Any]]:
        return CATALOG

    def download(
        self,
        repo: str,
        on_progress: Callable[[int, int], None],
        on_done: Callable[[str | None, Exception | None], None],
    ) -> None:
        """Resumable download in a background thread (huggingface_hub resumes partial files)."""
        if repo in self._downloads:
            return
        cancel = threading.Event()
        self._downloads[repo] = cancel
        total = next((c["size_bytes"] for c in CATALOG if c["repo"] == repo), 0)

        # Several byte bars can run for one download (e.g. Xet: download + reconstruction);
        # progress is the furthest any of them got, not their sum.
        bars: dict[int, int] = {}

        class _Progress(tqdm):  # type: ignore[misc]
            def update(self, n: float | None = 1) -> bool | None:
                if cancel.is_set():
                    raise DownloadCancelled(repo)
                result = super().update(n)
                if self.unit == "B" and n:
                    bars[id(self)] = int(self.n)
                    size = max(total, int(self.total or 0))
                    on_progress(max(bars.values()), size)
                return result

        def run() -> None:
            try:
                path = snapshot_download(
                    repo,
                    cache_dir=self.hf_cache_dir,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "*.model",
                        "*.txt",
                        "*.jinja",
                    ],
                    tqdm_class=_Progress,
                )
                on_done(path, None)
            except Exception as exc:
                on_done(None, exc)
            finally:
                self._downloads.pop(repo, None)

        threading.Thread(target=run, name=f"download-{repo}", daemon=True).start()

    def cancel_download(self, repo: str) -> bool:
        ev = self._downloads.get(repo)
        if ev is None:
            return False
        ev.set()
        return True


class DownloadCancelled(Exception):
    pass


def _incomplete(snapshot: Path) -> bool:
    """A sharded model whose index lists files that are not (yet) present."""
    idx = snapshot / "model.safetensors.index.json"
    if not idx.is_file():
        return False
    try:
        files = set(json.loads(idx.read_text())["weight_map"].values())
    except (OSError, ValueError, KeyError):
        return True
    return any(not (snapshot / f).exists() for f in files)
