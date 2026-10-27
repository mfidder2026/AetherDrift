"""Drift-Dense: unit planning, residency (real files, real MLX buffers) and end-to-end."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import mlx.core as mx
import numpy as np
import pytest
from conftest import SMALL_MODEL, requires_model

from aether.drift.index import build_index
from aether.drift.io_backend import PreadBackend
from aether.drift.residency import DenseResidency
from aether.drift.units import PAGE, choose_pinned, plan_dense
from aether.telemetry.events import EventBus
from aether.telemetry.stats import Stats

N_LAYERS = 6


def _write_model(tmp: Path) -> Path:
    """Toy dense model with random data, so reads can be checked byte for byte."""
    rng = np.random.default_rng(0)
    tensors: dict[str, np.ndarray] = {"model.embed_tokens.weight": rng.standard_normal((32, 8))}
    for i in range(N_LAYERS):
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = rng.standard_normal((16, 8))
        tensors[f"model.layers.{i}.mlp.up_proj.scales"] = rng.integers(
            0, 2**31, (16, 3), dtype=np.uint32
        )
        tensors[f"model.layers.{i}.input_layernorm.weight"] = rng.standard_normal((8,))
    arrays = {
        k: mx.array(v.astype(np.float16) if v.dtype == np.float64 else v)
        for k, v in tensors.items()
    }
    # Two shards, as real models have.
    names = sorted(arrays)
    mx.save_safetensors(
        str(tmp / "model-00001-of-00002.safetensors"), {k: arrays[k] for k in names[::2]}
    )
    mx.save_safetensors(
        str(tmp / "model-00002-of-00002.safetensors"), {k: arrays[k] for k in names[1::2]}
    )
    (tmp / "config.json").write_text(json.dumps({"num_hidden_layers": N_LAYERS}))
    return tmp


def _residency(tmp: Path, pinned: int, slots: int = 3, depth: int = 2):
    model = _write_model(tmp)
    plan = plan_dense(build_index(model))
    events = EventBus()
    res = DenseResidency(plan, pinned, slots, depth, PreadBackend(threads=4), events, Stats())
    res.load_pinned()
    return model, plan, res, events


def test_plan_layout(tmp_path: Path) -> None:
    plan = plan_dense(build_index(_write_model(tmp_path)))
    assert plan.layer_prefix == "model.layers."
    assert [u.layer for u in plan.units] == list(range(N_LAYERS))
    unit = plan.units[0]
    assert {t.name for t in unit.tensors} == {
        "mlp.up_proj.weight",
        "mlp.up_proj.scales",
        "input_layernorm.weight",
    }
    # Reads are page-aligned (F_NOCACHE only bypasses the page cache for aligned I/O).
    for span in (s for u in plan.units for s in u.spans):
        at_eof = span.file_offset + span.nbytes == os.path.getsize(span.file)
        assert span.file_offset % PAGE == 0 and (span.nbytes % PAGE == 0 or at_eof)
    assert unit.buffer_bytes >= sum(s.nbytes for s in unit.spans)
    for span in (s for u in plan.units for s in u.spans):
        assert span.buffer_offset % PAGE == 0
    assert unit.buffer_bytes >= sum(t.nbytes for t in unit.tensors)
    assert [t.name for t in plan.core] == ["model.embed_tokens.weight"]


def test_choose_pinned() -> None:
    from aether.drift.units import DensePlan, LayerUnit

    units = [LayerUnit(i, buffer_bytes=100) for i in range(10)]
    plan = DensePlan(core=[], units=units, layer_prefix="model.layers.")
    assert choose_pinned(plan, 300, slots=3) == 0  # only the ring fits
    assert choose_pinned(plan, 750, slots=3) == 4
    assert choose_pinned(plan, 299, slots=3) == -1  # not even the ring


def test_streamed_tensors_match_file(tmp_path: Path) -> None:
    model, plan, res, _ = _residency(tmp_path, pinned=2)
    reference = {}
    for f in model.glob("*.safetensors"):
        reference.update(mx.load(str(f)))
    for layer in range(N_LAYERS):  # pinned and streamed layers alike
        got = res.acquire(layer)
        for name, arr in got.items():
            want = reference[f"model.layers.{layer}.{name}"]
            assert arr.dtype == want.dtype and arr.shape == want.shape
            assert mx.array_equal(arr, want).item(), (layer, name)
        res.release(layer)
    res.close()


def test_prefetch_keeps_ring_full_and_evicts(tmp_path: Path) -> None:
    _, _, res, events = _residency(tmp_path, pinned=2, slots=3, depth=2)
    assert len(res.slots) == 3
    for _ in range(3):  # three "tokens": cyclic access 0..N-1
        for layer in range(N_LAYERS):
            res.acquire(layer)
            res.release(layer)
    types = [e["type"] for e in events.since(0)]
    assert types.count("evict") > 0
    # Streamed layers were prefetched before use, so (almost) everything was a hit.
    s = res.stats.snapshot()["drift"]
    assert s["hit_rate"] is not None and s["hit_rate"] > 0.5
    assert s["pinned_layers"] == 2 and len(s["layer_residency"]) == N_LAYERS
    assert sum(1 for v in s["layer_residency"] if v > 0) <= 2 + 3
    res.close()


def test_shrink_unpins_layers(tmp_path: Path) -> None:
    _, plan, res, events = _residency(tmp_path, pinned=4)
    before = res.resident_bytes
    assert res.request_shrink(plan.units[3].buffer_bytes)
    # While idle, repeated requests must not pile up (they would later unpin everything).
    assert not res.request_shrink(plan.units[3].buffer_bytes)
    got = res.acquire(0)
    assert got
    res.release(0)
    assert len(res.pinned) == 3 and 3 in res.streamed
    assert res.resident_bytes < before
    assert any(e["type"] == "unpinned" for e in events.since(0))
    res.close()


@requires_model
def test_drift_engine_matches_resident(tmp_path: Path) -> None:
    """M3 acceptance: forced Drift gives the same greedy output as plain mlx-lm."""
    from mlx_lm import generate, load

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aether",
            "--port",
            "0",
            "--data-dir",
            str(tmp_path),
            "--log-dir",
            str(tmp_path / "logs"),
            "--weights-budget",
            "0.15GB",
            "--model",
            SMALL_MODEL,
            "--parent-pid",
            str(os.getpid()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert proc.stdout
        url = f"http://127.0.0.1:{json.loads(proc.stdout.readline())['port']}"
        deadline = time.time() + 60
        while httpx.get(f"{url}/health").json()["model_state"] != "ready":
            assert time.time() < deadline
            time.sleep(0.1)
        stats = httpx.get(f"{url}/aether/v1/stats").json()
        assert stats["model"]["mode"] == "drift-dense"
        assert 0 < stats["drift"]["pinned_layers"] < stats["drift"]["layers_total"]

        model, tok = load(SMALL_MODEL)
        for q in ["Noem drie kleuren.", "Schrijf een haiku over de zee."]:
            msgs = [{"role": "user", "content": q}]
            ref = generate(
                model, tok, tok.apply_chat_template(msgs, add_generation_prompt=True), max_tokens=48
            )
            r = httpx.post(
                f"{url}/v1/chat/completions",
                timeout=60,
                json={"messages": msgs, "temperature": 0, "max_tokens": 48},
            ).json()
            assert r["choices"][0]["message"]["content"] == ref

        stats = httpx.get(f"{url}/aether/v1/stats").json()
        assert stats["drift"]["write_bytes_per_s"] == 0
        assert stats["drift"]["hit_rate"] is not None
    finally:
        proc.terminate()
        proc.wait(timeout=20)


@pytest.mark.parametrize("budget", ["0.01GB"])
@requires_model
def test_too_small_budget_is_a_clear_error(tmp_path: Path, budget: str) -> None:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aether",
            "--port",
            "0",
            "--data-dir",
            str(tmp_path),
            "--log-dir",
            str(tmp_path / "logs"),
            "--weights-budget",
            budget,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert proc.stdout
        url = f"http://127.0.0.1:{json.loads(proc.stdout.readline())['port']}"
        r = httpx.post(
            f"{url}/aether/v1/models/load", json={"id": SMALL_MODEL, "wait": True}, timeout=60
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "model_too_large"
        assert "Drift" in r.json()["error"]["message"]
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def test_unaligned_read_is_exact_including_file_end(tmp_path: Path) -> None:
    data = np.random.default_rng(1).integers(0, 256, 3 * 4096 + 777, dtype=np.uint8)
    f = tmp_path / "blob.bin"
    f.write_bytes(data.tobytes())
    backend = PreadBackend(threads=2, chunk_bytes=4096)
    from aether.drift.io_backend import ReadRequest

    for offset, n in [(5, 100), (4090, 5000), (len(data) - 900, 900), (0, len(data))]:
        dest = np.zeros(n, dtype=np.uint8)
        backend.submit_unaligned(ReadRequest(str(f), offset, n), memoryview(dest)).result()
        assert np.array_equal(dest, data[offset : offset + n]), (offset, n)
    backend.close()


def test_read_group_waits_for_every_part_before_failing() -> None:
    """A failed or cancelled part must not release a buffer that other parts still fill."""
    from concurrent.futures import CancelledError, Future

    from aether.drift.io_backend import _all_of

    a: Future[None] = Future()
    b: Future[None] = Future()
    group = _all_of([a, b])
    a.set_exception(OSError(5, "Input/output error"))
    assert not group.done()
    b.set_result(None)
    with pytest.raises(OSError):
        group.result(timeout=1)

    c: Future[None] = Future()
    d: Future[None] = Future()
    group = _all_of([c, d])
    c.cancel()
    d.set_result(None)
    with pytest.raises(CancelledError):
        group.result(timeout=1)


def test_layer_read_waits_for_every_span_before_failing(tmp_path: Path) -> None:
    """Drift-Dense: a failed span must not free the buffer while another span writes."""
    from concurrent.futures import Future

    from aether.drift.residency import read_unit

    plan = plan_dense(build_index(_write_model(tmp_path)))
    unit = next(u for u in plan.units if len(u.spans) >= 2)
    slow: Future[None] = Future()
    calls = iter([0, 1])

    class Backend:
        def submit(self, req, buf):  # noqa: ANN001, ANN202
            if next(calls, 1) == 0:
                f: Future[None] = Future()
                f.set_exception(OSError(5, "Input/output error"))
                return f
            return slow

    group = read_unit(
        Backend(), unit, type("Buf", (), {"host": np.zeros(unit.buffer_bytes, np.uint8)})()
    )  # type: ignore[arg-type]
    assert not group.done()
    slow.set_result(None)
    with pytest.raises(OSError):
        group.result(timeout=1)


def test_background_reads_use_their_own_threads(tmp_path: Path) -> None:
    import threading

    f = tmp_path / "blob.bin"
    f.write_bytes(bytes(range(256)) * 64)
    backend = PreadBackend(threads=1, background_threads=1)
    from aether.drift.io_backend import ReadRequest

    names: list[str] = []
    backend.on_read = lambda n, s: names.append(threading.current_thread().name)
    dest = np.zeros(100, dtype=np.uint8)
    backend.submit_unaligned(
        ReadRequest(str(f), 3, 100), memoryview(dest), background=True
    ).result()
    backend.submit_unaligned(ReadRequest(str(f), 3, 100), memoryview(dest)).result()
    assert names[0].startswith("aether-io-bg") and not names[1].startswith("aether-io-bg")
    backend.close()


def test_failed_layer_read_is_retried_not_reused(tmp_path: Path) -> None:
    """Regression: a failed read used to count as a hit and leave its slot in use forever."""
    from concurrent.futures import Future

    model, _, res, _ = _residency(tmp_path, pinned=0, slots=3, depth=2)
    inner = res.backend
    fail = {"on": True}

    class Flaky:
        def submit(self, *a, **kw):  # noqa: ANN002, ANN003, ANN202
            if not fail["on"]:
                return inner.submit(*a, **kw)
            f: Future[None] = Future()
            f.set_exception(OSError(5, "Input/output error"))
            return f

        def __getattr__(self, name: str):  # noqa: ANN204
            return getattr(inner, name)

    res.backend = Flaky()
    for s in res.slots:  # drop the reads started by load_pinned
        s.owner, s.future = None, None
    for _ in range(4):  # more failures than slots: none may stay "in use"
        with pytest.raises(OSError):
            res.acquire(0)
    assert not any(s.in_use for s in res.slots)
    fail["on"] = False
    reference = {}
    for f in model.glob("*.safetensors"):
        reference.update(mx.load(str(f)))
    for layer in range(N_LAYERS):
        got = res.acquire(layer)
        for name, arr in got.items():
            assert mx.array_equal(arr, reference[f"model.layers.{layer}.{name}"]).item()
        res.release(layer)
    res.close()
