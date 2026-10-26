"""Drift-MoE: expert planning, slots (real files, real MLX buffers) and end-to-end."""

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

from aether.drift.experts import ExpertCache, choose_slots, plan_moe
from aether.drift.index import build_index
from aether.drift.io_backend import PreadBackend
from aether.telemetry.events import EventBus
from aether.telemetry.stats import Stats

E, K, LAYERS = 16, 2, 3
MOE_MODEL = "mlx-community/Qwen3-30B-A3B-4bit"


def _write_moe(tmp: Path) -> Path:
    rng = np.random.default_rng(0)
    arrays: dict[str, mx.array] = {
        "model.embed_tokens.weight": mx.array(rng.standard_normal((8, 4)).astype(np.float16))
    }
    for layer in range(LAYERS):
        p = f"model.layers.{layer}.mlp"
        arrays[f"{p}.gate.weight"] = mx.array(rng.standard_normal((E, 4)).astype(np.float16))
        arrays[f"{p}.switch_mlp.up_proj.weight"] = mx.array(
            rng.integers(0, 2**31, (E, 6, 2), dtype=np.uint32)
        )
        arrays[f"{p}.switch_mlp.up_proj.scales"] = mx.array(
            rng.standard_normal((E, 6, 1)).astype(np.float16)
        )
    mx.save_safetensors(str(tmp / "model.safetensors"), arrays)
    (tmp / "config.json").write_text(
        json.dumps({"num_hidden_layers": LAYERS, "num_experts": E, "num_experts_per_tok": K})
    )
    return tmp


def _cache(tmp: Path, slots: int):
    model = _write_moe(tmp)
    plan = plan_moe(build_index(model))
    events = EventBus()
    return model, plan, ExpertCache(plan, slots, PreadBackend(threads=4), events, Stats()), events


def test_plan_moe(tmp_path: Path) -> None:
    plan = plan_moe(build_index(_write_moe(tmp_path)))
    assert sorted(plan.layers) == [0, 1, 2]
    lp = plan.layers[0]
    assert lp.module_path == "mlp.switch_mlp"
    assert {t.name for t in lp.tensors} == {"up_proj.weight", "up_proj.scales"}
    assert lp.bytes_per_expert == 6 * 2 * 4 + 6 * 1 * 2
    assert plan.prefill_bytes == E * lp.bytes_per_expert
    per_slot = plan.slot_bytes(1)
    assert choose_slots(plan, plan.resident_bytes + 5 * per_slot) == 5
    assert choose_slots(plan, plan.resident_bytes + per_slot) == -1  # fewer than k


def _reference(model: Path, layer: int, name: str) -> mx.array:
    return mx.load(str(model / "model.safetensors"))[f"model.layers.{layer}.mlp.switch_mlp.{name}"]


def test_slots_hold_the_right_experts(tmp_path: Path) -> None:
    model, _, cache, _ = _cache(tmp_path, slots=4)
    experts = np.array([3, 9, 15])
    mapping = np.array(cache.ensure(1, experts, decode=True))
    ls = cache.layers[1]
    for name in ("up_proj.weight", "up_proj.scales"):
        ref = _reference(model, 1, name)
        for e in experts:
            assert mx.array_equal(ls.tensors[name][int(mapping[e])], ref[int(e)]).item()


def test_lfu_keeps_frequent_experts(tmp_path: Path) -> None:
    _, _, cache, events = _cache(tmp_path, slots=3)
    for _ in range(5):  # expert 1 and 2 are popular
        cache.tick()
        cache.ensure(0, np.array([1, 2]), decode=True)
    cache.tick()
    cache.ensure(0, np.array([1, 7]), decode=True)  # 7 takes the free slot
    cache.tick()
    cache.ensure(0, np.array([1, 8]), decode=True)  # evicts 7 (rarely used), not 2
    ls = cache.layers[0]
    assert ls.slot_of[2] >= 0 and ls.slot_of[8] >= 0 and ls.slot_of[7] < 0
    assert any(e["type"] == "evict" for e in events.since(0))
    assert cache.stats.snapshot()["drift"]["hit_rate"] is None  # published by the model only


def test_prefetch_loads_next_layer(tmp_path: Path) -> None:
    model, _, cache, _ = _cache(tmp_path, slots=4)
    cache.ensure(0, np.array([1, 2]), decode=True, prefetch=(1, np.array([5, 6])))
    ls = cache.layers[1]
    assert ls.slot_of[5] >= 0 and ls.slot_of[6] >= 0
    mapping = np.array(cache.ensure(1, np.array([5, 6]), decode=True))  # waits for the reads
    ref = _reference(model, 1, "up_proj.scales")
    assert mx.array_equal(ls.tensors["up_proj.scales"][int(mapping[6])], ref[6]).item()


def test_prefill_buffer_holds_all_needed_experts(tmp_path: Path) -> None:
    model, _, cache, _ = _cache(tmp_path, slots=2)
    cache.ensure(2, np.array([4, 5]), decode=True)  # two experts already in slots
    needed = np.array([0, 4, 5, 9, 11, 15])  # more than the 2 slots
    full = cache.prefill_weights(2, needed)
    ref = _reference(model, 2, "up_proj.weight")
    for e in needed:
        assert mx.array_equal(full["up_proj.weight"][int(e)], ref[int(e)]).item()
    assert cache.layers[2].slot_of[4] >= 0  # prefill did not evict the decode set


def _moe_available() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(MOE_MODEL, "config.json"), str)


@pytest.mark.skipif(not _moe_available(), reason=f"{MOE_MODEL} not in the Hugging Face cache")
def test_drift_moe_matches_resident(tmp_path: Path) -> None:
    """M4 acceptance: forced Drift-MoE gives the same greedy output as plain mlx-lm."""
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
            "--weights-budget",
            "4GB",
            "--model",
            MOE_MODEL,
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
        deadline = time.time() + 180
        while httpx.get(f"{url}/health").json()["model_state"] != "ready":
            assert time.time() < deadline
            time.sleep(0.2)
        stats = httpx.get(f"{url}/aether/v1/stats").json()
        assert stats["model"]["mode"] == "drift-moe"
        assert stats["drift"]["expert_slots"] < stats["drift"]["experts_total"]

        model, tok = load(MOE_MODEL)
        for q in ["Noem drie kleuren.", "Schrijf een korte zin over de maan."]:
            msgs = [{"role": "user", "content": q}]
            ref = generate(
                model, tok, tok.apply_chat_template(msgs, add_generation_prompt=True), max_tokens=48
            )
            r = httpx.post(
                f"{url}/v1/chat/completions",
                timeout=300,
                json={"messages": msgs, "temperature": 0, "max_tokens": 48},
            ).json()
            assert r["choices"][0]["message"]["content"] == ref
        stats = httpx.get(f"{url}/aether/v1/stats").json()
        assert stats["drift"]["hit_rate"] is not None
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def test_unused_prefetches_do_not_leak_slots(tmp_path: Path) -> None:
    """Regression: predicted-but-unused experts used to keep their slot 'in flight' forever."""
    model, _, cache, _ = _cache(tmp_path, slots=3)
    rng = np.random.default_rng(3)
    ref = _reference(model, 1, "up_proj.weight")
    for _ in range(40):
        cache.tick()
        wrong_guess = rng.choice(E, K, replace=False)
        cache.ensure(0, rng.choice(E, K, replace=False), decode=True, prefetch=(1, wrong_guess))
        needed = rng.choice(E, K, replace=False)
        mapping = np.array(cache.ensure(1, needed, decode=True))
        for e in needed:
            assert mx.array_equal(
                cache.layers[1].tensors["up_proj.weight"][int(mapping[e])], ref[int(e)]
            ).item()
