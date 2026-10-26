from __future__ import annotations

from aether.config import GB, Mode, Settings
from aether.memory.budget import ModelShape, estimate_fit, kv_bytes_per_token, plan

RAM = 128 * GB
WORKING_SET = int(107.5 * GB)  # M3 Max 128 GB reports ~115.4e9 bytes


def test_kv_bytes_standard_attention() -> None:
    cfg = {
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "num_attention_heads": 14,
        "hidden_size": 896,
    }
    assert kv_bytes_per_token(cfg) == 2 * 24 * 2 * 64 * 2


def test_kv_bytes_mla() -> None:
    cfg = {"num_hidden_layers": 78, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    assert kv_bytes_per_token(cfg) == 78 * 576 * 2


def test_budget_respects_reserve_working_set_and_override() -> None:
    shape = ModelShape(GB, GB, 0, 100_000, 131072)
    b = plan(shape, RAM, WORKING_SET, Settings())
    assert b.usable_bytes == WORKING_SET - 2 * GB  # working set is the tighter limit
    assert b.context_tokens == 32 * 1024 and b.kv_bytes == 32 * 1024 * 100_000
    capped = plan(shape, RAM, WORKING_SET, Settings(weights_budget_gb=2))
    assert capped.weights_bytes == 2 * GB


def test_fast_mode_leaves_more_room_for_weights() -> None:
    shape = ModelShape(GB, GB, 0, 100_000, 131072)
    fast = plan(shape, RAM, WORKING_SET, Settings(mode=Mode.FAST))
    balanced = plan(shape, RAM, WORKING_SET, Settings())
    assert fast.weights_bytes > balanced.weights_bytes


def test_long_context_keeps_half_for_weights_and_quantizes_kv() -> None:
    shape = ModelShape(80 * GB, 80 * GB, 0, 200_000, 1 << 20)
    b = plan(shape, RAM, WORKING_SET, Settings(mode=Mode.LONG_CONTEXT))
    assert b.kv_bits == 8
    assert b.weights_bytes >= b.usable_bytes * 0.5 - 1


def test_fit_modes_for_glm53() -> None:
    # Numbers from bench/reports/M0.md
    glm = ModelShape(int(418.3e9), int(10.7e9), int(12.7e9), 78 * 576 * 2, 1 << 20)
    b = plan(glm, RAM, WORKING_SET, Settings())
    fit = estimate_fit(glm, b, 6.7e9)
    assert fit.mode == "drift-moe" and fit.estimated
    assert 0.5 < fit.expected_tps < 3  # honest expectation: ~0.5–2 tok/s

    small = ModelShape(int(0.28e9), int(0.28e9), 0, 12288, 32768)
    assert estimate_fit(small, plan(small, RAM, WORKING_SET, Settings()), None).mode == "resident"

    dense = ModelShape(150 * GB, 150 * GB, 0, 300_000, 32768)
    d = estimate_fit(dense, plan(dense, RAM, WORKING_SET, Settings()), 6.7e9)
    assert d.mode == "drift-dense" and d.expected_tps < 0.5


def test_too_large_when_resident_part_does_not_fit() -> None:
    huge = ModelShape(2000 * GB, 150 * GB, 20 * GB, 1000, 4096)
    fit = estimate_fit(huge, plan(huge, RAM, WORKING_SET, Settings()), 6.7e9)
    assert fit.mode == "too_large" and fit.expected_tps is None
