from __future__ import annotations

from aether.config import GB, Mode, Settings
from aether.memory.budget import (
    ModelShape,
    estimate_fit,
    kv_bytes_per_token,
    plan,
    usable_bytes,
)

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


def test_long_context_does_not_starve_a_drift_model() -> None:
    """GLM 5.3 in long-context mode used to reserve 44 GB of KV and keep 31 instead of 58
    expert slots per layer: ~40 % more SSD reads per token."""
    glm = ModelShape(int(418.3e9), int(10.7e9), int(12.7e9), 78 * 576 * 2, 1 << 20)
    long = plan(glm, RAM, WORKING_SET, Settings(mode=Mode.LONG_CONTEXT))
    balanced = plan(glm, RAM, WORKING_SET, Settings())
    assert long.context_tokens == 128 * 1024 > balanced.context_tokens
    assert long.kv_bytes < 6 * GB and long.weights_bytes > balanced.weights_bytes - 4 * GB


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


def test_os_reserve_scales_with_ram() -> None:
    no_limit = 1024 * GB  # working set not the constraint here
    s = Settings()
    assert usable_bytes(16 * GB, no_limit, s) == (16 - 4 - 2) * GB  # 4 GB floor
    assert usable_bytes(32 * GB, no_limit, s) == (32 - 8 - 2) * GB  # a quarter
    assert usable_bytes(96 * GB, no_limit, s) == (96 - 16 - 2) * GB  # 16 GB cap
    assert usable_bytes(512 * GB, no_limit, s) == (512 - 16 - 2) * GB
    assert usable_bytes(32 * GB, no_limit, Settings(os_reserve_gb=12)) == (32 - 12 - 2) * GB
