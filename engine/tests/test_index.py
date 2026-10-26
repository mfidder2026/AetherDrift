from __future__ import annotations

import json
from pathlib import Path

from conftest import write_safetensors

from aether.drift.index import build_index, load_or_build


def _moe_model(tmp: Path) -> Path:
    (tmp / "config.json").write_text(
        json.dumps(
            {
                "model_type": "toy_moe",
                "num_hidden_layers": 2,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "quantization": {"bits": 4, "group_size": 64},
            }
        )
    )
    write_safetensors(
        tmp / "model-00001-of-00002.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", [16, 8]),
            "model.layers.0.self_attn.q_proj.weight": ("BF16", [8, 8]),
            "model.layers.0.mlp.switch_mlp.up_proj.weight": ("U32", [4, 6, 2]),
            "model.layers.0.mlp.switch_mlp.up_proj.scales": ("BF16", [4, 6, 1]),
        },
    )
    write_safetensors(
        tmp / "model-00002-of-00002.safetensors",
        {
            "model.layers.1.self_attn.q_proj.weight": ("BF16", [8, 8]),
            "model.layers.1.mlp.switch_mlp.up_proj.weight": ("U32", [4, 6, 2]),
            "model.layers.1.mlp.switch_mlp.up_proj.scales": ("BF16", [4, 6, 1]),
            "lm_head.weight": ("BF16", [16, 8]),
        },
    )
    return tmp


def test_moe_index_classifies_experts(tmp_path: Path) -> None:
    idx = build_index(_moe_model(tmp_path))
    assert idx.is_moe and idx.n_experts == 4 and idx.n_layers == 2
    up = idx.tensors["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    assert up.stacked_experts and up.layer == 0 and up.nbytes == 4 * 6 * 2 * 4
    assert not idx.tensors["model.layers.0.self_attn.q_proj.weight"].stacked_experts
    assert idx.tensors["lm_head.weight"].layer is None
    # weight (48 B) + scales (12 B) per expert per layer
    assert idx.bytes_per_expert == 60
    assert idx.expert_bytes_per_token == 2 * 2 * 60
    assert idx.resident_bytes == idx.total_bytes - idx.expert_bytes


def test_offsets_point_at_tensor_data(tmp_path: Path) -> None:
    model = _moe_model(tmp_path)
    idx = build_index(model)
    t = idx.tensors["model.layers.1.mlp.switch_mlp.up_proj.scales"]
    size = Path(t.file).stat().st_size
    assert t.offset + t.nbytes <= size
    off, n = t.expert_slice(3, idx.n_experts)
    assert n == t.nbytes // 4 and off == t.offset + 3 * n


def test_dense_model_has_no_experts(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    write_safetensors(
        tmp_path / "model.safetensors",
        {
            "model.layers.0.mlp.up_proj.weight": ("F16", [4, 4]),
        },
    )
    idx = build_index(tmp_path)
    assert not idx.is_moe and idx.expert_bytes == 0 and idx.layer_bytes() == {0: 32}


def test_index_cache_roundtrip_and_invalidation(tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    model = _moe_model(tmp_path / "m")
    cache = tmp_path / "cache"
    a = load_or_build(model, cache)
    b = load_or_build(model, cache)
    assert a.tensors == b.tensors and len(list(cache.iterdir())) == 1
    write_safetensors(
        model / "model-00002-of-00002.safetensors",
        {
            "lm_head.weight": ("BF16", [32, 8]),
        },
    )
    c = load_or_build(model, cache)
    assert (
        c.fingerprint != a.fingerprint and "model.layers.1.self_attn.q_proj.weight" not in c.tensors
    )
