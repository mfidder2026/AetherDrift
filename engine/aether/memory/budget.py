"""Memory budget and fit estimates (start-prompt-v2 §6.2, §6.9).

All speed figures produced here are estimates and are labelled as such in the API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..config import GB, MODE_CONTEXT, MODE_KV_BITS, Settings, auto_os_reserve_gb

# Assumed effective GPU memory bandwidth for the resident estimate. Replaced by measured
# averages once a model has been used.
ASSUMED_MEMORY_BANDWIDTH = 250e9
ASSUMED_MOE_HIT_RATE = 0.5
MIN_LONG_CONTEXT_WEIGHT_SHARE = 0.5
# Long context for a model that streams via Drift: every GB of KV is a GB less of weights in
# RAM, i.e. speed, and reading in a prompt this long already takes about an hour.
LONG_CONTEXT_DRIFT_TOKENS = 128 * 1024


@dataclass(frozen=True, slots=True)
class ModelShape:
    total_bytes: int
    resident_bytes: int  # non-expert part
    expert_bytes_per_token: int  # 0 for dense
    kv_bytes_per_token: int
    max_context: int

    @property
    def is_moe(self) -> bool:
        return self.expert_bytes_per_token > 0

    @property
    def active_bytes_per_token(self) -> int:
        return (
            self.resident_bytes + self.expert_bytes_per_token if self.is_moe else self.total_bytes
        )


@dataclass(frozen=True, slots=True)
class Budget:
    usable_bytes: int  # min(RAM - reserve, Metal working set) - overhead
    kv_bytes: int
    weights_bytes: int
    context_tokens: int
    kv_bits: int | None


@dataclass(frozen=True, slots=True)
class FitEstimate:
    mode: str  # "resident" | "drift-dense" | "drift-moe" | "too_large"
    required_min_bytes: int
    available_bytes: int
    expected_tps: float | None
    estimated: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def kv_bytes_per_token(cfg: dict, dtype_bytes: int = 2) -> int:
    """KV cache bytes per token from the model config (MLA and standard attention)."""
    cfg = {**cfg.get("text_config", {}), **cfg}
    layers = int(cfg.get("num_hidden_layers") or 0)
    if cfg.get("kv_lora_rank"):  # MLA: compressed latent + rope key per layer
        per_layer = int(cfg["kv_lora_rank"]) + int(cfg.get("qk_rope_head_dim") or 0)
        return layers * per_layer * dtype_bytes
    heads = int(cfg.get("num_key_value_heads") or cfg.get("num_attention_heads") or 0)
    head_dim = int(cfg.get("head_dim") or 0) or (
        int(cfg.get("hidden_size") or 0) // max(1, int(cfg.get("num_attention_heads") or 1))
    )
    return 2 * layers * heads * head_dim * dtype_bytes


def usable_bytes(ram: int, working_set: int, settings: Settings) -> int:
    reserve_gb = settings.os_reserve_gb
    if reserve_gb is None:
        reserve_gb = auto_os_reserve_gb(ram)
    reserve = int(reserve_gb * GB)
    overhead = int(settings.engine_overhead_gb * GB)
    return max(0, min(ram - reserve, working_set) - overhead)


def plan(shape: ModelShape, ram: int, working_set: int, settings: Settings) -> Budget:
    usable = usable_bytes(ram, working_set, settings)
    kv_bits = MODE_KV_BITS[settings.mode]
    per_token = shape.kv_bytes_per_token * (kv_bits or 16) // 16
    target = MODE_CONTEXT[settings.mode]
    if target is None:  # long context: as much as fits while weights keep ≥ 50 %
        kv_room = max(usable - shape.total_bytes, int(usable * (1 - MIN_LONG_CONTEXT_WEIGHT_SHARE)))
        target = kv_room // max(1, per_token)
        if shape.total_bytes > usable:
            target = min(target, LONG_CONTEXT_DRIFT_TOKENS)
    context = max(1024, min(target, shape.max_context or target))
    kv = context * per_token
    weights = usable - kv
    if settings.weights_budget_gb is not None:
        weights = min(weights, int(settings.weights_budget_gb * GB))
    return Budget(usable, kv, max(0, weights), context, kv_bits)


def estimate_fit(shape: ModelShape, budget: Budget, ssd_bytes_per_s: float | None) -> FitEstimate:
    if shape.total_bytes <= budget.weights_bytes:
        return FitEstimate(
            "resident",
            shape.total_bytes + budget.kv_bytes,
            budget.usable_bytes,
            ASSUMED_MEMORY_BANDWIDTH / max(1, shape.active_bytes_per_token),
        )

    ssd = ssd_bytes_per_s or 0.0
    if shape.is_moe:
        # Always-resident part + room for one layer's worth of active experts.
        minimum = shape.resident_bytes + shape.expert_bytes_per_token
        if minimum + budget.kv_bytes > budget.usable_bytes:
            return FitEstimate("too_large", minimum + budget.kv_bytes, budget.usable_bytes, None)
        cache = budget.weights_bytes - shape.resident_bytes
        expert_total = shape.total_bytes - shape.resident_bytes
        hit = max(ASSUMED_MOE_HIT_RATE, cache / expert_total) if expert_total else 1.0
        streamed = (1 - min(hit, 1.0)) * shape.expert_bytes_per_token
        tps = ssd / streamed if ssd and streamed else None
        return FitEstimate("drift-moe", minimum + budget.kv_bytes, budget.usable_bytes, tps)

    streamed = shape.total_bytes - budget.weights_bytes
    tps = ssd / streamed if ssd else None
    return FitEstimate("drift-dense", budget.kv_bytes, budget.usable_bytes, tps)
