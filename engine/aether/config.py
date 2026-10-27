"""Engine configuration: fixed startup options plus settings that can change at runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

GB = 1024**3


class Mode(StrEnum):
    """User-facing presets (start-prompt-v2 §8.5)."""

    FAST = "snel"
    BALANCED = "balans"
    LONG_CONTEXT = "lange_context"


# Default context target per mode; LONG_CONTEXT is computed from the budget instead.
MODE_CONTEXT: dict[Mode, int | None] = {
    Mode.FAST: 8 * 1024,
    Mode.BALANCED: 32 * 1024,
    Mode.LONG_CONTEXT: None,
}
MODE_KV_BITS: dict[Mode, int | None] = {
    Mode.FAST: None,
    Mode.BALANCED: None,
    Mode.LONG_CONTEXT: 8,
}


class Thinking(StrEnum):
    """How long a reasoning model thinks before it answers."""

    AUTO = "auto"  # off when the model runs via Drift (a few tokens/s), short otherwise
    OFF = "off"
    LOW = "low"
    HIGH = "high"
    MAX = "max"  # the model's own default


# Request vocabularies: OpenAI `reasoning_effort`, Ollama `think` (bool or level).
_THINKING_ALIASES = {
    "none": Thinking.OFF,
    "minimal": Thinking.OFF,
    "false": Thinking.OFF,
    "medium": Thinking.HIGH,
    "xhigh": Thinking.MAX,
    "true": Thinking.MAX,  # "think: true" asks for the model's normal reasoning
}


def parse_thinking(value: str | bool | None) -> Thinking | None:
    """A request's thinking level; None when the request does not set one."""
    if value is None:
        return None
    key = str(value).lower()
    if key in _THINKING_ALIASES:
        return _THINKING_ALIASES[key]
    try:
        return Thinking(key)
    except ValueError:
        return None


def auto_os_reserve_gb(ram_bytes: int) -> float:
    """Memory kept free for macOS and other apps: a quarter of RAM, between 4 and 16 GB.

    A fixed 16 GB left nothing on a 16 GB Mac; on large Macs 16 GB is plenty.
    """
    return min(16.0, max(4.0, ram_bytes / GB / 4))


class Settings(BaseModel):
    """Runtime-changeable settings, exposed via GET/PUT /aether/v1/config."""

    mode: Mode = Mode.BALANCED
    share: bool = False
    allow_client_model_switch: bool = False
    os_reserve_gb: float | None = Field(None, ge=4.0, le=64.0, description="None = automatic")
    engine_overhead_gb: float = Field(2.0, ge=0.5, le=16.0)
    weights_budget_gb: float | None = Field(None, gt=0, description="Override; None = automatic")
    prefill_step_size: int = Field(2048, ge=128, le=16384)
    prompt_cache_entries: int = Field(3, ge=0, le=16)
    prefetch_experts: bool = True  # Drift-MoE: speculative prefetch of the next layer's experts
    # Drift-MoE: skip light experts that are not in RAM (faster, slightly less exact).
    turbo: bool = False
    thinking: Thinking = Thinking.AUTO


@dataclass
class EngineConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    data_dir: Path = field(
        default_factory=lambda: Path.home() / "Library/Application Support/AetherDrift"
    )
    log_dir: Path = field(default_factory=lambda: Path.home() / "Library/Logs/AetherDrift")
    hf_cache_dir: Path | None = None  # None = huggingface_hub default
    parent_pid: int | None = None
    api_key: str | None = None
    preload_model: str | None = None
    settings: Settings = field(default_factory=Settings)

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "Models"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"
