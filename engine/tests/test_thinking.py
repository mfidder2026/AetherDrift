"""Thinking levels: what reaches the chat template, and closing GLM's always-open block."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aether.config import EngineConfig, Thinking, parse_thinking
from aether.memory.budget import Budget
from aether.models.registry import LocalModel, Registry
from aether.runtime.worker import GenerateRequest, LoadedModel, Worker
from aether.telemetry.events import EventBus
from aether.telemetry.stats import Stats

THINK, END = 7, 8


class GlmLikeTokenizer:
    """Template that always opens <think> after the assistant tag, like GLM 5."""

    chat_template = "{% if reasoning_effort %}…{% endif %}{{ messages }}"
    has_thinking = True
    think_start = "<think>"
    think_end_tokens = (END,)

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
        self.kwargs = kwargs
        return [1, 2, 3, THINK]

    def rfind_think_start(self, tokens: list[int]) -> int:
        return max((i for i, t in enumerate(tokens) if t == THINK), default=-1)

    def rfind_think_end(self, tokens: list[int]) -> int:
        return max((i for i, t in enumerate(tokens) if t == END), default=-1)


def _worker(tmp_path: Path, mode: str, setting: Thinking = Thinking.AUTO) -> Worker:
    config = EngineConfig(data_dir=tmp_path)
    config.settings.thinking = setting
    worker = Worker(config, Registry(tmp_path, tmp_path), EventBus(), Stats())
    info = LocalModel("glm", str(tmp_path), "models_dir", 0, "glm_moe_dsa", True)
    worker.loaded = LoadedModel(info, None, GlmLikeTokenizer(), Budget(0, 0, 0, 4096, None), mode)
    return worker


def _prompt(worker: Worker, req: GenerateRequest) -> tuple[list[int], dict[str, Any]]:
    assert worker.loaded is not None
    tokens = worker._tokenize(req, worker._thinking(req))
    return tokens, worker.loaded.tokenizer.kwargs


MSGS = [{"role": "user", "content": "Hoi"}]


def test_auto_turns_thinking_off_in_drift(tmp_path: Path) -> None:
    tokens, kwargs = _prompt(_worker(tmp_path, "drift-moe"), GenerateRequest(messages=MSGS))
    assert kwargs["enable_thinking"] is False
    assert tokens[-2:] == [THINK, END]  # the open block is closed: answer right away


def test_auto_thinks_briefly_when_resident(tmp_path: Path) -> None:
    tokens, kwargs = _prompt(_worker(tmp_path, "resident"), GenerateRequest(messages=MSGS))
    assert kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "enable_thinking": True,
        "reasoning_effort": "low",
    }
    assert tokens[-1] == THINK


def test_request_overrides_setting(tmp_path: Path) -> None:
    worker = _worker(tmp_path, "drift-moe", Thinking.OFF)
    tokens, kwargs = _prompt(worker, GenerateRequest(messages=MSGS, thinking=Thinking.MAX))
    assert kwargs["enable_thinking"] is True
    assert "reasoning_effort" not in kwargs  # the model's own default
    assert tokens[-1] == THINK

    _, kwargs = _prompt(worker, GenerateRequest(messages=MSGS, thinking=Thinking.HIGH))
    assert kwargs["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("none", Thinking.OFF),
        ("minimal", Thinking.OFF),
        (False, Thinking.OFF),
        ("low", Thinking.LOW),
        ("medium", Thinking.HIGH),
        ("HIGH", Thinking.HIGH),
        (True, Thinking.MAX),
        ("max", Thinking.MAX),
        ("auto", Thinking.AUTO),
        ("loud", None),
    ],
)
def test_parse_thinking(value: Any, expected: Thinking | None) -> None:
    assert parse_thinking(value) == expected


def test_reasoning_effort_only_for_templates_that_know_it(tmp_path: Path) -> None:
    worker = _worker(tmp_path, "resident")
    assert worker.loaded is not None
    worker.loaded.tokenizer.chat_template = "{{ messages }}"  # e.g. Qwen3: on/off only
    _, kwargs = _prompt(worker, GenerateRequest(messages=MSGS))
    assert "reasoning_effort" not in kwargs and kwargs["enable_thinking"] is True


def test_python_renderer_without_options_still_works(tmp_path: Path) -> None:
    """mlx-lm's own renderers (DeepSeek V3.2) reject unknown keyword arguments."""

    class StrictRenderer(GlmLikeTokenizer):
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
            if "reasoning_effort" in kwargs:
                raise TypeError("unexpected keyword argument 'reasoning_effort'")
            return super().apply_chat_template(messages, **kwargs)

    worker = _worker(tmp_path, "resident")
    assert worker.loaded is not None
    worker.loaded.tokenizer = StrictRenderer()
    tokens, kwargs = _prompt(worker, GenerateRequest(messages=MSGS))
    assert tokens == [1, 2, 3, THINK] and "reasoning_effort" not in kwargs
