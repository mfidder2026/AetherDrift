"""API behaviour with a fake worker (no MLX): auth, SSE format, errors, config."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import httpx
import pytest

from aether.api.app import create_app
from aether.api.state import EngineState
from aether.config import EngineConfig, Settings
from aether.models.registry import Registry
from aether.system import SystemInfo
from aether.telemetry.events import EventBus
from aether.telemetry.stats import Stats


class FakeLoaded:
    class info:  # noqa: N801
        id = "fake/model"
        path = "/models/fake"


class FakeWorker:
    def __init__(self, tokens: list[str], error: dict[str, Any] | None = None) -> None:
        self.tokens = tokens
        self.error = error
        self.loaded = FakeLoaded()
        self.cancelled: list[str] = []

    def submit(self, kind: str, payload: Any = None) -> Future[Any]:
        fut: Future[Any] = Future()
        if kind == "generate":
            if self.error:
                payload.emit({"type": "error", **self.error})
            else:
                for t in self.tokens:
                    payload.emit({"type": "token", "text": t})
                payload.emit(
                    {
                        "type": "done",
                        "finish_reason": "stop",
                        "timings": {},
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": len(self.tokens),
                            "total_tokens": 3 + len(self.tokens),
                            "cached_tokens": 1,
                        },
                    }
                )
        fut.set_result(None)
        return fut

    def cancel(self, job_id: str, reason: str = "cancelled") -> bool:
        self.cancelled.append(job_id)
        return True


def make_engine(tmp_path: Path, worker: FakeWorker, **settings: Any) -> EngineState:
    cfg = EngineConfig(data_dir=tmp_path, api_key="secret", settings=Settings(**settings))
    return EngineState(
        cfg,
        Registry(tmp_path / "Models", tmp_path / "idx", tmp_path / "hf"),
        EventBus(),
        Stats(),
        worker,
        SystemInfo("chip", 128, 100, "27"),  # type: ignore[arg-type]
    )


def client(engine: EngineState, host: str = "127.0.0.1") -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(engine), client=(host, 50000))
    return httpx.AsyncClient(transport=transport, base_url="http://engine")


CHAT = {"messages": [{"role": "user", "content": "hoi"}]}


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_loopback_needs_no_key(tmp_path: Path) -> None:
    async def go() -> httpx.Response:
        async with client(make_engine(tmp_path, FakeWorker(["a"]))) as c:
            return await c.get("/health")

    assert run(go()).json()["status"] == "ok"


@pytest.mark.parametrize(
    ("share", "headers", "status", "code"),
    [
        (False, {"Authorization": "Bearer secret"}, 403, "sharing_off"),
        (True, {}, 401, "unauthorized"),
        (True, {"Authorization": "Bearer wrong"}, 401, "unauthorized"),
        (True, {"Authorization": "Bearer secret"}, 200, None),
    ],
)
def test_network_auth(
    tmp_path: Path, share: bool, headers: dict, status: int, code: str | None
) -> None:
    engine = make_engine(tmp_path, FakeWorker(["a"]), share=share)

    async def go() -> httpx.Response:
        async with client(engine, host="192.168.1.50") as c:
            return await c.get("/health", headers=headers)

    r = run(go())
    assert r.status_code == status
    if code:
        assert r.json()["error"]["code"] == code
    else:
        assert engine.clients["192.168.1.50"]["requests"] == 1


def test_chat_non_stream(tmp_path: Path) -> None:
    async def go() -> dict:
        async with client(make_engine(tmp_path, FakeWorker(["Hal", "lo"]))) as c:
            return (await c.post("/v1/chat/completions", json=CHAT)).json()

    body = run(go())
    assert body["choices"][0]["message"]["content"] == "Hallo"
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 1


def test_chat_stream_is_openai_sse(tmp_path: Path) -> None:
    async def go() -> list[str]:
        async with client(make_engine(tmp_path, FakeWorker(["Hal", "lo"]))) as c:
            body = {**CHAT, "stream": True, "stream_options": {"include_usage": True}}
            async with c.stream("POST", "/v1/chat/completions", json=body) as r:
                assert r.headers["content-type"].startswith("text/event-stream")
                return [line async for line in r.aiter_lines() if line]

    lines = run(go())
    assert all(line.startswith("data: ") for line in lines)
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "Hallo"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(c["choices"] and c["choices"][0]["finish_reason"] == "stop" for c in chunks)
    assert chunks[-1]["usage"]["completion_tokens"] == 2


def test_worker_error_becomes_http_error(tmp_path: Path) -> None:
    err = {"code": "context_too_long", "message": "te lang", "hint": "nieuw gesprek", "status": 400}

    async def go() -> httpx.Response:
        async with client(make_engine(tmp_path, FakeWorker([], error=err))) as c:
            return await c.post("/v1/chat/completions", json={**CHAT, "stream": True})

    r = run(go())
    assert r.status_code == 400
    assert r.json()["error"] == {
        "code": "context_too_long",
        "message": "te lang",
        "hint": "nieuw gesprek",
    }


def test_other_model_is_rejected_unless_allowed(tmp_path: Path) -> None:
    async def go() -> httpx.Response:
        async with client(make_engine(tmp_path, FakeWorker(["a"]))) as c:
            return await c.post("/v1/chat/completions", json={**CHAT, "model": "other/model"})

    r = run(go())
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_not_loaded"


def test_validation_error_format(tmp_path: Path) -> None:
    async def go() -> httpx.Response:
        async with client(make_engine(tmp_path, FakeWorker(["a"]))) as c:
            return await c.post("/v1/chat/completions", json={"messages": []})

    r = run(go())
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


def test_config_update_reports_reload(tmp_path: Path) -> None:
    async def go() -> dict:
        async with client(make_engine(tmp_path, FakeWorker(["a"]))) as c:
            return (await c.put("/aether/v1/config", json={"mode": "snel"})).json()

    body = run(go())
    assert body["mode"] == "snel" and body["requires_reload"] is True
