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
    base = "http://127.0.0.1:8080" if host == "127.0.0.1" else "http://studio.local:8080"
    return httpx.AsyncClient(transport=transport, base_url=base)


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


NET = "192.168.1.50"
KEY = {"Authorization": "Bearer secret"}


@pytest.mark.parametrize(
    ("method", "path", "headers", "status"),
    [
        ("GET", "/", {}, 200),  # webchat page: no key needed, holds no data
        ("GET", "/manifest.webmanifest", {}, 200),
        ("GET", "/v1/models", {}, 401),
        ("GET", "/v1/models", KEY, 200),
        ("GET", "/api/tags", KEY, 200),
        ("GET", "/aether/v1/config", KEY, 403),  # management stays on the Mac
        ("PUT", "/aether/v1/config", KEY, 403),
        ("POST", "/aether/v1/models/load", KEY, 403),
        ("GET", "/aether/v1/clients", KEY, 403),
    ],
)
def test_network_surface(
    tmp_path: Path, method: str, path: str, headers: dict, status: int
) -> None:
    engine = make_engine(tmp_path, FakeWorker(["a"]), share=True)

    async def go() -> httpx.Response:
        async with client(engine, host=NET) as c:
            return await c.request(method, path, headers=headers, json={})

    r = run(go())
    assert r.status_code == status
    if status == 403:
        assert r.json()["error"]["code"] == "local_only"


def test_webchat_page_has_strict_headers(tmp_path: Path) -> None:
    async def go() -> httpx.Response:
        async with client(make_engine(tmp_path, FakeWorker(["a"]))) as c:
            return await c.get("/")

    r = run(go())
    assert r.headers["content-type"].startswith("text/html")
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "/v1/chat/completions" in r.text


def test_network_client_tokens_are_counted(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeWorker(["Hal", "lo"]), share=True)

    async def go() -> None:
        async with client(engine, host=NET) as c:
            await c.post("/v1/chat/completions", json=CHAT, headers=KEY)
            await c.post("/api/chat", json={**CHAT, "stream": False}, headers=KEY)

    run(go())
    assert engine.clients[NET]["requests"] == 2
    assert engine.clients[NET]["tokens"] == 4


def test_ollama_chat_streams_ndjson(tmp_path: Path) -> None:
    async def go() -> list[dict]:
        engine = make_engine(tmp_path, FakeWorker(["Hal", "lo"]))
        # Ollama streams by default.
        async with client(engine) as c, c.stream("POST", "/api/chat", json=CHAT) as r:
            assert r.headers["content-type"].startswith("application/x-ndjson")
            return [json.loads(line) async for line in r.aiter_lines() if line]

    lines = run(go())
    assert [x["message"]["content"] for x in lines] == ["Hal", "lo", ""]
    assert [x["done"] for x in lines] == [False, False, True]
    final = lines[-1]
    assert final["done_reason"] == "stop" and final["eval_count"] == 2
    assert final["prompt_eval_count"] == 3 and final["model"] == "fake/model"


def test_ollama_generate_and_tags(tmp_path: Path) -> None:
    async def go() -> tuple[dict, dict, dict]:
        async with client(make_engine(tmp_path, FakeWorker(["4"]))) as c:
            gen = await c.post(
                "/api/generate", json={"prompt": "2+2?", "stream": False, "options": {"seed": 1}}
            )
            tags = await c.get("/api/tags")
            ver = await c.get("/api/version")
            return gen.json(), tags.json(), ver.json()

    gen, tags, ver = run(go())
    assert gen["response"] == "4" and gen["done"] is True
    assert tags == {"models": []}
    assert ver["version"]


def test_ollama_error_is_http_error(tmp_path: Path) -> None:
    worker = FakeWorker([], error={"code": "model_not_loaded", "message": "x", "status": 409})

    async def go() -> httpx.Response:
        async with client(make_engine(tmp_path, worker)) as c:
            return await c.post("/api/chat", json=CHAT)

    r = run(go())
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_not_loaded"


def test_text_completions(tmp_path: Path) -> None:
    async def go() -> dict:
        async with client(make_engine(tmp_path, FakeWorker(["b", "c"]))) as c:
            return (await c.post("/v1/completions", json={"prompt": "a"})).json()

    body = run(go())
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "bc"
    assert body["usage"]["completion_tokens"] == 2


def test_share_toggle_does_not_reload(tmp_path: Path) -> None:
    async def go() -> dict:
        async with client(make_engine(tmp_path, FakeWorker(["a"]))) as c:
            return (await c.put("/aether/v1/config", json={"share": True})).json()

    assert run(go())["requires_reload"] is False


def test_api_key_rotation(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeWorker(["a"]), share=True)
    new = "n" * 32

    async def go() -> tuple[int, int, int, int]:
        async with client(engine) as local, client(engine, host=NET) as net:
            denied = await net.put("/aether/v1/api-key", json={"key": new}, headers=KEY)
            ok = await local.put("/aether/v1/api-key", json={"key": new})
            old = await net.get("/v1/models", headers=KEY)
            fresh = await net.get("/v1/models", headers={"Authorization": f"Bearer {new}"})
            return denied.status_code, ok.status_code, old.status_code, fresh.status_code

    assert run(go()) == (403, 204, 401, 200)


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({}, 200),
        ({"Origin": "http://127.0.0.1:8080"}, 200),  # the webchat itself
        ({"Origin": "http://localhost:3000"}, 200),
        ({"Origin": "https://evil.example"}, 403),  # CSRF from a web page
        ({"Host": "evil.example:8080"}, 403),  # DNS rebinding
        ({"Host": "localhost:8080"}, 200),
    ],
)
def test_loopback_refuses_web_pages(tmp_path: Path, headers: dict, status: int) -> None:
    async def go() -> httpx.Response:
        async with client(make_engine(tmp_path, FakeWorker(["a"]))) as c:
            return await c.post("/aether/v1/models/unload", headers=headers)

    r = run(go())
    if status == 403:
        assert r.status_code == 403 and r.json()["error"]["code"] == "foreign_origin"
    else:
        assert r.status_code != 403


def test_chat_api_refuses_model_paths(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeWorker(["a"]), allow_client_model_switch=True)

    async def go() -> httpx.Response:
        async with client(engine) as c:
            return await c.post("/v1/chat/completions", json={**CHAT, "model": "/Users/x/m"})

    r = run(go())
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


def test_failed_share_start_is_not_saved(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeWorker(["a"]))

    async def refuse(_: bool) -> None:
        raise OSError(48, "Address already in use")

    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(engine, refuse), client=("127.0.0.1", 1))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8080") as c:
            return await c.put("/aether/v1/config", json={"share": True, "mode": "snel"})

    r = run(go())
    assert r.status_code == 409 and r.json()["error"]["code"] == "share_failed"
    assert engine.config.settings.share is False
    assert engine.config.settings.mode == "snel"  # the rest of the change still applies


def test_ollama_accepts_json_without_content_type(tmp_path: Path) -> None:
    async def go() -> httpx.Response:
        async with client(make_engine(tmp_path, FakeWorker(["hoi"]))) as c:
            body = json.dumps({**CHAT, "stream": False})
            headers = {"Content-Type": "application/x-www-form-urlencoded"}  # what curl -d sends
            return await c.post("/api/chat", content=body, headers=headers)

    r = run(go())
    assert r.status_code == 200 and r.json()["message"]["content"] == "hoi"
