"""OpenAI-compatible endpoints (start-prompt-v2 §7.2)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from ..runtime.errors import EngineError
from ..runtime.worker import GenerateRequest
from .sse import sse_data, sse_response
from .state import EngineState

router = APIRouter()


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[dict[str, Any]] | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "default"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    stop: str | list[str] | None = None
    seed: int | None = None


def _engine(request: Request) -> EngineState:
    return request.app.state.engine


def _text_content(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, list):  # OpenAI content parts; text only in v1
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content or ""


async def ensure_model(engine: EngineState, requested: str) -> str:
    """Returns the id of the model that will serve the request."""
    loaded = engine.worker.loaded
    if requested in ("", "default") or (loaded and requested in (loaded.info.id, loaded.info.path)):
        if loaded is None and engine.stats.model["state"] != "loading":
            raise EngineError(
                "model_not_loaded",
                "Er is nog geen model geladen.",
                hint="Kies een model.",
                status=409,
            )
        return loaded.info.id if loaded else (engine.stats.model["id"] or "default")
    if requested.startswith(("/", "~", ".")):
        # Chat clients pick from /v1/models; loading arbitrary folders is for the Mac itself.
        raise EngineError(
            "invalid_request", "Gebruik een model-id uit /v1/models, geen pad.", status=400
        )
    if not engine.config.settings.allow_client_model_switch:
        current = loaded.info.id if loaded else None
        raise EngineError(
            "model_not_loaded",
            f"Model '{requested}' is niet geladen (geladen: {current or 'geen'}).",
            hint="Gebruik model 'default' of zet 'Clients mogen van model wisselen' aan.",
            status=409,
        )
    await asyncio.wrap_future(engine.worker.submit("load", requested))
    return requested


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    loaded = engine.worker.loaded
    data = [
        {
            "id": m.id,
            "object": "model",
            "owned_by": "local",
            "created": 0,
            "loaded": bool(loaded and loaded.info.path == m.path),
        }
        for m in await asyncio.to_thread(engine.registry.scan)
    ]
    if loaded:
        data.insert(
            0,
            {
                "id": "default",
                "object": "model",
                "owned_by": "local",
                "created": 0,
                "loaded": True,
                "alias_of": loaded.info.id,
            },
        )
    return {"object": "list", "data": data}


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    engine = _engine(request)
    model_id = await ensure_model(engine, body.model)
    stops = [body.stop] if isinstance(body.stop, str) else list(body.stop or [])
    gen = GenerateRequest(
        messages=[
            {**m.model_dump(exclude_none=True), "content": _text_content(m.content)}
            for m in body.messages
        ],
        max_tokens=body.max_completion_tokens or body.max_tokens or 2048,
        temperature=0.7 if body.temperature is None else body.temperature,
        top_p=body.top_p or 1.0,
        stop=stops,
        seed=body.seed,
    )
    job, first, events = await engine.begin(gen, request)
    created = int(time.time())
    cid = f"chatcmpl-{job.id}"
    host = request.client.host if request.client else None

    if not body.stream:
        parts: list[str] = []
        ev = first
        while ev["type"] == "token":
            parts.append(ev["text"])
            ev = await engine.next_event(job, events, request)
        if ev["type"] == "error":
            raise EngineError(ev["code"], ev["message"], ev.get("hint"), status=_status(ev))
        engine.count_tokens(host, ev)
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(parts)},
                    "finish_reason": ev["finish_reason"],
                }
            ],
            "usage": _openai_usage(ev["usage"]),
            "aether": {"timings": ev.get("timings"), "generation_id": job.id},
        }

    include_usage = bool(body.stream_options and body.stream_options.include_usage)

    def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> str:
        return sse_data(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **extra,
            }
        )

    async def stream() -> AsyncIterator[str]:
        finished = False
        try:
            yield chunk({"role": "assistant", "content": ""}, aether={"generation_id": job.id})
            ev = first
            while True:
                if ev["type"] == "token":
                    yield chunk({"content": ev["text"]})
                elif ev["type"] == "done":
                    finished = True
                    engine.count_tokens(host, ev)
                    yield chunk({}, ev["finish_reason"], aether={"timings": ev.get("timings")})
                    if include_usage:
                        yield sse_data(
                            {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [],
                                "usage": _openai_usage(ev["usage"]),
                            }
                        )
                    break
                elif ev["type"] == "error":
                    finished = True
                    yield sse_data({"error": {k: ev.get(k) for k in ("code", "message", "hint")}})
                    break
                ev = await events.get()
            yield sse_data("[DONE]")
        finally:
            if not finished:  # client went away: stop generating
                engine.worker.cancel(job.id, "cancelled")

    return sse_response(stream())


def _openai_usage(u: dict[str, int]) -> dict[str, Any]:
    return {
        "prompt_tokens": u["prompt_tokens"],
        "completion_tokens": u["completion_tokens"],
        "total_tokens": u["total_tokens"],
        "prompt_tokens_details": {"cached_tokens": u["cached_tokens"]},
    }


def _status(ev: dict[str, Any]) -> int:
    return int(ev.get("status") or 500)


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "default"
    prompt: str
    stream: bool = False
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    max_tokens: int | None = Field(None, ge=1)
    stop: str | list[str] | None = None
    seed: int | None = None


@router.post("/v1/completions", response_model=None)
async def completions(body: CompletionRequest, request: Request) -> Any:
    """Raw text completion: the prompt is used as-is, without a chat template."""
    engine = _engine(request)
    model_id = await ensure_model(engine, body.model)
    gen = GenerateRequest(
        prompt=body.prompt,
        max_tokens=body.max_tokens or 256,
        temperature=0.7 if body.temperature is None else body.temperature,
        top_p=body.top_p or 1.0,
        stop=[body.stop] if isinstance(body.stop, str) else list(body.stop or []),
        seed=body.seed,
    )
    job, first, events = await engine.begin(gen, request)
    created = int(time.time())
    cid = f"cmpl-{job.id}"
    host = request.client.host if request.client else None

    def payload(text: str, finish: str | None) -> dict[str, Any]:
        return {
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": finish}],
        }

    if not body.stream:
        parts: list[str] = []
        ev = first
        while ev["type"] == "token":
            parts.append(ev["text"])
            ev = await engine.next_event(job, events, request)
        if ev["type"] == "error":
            raise EngineError(ev["code"], ev["message"], ev.get("hint"), status=_status(ev))
        engine.count_tokens(host, ev)
        return {**payload("".join(parts), ev["finish_reason"]), "usage": _openai_usage(ev["usage"])}

    async def stream() -> AsyncIterator[str]:
        finished = False
        try:
            ev = first
            while True:
                if ev["type"] == "token":
                    yield sse_data(payload(ev["text"], None))
                elif ev["type"] == "done":
                    finished = True
                    engine.count_tokens(host, ev)
                    yield sse_data(payload("", ev["finish_reason"]))
                    break
                else:
                    finished = True
                    yield sse_data({"error": {k: ev.get(k) for k in ("code", "message", "hint")}})
                    break
                ev = await events.get()
            yield sse_data("[DONE]")
        finally:
            if not finished:
                engine.worker.cancel(job.id, "cancelled")

    return sse_response(stream())
