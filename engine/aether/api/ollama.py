"""Ollama-compatible endpoints (start-prompt-v2 §7.3): /api/chat, /api/generate, /api/tags.

Streaming is NDJSON, one JSON object per line, as Ollama does. Only the fields clients rely
on are filled in; unknown request fields are ignored.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .. import __version__
from ..config import parse_thinking
from ..runtime.errors import EngineError
from ..runtime.worker import GenerateRequest
from .openai import ensure_model
from .state import EngineState

router = APIRouter()


class Options(BaseModel):
    model_config = ConfigDict(extra="ignore")
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    num_predict: int | None = None
    stop: list[str] | None = None
    seed: int | None = None


class OllamaMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str = ""


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "default"
    messages: list[OllamaMessage] = Field(min_length=1)
    stream: bool = True
    options: Options | None = None
    think: bool | str | None = None


class GenerateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "default"
    prompt: str = ""
    system: str | None = None
    raw: bool = False
    stream: bool = True
    options: Options | None = None
    think: bool | str | None = None


def _engine(request: Request) -> EngineState:
    return request.app.state.engine


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _request(opts: Options | None, think: bool | str | None, **kw: Any) -> GenerateRequest:
    o = opts or Options()
    return GenerateRequest(
        # Ollama: -1 (or nothing) generates until the context is full.
        max_tokens=o.num_predict if o.num_predict and o.num_predict > 0 else None,
        temperature=0.8 if o.temperature is None else o.temperature,  # Ollama's default
        top_p=o.top_p or 1.0,
        stop=list(o.stop or []),
        seed=o.seed,
        thinking=parse_thinking(think),
        **kw,
    )


def _final(done: dict[str, Any], started: float) -> dict[str, Any]:
    usage, timings = done["usage"], done.get("timings") or {}
    total_ns = int((time.perf_counter() - started) * 1e9)
    tps = timings.get("decode_tps") or 0
    return {
        "done": True,
        "done_reason": done["finish_reason"],
        "total_duration": total_ns,
        "prompt_eval_count": usage["prompt_tokens"],
        "eval_count": usage["completion_tokens"],
        "eval_duration": int(usage["completion_tokens"] / tps * 1e9) if tps else 0,
    }


async def _respond(
    request: Request, gen: GenerateRequest, model: str, part: Any, stream: bool
) -> Any:
    """Runs the generation; `part(text)` builds the per-chunk payload (message or response)."""
    engine = _engine(request)
    started = time.perf_counter()
    model_id = await ensure_model(engine, model)
    job, first, events = await engine.begin(gen, request)
    host = request.client.host if request.client else None

    if not stream:
        parts: list[str] = []
        ev = first
        while ev["type"] == "token":
            parts.append(ev["text"])
            ev = await engine.next_event(job, events, request)
        if ev["type"] == "error":
            raise EngineError(
                ev["code"], ev["message"], ev.get("hint"), int(ev.get("status") or 500)
            )
        engine.count_tokens(host, ev)
        return {
            "model": model_id,
            "created_at": _now(),
            **part("".join(parts)),
            **_final(ev, started),
        }

    async def lines() -> AsyncIterator[str]:
        finished = False
        try:
            ev = first
            while True:
                if ev["type"] == "token":
                    chunk = {
                        "model": model_id,
                        "created_at": _now(),
                        **part(ev["text"]),
                        "done": False,
                    }
                    yield json.dumps(chunk) + "\n"
                elif ev["type"] == "done":
                    finished = True
                    engine.count_tokens(host, ev)
                    final = {
                        "model": model_id,
                        "created_at": _now(),
                        **part(""),
                        **_final(ev, started),
                    }
                    yield json.dumps(final) + "\n"
                    break
                else:
                    finished = True
                    yield json.dumps({"error": ev.get("message", "error")}) + "\n"
                    break
                ev = await events.get()
        finally:
            if not finished:  # client went away: stop generating
                engine.worker.cancel(job.id, "cancelled")

    return StreamingResponse(lines(), media_type="application/x-ndjson")


async def _body[BodyT: BaseModel](request: Request, model: type[BodyT]) -> BodyT:
    """Ollama accepts JSON without a Content-Type (`curl -d '{…}'`); so do we."""
    try:
        return model.model_validate(json.loads(await request.body() or b"{}"))
    except (ValueError, ValidationError) as exc:
        raise EngineError("invalid_request", f"Ongeldige JSON: {exc}", status=400) from exc


@router.post("/api/chat", response_model=None)
async def chat(request: Request) -> Any:
    body = await _body(request, ChatBody)
    gen = _request(body.options, body.think, messages=[m.model_dump() for m in body.messages])
    return await _respond(
        request,
        gen,
        body.model,
        lambda text: {"message": {"role": "assistant", "content": text}},
        body.stream,
    )


@router.post("/api/generate", response_model=None)
async def generate(request: Request) -> Any:
    body = await _body(request, GenerateBody)
    if not body.prompt:  # Ollama loads the model and returns immediately
        model_id = await ensure_model(_engine(request), body.model)
        return {"model": model_id, "created_at": _now(), "response": "", "done": True}
    if body.raw:
        gen = _request(body.options, body.think, prompt=body.prompt)
    else:
        messages = [{"role": "system", "content": body.system}] if body.system else []
        messages.append({"role": "user", "content": body.prompt})
        gen = _request(body.options, body.think, messages=messages)
    return await _respond(request, gen, body.model, lambda text: {"response": text}, body.stream)


def _tag(model_id: str, size: int, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": model_id,
        "model": model_id,
        "modified_at": _now(),
        "size": size,
        "digest": "",
        "details": {"format": "mlx", **(details or {})},
    }


@router.get("/api/tags")
async def tags(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    models = await asyncio.to_thread(engine.registry.scan)
    return {"models": [_tag(m.id, m.size_bytes, {"family": m.model_type or ""}) for m in models]}


@router.get("/api/ps")
async def running(request: Request) -> dict[str, Any]:
    loaded = _engine(request).worker.loaded
    return {"models": [_tag(loaded.info.id, 0)] if loaded else []}


@router.get("/api/version")
async def version() -> dict[str, str]:
    return {"version": __version__}
