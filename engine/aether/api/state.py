"""Shared engine state handed to the API routes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from starlette.requests import Request

from ..config import EngineConfig
from ..models.registry import Registry
from ..runtime.errors import EngineError
from ..runtime.worker import GenerateRequest, Job, Worker
from ..system import SystemInfo
from ..telemetry.events import EventBus
from ..telemetry.stats import Stats

DISCONNECT_POLL_S = 0.25


@dataclass
class EngineState:
    config: EngineConfig
    registry: Registry
    events: EventBus
    stats: Stats
    worker: Worker
    system: SystemInfo
    clients: dict[str, dict[str, Any]] = field(default_factory=dict)

    def start_job(self, request: GenerateRequest) -> tuple[Job, asyncio.Queue[dict[str, Any]]]:
        """Queue a generation; its events arrive on the returned asyncio queue."""
        loop = asyncio.get_running_loop()
        out: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def emit(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(out.put_nowait, event)

        job = Job(request=request, emit=emit)
        self.worker.submit("generate", job)
        return job, out

    async def begin(
        self, request: GenerateRequest, client: Request | None = None
    ) -> tuple[Job, dict[str, Any], asyncio.Queue[dict[str, Any]]]:
        """Queue a generation and wait for its first event.

        Errors before the first token become proper HTTP errors instead of a broken stream.
        A client that leaves while queued or during prefill cancels its job.
        """
        job, events = self.start_job(request)
        try:
            first = await self.next_event(job, events, client)
        except asyncio.CancelledError:
            self.worker.cancel(job.id)
            raise
        if first["type"] == "error":
            raise EngineError(
                first["code"], first["message"], first.get("hint"), int(first.get("status") or 500)
            )
        return job, first, events

    async def next_event(
        self, job: Job, events: asyncio.Queue[dict[str, Any]], client: Request | None
    ) -> dict[str, Any]:
        """The job's next event; cancels the job when the HTTP client has gone away.

        uvicorn does not cancel a handler when its client disconnects, so a request that is
        queued, prefilling or collecting a non-streamed answer would otherwise keep the
        generator busy for nobody.
        """
        while True:
            try:
                return await asyncio.wait_for(events.get(), timeout=DISCONNECT_POLL_S)
            except TimeoutError:
                if client is not None and await client.is_disconnected():
                    self.worker.cancel(job.id, "cancelled")
                    raise asyncio.CancelledError from None

    def count_tokens(self, host: str | None, done: dict[str, Any]) -> None:
        """Per-client token totals for the Connect sheet (network clients only)."""
        client = self.clients.get(host or "")
        if client is not None:
            client["tokens"] = client.get("tokens", 0) + done["usage"]["completion_tokens"]
