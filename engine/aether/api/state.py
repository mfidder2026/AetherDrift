"""Shared engine state handed to the API routes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..config import EngineConfig
from ..models.registry import Registry
from ..runtime.worker import GenerateRequest, Job, Worker
from ..system import SystemInfo
from ..telemetry.events import EventBus
from ..telemetry.stats import Stats


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
