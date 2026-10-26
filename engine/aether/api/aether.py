"""AetherDrift management and telemetry endpoints (start-prompt-v2 §7.4)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from .. import __version__
from ..config import Settings
from ..memory.budget import FitEstimate, ModelShape, estimate_fit, plan
from ..runtime.errors import EngineError
from ..runtime.worker import SUPPORTED_MODES
from ..system import quick_ssd_test
from .sse import sse_data, sse_response
from .state import EngineState

router = APIRouter()
STATS_INTERVAL_S = 0.25


def _engine(request: Request) -> EngineState:
    return request.app.state.engine


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    return {"status": "ok", "version": __version__, "model_state": engine.stats.model["state"]}


@router.get("/aether/v1/system")
async def system(request: Request) -> dict[str, Any]:
    return _engine(request).system.to_dict()


@router.post("/aether/v1/system/ssd-test")
async def ssd_test(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    speed = await asyncio.to_thread(quick_ssd_test, engine.config.data_dir)
    engine.system.ssd_read_bytes_per_s = speed
    engine.events.publish("ssd_test", read_bytes_per_s=int(speed))
    return engine.system.to_dict()


def _describe_models(engine: EngineState) -> list[dict[str, Any]]:
    sysinfo = engine.system
    loaded = engine.worker.loaded
    out = []
    for m in engine.registry.scan():
        entry = m.to_dict()
        entry["loaded"] = bool(loaded and loaded.info.path == m.path)
        try:
            shape = engine.registry.shape(m)
            budget = plan(
                shape, sysinfo.ram_bytes, sysinfo.metal_working_set_bytes, engine.config.settings
            )
            entry["fit"] = _with_loadable(estimate_fit(shape, budget, sysinfo.ssd_read_bytes_per_s))
        except Exception as exc:  # unreadable/odd model: still list it
            entry["fit"] = None
            entry["fit_error"] = str(exc)
        out.append(entry)
    return out


@router.get("/aether/v1/models")
async def models(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    local = await asyncio.to_thread(_describe_models, engine)
    have = {m["id"] for m in local}
    catalog = [
        {**c, "downloaded": c["repo"] in have, "fit": _catalog_fit(engine, c)}
        for c in engine.registry.catalog()
    ]
    return {"local": local, "catalog": catalog}


def _catalog_fit(engine: EngineState, entry: dict[str, Any]) -> dict[str, Any]:
    """Fit for a model that may not be downloaded yet, from the catalog's shape data."""
    shape = ModelShape(
        total_bytes=entry["size_bytes"],
        resident_bytes=entry["resident_bytes"],
        expert_bytes_per_token=entry["expert_bytes_per_token"],
        kv_bytes_per_token=entry["kv_bytes_per_token"],
        max_context=entry["max_context"],
    )
    sysinfo = engine.system
    budget = plan(shape, sysinfo.ram_bytes, sysinfo.metal_working_set_bytes, engine.config.settings)
    return _with_loadable(estimate_fit(shape, budget, sysinfo.ssd_read_bytes_per_s))


def _with_loadable(fit: FitEstimate) -> dict[str, Any]:
    return {**fit.to_dict(), "loadable": fit.mode in SUPPORTED_MODES}


class LoadBody(BaseModel):
    id: str
    wait: bool = False


@router.post("/aether/v1/models/load", status_code=202)
async def load(body: LoadBody, request: Request) -> dict[str, Any]:
    engine = _engine(request)
    fut = engine.worker.submit("load", body.id)
    if body.wait:
        return await asyncio.wrap_future(fut)

    def _report(f: Any) -> None:
        if (exc := f.exception()) is not None:
            err = (
                exc.to_dict()
                if isinstance(exc, EngineError)
                else {"code": "internal", "message": str(exc)}
            )
            engine.stats.update("model", state="error")
            engine.events.publish("error", source="load", model=body.id, **err)

    fut.add_done_callback(_report)
    return {"id": body.id, "state": "loading"}


@router.post("/aether/v1/models/unload")
async def unload(request: Request) -> dict[str, Any]:
    await asyncio.wrap_future(_engine(request).worker.submit("unload"))
    return {"state": "none"}


class DownloadBody(BaseModel):
    repo: str


@router.post("/aether/v1/models/download", status_code=202)
async def download(body: DownloadBody, request: Request) -> dict[str, Any]:
    engine = _engine(request)
    last = [0.0]

    def progress(done: int, total: int) -> None:
        frac = done / total if total else None
        if frac is None or frac - last[0] >= 0.005:
            last[0] = frac or 0.0
            engine.events.publish(
                "download_progress",
                repo=body.repo,
                bytes=done,
                total=total,
                progress=round(frac, 4) if frac is not None else None,
            )

    def done(path: str | None, exc: Exception | None) -> None:
        if exc is None:
            engine.events.publish("download_done", repo=body.repo, path=path)
        else:
            engine.events.publish(
                "download_failed", repo=body.repo, code="download_interrupted", message=str(exc)
            )

    engine.registry.download(body.repo, progress, done)
    return {"repo": body.repo, "state": "downloading"}


@router.delete("/aether/v1/models/download/{repo:path}")
async def cancel_download(repo: str, request: Request) -> dict[str, Any]:
    if not _engine(request).registry.cancel_download(repo):
        raise EngineError("not_found", f"Geen lopende download voor '{repo}'.", status=404)
    return {"repo": repo, "state": "cancelled"}


@router.post("/aether/v1/generations/{gen_id}/cancel")
async def cancel_generation(gen_id: str, request: Request) -> dict[str, Any]:
    if not _engine(request).worker.cancel(gen_id):
        raise EngineError("not_found", f"Generatie '{gen_id}' bestaat niet (meer).", status=404)
    return {"id": gen_id, "state": "cancelling"}


@router.get("/aether/v1/stats")
async def stats(request: Request) -> dict[str, Any]:
    return _engine(request).stats.snapshot()


@router.get("/aether/v1/stats/stream")
async def stats_stream(request: Request) -> Any:
    engine = _engine(request)

    async def gen() -> AsyncIterator[str]:
        while True:
            yield sse_data(await asyncio.to_thread(engine.stats.snapshot))
            await asyncio.sleep(STATS_INTERVAL_S)

    return sse_response(gen())


@router.get("/aether/v1/events/stream")
async def events_stream(request: Request, since: int | None = None) -> Any:
    engine = _engine(request)

    async def gen() -> AsyncIterator[str]:
        async for batch in engine.events.stream(since=since):
            for event in batch:
                yield sse_data(event)

    return sse_response(gen())


@router.get("/aether/v1/config")
async def get_config(request: Request) -> dict[str, Any]:
    return _engine(request).config.settings.model_dump()


@router.put("/aether/v1/config")
async def put_config(body: dict[str, Any], request: Request) -> dict[str, Any]:
    engine = _engine(request)
    old = engine.config.settings
    new = Settings.model_validate({**old.model_dump(), **body})
    engine.config.settings = new
    if new.share != old.share:
        await request.app.state.on_share_changed(new.share)
    budget_keys = ("mode", "os_reserve_gb", "engine_overhead_gb", "weights_budget_gb")
    needs_reload = engine.worker.loaded is not None and any(
        getattr(old, k) != getattr(new, k) for k in budget_keys
    )
    engine.events.publish(
        "config_changed", changed=sorted(k for k in body if k in Settings.model_fields)
    )
    return {**new.model_dump(), "requires_reload": needs_reload}


@router.get("/aether/v1/clients")
async def clients(request: Request) -> dict[str, Any]:
    return {
        "clients": sorted(
            _engine(request).clients.values(), key=lambda c: c.get("last_request", 0), reverse=True
        )
    }
