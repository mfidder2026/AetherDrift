"""aether-engine CLI.

Prints exactly one JSON line on stdout when ready (start-prompt-v2 §7.5); everything else
goes to stderr and the log file.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import logging.handlers
import os
import secrets
import socket
import sys
from pathlib import Path

import uvicorn
from starlette.types import ASGIApp

from . import __version__
from .api.app import create_app
from .api.state import EngineState
from .config import EngineConfig, Mode, Settings
from .memory.guard import MemoryGuard
from .models.registry import Registry
from .runtime.worker import Worker
from .system import detect
from .telemetry.events import EventBus
from .telemetry.stats import Stats

log = logging.getLogger("aether")
PORT_ATTEMPTS = 20


def parse_size(text: str) -> float:
    """'2GB', '512MB', '1.5' (GB) → GB as float."""
    t = text.strip().upper().removesuffix("B")
    if t.endswith("G"):
        return float(t[:-1])
    if t.endswith("M"):
        return float(t[:-1]) / 1024
    return float(t)


def parse_args(argv: list[str] | None = None) -> EngineConfig:
    p = argparse.ArgumentParser(prog="aether-engine", description="AetherDrift inference engine")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--hf-cache-dir", type=Path)
    p.add_argument("--parent-pid", type=int)
    p.add_argument("--api-key-stdin", action="store_true", help="read the API key from stdin")
    p.add_argument("--share", action="store_true", help="also listen on the network")
    p.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.BALANCED.value)
    p.add_argument("--weights-budget", help="cap for weights, e.g. 2GB")
    p.add_argument("--os-reserve-gb", type=float, help="default: automatic (a quarter of RAM)")
    p.add_argument("--model", help="model id or path to load at startup")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)

    cfg = EngineConfig(host=a.host, port=a.port, parent_pid=a.parent_pid, preload_model=a.model)
    if a.data_dir:
        cfg.data_dir = a.data_dir
    cfg.hf_cache_dir = a.hf_cache_dir
    cfg.settings = Settings(
        mode=Mode(a.mode),
        share=a.share,
        os_reserve_gb=a.os_reserve_gb,
        weights_budget_gb=parse_size(a.weights_budget) if a.weights_budget else None,
    )
    if a.api_key_stdin:
        cfg.api_key = sys.stdin.readline().strip() or None
    cfg.api_key = cfg.api_key or os.environ.get("AETHER_API_KEY") or None
    if cfg.settings.share and not cfg.api_key:
        cfg.api_key = secrets.token_urlsafe(32)
        print(f"Generated API key: {cfg.api_key}", file=sys.stderr)
    logging.getLogger().setLevel(a.log_level.upper())
    return cfg


def setup_logging(cfg: EngineConfig) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(fmt)
    root.addHandler(stderr)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    file = logging.handlers.RotatingFileHandler(
        cfg.log_dir / "engine.log", maxBytes=10 * 1024**2, backupCount=3
    )
    file.setFormatter(fmt)
    root.addHandler(file)
    # Events can be high-frequency: log file only, not stderr.
    ev = logging.getLogger("aether.events")
    ev.propagate = False
    ev.addHandler(file)
    ev.setLevel(logging.INFO)


def port_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(host: str, wanted: int) -> int:
    if wanted == 0:
        with socket.socket() as s:
            s.bind((host, 0))
            return s.getsockname()[1]
    for port in range(wanted, wanted + PORT_ATTEMPTS):
        if port_free(host, port) and port_free("0.0.0.0", port):
            return port
    raise SystemExit(f"no free port in {wanted}-{wanted + PORT_ATTEMPTS - 1}")


def _server(app: ASGIApp, host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        lifespan="off",
        timeout_graceful_shutdown=2,
    )
    return uvicorn.Server(config)


async def _wait_started(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    while not server.started:
        if task.done():
            task.result()  # re-raise bind errors
            raise RuntimeError("server stopped during startup")
        await asyncio.sleep(0.01)


async def _watch_parent(pid: int, stop: asyncio.Event) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log.warning("parent process %d is gone; shutting down", pid)
            stop.set()
            return
        except PermissionError:
            pass
        await asyncio.sleep(1.0)


async def serve(cfg: EngineConfig) -> None:
    events, stats = EventBus(), Stats()
    registry = Registry(cfg.models_dir, cfg.index_dir, cfg.hf_cache_dir)
    worker = Worker(cfg, registry, events, stats)
    worker.start()
    await asyncio.to_thread(worker.wait_ready)
    ram = int(worker.device_info.get("memory_size", 0))
    system = detect(
        int(worker.device_info.get("max_recommended_working_set_size", ram)), cfg.data_dir
    )
    engine = EngineState(cfg, registry, events, stats, worker, system)

    port = pick_port(cfg.host, cfg.port)
    network: dict[str, tuple[uvicorn.Server, asyncio.Task[None]]] = {}

    async def on_share_changed(on: bool) -> None:
        if on and "srv" not in network:
            srv = _server(app, "0.0.0.0", port)
            task = asyncio.create_task(srv.serve())
            await _wait_started(srv, task)
            network["srv"] = (srv, task)
            events.publish("share_changed", share=True, port=port)
        elif not on and "srv" in network:
            srv, task = network.pop("srv")
            srv.should_exit = True
            await task
            events.publish("share_changed", share=False)

    app = create_app(engine, on_share_changed)
    local = _server(app, cfg.host, port)
    local_task = asyncio.create_task(local.serve())
    await _wait_started(local, local_task)
    if cfg.settings.share:
        await on_share_changed(True)

    ready = {"event": "ready", "port": port, "version": __version__, "pid": os.getpid()}
    sys.stdout.write(json.dumps(ready) + "\n")
    sys.stdout.flush()
    log.info("engine ready on %s:%d (shared: %s)", cfg.host, port, cfg.settings.share)

    stop = asyncio.Event()
    background = [asyncio.create_task(MemoryGuard(worker, events).run())]
    if cfg.parent_pid:
        background.append(asyncio.create_task(_watch_parent(cfg.parent_pid, stop)))
    if cfg.preload_model:
        worker.submit("load", cfg.preload_model)

    stopper = asyncio.create_task(stop.wait())
    await asyncio.wait({local_task, stopper}, return_when=asyncio.FIRST_COMPLETED)
    local.should_exit = True
    if "srv" in network:
        network["srv"][0].should_exit = True
    for t in background:
        t.cancel()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(local_task, timeout=3)


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    setup_logging(cfg)
    log.info(
        "aether-engine %s starting (budget cap: %s GB)",
        __version__,
        cfg.settings.weights_budget_gb or "auto",
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(cfg))
    # The MLX thread is a daemon; exit without waiting for in-flight GPU work.
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
