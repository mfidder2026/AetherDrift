"""The generator thread: sole owner of MLX (start-prompt-v2 B8).

Commands (load, unload, generate) are queued and executed one at a time. Results are sent
back through callbacks that the API layer bridges onto its asyncio loop.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..config import EngineConfig
from ..memory.budget import Budget, estimate_fit, plan
from ..mlx_params import flat_params
from ..models.registry import LocalModel, Registry
from ..power import KeepAwake
from ..telemetry.events import EventBus
from ..telemetry.stats import Stats
from .errors import EngineError

log = logging.getLogger("aether.worker")
IDLE_TICK_S = 0.25
# Execution modes this engine build can run. Drift modes arrive in M3/M4.
SUPPORTED_MODES = frozenset({"resident", "drift-dense", "drift-moe"})
# Drift-Dense: layers read ahead while the current one computes (start-prompt-v2 §6.6).
PREFETCH_DEPTH = 2
# Drift mode keeps MLX's buffer cache small: room to recycle a couple of layer-sized
# temporaries, nothing more, so memory stays within the budget.
DRIFT_CACHE_BASE = 256 * 1024**2
MIN_PROMPT_REUSE = 32  # tokens
MEMORY_WAIT_S = 60.0  # e.g. a previous engine process still releasing ~100 GB
_ids = itertools.count(1)


@dataclass
class GenerateRequest:
    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    model: str | None = None


@dataclass
class Job:
    request: GenerateRequest
    emit: Callable[[dict[str, Any]], None]
    id: str = field(default_factory=lambda: f"gen-{next(_ids)}")
    cancelled: threading.Event = field(default_factory=threading.Event)
    cancel_reason: str = "cancelled"
    started: float = field(default_factory=time.perf_counter)

    def cancel(self, reason: str = "cancelled") -> None:
        self.cancel_reason = reason
        self.cancelled.set()


@dataclass
class LoadedModel:
    info: LocalModel
    model: Any
    tokenizer: Any
    budget: Budget
    mode: str
    drift: Any = None  # DenseResidency in Drift mode
    backend: Any = None  # PreadBackend in Drift mode


class Worker(threading.Thread):
    def __init__(self, config: EngineConfig, registry: Registry, events: EventBus, stats: Stats):
        super().__init__(name="aether-mlx", daemon=True)
        self.config = config
        self.registry = registry
        self.events = events
        self.stats = stats
        self.loaded: LoadedModel | None = None
        self.device_info: dict[str, Any] = {}
        self._commands: queue.Queue[tuple[str, Any, Future[Any]]] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._current: Job | None = None
        self.keep_awake = KeepAwake()
        self._ready = threading.Event()
        self._prompt_cache: Any = None
        self._default_cache_limit: int | None = None

    # ---- public API (any thread) -------------------------------------------------------

    def wait_ready(self, timeout: float = 30) -> None:
        if not self._ready.wait(timeout):
            raise RuntimeError("MLX worker did not start")

    def submit(self, kind: str, payload: Any = None) -> Future[Any]:
        fut: Future[Any] = Future()
        if kind == "generate":
            self._jobs[payload.id] = payload
            self._publish_queue_depth()
        self._commands.put((kind, payload, fut))
        return fut

    def cancel(self, job_id: str, reason: str = "cancelled") -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.cancel(reason)
        return True

    def relieve_memory(self, fraction: float) -> int:
        """Ask Drift to give back part of its pinned weights; returns bytes requested."""
        loaded = self.loaded
        if loaded is None or loaded.mode != "drift-dense":
            return 0  # expert slots are fixed-size tensors; shrinking them is not supported yet
        want = int(loaded.drift.pinned_bytes() * fraction)
        if want and loaded.drift.request_shrink(want):
            return want
        return 0

    def cancel_current(self, reason: str) -> None:
        if self._current is not None:
            self._current.cancel(reason)

    @property
    def queue_depth(self) -> int:
        return max(0, len(self._jobs) - (1 if self._current else 0))

    # ---- thread ------------------------------------------------------------------------

    def run(self) -> None:
        self.device_info = dict(mx.device_info())
        self._ready.set()
        while True:
            try:
                kind, payload, fut = self._commands.get(timeout=IDLE_TICK_S)
            except queue.Empty:
                self._publish_memory()
                continue
            try:
                if kind == "generate":
                    with self.keep_awake:
                        self._generate(payload)
                    fut.set_result(None)
                elif kind == "load":
                    fut.set_result(self._load(payload))
                elif kind == "unload":
                    self._unload()
                    fut.set_result(None)
                elif kind == "call":  # run an arbitrary function on the MLX thread
                    fut.set_result(payload())
                else:
                    raise ValueError(kind)
            except Exception as exc:
                if not isinstance(exc, EngineError):
                    log.exception("worker command %s failed", kind)
                if kind == "load":  # every caller (API, CLI preload) sees load failures
                    err = (
                        exc.to_dict()
                        if isinstance(exc, EngineError)
                        else {"code": "internal", "message": str(exc), "hint": None}
                    )
                    self.events.publish("error", source="load", model=payload, **err)
                fut.set_exception(exc)
            finally:
                self._publish_memory()

    # ---- commands (MLX thread only) ----------------------------------------------------

    def _publish_memory(self) -> None:
        self.stats.update(
            "memory",
            mlx_active_bytes=mx.get_active_memory(),
            mlx_cache_bytes=mx.get_cache_memory(),
            mlx_peak_bytes=mx.get_peak_memory(),
        )

    def _publish_queue_depth(self) -> None:
        self.stats.update("generation", queue_depth=self.queue_depth)

    def _unload(self) -> None:
        if self.loaded is None:
            return
        model_id = self.loaded.info.id
        if self.loaded.drift is not None:
            self.loaded.drift.close()
            self.loaded.backend.close()
            if self._default_cache_limit is not None:
                mx.set_cache_limit(self._default_cache_limit)
                self._default_cache_limit = None
        self.loaded = None
        self._prompt_cache = None
        import gc

        gc.collect()
        mx.clear_cache()
        self.stats.update("model", id=None, state="none", mode=None, progress=None)
        self.stats.update("memory", weights_bytes=0, kv_bytes=0, budget_bytes=None)
        self.stats.update("generation", context_max=None, context_tokens=0)
        self.stats.update(
            "drift",
            layers_total=None,
            current_layer=None,
            layer_residency=[],
            hit_rate=None,
            pinned_layers=None,
            stall_seconds=None,
        )
        self.events.publish("model_unloaded", model=model_id)

    def _load(self, model_id: str) -> dict[str, Any]:
        from mlx_lm.models.cache import LRUPromptCache

        info = self.registry.resolve(model_id)
        if info is None:
            raise EngineError("model_not_found", f"Model '{model_id}' niet gevonden.", status=404)
        if self.loaded is not None and self.loaded.info.path == info.path:
            return self._model_status()

        shape = self.registry.shape(info)
        ram = int(self.device_info.get("memory_size", 0))
        working_set = int(self.device_info.get("max_recommended_working_set_size", ram))
        settings = self.config.settings
        budget = plan(shape, ram, working_set, settings)
        fit = estimate_fit(shape, budget, None)
        if fit.mode not in SUPPORTED_MODES:
            gb = 1024**3
            hint = "Kies een kleiner model of maak geheugen vrij."
            raise EngineError(
                "model_too_large",
                f"Dit model heeft {shape.total_bytes / gb:.0f} GB nodig; "
                f"er is {budget.weights_bytes / gb:.0f} GB beschikbaar voor gewichten.",
                hint=hint,
                status=409,
            )

        # Never two models in memory at once.
        self._unload()
        # Drift fills its weights budget (pinned layers or expert slots); resident loads it all.
        need = shape.total_bytes if fit.mode == "resident" else budget.weights_bytes
        self._ensure_memory(info.id, min(need, shape.total_bytes), budget)
        self.stats.update("model", id=info.id, state="loading", mode=fit.mode, progress=0.0)
        self.events.publish("model_progress", model=info.id, progress=0.0, bytes=0)
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        try:
            if fit.mode == "drift-dense":
                model, tokenizer, drift, backend, total = self._load_drift_dense(info, budget)
            elif fit.mode == "drift-moe":
                model, tokenizer, drift, backend, total = self._load_drift_moe(info, budget)
            else:
                model, tokenizer, total = self._load_resident(info)
                drift = backend = None
        except Exception:
            # The previous model is already unloaded: report the failure, never hang in
            # "loading" (the app would show "Model laden…" forever).
            self.stats.update("model", state="error", progress=None)
            import gc

            gc.collect()
            mx.clear_cache()
            raise
        elapsed = time.perf_counter() - t0

        self._prompt_cache = LRUPromptCache(
            max_size=max(1, settings.prompt_cache_entries), max_bytes=max(1, budget.kv_bytes)
        )
        self.loaded = LoadedModel(info, model, tokenizer, budget, fit.mode, drift, backend)
        self.stats.update("model", state="ready", progress=1.0)
        self.stats.update(
            "memory", budget_bytes=budget.usable_bytes, weights_bytes=total, kv_bytes=0
        )
        self.stats.update("generation", context_max=budget.context_tokens, context_tokens=0)
        self.events.publish(
            "model_loaded",
            model=info.id,
            mode=fit.mode,
            bytes=total,
            ms=int(elapsed * 1000),
            read_bytes_per_s=int(total / elapsed) if elapsed else None,
        )
        return self._model_status()

    def _ensure_memory(self, model_id: str, need: int, budget: Budget) -> None:
        """Refuse to allocate what macOS cannot give without swapping (never freeze)."""
        from ..memory.guard import wait_for_memory

        if wait_for_memory(need, timeout_s=MEMORY_WAIT_S):
            return
        import psutil

        gb = 1024**3
        free = psutil.virtual_memory().available
        raise EngineError(
            "not_enough_memory",
            f"Er is nu {free / gb:.0f} GB vrij; dit model heeft {need / gb:.0f} GB nodig.",
            hint="Sluit andere apps of verlaag het geheugen voor modelgewichten.",
            status=409,
        )

    def _progress(self, model_id: str, fraction: float, done_bytes: int) -> None:
        self.stats.update("model", progress=fraction)
        self.events.publish(
            "model_progress", model=model_id, progress=round(fraction, 4), bytes=done_bytes
        )

    def _load_resident(self, info: LocalModel) -> tuple[Any, Any, int]:
        from mlx_lm import load as mlx_load

        loaded = mlx_load(info.path, lazy=True)
        model, tokenizer = loaded[0], loaded[1]
        # Materialise weights in batches so we can report real progress.
        params = flat_params(model)
        total = sum(v.nbytes for _, v in params) or 1
        done = 0
        last_report = 0.0
        batch: list[mx.array] = []
        batch_bytes = 0
        for _, v in params:
            batch.append(v)
            batch_bytes += v.nbytes
            if batch_bytes >= 256 * 1024**2:
                mx.eval(batch)
                done += batch_bytes
                batch, batch_bytes = [], 0
                now = time.perf_counter()
                if now - last_report > 0.1:
                    last_report = now
                    self._progress(info.id, done / total, done)
        mx.eval(batch)
        return model, tokenizer, total

    def _load_drift_dense(self, info: LocalModel, budget: Budget) -> tuple[Any, ...]:
        from ..drift.io_backend import PreadBackend
        from ..drift.model import build_drift_model
        from ..drift.residency import DenseResidency
        from ..drift.units import choose_pinned, plan_dense

        index = self.registry.index(info)
        dense = plan_dense(index)
        slots = PREFETCH_DEPTH + 1
        pinned = choose_pinned(dense, budget.weights_bytes, slots)
        if pinned < 0:
            gb = 1024**3
            need = dense.core_bytes + slots * dense.max_unit_bytes
            raise EngineError(
                "model_too_large",
                f"Zelfs in Drift-modus heeft dit model minimaal {need / gb:.1f} GB nodig; "
                f"er is {budget.weights_bytes / gb:.1f} GB beschikbaar.",
                hint="Kies een kleiner model of maak geheugen vrij.",
                status=409,
            )
        self._default_cache_limit = mx.set_cache_limit(DRIFT_CACHE_BASE + 2 * dense.max_unit_bytes)
        backend = PreadBackend()
        backend.on_read = self.stats.record_read
        drift = DenseResidency(
            dense, pinned, slots, PREFETCH_DEPTH, backend, self.events, self.stats
        )
        try:
            model = build_drift_model(info.path, drift)
            pinned_total = sum(u.buffer_bytes for u in dense.units[:pinned]) or 1
            drift.load_pinned(lambda f: self._progress(info.id, f, int(f * pinned_total)))
            tokenizer = _tokenizer(info.path)
        except Exception:
            drift.close()
            backend.close()
            mx.set_cache_limit(self._default_cache_limit)
            self._default_cache_limit = None
            raise
        self.events.publish(
            "drift_plan",
            model=info.id,
            layers=len(dense.units),
            pinned_layers=pinned,
            slots=slots,
            unit_bytes=dense.max_unit_bytes,
            resident_bytes=drift.resident_bytes,
        )
        return model, tokenizer, drift, backend, drift.resident_bytes

    def _load_drift_moe(self, info: LocalModel, budget: Budget) -> tuple[Any, ...]:
        from ..drift.experts import ExpertCache, choose_slots, plan_moe
        from ..drift.io_backend import PreadBackend
        from ..drift.model import build_moe_model

        plan = plan_moe(self.registry.index(info))
        # Prefill: one layer's full expert buffer + activations.
        transient = plan.prefill_bytes + 512 * 1024**2
        slots = choose_slots(plan, budget.weights_bytes, transient)
        if slots < 0:
            gb = 1024**3
            need = plan.resident_bytes + plan.slot_bytes(plan.experts_per_token) + transient
            raise EngineError(
                "model_too_large",
                f"Zelfs in Drift-modus heeft dit model minimaal {need / gb:.1f} GB nodig; "
                f"er is {budget.weights_bytes / gb:.1f} GB beschikbaar.",
                hint="Kies een kleiner model of maak geheugen vrij.",
                status=409,
            )
        self._default_cache_limit = mx.set_cache_limit(DRIFT_CACHE_BASE + 512 * 1024**2)
        backend = PreadBackend()
        backend.on_read = self.stats.record_read
        cache = ExpertCache(plan, slots, backend, self.events, self.stats)
        cache.prefetch_enabled = lambda: self.config.settings.prefetch_experts
        try:
            model = build_moe_model(info.path, cache, lambda f: self._progress(info.id, f, 0))
            tokenizer = _tokenizer(info.path)
        except Exception:
            cache.close()
            backend.close()
            mx.set_cache_limit(self._default_cache_limit)
            self._default_cache_limit = None
            raise
        resident = plan.resident_bytes + cache.resident_bytes + plan.prefill_bytes
        self.events.publish(
            "drift_plan",
            model=info.id,
            layers=len(cache.order),
            slots=slots,
            experts=plan.n_experts,
            resident_bytes=resident,
        )
        return model, tokenizer, cache, backend, resident

    def _model_status(self) -> dict[str, Any]:
        if self.loaded is None:
            return {"id": None, "state": "none"}
        b = self.loaded.budget
        return {
            "id": self.loaded.info.id,
            "state": "ready",
            "mode": self.loaded.mode,
            "context_max": b.context_tokens,
            "kv_bits": b.kv_bits,
        }

    def _tokenize(self, req: GenerateRequest) -> list[int]:
        assert self.loaded is not None
        tok = self.loaded.tokenizer
        if req.messages is not None:
            if getattr(tok, "chat_template", None):
                return list(
                    tok.apply_chat_template(req.messages, add_generation_prompt=True, tokenize=True)
                )
            text = "\n".join(f"{m['role']}: {m.get('content') or ''}" for m in req.messages)
            return list(tok.encode(text + "\nassistant:"))
        return list(tok.encode(req.prompt or ""))

    def _generate(self, job: Job) -> None:
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        self._current = job
        self._publish_queue_depth()
        try:
            if job.cancelled.is_set():
                job.emit(
                    {"type": "done", "finish_reason": job.cancel_reason, "usage": _usage(0, 0, 0)}
                )
                return
            if self.loaded is None:
                raise EngineError(
                    "model_not_loaded",
                    "Er is nog geen model geladen.",
                    hint="Kies een model.",
                    status=409,
                )
            lm = self.loaded
            prompt = self._tokenize(job.request)
            ctx_max = lm.budget.context_tokens
            if len(prompt) >= ctx_max:
                raise EngineError(
                    "context_too_long",
                    f"Het gesprek is {len(prompt)} tokens; het maximum in deze modus is {ctx_max}.",
                    hint="Start een nieuw gesprek of kies modus Lange context.",
                    status=400,
                )
            max_tokens = min(job.request.max_tokens, ctx_max - len(prompt))

            use_cache = self.config.settings.prompt_cache_entries > 0
            cache, rest = (
                self._prompt_cache.fetch_nearest_cache(lm.info.path, prompt)
                if use_cache
                else (None, prompt)
            )
            # Reusing a tiny prefix saves nothing but splits the prefill, which changes
            # rounding (and so greedy output) compared to a single pass.
            if cache is None or len(prompt) - len(rest) < MIN_PROMPT_REUSE:
                cache, rest = make_prompt_cache(lm.model), prompt
            if not rest:  # exact hit: re-run the last token to get fresh logits
                if can_trim_prompt_cache(cache):
                    trim_prompt_cache(cache, 1)
                    rest = prompt[-1:]
                else:
                    cache, rest = make_prompt_cache(lm.model), prompt
            cached = len(prompt) - len(rest)
            # The working cache is a copy; stored entries + this request together must fit
            # the KV budget, so make room before generating.
            per_token = lm.budget.kv_bytes // max(1, lm.budget.context_tokens)
            working = per_token * (len(prompt) + max_tokens)
            self._prompt_cache.trim_to(n_bytes=max(0, lm.budget.kv_bytes - working))

            if job.request.seed is not None:
                mx.random.seed(job.request.seed)
            sampler = make_sampler(temp=job.request.temperature, top_p=job.request.top_p)

            self.stats.update("generation", active=True, ttft_ms=None, context_tokens=len(prompt))
            self.events.publish(
                "generation_started", id=job.id, prompt_tokens=len(prompt), cached_tokens=cached
            )
            key = list(prompt)
            # The template may open <think> itself (GLM); clients then still see a whole block.
            if _opens_thinking(lm.tokenizer, prompt):
                job.emit({"type": "token", "text": lm.tokenizer.think_start})
            text = ""
            emitted = 0
            n = 0
            finish = "length"
            first_token_at: float | None = None
            last = None
            for resp in stream_generate(
                lm.model,
                lm.tokenizer,
                rest,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=cache,
                prefill_step_size=self.config.settings.prefill_step_size,
                kv_bits=lm.budget.kv_bits,
            ):
                last = resp
                n += 1
                key.append(resp.token)
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    self.stats.update(
                        "generation",
                        ttft_ms=round((first_token_at - job.started) * 1000),
                        prefill_tps=round(resp.prompt_tps, 1),
                    )
                text += resp.text
                # Hold back text that could be the start of a stop sequence.
                stop_at = _find_stop(text, job.request.stop)
                if stop_at is not None:
                    if stop_at > emitted:
                        job.emit({"type": "token", "text": text[emitted:stop_at]})
                    finish = "stop"
                    break
                safe = len(text) - _stop_holdback(text, job.request.stop)
                if safe > emitted:
                    job.emit({"type": "token", "text": text[emitted:safe]})
                    emitted = safe
                if n % 8 == 0:
                    self.stats.update(
                        "generation",
                        decode_tps=round(resp.generation_tps, 2),
                        context_tokens=len(key),
                    )
                if job.cancelled.is_set():
                    finish = job.cancel_reason
                    break
                if resp.finish_reason is not None:
                    finish = resp.finish_reason
                    break
            if finish != "stop" and len(text) > emitted:
                job.emit({"type": "token", "text": text[emitted:]})

            if use_cache:
                self._prompt_cache.insert_cache(lm.info.path, key, cache)
            kv_now = sum(c.nbytes for c in cache)
            self.stats.update("memory", kv_bytes=kv_now)
            decode_tps = round(last.generation_tps, 2) if last else None
            self.stats.update("generation", decode_tps=decode_tps, context_tokens=len(key))
            usage = _usage(len(prompt), n, cached)
            timings = {
                "ttft_ms": round((first_token_at - job.started) * 1000) if first_token_at else None,
                "prefill_tps": round(last.prompt_tps, 1) if last else None,
                "decode_tps": decode_tps,
            }
            self.events.publish(
                "generation_done", id=job.id, finish_reason=finish, **usage, **timings
            )
            job.emit({"type": "done", "finish_reason": finish, "usage": usage, "timings": timings})
        except EngineError as exc:
            job.emit({"type": "error", **exc.to_dict(), "status": exc.status})
        except Exception as exc:
            log.exception("generation failed")
            job.emit({"type": "error", "code": "internal", "message": str(exc), "status": 500})
        finally:
            self._current = None
            self._jobs.pop(job.id, None)
            self.stats.update("generation", active=False)
            self._publish_queue_depth()


def _tokenizer(path: str) -> Any:
    """Tokenizer with the model's own stop tokens, exactly as mlx_lm.load sets them up.

    Models such as GLM list several end-of-turn tokens in config.json; without them a reply
    runs on into an invented next user turn.
    """
    import json

    from mlx_lm.utils import load_tokenizer

    cfg = json.loads((Path(path) / "config.json").read_text())
    eos = cfg.get("eos_token_id", cfg.get("text_config", {}).get("eos_token_id"))
    return load_tokenizer(Path(path), eos_token_ids=eos)


def _opens_thinking(tokenizer: Any, prompt: list[int]) -> bool:
    """True when the chat template already opened a reasoning block in the prompt."""
    if not getattr(tokenizer, "has_thinking", False):
        return False
    start = tokenizer.rfind_think_start(prompt)
    return start >= 0 and tokenizer.rfind_think_end(prompt) < start


def _usage(prompt: int, completion: int, cached: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cached_tokens": cached,
    }


def _find_stop(text: str, stops: list[str]) -> int | None:
    hits = [i for s in stops if s and (i := text.find(s)) >= 0]
    return min(hits) if hits else None


def _stop_holdback(text: str, stops: list[str]) -> int:
    """Length of the longest suffix of text that is a proper prefix of a stop sequence."""
    best = 0
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:k]):
                best = max(best, k)
                break
    return best
