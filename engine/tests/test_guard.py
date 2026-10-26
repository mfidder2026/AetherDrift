"""Memory guard decisions with simulated pressure (real pressure cannot be produced safely)."""

from __future__ import annotations

import pytest

from aether.memory import guard as guard_mod
from aether.memory.guard import MemoryGuard
from aether.telemetry.events import EventBus


class FakeWorker:
    def __init__(self) -> None:
        self.loaded = object()
        self.relieved: list[float] = []
        self.cancelled: list[str] = []

    def relieve_memory(self, fraction: float) -> int:
        self.relieved.append(fraction)
        return 2 * 1024**3

    def cancel_current(self, reason: str) -> None:
        self.cancelled.append(reason)


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch):
    level = {"value": "normal"}
    clock = {"t": 1000.0}
    monkeypatch.setattr(guard_mod, "memory_pressure", lambda: level["value"])
    monkeypatch.setattr(guard_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(MemoryGuard, "_swap_growing", lambda self: False)
    worker, events = FakeWorker(), EventBus()
    return MemoryGuard(worker, events), worker, events, level, clock  # type: ignore[arg-type]


def test_warning_shrinks_drift_with_cooldown(setup) -> None:
    guard, worker, events, level, clock = setup
    level["value"] = "warning"
    guard.check()
    guard.check()  # within cooldown: no second shrink
    assert worker.relieved == [0.10]
    clock["t"] += 5
    guard.check()
    assert worker.relieved == [0.10, 0.10]
    assert worker.cancelled == []
    actions = [e["action"] for e in events.since(0) if e["type"] == "pressure"]
    assert actions and "Drift-cache verkleind" in actions[0]


def test_critical_stops_generation_once(setup) -> None:
    guard, worker, events, level, clock = setup
    level["value"] = "critical"
    guard.check()
    clock["t"] += 5
    guard.check()
    assert worker.cancelled == ["memory"]


def test_normal_does_nothing(setup) -> None:
    guard, worker, events, level, _ = setup
    guard.check()
    assert worker.relieved == [] and events.since(0) == []


def test_wait_for_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    available = iter([1, 2, 10])
    monkeypatch.setattr(
        guard_mod.psutil, "virtual_memory", lambda: SimpleNamespace(available=next(available))
    )
    assert guard_mod.wait_for_memory(5, timeout_s=5, poll_s=0)
    monkeypatch.setattr(guard_mod.psutil, "virtual_memory", lambda: SimpleNamespace(available=1))
    assert not guard_mod.wait_for_memory(5, timeout_s=0.01, poll_s=0)
