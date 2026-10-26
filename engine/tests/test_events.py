from __future__ import annotations

import asyncio
import threading

from aether.runtime.worker import _find_stop, _stop_holdback
from aether.telemetry.events import EventBus, throttle


def test_ring_buffer_and_since() -> None:
    bus = EventBus(ring_size=3)
    for i in range(5):
        bus.publish("x", i=i)
    assert [e["i"] for e in bus.since(0)] == [2, 3, 4]
    assert [e["i"] for e in bus.since(4)] == [4]


def test_throttle_folds_load_evict_storm_into_summary() -> None:
    batch = [{"seq": i, "ts": 0, "type": "load", "bytes": 10} for i in range(100)]
    batch.append({"seq": 100, "ts": 0, "type": "pressure", "level": "warning"})
    out = throttle(batch, 0.25)
    types = [e["type"] for e in out]
    assert types.count("summary") == 1 and "pressure" in types and "load" not in types
    summary = next(e for e in out if e["type"] == "summary")
    assert summary["loads"] == 100 and summary["bytes"] == 1000


def test_throttle_keeps_small_batches() -> None:
    batch = [{"seq": 1, "ts": 0, "type": "load"}, {"seq": 2, "ts": 0, "type": "evict"}]
    assert throttle(batch, 0.25) == batch


def test_stream_receives_events_from_other_threads() -> None:
    bus = EventBus()

    async def main() -> list[dict]:
        stream = bus.stream(tick=0.01)
        task = asyncio.ensure_future(stream.__anext__())
        await asyncio.sleep(0.05)
        threading.Thread(target=lambda: bus.publish("model_loaded", model="m")).start()
        batch = await asyncio.wait_for(task, 2)
        await stream.aclose()
        return batch

    batch = asyncio.run(main())
    assert batch[0]["type"] == "model_loaded" and batch[0]["seq"] == 1


def test_stop_sequences() -> None:
    assert _find_stop("hello END world", ["END"]) == 6
    assert _find_stop("hello", ["END"]) is None
    assert _stop_holdback("hello EN", ["END"]) == 2
    assert _stop_holdback("hello", ["END"]) == 0
