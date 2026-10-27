"""Integration tests against the real engine process with a small model (M1 acceptance)."""

from __future__ import annotations

import json
import threading
import time

import httpx
from conftest import SMALL_MODEL, RunningEngine, requires_model

pytestmark = requires_model


def chat(eng: RunningEngine, messages: list[dict], **kw: object) -> dict:
    body = {"messages": messages, "temperature": 0, "max_tokens": 24, **kw}
    r = httpx.post(f"{eng.url}/v1/chat/completions", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def test_handshake(engine_proc: RunningEngine) -> None:
    assert engine_proc.ready["version"] and engine_proc.ready["pid"] == engine_proc.proc.pid


def test_greedy_output_matches_plain_mlx_lm(engine_proc: RunningEngine) -> None:
    from mlx_lm import generate, load

    messages = [{"role": "user", "content": "Noem drie kleuren."}]
    ours = chat(engine_proc, messages)["choices"][0]["message"]["content"]
    model, tok = load(SMALL_MODEL)
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    ref = generate(model, tok, prompt, max_tokens=24)
    assert ours == ref


def test_follow_up_only_prefills_new_tokens(engine_proc: RunningEngine) -> None:
    first = [{"role": "user", "content": "Wat is de hoofdstad van Spanje?"}]
    a = chat(engine_proc, first)
    reply = a["choices"][0]["message"]["content"]
    follow = [
        *first,
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "En van Portugal?"},
    ]
    b = chat(engine_proc, follow)
    usage = b["usage"]
    new_tokens = usage["prompt_tokens"] - usage["prompt_tokens_details"]["cached_tokens"]
    # Everything up to the previous answer is reused; only the new turn is prefilled.
    assert usage["prompt_tokens_details"]["cached_tokens"] >= a["usage"]["prompt_tokens"]
    assert new_tokens < 20


def test_stop_sequence(engine_proc: RunningEngine) -> None:
    msgs = [{"role": "user", "content": "Tel van 1 tot 10, gescheiden door komma's."}]
    out = chat(engine_proc, msgs, stop=["5"], max_tokens=60)
    choice = out["choices"][0]
    assert choice["finish_reason"] == "stop" and "5" not in choice["message"]["content"]


def test_cancel_endpoint_stops_quickly(engine_proc: RunningEngine) -> None:
    # Deterministic and long: counting to 2000 cannot end within the test window.
    body = {
        "messages": [
            {"role": "user", "content": "Tel van 1 tot 2000, elk getal op een eigen regel."}
        ],
        "stream": True,
        "max_tokens": 4000,
        "temperature": 0,
    }
    finish: dict[str, object] = {}

    def consume() -> None:
        with httpx.stream(
            "POST", f"{engine_proc.url}/v1/chat/completions", json=body, timeout=60
        ) as r:
            for line in r.iter_lines():
                if not line.startswith("data: {"):
                    continue
                c = json.loads(line[6:])
                if "aether" in c and "generation_id" in c["aether"]:
                    finish["id"] = c["aether"]["generation_id"]
                if c["choices"] and c["choices"][0]["finish_reason"]:
                    finish["reason"] = c["choices"][0]["finish_reason"]
                    finish["at"] = time.perf_counter()

    t = threading.Thread(target=consume)
    t.start()
    while "id" not in finish:
        time.sleep(0.01)
    time.sleep(0.2)
    sent = time.perf_counter()
    httpx.post(f"{engine_proc.url}/aether/v1/generations/{finish['id']}/cancel").raise_for_status()
    t.join(10)
    assert finish["reason"] == "cancelled"
    assert float(finish["at"]) - sent < 0.25  # §9: stop < 250 ms in resident mode


def test_official_openai_client(engine_proc: RunningEngine) -> None:
    from openai import OpenAI

    oc = OpenAI(base_url=f"{engine_proc.url}/v1", api_key="unused-on-loopback")
    assert any(m.id == "default" for m in oc.models.list().data)
    r = oc.chat.completions.create(
        model="default",
        max_tokens=8,
        temperature=0,
        messages=[{"role": "user", "content": "Zeg hallo."}],
    )
    assert r.choices[0].message.content
    parts = [
        c.choices[0].delta.content or ""
        for c in oc.chat.completions.create(
            model="default",
            max_tokens=8,
            temperature=0,
            stream=True,
            messages=[{"role": "user", "content": "Zeg hallo."}],
        )
        if c.choices
    ]
    assert "".join(parts) == r.choices[0].message.content


def test_stats_and_events_streams(engine_proc: RunningEngine) -> None:
    with httpx.stream("GET", f"{engine_proc.url}/aether/v1/stats/stream", timeout=5) as r:
        snap = json.loads(next(line for line in r.iter_lines() if line.startswith("data: "))[6:])
    assert snap["model"]["state"] == "ready" and snap["memory"]["weights_bytes"] > 0
    assert snap["drift"]["write_bytes_per_s"] == 0
    with httpx.stream("GET", f"{engine_proc.url}/aether/v1/events/stream?since=0", timeout=5) as r:
        ev = json.loads(next(line for line in r.iter_lines() if line.startswith("data: "))[6:])
    assert ev["seq"] >= 1 and "type" in ev


def test_models_endpoint_has_fit(engine_proc: RunningEngine) -> None:
    body = httpx.get(f"{engine_proc.url}/aether/v1/models", timeout=30).json()
    small = next(m for m in body["local"] if m["id"] == SMALL_MODEL)
    assert small["loaded"] and small["fit"]["mode"] == "resident"
    assert small["fit"]["loadable"]
    glm = next(c for c in body["catalog"] if "GLM-5.3" in c["repo"])
    assert glm["size_bytes"] > 400e9
    assert glm["fit"]["mode"] == "drift-moe" and glm["fit"]["loadable"]


def test_context_too_long(engine_proc: RunningEngine) -> None:
    httpx.put(f"{engine_proc.url}/aether/v1/config", json={"mode": "snel"}).raise_for_status()
    try:
        # Context max only changes on reload; the loaded model keeps 32k. Use a huge prompt.
        long = "woord " * 40000
        r = httpx.post(
            f"{engine_proc.url}/v1/chat/completions",
            timeout=60,
            json={"messages": [{"role": "user", "content": long}], "max_tokens": 1},
        )
        assert r.status_code == 400 and r.json()["error"]["code"] == "context_too_long"
    finally:
        httpx.put(f"{engine_proc.url}/aether/v1/config", json={"mode": "balans"})


def test_failed_load_reports_error_and_keeps_engine_usable(tmp_path) -> None:
    """Regression: a failing preload left model_state at "loading" forever."""
    import os
    import subprocess
    import sys

    bad = tmp_path / "broken-model"
    bad.mkdir()
    (bad / "config.json").write_text('{"model_type": "does_not_exist", "quantization": {}}')
    from conftest import write_safetensors

    write_safetensors(bad / "model.safetensors", {"model.layers.0.w.weight": ("F16", [4, 4])})
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aether",
            "--port",
            "0",
            "--data-dir",
            str(tmp_path / "d"),
            "--log-dir",
            str(tmp_path / "d" / "logs"),
            "--model",
            str(bad),
            "--parent-pid",
            str(os.getpid()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert proc.stdout
        url = f"http://127.0.0.1:{json.loads(proc.stdout.readline())['port']}"
        events = []
        with httpx.stream("GET", f"{url}/aether/v1/events/stream?since=0", timeout=30) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
                    if events[-1]["type"] == "error":
                        break
        assert events[-1]["source"] == "load" and events[-1]["message"]
        # Never stuck in "loading": either no model ("none") or a visible error.
        assert httpx.get(f"{url}/health").json()["model_state"] in ("none", "error")
    finally:
        proc.terminate()
        proc.wait(timeout=20)
