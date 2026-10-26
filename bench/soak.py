"""M4 soak test: continuous generation for a long time; the Mac must stay responsive and
memory must not grow.

Usage:
  uv run --project engine python bench/soak.py --model REPO --minutes 60 [--weights-budget 8GB]

Every second it samples the engine's memory footprint, swap, macOS memory pressure and two
responsiveness probes: how long starting a trivial process takes, and how late a 20 ms sleep
wakes up. A Mac that beachballs shows up as large spikes in both.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import psutil
from drift import phys_footprint  # bench/drift.py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from aether.system import memory_pressure  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"
PROMPTS = [
    "Schrijf een uitgebreid verhaal over een vuurtorenwachter.",
    "Leg stap voor stap uit hoe een compiler werkt.",
    "Vergelijk drie programmeertalen en geef voor- en nadelen.",
    "Beschrijf de geschiedenis van de Nederlandse waterbouw.",
    "Geef een gedetailleerd recept voor een driegangenmenu.",
]


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))] if s else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--weights-budget")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "aether",
        "--port",
        "0",
        "--data-dir",
        tempfile.mkdtemp(prefix="aether-soak-"),
        "--model",
        a.model,
    ]
    if a.weights_budget:
        cmd += ["--weights-budget", a.weights_budget]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    assert proc.stdout
    url = f"http://127.0.0.1:{json.loads(proc.stdout.readline())['port']}"
    t_load = time.time()
    while (state := httpx.get(f"{url}/health").json()["model_state"]) != "ready":
        if state == "error":
            raise SystemExit("model failed to load")
        time.sleep(1)
    load_s = time.time() - t_load
    mode = httpx.get(f"{url}/aether/v1/stats").json()["model"]["mode"]

    samples: list[dict] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            subprocess.run(["/usr/bin/true"], check=False)
            spawn_ms = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            time.sleep(0.02)
            sleep_late_ms = (time.perf_counter() - t1) * 1000 - 20
            samples.append(
                {
                    "t": time.time(),
                    "footprint": phys_footprint(proc.pid),
                    "swap": psutil.swap_memory().used,
                    "pressure": memory_pressure(),
                    "available": psutil.virtual_memory().available,
                    "spawn_ms": spawn_ms,
                    "sleep_late_ms": sleep_late_ms,
                }
            )
            time.sleep(max(0.0, 1.0 - (time.perf_counter() - t0)))

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    deadline = time.time() + a.minutes * 60
    generations, tokens, errors, tps = 0, 0, 0, []
    i = 0
    try:
        while time.time() < deadline:
            r = httpx.post(
                f"{url}/v1/chat/completions",
                timeout=1800,
                json={
                    "messages": [
                        {"role": "user", "content": f"{PROMPTS[i % len(PROMPTS)]} (#{i})"}
                    ],
                    "max_tokens": a.max_tokens,
                    "temperature": 0.7,
                },
            )
            i += 1
            if r.status_code != 200:
                errors += 1
                continue
            body = r.json()
            generations += 1
            tokens += body["usage"]["completion_tokens"]
            if body["aether"]["timings"]["decode_tps"]:
                tps.append(body["aether"]["timings"]["decode_tps"])
            elapsed = a.minutes * 60 - (deadline - time.time())
            print(
                f"\r{elapsed / 60:5.1f} min · {generations} antwoorden · {tokens} tokens",
                end="",
                flush=True,
            )
        final = httpx.get(f"{url}/aether/v1/stats").json()
    finally:
        stop.set()
        sampler.join(timeout=5)
        proc.terminate()
        proc.wait(timeout=60)
    print()

    gb = 1024**3
    n = max(1, len(samples) // 10)
    fp_first = statistics.mean(s["footprint"] for s in samples[:n])
    fp_last = statistics.mean(s["footprint"] for s in samples[-n:])
    swap_growth = samples[-1]["swap"] - samples[0]["swap"]
    pressure = {
        lvl: sum(1 for s in samples if s["pressure"] == lvl)
        for lvl in ("normal", "warning", "critical")
    }
    spawn = [s["spawn_ms"] for s in samples]
    late = [s["sleep_late_ms"] for s in samples]
    stamp = dt.date.today().isoformat()
    name = f"{stamp}-M4-soak-{a.model.split('/')[-1]}"
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{name}.json").write_text(
        json.dumps(
            {
                "model": a.model,
                "mode": mode,
                "minutes": a.minutes,
                "load_s": load_s,
                "generations": generations,
                "tokens": tokens,
                "errors": errors,
                "tps": tps,
                "hit_rate": final["drift"]["hit_rate"],
                "samples": samples,
            }
        )
    )
    md = f"""# Soak-test: {a.model} ({stamp})

{a.minutes:.0f} minuten continu genereren · modus {mode} · laden {load_s:.0f} s

| Meting | Waarde |
|---|---|
| Antwoorden / tokens / fouten | {generations} / {tokens} / {errors} |
| Decode (mediaan, min–max) | {statistics.median(tps) if tps else 0:.2f} tok/s \
({min(tps) if tps else 0:.2f}–{max(tps) if tps else 0:.2f}) |
| Cache-hitrate (eind) | {final["drift"]["hit_rate"]} |
| Engine-footprint: eerste 10 % → laatste 10 % | {fp_first / gb:.2f} → {fp_last / gb:.2f} GB \
(groei {(fp_last - fp_first) / gb:+.2f} GB) |
| Piek-footprint | {max(s["footprint"] for s in samples) / gb:.2f} GB |
| Swapgroei | {swap_growth / 1024**2:.0f} MB |
| Geheugendruk (samples normaal / krap / kritiek) | {pressure["normal"]} / \
{pressure["warning"]} / {pressure["critical"]} |
| Min. vrij geheugen systeem | {min(s["available"] for s in samples) / gb:.1f} GB |
| **Responsiviteit: proces starten** p50 / p99 / max | {pct(spawn, 0.5):.1f} / \
{pct(spawn, 0.99):.1f} / {max(spawn):.1f} ms |
| **Responsiviteit: 20 ms-sleep te laat** p50 / p99 / max | {pct(late, 0.5):.1f} / \
{pct(late, 0.99):.1f} / {max(late):.1f} ms |
"""
    (REPORTS / f"{name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
