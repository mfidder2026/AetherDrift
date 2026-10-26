"""M1 benchmark: decode speed through the engine (HTTP + SSE) vs plain mlx-lm.

Usage:  uv run --project engine python bench/generate.py --model REPO [--tokens 256] [--runs 3]

Starts the engine as a subprocess (like the app does), measures client-side decode tok/s
(first to last streamed token) and compares with mlx_lm.stream_generate in-process, run
after the engine has exited so both get the GPU to themselves.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPORTS = Path(__file__).resolve().parent / "reports"
PROMPT = "Schrijf een uitgebreid essay over de geschiedenis van de Nederlandse waterbouw."


def engine_runs(model: str, tokens: int, runs: int) -> dict:
    data = tempfile.mkdtemp(prefix="aether-bench-")
    proc = subprocess.Popen(
        [sys.executable, "-m", "aether", "--port", "0", "--data-dir", data, "--model", model],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert proc.stdout
        t_start = time.perf_counter()
        ready = json.loads(proc.stdout.readline())
        ready_s = time.perf_counter() - t_start
        url = f"http://127.0.0.1:{ready['port']}"
        while httpx.get(f"{url}/health").json()["model_state"] != "ready":
            time.sleep(0.05)
        load_s = time.perf_counter() - t_start
        results = []
        for i in range(runs + 1):  # first run warms up kernels
            body = {
                "messages": [{"role": "user", "content": f"{PROMPT} (#{i})"}],
                "stream": True,
                "temperature": 0,
                "max_tokens": tokens,
                "stream_options": {"include_usage": True},
            }
            sent = time.perf_counter()
            first = last = None
            n = 0
            usage = timings = None
            with httpx.stream("POST", f"{url}/v1/chat/completions", json=body, timeout=600) as r:
                for line in r.iter_lines():
                    if not line.startswith("data: {"):
                        continue
                    c = json.loads(line[6:])
                    if c.get("usage"):
                        usage = c["usage"]
                    if c["choices"] and c["choices"][0]["delta"].get("content"):
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        n += 1
                    if c.get("aether", {}).get("timings"):
                        timings = c["aether"]["timings"]
            if i == 0:
                continue
            tokens_out = usage["completion_tokens"]
            results.append(
                {
                    "client_ttft_ms": (first - sent) * 1000,
                    "client_decode_tps": (tokens_out - 1) / (last - first),
                    "engine_decode_tps": timings["decode_tps"],
                    "tokens": tokens_out,
                }
            )
        return {"ready_s": ready_s, "load_s": load_s, "runs": results}
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def baseline_runs(model: str, tokens: int, runs: int) -> list[dict]:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    m, tok = load(model)
    out = []
    for i in range(runs + 1):
        msgs = [{"role": "user", "content": f"{PROMPT} (#{i})"}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        last = None
        for resp in stream_generate(m, tok, prompt, max_tokens=tokens, sampler=make_sampler(0.0)):
            last = resp
        if i:
            out.append({"decode_tps": last.generation_tps, "tokens": last.generation_tokens})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    eng = engine_runs(a.model, a.tokens, a.runs)
    base = baseline_runs(a.model, a.tokens, a.runs)
    e_tps = statistics.median(r["client_decode_tps"] for r in eng["runs"])
    b_tps = statistics.median(r["decode_tps"] for r in base)
    overhead = 1 - e_tps / b_tps
    ttft = statistics.median(r["client_ttft_ms"] for r in eng["runs"])

    REPORTS.mkdir(exist_ok=True)
    stamp = dt.date.today().isoformat()
    name = a.model.split("/")[-1]
    result = {"model": a.model, "engine": eng, "baseline": base, "overhead": overhead}
    (REPORTS / f"{stamp}-M1-generate-{name}.json").write_text(json.dumps(result, indent=2))
    verdict = "OK" if overhead <= 0.05 else "BOVEN DOEL"
    md = f"""# Generatie-benchmark: {a.model} ({stamp})

| Meting | Waarde |
|---|---|
| Engine start → `ready` | {eng["ready_s"]:.2f} s |
| Engine start → model geladen | {eng["load_s"]:.2f} s |
| TTFT (client, mediaan) | {ttft:.0f} ms |
| Decode via engine (client-side, HTTP+SSE, mediaan) | {e_tps:.1f} tok/s |
| Decode kale mlx-lm (mediaan) | {b_tps:.1f} tok/s |
| **Overhead** | **{overhead * 100:.1f} %** (doel ≤ 5 %: {verdict}) |

{a.runs} runs van {a.tokens} tokens, temperature 0, na één opwarm-run.
"""
    (REPORTS / f"{stamp}-M1-generate-{name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
