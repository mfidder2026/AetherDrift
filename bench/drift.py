"""Drift benchmark (M3/M4): speed, SSD utilisation and memory discipline.

Usage:
  uv run --project engine python bench/drift.py --model REPO --weights-budget 2GB

Starts the engine with a weights cap so the model must stream, drops the model files from the
page cache before every run (so the SSD is measured, not RAM), and samples the engine's
physical memory footprint every 50 ms.

I/O efficiency = achieved decode tok/s ÷ the I/O bound (SSD bandwidth ÷ streamed bytes per
token). 100 % means compute is completely hidden behind reading.
"""

from __future__ import annotations

import argparse
import ctypes
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from aether.drift.index import build_index  # noqa: E402
from aether.drift.io_backend import invalidate_page_cache  # noqa: E402
from aether.drift.units import plan_dense  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"
PROMPT = "Schrijf een uitgebreid essay over de geschiedenis van de Nederlandse waterbouw."
SSD_REFERENCE = 6.77e9  # best sequential read, bench/reports/2026-10-26-M0-ssd.md

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def phys_footprint(pid: int) -> int:
    """Same number Activity Monitor shows as "Memory" (includes Metal allocations)."""
    buf = (ctypes.c_uint64 * 40)()
    if _libproc.proc_pid_rusage(pid, 2, ctypes.byref(buf)) != 0:  # RUSAGE_INFO_V2
        return 0
    return int(buf[2 + 7])  # skip 16-byte uuid; ri_phys_footprint is the 8th uint64


class Sampler(threading.Thread):
    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.samples: list[int] = []
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            self.samples.append(phys_footprint(self.pid))
            time.sleep(0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights-budget", required=True, help="e.g. 2GB")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-invalidate", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(a.model))
    files = [str(p.resolve()) for p in path.glob("*.safetensors")]

    data = tempfile.mkdtemp(prefix="aether-drift-")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aether",
            "--port",
            "0",
            "--data-dir",
            data,
            "--weights-budget",
            a.weights_budget,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    swap_before = psutil.swap_memory().used
    sampler = Sampler(proc.pid)
    try:
        assert proc.stdout
        url = f"http://127.0.0.1:{json.loads(proc.stdout.readline())['port']}"
        baseline = phys_footprint(proc.pid)
        sampler.start()
        if not a.no_invalidate:
            for f in files:
                invalidate_page_cache(f)
        httpx.post(
            f"{url}/aether/v1/models/load", json={"id": str(path), "wait": True}, timeout=600
        ).raise_for_status()
        stats = httpx.get(f"{url}/aether/v1/stats").json()
        mode = stats["model"]["mode"]
        if not mode.startswith("drift"):
            raise SystemExit(f"model loaded in {mode} mode; lower the budget")
        pinned = stats["drift"]["pinned_layers"]
        layers = stats["drift"]["layers_total"]
        if mode == "drift-dense":
            plan = plan_dense(build_index(path))
            streamed_bytes = sum(u.nbytes for u in plan.units[pinned:])
        else:
            streamed_bytes = 0  # MoE: depends on the hit rate; reported from measurements

        runs = []
        for i in range(a.runs):
            if not a.no_invalidate:
                for f in files:
                    invalidate_page_cache(f)
            read_samples: list[float] = []
            done = threading.Event()

            def poll(
                done: threading.Event = done, read_samples: list[float] = read_samples
            ) -> None:
                while not done.is_set():
                    s = httpx.get(f"{url}/aether/v1/stats").json()
                    if s["generation"]["active"]:
                        read_samples.append(s["drift"]["read_bytes_per_s"])
                    time.sleep(0.25)

            poller = threading.Thread(target=poll, daemon=True)
            poller.start()
            r = httpx.post(
                f"{url}/v1/chat/completions",
                timeout=600,
                json={
                    "messages": [{"role": "user", "content": f"{PROMPT} (#{i})"}],
                    "temperature": 0,
                    "max_tokens": a.tokens,
                },
            ).json()
            done.set()
            poller.join()
            t = r["aether"]["timings"]
            bound = SSD_REFERENCE / streamed_bytes if streamed_bytes else float("nan")
            runs.append(
                {
                    "decode_tps": t["decode_tps"],
                    "ttft_ms": t["ttft_ms"],
                    "io_bound_tps": bound,
                    "efficiency": t["decode_tps"] / bound,
                    "read_gbps_avg": statistics.mean(read_samples) / 1e9 if read_samples else None,
                }
            )
        final = httpx.get(f"{url}/aether/v1/stats").json()
    finally:
        sampler.stop.set()
        proc.terminate()
        proc.wait(timeout=20)
    swap_after = psutil.swap_memory().used

    budget_bytes = float(a.weights_budget.upper().removesuffix("B").removesuffix("G")) * 1024**3
    peak_fp = max(sampler.samples) if sampler.samples else 0
    mlx_peak = final["memory"]["mlx_peak_bytes"]
    kv = final["memory"]["kv_bytes"]
    eff = statistics.median(r["efficiency"] for r in runs) if streamed_bytes else None
    tps = statistics.median(r["decode_tps"] for r in runs)
    reads = [r["read_gbps_avg"] for r in runs if r["read_gbps_avg"]]
    gb = 1024**3
    result = {
        "model": a.model,
        "weights_budget": a.weights_budget,
        "layers": layers,
        "pinned": pinned,
        "streamed_bytes_per_token": streamed_bytes,
        "runs": runs,
        "hit_rate": final["drift"]["hit_rate"],
        "stall_seconds": final["drift"]["stall_seconds"],
        "mlx_peak_bytes": mlx_peak,
        "kv_bytes": kv,
        "baseline_footprint": baseline,
        "peak_footprint": peak_fp,
        "swap_growth": swap_after - swap_before,
        "page_cache_invalidated": not a.no_invalidate,
    }
    REPORTS.mkdir(exist_ok=True)
    stamp = dt.date.today().isoformat()
    name = f"{stamp}-{'M3' if mode == 'drift-dense' else 'M4'}-drift-{a.model.split('/')[-1]}"
    (REPORTS / f"{name}.json").write_text(json.dumps(result, indent=2))
    limit = 1.05 * (budget_bytes + kv)
    read_avg = statistics.mean(reads) if reads else 0.0
    if mode == "drift-dense":
        head = (
            f"{layers} lagen, waarvan {pinned} gepind · {streamed_bytes / gb:.2f} GB gestreamd "
            "per token"
        )
        io_rows = (
            f"| I/O-grens bij {SSD_REFERENCE / 1e9:.2f} GB/s | "
            f"{runs[0]['io_bound_tps']:.2f} tok/s |\n"
            f"| **I/O-efficiëntie** | **{eff * 100:.0f} %** (doel ≥ 80 %) |\n"
        )
    else:
        slots = final["drift"].get("expert_slots")
        total = final["drift"].get("experts_total")
        head = f"{layers} lagen · {slots} van {total} experts per laag in RAM"
        io_rows = (
            f"| **SSD-benutting tijdens decode** | "
            f"**{read_avg / (SSD_REFERENCE / 1e9) * 100:.0f} %** "
            f"van {SSD_REFERENCE / 1e9:.2f} GB/s |\n"
            f"| GB van SSD per token (gem.) | {read_avg / tps:.2f} |\n"
        )
    md = f"""# Drift-benchmark ({mode}): {a.model} ({stamp})

Budget voor gewichten: {a.weights_budget} · {head} · page cache geleegd: \
{"ja" if not a.no_invalidate else "nee"}

| Meting | Waarde |
|---|---|
| Decode (mediaan) | {tps:.2f} tok/s |
{io_rows}| SSD-lezen tijdens decode (gem.) | {read_avg:.2f} GB/s |
| Cache-hitrate / totale wachttijd op I/O | {final["drift"]["hit_rate"]} / \
{final["drift"]["stall_seconds"]} s |
| MLX-piekgeheugen | {mlx_peak / gb:.2f} GB (budget + KV + 5 % = {limit / gb:.2f} GB) |
| Proces-footprint: basis → piek (50 ms-samples) | {baseline / gb:.2f} → {peak_fp / gb:.2f} GB |
| Swapgroei | {(swap_after - swap_before) / 1024**2:.0f} MB |
"""
    (REPORTS / f"{name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
