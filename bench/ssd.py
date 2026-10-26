"""M0 SSD benchmark: sequential/random read throughput vs block size and queue depth.

Usage:  uv run --project engine python bench/ssd.py [--size-gb 12] [--dir PATH]

Writes a temporary test file with F_NOCACHE (so it is not served from the page cache),
runs pread tests with F_NOCACHE, runs one mmap test on an untouched region, deletes the
file and writes JSON + Markdown to bench/reports/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import mmap
import os
import platform
import random
import subprocess
import tempfile
import threading
import time
from pathlib import Path

MB = 1024 * 1024
GB = 1024 * MB
PAGE = 16 * 1024
REPORTS = Path(__file__).resolve().parent / "reports"


def sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def write_file(path: Path, size: int) -> float:
    block = os.urandom(64 * MB)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    start = time.perf_counter()
    try:
        written = 0
        while written < size:
            written += os.write(fd, block[: min(len(block), size - written)])
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    finally:
        os.close(fd)
    return size / (time.perf_counter() - start)


def pread_test(
    path: Path, region: tuple[int, int], block: int, qd: int, total: int, pattern: str
) -> dict:
    lo, hi = region
    n_blocks = total // block
    if pattern == "seq":
        offsets = [lo + (i * block) % (hi - lo - block) for i in range(n_blocks)]
    else:
        rng = random.Random(block * 31 + qd)
        slots = (hi - lo - block) // PAGE
        offsets = [lo + rng.randrange(slots) * PAGE for _ in range(n_blocks)]

    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    lock = threading.Lock()
    next_idx = [0]
    latencies: list[float] = []

    def worker() -> None:
        buf = bytearray(block)
        view = memoryview(buf)
        local: list[float] = []
        while True:
            with lock:
                i = next_idx[0]
                next_idx[0] += 1
            if i >= n_blocks:
                break
            t0 = time.perf_counter()
            done = 0
            while done < block:
                done += os.preadv(fd, [view[done:]], offsets[i] + done)
            local.append(time.perf_counter() - t0)
        with lock:
            latencies.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(qd)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    os.close(fd)
    latencies.sort()
    return {
        "pattern": pattern,
        "block_mb": block / MB,
        "qd": qd,
        "gbps": n_blocks * block / elapsed / 1e9,
        "p50_ms": latencies[len(latencies) // 2] * 1e3,
        "p95_ms": latencies[int(len(latencies) * 0.95)] * 1e3,
    }


def mmap_test(path: Path, region: tuple[int, int], block: int) -> dict:
    lo, hi = region
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        start = time.perf_counter()
        pos = lo
        while pos + block <= hi:
            mm.madvise(mmap.MADV_WILLNEED, pos, block)
            _ = bytes(mm[pos : pos + block])  # copy out, like materialising a weight
            mm.madvise(mmap.MADV_DONTNEED, pos, block)
            pos += block
        elapsed = time.perf_counter() - start
        mm.close()
    return {
        "pattern": "mmap-seq",
        "block_mb": block / MB,
        "qd": 1,
        "gbps": (pos - lo) / elapsed / 1e9,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-gb", type=int, default=12)
    ap.add_argument("--per-test-gb", type=float, default=1.0)
    ap.add_argument(
        "--dir", default=None, help="directory on the volume to test (default: models volume)"
    )
    args = ap.parse_args()

    size = args.size_gb * GB
    mmap_region = (size - 4 * GB, size)
    pread_region = (0, size - 4 * GB)
    per_test = int(args.per_test_gb * GB)

    base = Path(args.dir or Path.home() / "Library/Caches/AetherDrift")
    base.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="ssd-bench-", suffix=".bin", dir=base)
    os.close(fd)
    path = Path(name)
    results: list[dict] = []
    try:
        print(f"Writing {args.size_gb} GB test file to {path} …", flush=True)
        write_bps = write_file(path, size)
        print(f"  write: {write_bps / 1e9:.2f} GB/s", flush=True)

        for pattern, blocks in (("seq", [1, 4, 16, 64]), ("rand", [1, 4, 16])):
            for block_mb in blocks:
                for qd in (1, 4, 8, 16, 32):
                    r = pread_test(path, pread_region, block_mb * MB, qd, per_test, pattern)
                    results.append(r)
                    print(
                        f"  {pattern:4} {block_mb:>3} MB  qd={qd:<2}  {r['gbps']:5.2f} GB/s"
                        f"  p50 {r['p50_ms']:6.1f} ms  p95 {r['p95_ms']:6.1f} ms",
                        flush=True,
                    )
        r = mmap_test(path, mmap_region, 16 * MB)
        results.append(r)
        print(f"  mmap 16 MB  qd=1   {r['gbps']:5.2f} GB/s", flush=True)
    finally:
        path.unlink(missing_ok=True)

    best_seq = max((r for r in results if r["pattern"] == "seq"), key=lambda r: r["gbps"])
    best_rand = max((r for r in results if r["pattern"] == "rand"), key=lambda r: r["gbps"])
    meta = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "chip": sysctl("machdep.cpu.brand_string"),
        "ram_bytes": int(sysctl("hw.memsize")),
        "macos": platform.mac_ver()[0],
        "write_gbps": write_bps / 1e9,
        "best_seq": best_seq,
        "best_rand": best_rand,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    (REPORTS / f"{stamp}-M0-ssd.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2)
    )

    lines = [
        f"# SSD-benchmark ({stamp})",
        "",
        f"- Machine: {meta['chip']}, {meta['ram_bytes'] // GB} GB RAM, macOS {meta['macos']}",
        f"- Schrijven (F_NOCACHE): {meta['write_gbps']:.2f} GB/s",
        f"- Beste sequentieel lezen: **{best_seq['gbps']:.2f} GB/s** "
        f"({best_seq['block_mb']:.0f} MB, qd {best_seq['qd']})",
        f"- Beste random lezen: **{best_rand['gbps']:.2f} GB/s** "
        f"({best_rand['block_mb']:.0f} MB, qd {best_rand['qd']})",
        "",
        "| Patroon | Blok (MB) | QD | GB/s | p50 (ms) | p95 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        p50 = f"{r['p50_ms']:.1f}" if "p50_ms" in r else "—"
        p95 = f"{r['p95_ms']:.1f}" if "p95_ms" in r else "—"
        cells = [r["pattern"], f"{r['block_mb']:.0f}", r["qd"], f"{r['gbps']:.2f}", p50, p95]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    (REPORTS / f"{stamp}-M0-ssd.md").write_text("\n".join(lines) + "\n")
    print(f"\nReport: {REPORTS / f'{stamp}-M0-ssd.md'}")


if __name__ == "__main__":
    main()
