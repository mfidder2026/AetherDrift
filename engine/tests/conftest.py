from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SMALL_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "U32": 4, "U8": 1}


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """Minimal safetensors writer: zero-filled tensors with the given dtype/shape."""
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        n = DTYPE_BYTES[dtype]
        for s in shape:
            n *= s
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\0" * offset)


def small_model_available() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(SMALL_MODEL, "model.safetensors"), str)


requires_model = pytest.mark.skipif(
    not small_model_available(), reason=f"{SMALL_MODEL} not in the Hugging Face cache"
)


class RunningEngine:
    def __init__(self, proc: subprocess.Popen[str], port: int, ready: dict) -> None:
        self.proc = proc
        self.port = port
        self.ready = ready
        self.url = f"http://127.0.0.1:{port}"


@pytest.fixture(scope="session")
def engine_proc(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RunningEngine]:
    """The real engine as a subprocess, exactly as the app starts it."""
    if not small_model_available():
        pytest.skip(f"{SMALL_MODEL} not in the Hugging Face cache")
    data = tmp_path_factory.mktemp("aether-data")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aether",
            "--port",
            "0",
            "--data-dir",
            str(data),
            "--log-dir",
            str(data / "logs"),
            "--model",
            SMALL_MODEL,
            "--parent-pid",
            str(os.getpid()),
            "--api-key-stdin",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write("test-key\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    ready = json.loads(line)
    assert ready["event"] == "ready"
    eng = RunningEngine(proc, ready["port"], ready)
    _wait_model(eng)
    yield eng
    proc.terminate()
    proc.wait(timeout=10)


def _wait_model(eng: RunningEngine, timeout: float = 60) -> None:
    import time

    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        state = httpx.get(f"{eng.url}/health").json()["model_state"]
        if state == "ready":
            return
        if state == "error":
            raise RuntimeError("model failed to load")
        time.sleep(0.1)
    raise TimeoutError("model did not load")
