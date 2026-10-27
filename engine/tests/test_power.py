"""The Mac stays awake while generating (a real IOKit assertion, released afterwards)."""

from __future__ import annotations

import subprocess

from aether.power import KeepAwake


def _assertions() -> str:
    return subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True).stdout


def test_keep_awake_is_reference_counted() -> None:
    k = KeepAwake("AetherDrift test assertion")
    with k:
        with k:
            assert k.active
        assert k.active  # one holder left
        assert "AetherDrift test assertion" in _assertions()
    assert not k.active
    assert "AetherDrift test assertion" not in _assertions()
