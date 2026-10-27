"""Keeps the Mac from idle-sleeping while the engine is generating.

Without this a long Drift answer (GLM 5.3: minutes) stops halfway when nobody touches the
Mac, also when the request came from another device. IOKit power assertions via ctypes; the
display may still sleep.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import threading
from typing import Any

log = logging.getLogger("aether.power")
_ASSERTION = "PreventUserIdleSystemSleep"
_LEVEL_ON = 255


class _IOKit:
    def __init__(self) -> None:
        iokit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("IOKit") or "IOKit")
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation") or "CoreFoundation")
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        iokit.IOPMAssertionCreateWithName.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        iokit.IOPMAssertionRelease.argtypes = [ctypes.c_uint32]
        self.iokit, self.cf = iokit, cf

    def _cfstr(self, s: str) -> Any:
        return self.cf.CFStringCreateWithCString(None, s.encode(), 0x08000100)  # UTF-8

    def create(self, reason: str) -> int | None:
        kind, name = self._cfstr(_ASSERTION), self._cfstr(reason)
        out = ctypes.c_uint32(0)
        try:
            rc = self.iokit.IOPMAssertionCreateWithName(kind, _LEVEL_ON, name, ctypes.byref(out))
        finally:
            self.cf.CFRelease(kind)
            self.cf.CFRelease(name)
        return out.value if rc == 0 else None

    def release(self, assertion: int) -> None:
        self.iokit.IOPMAssertionRelease(assertion)


class KeepAwake:
    """Reference-counted: awake while at least one holder is active. Never raises."""

    def __init__(self, reason: str = "AetherDrift genereert een antwoord") -> None:
        self.reason = reason
        self._lock = threading.Lock()
        self._holders = 0
        self._assertion: int | None = None
        try:
            self._iokit: _IOKit | None = _IOKit()
        except OSError:
            log.warning("IOKit unavailable; the Mac may sleep during long answers")
            self._iokit = None

    @property
    def active(self) -> bool:
        return self._assertion is not None

    def __enter__(self) -> KeepAwake:
        with self._lock:
            self._holders += 1
            if self._holders == 1 and self._iokit is not None:
                self._assertion = self._iokit.create(self.reason)
        return self

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self._holders -= 1
            if self._holders == 0 and self._assertion is not None and self._iokit is not None:
                self._iokit.release(self._assertion)
                self._assertion = None
