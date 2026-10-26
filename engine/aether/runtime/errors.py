"""Errors with a machine-readable code so the app can show a friendly, actionable message."""

from __future__ import annotations

from typing import Any


class EngineError(Exception):
    def __init__(self, code: str, message: str, hint: str | None = None, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint}
