"""Loopback requests need no key; network requests need sharing on and a Bearer key (§7.1).

Pure ASGI middleware (not BaseHTTPMiddleware) so client disconnects reach streaming
responses directly and cancel generations promptly.
"""

from __future__ import annotations

import hmac
import json
import time
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class AuthMiddleware:
    def __init__(self, app: ASGIApp, engine: Any) -> None:
        self.app = app
        self.engine = engine

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        host = (scope.get("client") or ("", 0))[0]
        if host not in LOOPBACK:
            settings = self.engine.config.settings
            if not settings.share:
                await _reject(send, 403, "sharing_off", "Delen op het netwerk staat uit.")
                return
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode(errors="replace")
            token = auth[7:] if auth.lower().startswith("bearer ") else ""
            key = self.engine.config.api_key or ""
            if not key or not hmac.compare_digest(token.encode(), key.encode()):
                await _reject(
                    send,
                    401,
                    "unauthorized",
                    "Ongeldige of ontbrekende API-sleutel.",
                    "Gebruik 'Authorization: Bearer <sleutel>' uit het Verbind-scherm.",
                )
                return
            client = self.engine.clients.setdefault(host, {"ip": host, "requests": 0})
            client["requests"] += 1
            client["last_request"] = time.time()
        await self.app(scope, receive, send)


async def _reject(
    send: Send, status: int, code: str, message: str, hint: str | None = None
) -> None:
    body = json.dumps({"error": {"code": code, "message": message, "hint": hint}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
