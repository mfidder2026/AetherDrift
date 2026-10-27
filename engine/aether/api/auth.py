"""Loopback requests need no key; network requests need sharing on and a Bearer key (§7.1).

Over the network only the chat APIs are reachable. Management (/aether/v1/*: loading models,
settings, the key itself) stays loopback-only, so a shared Mac cannot be reconfigured by the
devices it serves. The webchat page itself needs no key: it holds no data, and a browser
cannot send a header when it opens a URL; the page sends the key with its API calls.

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
LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "[::1]"}
PUBLIC_PATHS = {"/", "/favicon.ico", "/manifest.webmanifest"}
NETWORK_PREFIXES = ("/v1/", "/api/")
NETWORK_PATHS = {"/health"}


class AuthMiddleware:
    def __init__(self, app: ASGIApp, engine: Any) -> None:
        self.app = app
        self.engine = engine

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        host = (scope.get("client") or ("", 0))[0]
        if host in LOOPBACK and not _same_machine(scope):
            await _reject(
                send, 403, "foreign_origin", "Verzoeken van andere websites worden geweigerd."
            )
            return
        if host not in LOOPBACK:
            settings = self.engine.config.settings
            if not settings.share:
                await _reject(send, 403, "sharing_off", "Delen op het netwerk staat uit.")
                return
            path = scope.get("path", "")
            if path in PUBLIC_PATHS and scope.get("method") in ("GET", "HEAD"):
                await self.app(scope, receive, send)
                return
            if not (path in NETWORK_PATHS or path.startswith(NETWORK_PREFIXES)):
                await _reject(
                    send, 403, "local_only", "Dit endpoint is alleen beschikbaar op de Mac zelf."
                )
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
            client = self.engine.clients.setdefault(host, {"ip": host, "requests": 0, "tokens": 0})
            client["requests"] += 1
            client["last_request"] = time.time()
        await self.app(scope, receive, send)


def _hostname(value: str) -> str:
    """'127.0.0.1:8080' → '127.0.0.1', '[::1]:8080' → '[::1]', 'http://x:1' → 'x'."""
    value = value.split("://", 1)[-1].split("/", 1)[0]
    if value.startswith("["):
        return value.split("]", 1)[0] + "]"
    return value.rsplit(":", 1)[0] if ":" in value else value


def _same_machine(scope: Scope) -> bool:
    """Loopback needs no key, so it must really come from this Mac and not from a web page.

    A page in the user's browser can send requests to 127.0.0.1 (CSRF) or reach it under its
    own domain name (DNS rebinding). Both are refused: the Host must name the loopback
    address, and a browser Origin, if any, must be a loopback page (such as the webchat).
    """
    headers = dict(scope.get("headers") or [])
    host = headers.get(b"host", b"").decode(errors="replace")
    if host and _hostname(host).lower() not in LOCAL_HOSTNAMES:
        return False
    origin = headers.get(b"origin")
    return origin is None or _hostname(origin.decode(errors="replace")).lower() in LOCAL_HOSTNAMES


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
