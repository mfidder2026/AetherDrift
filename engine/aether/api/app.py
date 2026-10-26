"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import __version__
from ..runtime.errors import EngineError
from . import aether, openai
from .auth import AuthMiddleware
from .state import EngineState


async def _noop_share(_: bool) -> None:
    return None


def create_app(
    engine: EngineState, on_share_changed: Callable[[bool], Awaitable[None]] = _noop_share
) -> FastAPI:
    app = FastAPI(title="AetherDrift Engine", version=__version__, docs_url="/docs", redoc_url=None)
    app.state.engine = engine
    app.state.on_share_changed = on_share_changed
    app.add_middleware(AuthMiddleware, engine=engine)
    app.include_router(openai.router)
    app.include_router(aether.router)

    @app.exception_handler(EngineError)
    async def engine_error(_: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse({"error": exc.to_dict()}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_request",
                    "message": f"{where}: {first.get('msg', '')}",
                    "hint": None,
                }
            },
            status_code=400,
        )

    return app
