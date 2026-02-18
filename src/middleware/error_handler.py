"""
Global error handler for consistent API error responses.
"""
from __future__ import annotations

import traceback
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.utils.logging import get_logger

logger = get_logger("superai.errors")


class APIError(Exception):
    """Structured API error."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "internal_error",
        param: str | None = None,
        code: str | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code
        super().__init__(message)


def _build_error_response(
    status_code: int,
    message: str,
    error_type: str = "internal_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """Build OpenAI-compatible error response."""
    body: dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code or str(status_code),
        }
    }
    return JSONResponse(status_code=status_code, content=body)


def error_handler(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        logger.error(
            "API error",
            status_code=exc.status_code,
            error_type=exc.error_type,
            message=exc.message,
            path=str(request.url),
        )
        return _build_error_response(
            exc.status_code, exc.message, exc.error_type, exc.param, exc.code
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return _build_error_response(
            exc.status_code,
            str(exc.detail),
            error_type="invalid_request_error" if exc.status_code < 500 else "server_error",
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning("Validation error", message=str(exc), path=str(request.url))
        return _build_error_response(400, str(exc), "invalid_request_error")

    @app.exception_handler(NotImplementedError)
    async def not_implemented_handler(request: Request, exc: NotImplementedError) -> JSONResponse:
        return _build_error_response(501, str(exc), "not_implemented")

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        msg = str(exc)
        if "queue is full" in msg.lower():
            return _build_error_response(503, msg, "server_overloaded")
        if "no healthy engine" in msg.lower():
            return _build_error_response(503, msg, "service_unavailable")
        logger.error("Runtime error", message=msg, path=str(request.url))
        return _build_error_response(500, msg, "internal_error")

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled exception",
            error_type=type(exc).__name__,
            path=str(request.url),
        )
        return _build_error_response(
            500,
            "An internal error occurred. Please try again later.",
            "internal_error",
        )