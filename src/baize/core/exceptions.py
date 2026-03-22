"""Global exception handlers for the FastAPI application."""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from baize.core.schemas import ErrorCode

logger = logging.getLogger(__name__)

_HTTP_STATUS_TO_ERROR_CODE: dict[int, int] = {
    400: ErrorCode.BAD_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    500: ErrorCode.INTERNAL_ERROR,
}


def _error_body(code: int, message: str) -> dict:
    return {"code": code, "data": None, "message": message}


async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    error_code = _HTTP_STATUS_TO_ERROR_CODE.get(exc.status_code, exc.status_code * 100)
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content=_error_body(error_code, message))


async def _validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = " -> ".join(str(p) for p in first.get("loc", []))
        msg = first.get("msg", "Validation error")
        message = f"{loc}: {msg}" if loc else msg
    else:
        message = "Request validation failed"
    return JSONResponse(status_code=422, content=_error_body(ErrorCode.BAD_REQUEST, message))


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=_error_body(ErrorCode.INTERNAL_ERROR, "Internal server error"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all global exception handlers on the given FastAPI instance.

    Args:
        app: The FastAPI application to attach handlers to.
    """
    app.add_exception_handler(HTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
