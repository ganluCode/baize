"""Unified API response schemas and error code constants."""

from enum import IntEnum
from typing import Any

from pydantic import BaseModel


class ErrorCode(IntEnum):
    """Standard error codes for the Baize API."""

    SUCCESS = 0
    BAD_REQUEST = 40000
    UNAUTHORIZED = 40100
    FORBIDDEN = 40300
    NOT_FOUND = 40400
    INTERNAL_ERROR = 50000


class ApiResponse[T](BaseModel):
    """Unified API response envelope.

    Attributes:
        code: Numeric status code (0 = success, non-zero = error).
        data: Response payload; None for error responses.
        message: Human-readable status message.
    """

    code: int
    data: T | None
    message: str


def success(data: Any) -> ApiResponse[Any]:
    """Build a successful ApiResponse with code=0.

    Args:
        data: The response payload (may be None).

    Returns:
        ApiResponse with code=0 and the given data.
    """
    return ApiResponse(code=ErrorCode.SUCCESS, data=data, message="ok")


def error(code: ErrorCode, message: str) -> ApiResponse[None]:
    """Build an error ApiResponse with data=None.

    Args:
        code: The ErrorCode to use.
        message: Human-readable error description.

    Returns:
        ApiResponse with the given error code and message, data=None.
    """
    return ApiResponse(code=code, data=None, message=message)
