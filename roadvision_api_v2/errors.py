"""Structured API error classes and safe JSON error handlers for RoadVision AI 2.0."""

from __future__ import annotations

from typing import Any, Dict, Optional

from flask import jsonify
from werkzeug.exceptions import HTTPException


class APIError(Exception):
    """Base structured error for the V2 REST API. Never exposes tracebacks or private paths."""

    status_code: int = 400
    error_code: str = "BAD_REQUEST"

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        if status_code is not None:
            self.status_code = int(status_code)
        if error_code is not None:
            self.error_code = str(error_code)
        self.details = details or {}

    def to_response(self):
        payload: Dict[str, Any] = {
            "error": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code,
        }
        if self.details:
            payload["details"] = self.details
        return jsonify(payload), self.status_code


class BadRequestError(APIError):
    status_code = 400
    error_code = "INVALID_REQUEST"


class NotFoundError(APIError):
    status_code = 404
    error_code = "NOT_FOUND"


class InvalidJobStateError(APIError):
    status_code = 409
    error_code = "INVALID_JOB_STATE"


class UploadTooLargeError(APIError):
    status_code = 413
    error_code = "UPLOAD_TOO_LARGE"


class UnsupportedMediaError(APIError):
    status_code = 415
    error_code = "UNSUPPORTED_MEDIA"


class QueueFullError(APIError):
    status_code = 429
    error_code = "QUEUE_CAPACITY_REACHED"


class ServiceUnavailableError(APIError):
    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"


def sanitize_error_message(exc: Exception, max_len: int = 240) -> str:
    """Return a safe, path-free error summary suitable for API / DB storage."""
    raw = f"{type(exc).__name__}: {str(exc)}"
    # Strip Windows/POSIX directory paths if accidentally embedded
    cleaned = raw.replace("\\", "/")
    if ":/" in cleaned or "/Users/" in cleaned or "/home/" in cleaned:
        cleaned = f"{type(exc).__name__}: internal processing failure (details logged server-side)"
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3] + "..."
    return cleaned


def register_v2_error_handlers(bp) -> None:
    """Attach JSON error handlers to the V2 Flask Blueprint."""

    @bp.errorhandler(APIError)
    def handle_api_error(err: APIError):
        return err.to_response()

    @bp.errorhandler(HTTPException)
    def handle_http_error(err: HTTPException):
        code = err.code or 500
        return (
            jsonify(
                {
                    "error": err.description or "HTTP error",
                    "error_code": f"HTTP_{code}",
                    "status_code": code,
                }
            ),
            code,
        )

    @bp.errorhandler(Exception)
    def handle_unexpected_error(err: Exception):
        return (
            jsonify(
                {
                    "error": "Unexpected internal server error",
                    "error_code": "INTERNAL_ERROR",
                    "status_code": 500,
                }
            ),
            500,
        )
