from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.schemas.errors import ErrorResponse


class AppError(Exception):
    code = "APP_ERROR"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class StartupError(AppError):
    code = "STARTUP_ERROR"


class ValidationAppError(AppError):
    code = "VALIDATION_ERROR"
    status_code = status.HTTP_400_BAD_REQUEST


class LlmUnavailableError(AppError):
    code = "LLM_UNAVAILABLE"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class LlmResponseError(AppError):
    code = "LLM_BAD_RESPONSE"
    status_code = status.HTTP_502_BAD_GATEWAY


class AsrUnavailableError(AppError):
    code = "ASR_UNAVAILABLE"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class AsrApiError(AppError):
    code = "ASR_API_ERROR"
    status_code = status.HTTP_502_BAD_GATEWAY


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("{}: {}", exc.code, exc.message)
        body = ErrorResponse(code=exc.code, message=exc.message, details=exc.details)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: {}", exc)
        body = ErrorResponse(code="INTERNAL_ERROR", message="Internal server error", details={})
        return JSONResponse(status_code=500, content=body.model_dump())

