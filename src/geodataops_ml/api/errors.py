"""Errores estructurados: todas las respuestas de error comparten el mismo formato.

{"error": {"codigo": "...", "mensaje": "...", "detalle": ...}}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 400
    code = "solicitud_invalida"

    def __init__(self, message: str, details: Any = None, *, code: str | None = None):
        super().__init__(message)
        self.message, self.details = message, details
        if code:
            self.code = code


class NotFound(AppError):
    status_code, code = 404, "no_encontrado"


class Unprocessable(AppError):
    status_code, code = 422, "datos_insuficientes"


class Conflict(AppError):
    status_code, code = 409, "conflicto"


class Unauthorized(AppError):
    status_code, code = 401, "no_autorizado"


class ServiceUnavailable(AppError):
    status_code, code = 503, "servicio_no_disponible"


class UpstreamError(AppError):
    status_code, code = 502, "error_proveedor"


class UpstreamTimeout(AppError):
    status_code, code = 504, "tiempo_agotado"


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {"error": {"codigo": code, "mensaje": message, "detalle": jsonable_encoder(details)}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message, exc.details), exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"campo": ".".join(str(p) for p in e["loc"]), "mensaje": e["msg"], "tipo": e["type"]}
            for e in exc.errors()
        ]
        return JSONResponse(error_body("validacion", "Parámetros inválidos", details), 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(error_body("http_error", str(exc.detail)), exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Error no controlado")  # el detalle queda en logs, no en la respuesta
        return JSONResponse(error_body("error_interno", "Error interno del servidor"), 500)
