"""Endpoints REST de indicadores, modelos y predicciones."""

from __future__ import annotations

import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Query, Request

from geodataops_ml import __version__
from geodataops_ml.api.errors import Unauthorized
from geodataops_ml.api.schemas import (
    Cvegeo,
    ErrorResponse,
    Indicador,
    IndicadorCatalogo,
    Municipio,
    PaginaObservaciones,
    PrediccionRequest,
    PrediccionResponse,
    Salud,
)
from geodataops_ml.data.repository import ObservationQuery
from geodataops_ml.services import GeoService

ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (404, 409, 422, 500, 503)
}
router = APIRouter(prefix="/api/v1", responses=ERRORS)


def get_service(request: Request) -> GeoService:
    return request.app.state.service


Service = Annotated[GeoService, Depends(get_service)]


@router.get("/salud", response_model=Salud, tags=["sistema"])
async def salud(svc: Service, request: Request) -> dict[str, Any]:
    agent = request.app.state.agent
    return {
        "estado": "ok" if svc.bundle else "degradado",
        "version": __version__,
        "fuente_datos": svc.source(),
        "modelo_version": svc.bundle.version if svc.bundle else None,
        "llm_proveedor": svc.settings.llm_provider,
        "llm_configurado": agent is not None,
    }


@router.get("/municipios", response_model=list[Municipio], tags=["indicadores"])
async def municipios(svc: Service) -> list[dict[str, Any]]:
    return await svc.municipios()


@router.get("/indicadores", response_model=list[IndicadorCatalogo], tags=["indicadores"])
async def catalogo(svc: Service) -> list[dict[str, Any]]:
    return await svc.catalog()


@router.get("/observaciones", response_model=PaginaObservaciones, tags=["indicadores"])
async def observaciones(
    svc: Service,
    cvegeo: Annotated[Cvegeo | None, Query()] = None,
    indicador: Annotated[list[Indicador] | None, Query(max_length=20)] = None,
    periodo_desde: Annotated[int | None, Query(ge=1000, le=9999)] = None,
    periodo_hasta: Annotated[int | None, Query(ge=1000, le=9999)] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    q = ObservationQuery(
        cvegeo=cvegeo, indicadores=tuple(indicador or ()), periodo_desde=periodo_desde,
        periodo_hasta=periodo_hasta, limit=limit, offset=offset,
    )
    return await svc.observations(q)


@router.get("/modelos", tags=["modelos"])
async def modelos(svc: Service) -> list[dict[str, Any]]:
    return svc.registry.list()


@router.get("/modelos/actual", tags=["modelos"])
async def modelo_actual(svc: Service) -> dict[str, Any]:
    return svc.model_metadata()


@router.get("/modelos/{version}", tags=["modelos"])
async def modelo(version: str, svc: Service) -> dict[str, Any]:
    return svc.model_metadata(version)


@router.post("/modelos/entrenar", tags=["modelos"], status_code=201,
             responses={401: {"model": ErrorResponse}})
async def entrenar(
    svc: Service, x_admin_token: Annotated[str | None, Header()] = None
) -> dict[str, Any]:
    """Entrena con el conjunto configurado, registra la versión y la activa."""
    expected = svc.settings.admin_token
    if expected is not None and not (
        x_admin_token and secrets.compare_digest(x_admin_token, expected.get_secret_value())
    ):
        raise Unauthorized("X-Admin-Token inválido o ausente")
    return await svc.train_model()


@router.post("/predicciones", response_model=PrediccionResponse, tags=["predicciones"])
async def predecir(body: PrediccionRequest, svc: Service) -> dict[str, Any]:
    return await svc.predict(
        cvegeo=body.cvegeo,
        periodo=body.periodo,
        caracteristicas=body.caracteristicas.model_dump() if body.caracteristicas else None,
        fuente_datos=body.fuente_datos,
    )
