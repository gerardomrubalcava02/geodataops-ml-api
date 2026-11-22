"""Herramientas expuestas al LLM. Los argumentos se validan con Pydantic antes de ejecutar
cualquier consulta y el esquema JSON enviado al modelo se deriva de esos mismos modelos.
Todas llaman a GeoService: el LLM ve exactamente los datos que sirve la API REST."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from geodataops_ml.api.errors import AppError
from geodataops_ml.data.repository import ObservationQuery
from geodataops_ml.llm.provider import ToolSpec
from geodataops_ml.services import GeoService


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SinArgs(_Args):
    pass


class ConsultarIndicadoresArgs(_Args):
    cvegeo: str | None = Field(default=None, pattern=r"^[0-9]{5}$", description="Clave municipal de 5 dígitos")
    indicadores: list[str] = Field(default_factory=list, max_length=10, description="Ej. ['poblacion_total']")
    periodo_desde: int | None = Field(default=None, ge=1000, le=9999)
    periodo_hasta: int | None = Field(default=None, ge=1000, le=9999)
    limite: int = Field(default=50, ge=1, le=200)


class PredecirArgs(_Args):
    cvegeo: str = Field(pattern=r"^[0-9]{5}$")
    periodo: int = Field(ge=1000, le=9999, description="Año base t; se estima t+1")
    fuente_datos: Literal["entrenamiento", "operativa"] = "entrenamiento"


class ToolError(Exception):
    pass


Handler = Callable[[GeoService, Any], Awaitable[Any]]


async def _catalogo(svc: GeoService, _: SinArgs) -> Any:
    return {"fuente": svc.source(), "indicadores": await svc.catalog()}


async def _municipios(svc: GeoService, _: SinArgs) -> Any:
    items = await svc.municipios()
    return {"fuente": svc.source(), "total": len(items), "municipios": items[:200]}


async def _consultar(svc: GeoService, a: ConsultarIndicadoresArgs) -> Any:
    q = ObservationQuery(
        cvegeo=a.cvegeo, indicadores=tuple(a.indicadores), periodo_desde=a.periodo_desde,
        periodo_hasta=a.periodo_hasta, limit=a.limite,
    )
    return await svc.observations(q)


async def _predecir(svc: GeoService, a: PredecirArgs) -> Any:
    return await svc.predict(cvegeo=a.cvegeo, periodo=a.periodo, fuente_datos=a.fuente_datos)


async def _metricas(svc: GeoService, _: SinArgs) -> Any:
    m = svc.model_metadata()
    sel = m["modelo_seleccionado"]
    return {
        "version": m["version"], "algoritmo": m["algoritmo"], "datos": m["datos"],
        "objetivo": m["objetivo"], "particion": m["particion"],
        "prueba_modelo": m["prueba"][sel],
        "prueba_mejor_baseline": {m["mejor_baseline_prueba"]: m["prueba"][m["mejor_baseline_prueba"]]},
        "mejora_mae_vs_mejor_baseline_pct": m["mejora_mae_vs_mejor_baseline_pct"],
    }


TOOLS: dict[str, tuple[str, type[_Args], Handler]] = {
    "listar_indicadores": (
        "Catálogo de indicadores disponibles en la fuente operativa: unidad, cobertura de "
        "municipios y periodos. Úsala primero si no sabes qué indicadores existen.",
        SinArgs, _catalogo,
    ),
    "listar_municipios": (
        "Municipios (cvegeo) disponibles en la fuente operativa con su rango de periodos.",
        SinArgs, _municipios,
    ),
    "consultar_indicadores": (
        "Observaciones reales de la fuente operativa filtradas por municipio, indicadores y "
        "rango de años. Devuelve también la procedencia (fuente) y si es sintética.",
        ConsultarIndicadoresArgs, _consultar,
    ),
    "predecir_crecimiento": (
        "Predice el crecimiento poblacional t→t+1 de un municipio con el modelo activo. "
        "Requiere poblaciones de t, t-1, t-2 y covariables de t. Por defecto usa el conjunto "
        "de entrenamiento (sintético); 'operativa' usa los datos servidos por la API.",
        PredecirArgs, _predecir,
    ),
    "metricas_modelo": (
        "Versión, datos de entrenamiento y métricas de prueba del modelo activo frente al "
        "mejor baseline.",
        SinArgs, _metricas,
    ),
}


def tool_specs() -> list[ToolSpec]:
    specs = []
    for name, (desc, model, _) in TOOLS.items():
        schema = model.model_json_schema()
        schema.pop("title", None)
        specs.append(ToolSpec(name=name, description=desc, input_schema=schema))
    return specs


async def run_tool(svc: GeoService, name: str, arguments: Any) -> str:
    """Valida y ejecuta. Devuelve JSON. Lanza ToolError con un mensaje útil para el modelo."""
    if name not in TOOLS:
        raise ToolError(f"herramienta desconocida: {name}")
    _, model, handler = TOOLS[name]
    try:
        args = model.model_validate(arguments if arguments is not None else {})
    except ValidationError as exc:
        details = [{"campo": ".".join(map(str, e["loc"])), "error": e["msg"]} for e in exc.errors()]
        raise ToolError(json.dumps({"argumentos_invalidos": details}, ensure_ascii=False)) from exc
    try:
        result = await handler(svc, args)
    except AppError as exc:
        raise ToolError(json.dumps({"codigo": exc.code, "mensaje": exc.message, "detalle": exc.details},
                                   ensure_ascii=False, default=str)) from exc
    return json.dumps(result, ensure_ascii=False, default=str)
