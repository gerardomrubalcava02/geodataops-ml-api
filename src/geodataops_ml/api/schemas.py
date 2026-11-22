"""Modelos Pydantic de la API (validación de entrada y documentación OpenAPI)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Cvegeo = Annotated[str, Field(pattern=r"^[0-9]{5}$", examples=["99001"], description="Clave INEGI de 5 dígitos")]
Periodo = Annotated[int, Field(ge=1000, le=9999, examples=[2023])]
Indicador = Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$", max_length=64)]


class ErrorDetail(BaseModel):
    codigo: str
    mensaje: str
    detalle: Any = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Fuente(BaseModel):
    tipo: str
    ubicacion: str
    sintetico: bool
    sha256: str | None = None
    extra: dict[str, Any] = {}


class Observacion(BaseModel):
    cvegeo: str
    periodo: int
    indicador: str
    valor: float
    unidad: str
    fuente: str


class PaginaObservaciones(BaseModel):
    total: int
    limit: int
    offset: int
    fuente: Fuente
    items: list[Observacion]


class Municipio(BaseModel):
    cvegeo: str
    nombre: str | None = None
    periodo_min: int
    periodo_max: int
    n_periodos: int
    indicadores: list[str]


class IndicadorCatalogo(BaseModel):
    indicador: str
    unidad: str
    observaciones: int
    municipios: int
    periodo_min: int
    periodo_max: int


class Caracteristicas(BaseModel):
    """Características en el año base t. Los rangos rechazan entradas físicamente absurdas."""

    model_config = ConfigDict(extra="forbid")

    log_poblacion: float = Field(ge=0, le=20, description="ln(población total en t)")
    crecimiento: float = Field(ge=-0.5, le=0.5, description="P[t]/P[t-1] - 1")
    crecimiento_lag1: float = Field(ge=-0.5, le=0.5, description="P[t-1]/P[t-2] - 1")
    razon_masculinidad: float = Field(ge=50, le=200, description="hombres por cada 100 mujeres")
    tasa_ocupacion: float = Field(ge=0, le=100)
    escolaridad_promedio: float = Field(ge=0, le=25)
    hogares_con_internet: float = Field(ge=0, le=100)


class PrediccionRequest(BaseModel):
    """Dos modos excluyentes: por municipio (`cvegeo` + `periodo`) o con `caracteristicas`."""

    model_config = ConfigDict(extra="forbid")

    cvegeo: Cvegeo | None = None
    periodo: Periodo | None = None
    caracteristicas: Caracteristicas | None = None
    fuente_datos: Literal["entrenamiento", "operativa"] = Field(
        default="entrenamiento",
        description="Historial usado en modo municipio: el conjunto de entrenamiento del modelo "
        "o la fuente operativa de la API (ETL/fixtures/PostgreSQL).",
    )

    @model_validator(mode="after")
    def _one_mode(self) -> PrediccionRequest:
        by_mun = self.cvegeo is not None or self.periodo is not None
        if by_mun and self.caracteristicas is not None:
            raise ValueError("use cvegeo+periodo o caracteristicas, no ambos")
        if by_mun and (self.cvegeo is None or self.periodo is None):
            raise ValueError("el modo municipio requiere cvegeo y periodo")
        if not by_mun and self.caracteristicas is None:
            raise ValueError("indique cvegeo+periodo o caracteristicas")
        return self


class PrediccionResponse(BaseModel):
    modelo_version: str
    algoritmo: str
    datos_entrenamiento_sinteticos: bool
    crecimiento_esperado: float
    cvegeo: str | None = None
    periodo_base: int | None = None
    periodo_objetivo: int | None = None
    poblacion_base: int | None = None
    poblacion_estimada: int | None = None
    caracteristicas: dict[str, float]
    advertencia: str | None = None


class Salud(BaseModel):
    estado: Literal["ok", "degradado"]
    version: str
    fuente_datos: Fuente
    modelo_version: str | None
    llm_proveedor: str
    llm_configurado: bool
