"""Capa de servicio compartida por la API REST y las herramientas del LLM.

Concentra reglas de negocio para que ambos canales devuelvan exactamente los mismos datos.
Las operaciones de CPU (entrenar, predecir) se ejecutan en un hilo para no bloquear el
event loop; el entrenamiento se serializa con un lock.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any

from anyio import to_thread

from geodataops_ml.api.errors import Conflict, NotFound, ServiceUnavailable, Unprocessable
from geodataops_ml.config import Settings
from geodataops_ml.data.contract import LoadedDataset, load_csv
from geodataops_ml.data.repository import IndicatorRepository, ObservationQuery
from geodataops_ml.ml.features import InsufficientHistoryError
from geodataops_ml.ml.predict import predict_for_municipio, predict_from_features
from geodataops_ml.ml.registry import ModelBundle, ModelNotFoundError, ModelRegistry
from geodataops_ml.ml.train import TrainConfig, train

log = logging.getLogger(__name__)

SYNTHETIC_WARNING = (
    "Modelo entrenado con datos demostrativos SINTÉTICOS: la predicción ilustra el flujo "
    "de ML y no es una proyección demográfica real."
)


class GeoService:
    def __init__(self, settings: Settings, repo: IndicatorRepository, registry: ModelRegistry):
        self.settings = settings
        self.repo = repo
        self.registry = registry
        self.bundle: ModelBundle | None = None
        self._training_data: LoadedDataset | None = None
        self._train_lock = asyncio.Lock()

    # ---------- modelo ----------
    def load_latest_model(self) -> None:
        try:
            self.bundle = self.registry.load()
            log.info("Modelo cargado: %s", self.bundle.version)
        except ModelNotFoundError:
            log.warning("No hay modelo entrenado; /predicciones responderá 503")

    def require_model(self) -> ModelBundle:
        if self.bundle is None:
            raise ServiceUnavailable(
                "No hay modelo entrenado. Ejecute `geodataops-ml train` o POST /api/v1/modelos/entrenar",
                code="modelo_no_disponible",
            )
        return self.bundle

    async def training_data(self) -> LoadedDataset:
        if self._training_data is None:
            self._training_data = await to_thread.run_sync(load_csv, self.settings.training_csv_path)
        return self._training_data

    async def train_model(self) -> dict[str, Any]:
        if self._train_lock.locked():
            raise Conflict("Ya hay un entrenamiento en curso")
        async with self._train_lock:
            self._training_data = None  # releer por si el archivo cambió
            data = await self.training_data()

            def _run() -> ModelBundle:
                return self.registry.save(train(data.frame, TrainConfig()), data)

            self.bundle = await to_thread.run_sync(_run)
            return self.bundle.metadata

    def model_metadata(self, version: str | None = None) -> dict[str, Any]:
        if version is None:
            return self.require_model().metadata
        try:
            return self.registry.load(version).metadata
        except ModelNotFoundError as exc:
            raise NotFound(str(exc)) from exc

    # ---------- predicción ----------
    async def predict(
        self,
        *,
        cvegeo: str | None = None,
        periodo: int | None = None,
        caracteristicas: dict[str, float] | None = None,
        fuente_datos: str = "entrenamiento",
    ) -> dict[str, Any]:
        bundle = self.require_model()
        synthetic = bool(bundle.metadata["datos"]["sintetico"])
        base = {
            "modelo_version": bundle.version,
            "algoritmo": bundle.metadata["algoritmo"],
            "datos_entrenamiento_sinteticos": synthetic,
            "advertencia": SYNTHETIC_WARNING if synthetic else None,
        }
        if caracteristicas is not None:
            growth = await to_thread.run_sync(predict_from_features, bundle, caracteristicas)
            return {**base, "crecimiento_esperado": growth, "caracteristicas": caracteristicas}

        assert cvegeo is not None and periodo is not None
        if fuente_datos == "entrenamiento":
            frame = (await self.training_data()).frame
        else:
            frame = await self.repo.frame()
        if not (frame["cvegeo"] == cvegeo).any():
            raise NotFound(f"El municipio {cvegeo} no existe en la fuente '{fuente_datos}'")
        try:
            result = await to_thread.run_sync(predict_for_municipio, bundle, frame, cvegeo, periodo)
        except InsufficientHistoryError as exc:
            raise Unprocessable(
                str(exc),
                {"cvegeo": cvegeo, "periodo": periodo, "faltantes": exc.missing,
                 "requisito": "poblaciones de t, t-1, t-2 y covariables de t"},
            ) from exc
        except ValueError as exc:  # p. ej. la fuente no tiene las covariables del modelo
            raise Unprocessable(str(exc)) from exc
        return {**base, **result}

    # ---------- indicadores ----------
    def source(self) -> dict[str, Any]:
        info = asdict(self.repo.info)
        return {
            "tipo": info["kind"], "ubicacion": info["location"], "sintetico": info["is_synthetic"],
            "sha256": info["sha256"], "extra": info["extra"],
        }

    async def observations(self, q: ObservationQuery) -> dict[str, Any]:
        items, total = await self.repo.observations(q)
        return {"total": total, "limit": q.limit, "offset": q.offset, "fuente": self.source(), "items": items}

    async def municipios(self) -> list[dict[str, Any]]:
        return await self.repo.municipios()

    async def catalog(self) -> list[dict[str, Any]]:
        return await self.repo.catalog()
