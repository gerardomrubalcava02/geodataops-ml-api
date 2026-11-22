"""Fábrica de la aplicación FastAPI."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from geodataops_ml import __version__
from geodataops_ml.api.chat import router as chat_router
from geodataops_ml.api.errors import install_error_handlers
from geodataops_ml.api.routes import router
from geodataops_ml.config import Settings, get_settings
from geodataops_ml.data.sources import open_repository
from geodataops_ml.llm.factory import build_agent
from geodataops_ml.llm.provider import LLMProvider
from geodataops_ml.ml.registry import ModelRegistry
from geodataops_ml.services import GeoService

DESCRIPTION = """
API de GeoDataOps para consultar indicadores municipales (contrato CSV v1 del ETL),
entrenar y servir un modelo de crecimiento poblacional y conversar con un asistente LLM
que responde con herramientas sobre los datos disponibles.

**Aviso:** el modelo por defecto se entrena con un panel DEMOSTRATIVO SINTÉTICO.
"""


def create_app(settings: Settings | None = None, llm_provider: LLMProvider | None = None) -> FastAPI:
    """`llm_provider` permite inyectar un proveedor (p. ej. guionizado en pruebas)."""
    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        repo = await open_repository(settings)
        service = GeoService(settings, repo, ModelRegistry(settings.model_dir))
        service.load_latest_model()
        app.state.service = service
        app.state.agent = build_agent(settings, service, llm_provider)
        try:
            yield
        finally:
            if app.state.agent is not None:
                await app.state.agent.aclose()
            await repo.close()

    app = FastAPI(
        title="GeoDataOps ML API", version=__version__, description=DESCRIPTION, lifespan=lifespan
    )
    install_error_handlers(app)
    app.include_router(router)
    app.include_router(chat_router)
    return app
