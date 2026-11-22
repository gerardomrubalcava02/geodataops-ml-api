"""Adaptador de fuentes: decide de dónde vienen los indicadores servidos por la API.

Orden en modo `auto`, evaluado AL ARRANCAR: salida del ETL (contrato v1) -> fixtures propios.
Así el backend arranca sin esperar al ETL. Una vez elegido un CSV, las nuevas publicaciones
de ese archivo se recargan solas; si el ETL publica por primera vez después de arrancar
en modo fixtures, hay que reiniciar la API (ver README).
"""

from __future__ import annotations

import logging

from geodataops_ml.config import Settings
from geodataops_ml.data.contract import ContractError
from geodataops_ml.data.repository import CsvRepository, IndicatorRepository, PostgresRepository

log = logging.getLogger(__name__)


async def open_repository(settings: Settings) -> IndicatorRepository:
    mode = settings.data_source

    if mode == "postgres":
        if settings.database_url is None:
            raise RuntimeError("data_source=postgres requiere GEODATAOPS_ML_DATABASE_URL")
        return await PostgresRepository.open(
            settings.database_url.get_secret_value(), settings.postgres_dataset
        )

    if mode == "csv":
        if settings.csv_path is None:
            raise RuntimeError("data_source=csv requiere GEODATAOPS_ML_CSV_PATH")
        return CsvRepository(settings.csv_path, kind="csv")

    if mode in ("auto", "etl_csv"):
        path = settings.etl_csv_path
        if path.exists():
            try:
                return CsvRepository(path, kind="etl_csv")
            except ContractError:
                if mode == "etl_csv":
                    raise
                log.exception("La salida del ETL no cumple el contrato; se usan fixtures")
        elif mode == "etl_csv":
            raise FileNotFoundError(path)
        else:
            log.warning("No existe %s; se usan fixtures propios", path)

    return CsvRepository(settings.fixture_csv_path, kind="fixtures")
