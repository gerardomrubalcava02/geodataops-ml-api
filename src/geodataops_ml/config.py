"""Configuración centralizada. Todas las variables usan el prefijo GEODATAOPS_ML_."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_ETL_CSV = WORKSPACE_ROOT / "geodataops-etl" / "output" / "municipios_indicadores.csv"
DEFAULT_FIXTURE_CSV = PROJECT_ROOT / "data" / "fixtures" / "municipios_indicadores_fixture.csv"
DEFAULT_TRAINING_CSV = PROJECT_ROOT / "data" / "synthetic" / "panel_sintetico.csv"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEODATAOPS_ML_", env_file=".env", extra="ignore")

    # Fuente de indicadores servidos por la API.
    # auto: CSV del ETL si existe; si no, fixtures propios.
    data_source: Literal["auto", "etl_csv", "fixtures", "csv", "postgres"] = "auto"
    csv_path: Path | None = None
    etl_csv_path: Path = DEFAULT_ETL_CSV
    fixture_csv_path: Path = DEFAULT_FIXTURE_CSV
    database_url: SecretStr | None = None
    postgres_dataset: str = "jalisco"

    # Machine learning.
    training_csv_path: Path = DEFAULT_TRAINING_CSV
    model_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    admin_token: SecretStr | None = Field(
        default=None, description="Si se define, POST /modelos/entrenar exige X-Admin-Token."
    )

    # LLM.
    llm_provider: Literal["none", "anthropic", "openai_compatible"] = "none"
    llm_model: str = "claude-sonnet-5-5"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_timeout_s: float = Field(default=30.0, gt=0, le=300)
    llm_max_tool_rounds: int = Field(default=4, ge=1, le=10)
    llm_max_tokens: int = Field(default=1024, ge=64, le=8192)


@lru_cache
def get_settings() -> Settings:
    return Settings()
