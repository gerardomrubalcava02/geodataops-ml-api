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
    # "none" desactiva /chat (503). "anthropic" usa el SDK oficial; la API key se toma de
    # GEODATAOPS_ML_LLM_API_KEY o, si no se define, de las credenciales estándar del SDK.
    llm_provider: Literal["none", "anthropic"] = "none"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high"] = "low"
    llm_api_key: SecretStr | None = None
    llm_request_timeout_s: float = Field(default=60.0, gt=0, le=600)
    llm_total_timeout_s: float = Field(default=120.0, gt=0, le=900)
    llm_tool_timeout_s: float = Field(default=10.0, gt=0, le=120)
    llm_max_tool_rounds: int = Field(default=5, ge=1, le=10)
    llm_max_tokens: int = Field(default=16000, ge=256, le=64000)
    llm_server_fallbacks: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
