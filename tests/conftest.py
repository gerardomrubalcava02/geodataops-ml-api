from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geodataops_ml.api.app import create_app
from geodataops_ml.config import Settings
from geodataops_ml.data.contract import load_csv, write_csv
from geodataops_ml.data.synthetic import generate_panel
from geodataops_ml.ml.registry import ModelRegistry
from geodataops_ml.ml.train import TrainConfig, train


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("data")
    write_csv(generate_panel(n_municipios=40, periodo_inicio=2005, periodo_fin=2016, seed=11), d / "panel.csv")
    write_csv(generate_panel(n_municipios=4, periodo_inicio=2016, periodo_fin=2021, seed=3), d / "fixture.csv")
    return d


@pytest.fixture(scope="session")
def trained_model_dir(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    model_dir = tmp_path_factory.mktemp("models")
    ds = load_csv(data_dir / "panel.csv")
    ModelRegistry(model_dir).save(train(ds.frame, TrainConfig(test_periodos=3)), ds)
    return model_dir


def make_settings(data_dir: Path, model_dir: Path, **kw) -> Settings:
    base = dict(
        data_source="csv",
        csv_path=data_dir / "fixture.csv",
        training_csv_path=data_dir / "panel.csv",
        model_dir=model_dir,
        llm_provider="none",
    )
    return Settings(_env_file=None, **{**base, **kw})


@pytest.fixture
def settings(data_dir: Path, trained_model_dir: Path) -> Settings:
    return make_settings(data_dir, trained_model_dir)


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="session")
def event_loop_policy():
    """psycopg async requiere SelectorEventLoop en Windows (igual que `geodataops-ml serve`)."""
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()
