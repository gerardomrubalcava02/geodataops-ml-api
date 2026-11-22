"""Integración con el esquema real del ETL en PostgreSQL/PostGIS.

Se ejecuta solo si GEODATAOPS_ML_TEST_DATABASE_URL apunta a un servidor con PostGIS donde
se puedan crear bases. Crea una base temporal, aplica `geodataops-etl/src/geodataops/schema.sql`
(lectura, sin modificar el repositorio del ETL), carga su CSV publicado y sus geometrías y
compara PostgresRepository contra el repositorio CSV.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from geodataops_ml.config import DEFAULT_ETL_CSV, WORKSPACE_ROOT
from geodataops_ml.data.contract import load_csv
from geodataops_ml.data.repository import CsvRepository, ObservationQuery, PostgresRepository

DSN = os.environ.get("GEODATAOPS_ML_TEST_DATABASE_URL")
ETL = WORKSPACE_ROOT / "geodataops-etl"
SCHEMA = ETL / "src" / "geodataops" / "schema.sql"
GEOJSON = ETL / "data" / "jalisco" / "municipios.geojson"

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DSN, reason="GEODATAOPS_ML_TEST_DATABASE_URL no definida"),
    pytest.mark.skipif(not (SCHEMA.exists() and DEFAULT_ETL_CSV.exists() and GEOJSON.exists()),
                       reason="artefactos del ETL no disponibles"),
]


@pytest.fixture(scope="module")
def etl_database():
    import psycopg
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    dbname = f"gdo_ml_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{dbname}"')
    dsn = make_conninfo(**{**conninfo_to_dict(DSN), "dbname": dbname})
    try:
        ds = load_csv(DEFAULT_ETL_CSV)
        features = json.loads(GEOJSON.read_text("utf-8"))["features"]
        run_id = uuid.uuid4()
        with psycopg.connect(dsn) as conn:
            conn.execute(SCHEMA.read_text("utf-8"))
            conn.execute(
                "INSERT INTO raw.etl_runs (run_id, dataset, status, finished_at) "
                "VALUES (%s, 'jalisco', 'succeeded', clock_timestamp())", (run_id,))
            for f in features:
                conn.execute(
                    "INSERT INTO staging.municipalities (dataset, cvegeo, name, geometry, last_run_id) VALUES "
                    "('jalisco', %s, %s, ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)), %s)",
                    (f["properties"]["cvegeo"], f["properties"]["nomgeo"], json.dumps(f["geometry"]), run_id))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO staging.indicators (dataset, cvegeo, periodo, indicador, valor, unidad, fuente,"
                    " last_run_id) VALUES ('jalisco', %s, %s, %s, %s, %s, %s, %s)",
                    [(*r, run_id) for r in ds.frame.itertuples(index=False)])
        yield dsn
    finally:
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{dbname}" WITH (FORCE)')


async def test_postgres_matches_csv(etl_database):
    pg = await PostgresRepository.open(etl_database, "jalisco")
    csv = CsvRepository(DEFAULT_ETL_CSV, kind="etl_csv")
    try:
        assert pg.info.is_synthetic is False and pg.info.extra["filas"] == csv.info.extra["filas"]
        for q in [
            ObservationQuery(),
            ObservationQuery(cvegeo="14039", indicadores=("poblacion_total",)),
            ObservationQuery(periodo_desde=2020, limit=4, offset=2),
            ObservationQuery(offset=10_000),  # página vacía: el total se conserva
        ]:
            assert await pg.observations(q) == await csv.observations(q), q
        pg_cat = {c["indicador"]: c for c in await pg.catalog()}
        csv_cat = {c["indicador"]: c for c in await csv.catalog()}
        assert pg_cat == csv_cat
        pg_mun = await pg.municipios()
        assert [m["cvegeo"] for m in pg_mun] == [m["cvegeo"] for m in await csv.municipios()]
        assert all(m["nombre"] for m in pg_mun)  # PostgreSQL aporta el nombre del municipio
        frame = await pg.frame()
        assert frame.equals((await csv.frame()).reset_index(drop=True))
    finally:
        await pg.close()


async def test_unknown_dataset_is_rejected(etl_database):
    with pytest.raises(LookupError):
        await PostgresRepository.open(etl_database, "no_existe")
