# GeoDataOps ML API

[![CI](https://github.com/gerardomrubalcava02/geodataops-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/gerardomrubalcava02/geodataops-ml-api/actions/workflows/ci.yml)

*[Versión en español](README.es.md)*

Backend service of the **GeoDataOps** platform. It serves Mexican municipal indicators produced by a separate ETL project, trains and serves a population-growth model with scikit-learn, and exposes a tool-calling LLM assistant whose answers come only from the data the API serves.

Stack: Python 3.12 · FastAPI · Pydantic v2 · scikit-learn · pandas · psycopg 3 (async) · Anthropic SDK · Streamlit (demo UI) · uv · Docker · GitHub Actions.

This repository was built alongside a sibling project, `geodataops-etl`, which loads the data into PostgreSQL/PostGIS. The two projects meet at a file contract (CSV v1: `cvegeo,periodo,indicador,valor,unidad,fuente`) and at the SQL view `analytics.municipal_indicators`.

## What is real, what is synthetic, what is simulated

The point of this project is the engineering. The table below states plainly what each part rests on.

| | What it covers | Evidence |
|---|---|---|
| **Real** | Reading and strictly validating the ETL's CSV (contract v1) and its PostgreSQL view. The real ETL sample (3 Jalisco municipalities, 2010 and 2020 censuses, population only) is served by the API. | Unit tests in CI; container smoke test in CI with the real sample mounted |
| **Real, by design rejected** | That sample **cannot** train or feed the model: it has 2 census years 10 years apart and no covariates. Predictions on it return **422 `datos_insuficientes`**, listing the missing features. | Test `test_real_etl_census_sample_cannot_feed_model`; container smoke test |
| **Synthetic** | The model is trained on a **synthetic panel** (120 fake municipalities with keys `99xxx`, 2005-2024) from a known generating process (`src/geodataops_ml/data/synthetic.py`). Its metrics show the technical workflow works. **They are not evidence of predictive skill on real population.** Every prediction carries `datos_entrenamiento_sinteticos: true` and a warning. | `artifacts/models/*/metadata.json`; container smoke test |
| **Simulated** | The LLM assistant was tested **only with a scripted provider**. The tool loop, argument validation, SSE events, timeouts and limits are tested. **No call to the real Anthropic API has been made or verified.** | `tests/test_llm.py` |
| **Verified locally only** | PostgreSQL parity, meaning the same results from SQL and CSV against the ETL's real `schema.sql`, plus the API running in `postgres` mode. Checked once on the development machine with portable PostgreSQL 17.12 + PostGIS 3.7.0 binaries. The 2 `postgres` tests are **not run in CI**; that belongs to the next iteration (ETL-API integration). | `tests/test_postgres.py`, run manually (see [Tests](#tests-and-ci)) |

## Architecture

```
            geodataops-etl ──► CSV contract v1 ─────┐        PostgreSQL/PostGIS
                                                    │   analytics.municipal_indicators
                                                    ▼              │ (psycopg async)
┌──────────────────────────────────────────────────────────────────▼───────────┐
│ data/     contract.py    strict v1 validation, row-level errors              │
│           repository.py  CsvRepository (reloads on publish) · Postgres...    │
│           sources.py     auto | etl_csv | csv | fixtures | postgres          │
│           synthetic.py   demonstration panel generator                       │
│ ml/       features.py (leak-free) · train.py (temporal CV, baselines)        │
│           registry.py (versioned joblib + metadata.json) · predict.py        │
│ services.py  GeoService — one data path shared by REST and the LLM tools     │
│ llm/      provider.py (LLMProvider, AnthropicProvider) · tools.py · agent.py │
│ api/      FastAPI routes · chat (JSON + Server-Sent Events) · errors         │
└──────────────────────────────────────────┬───────────────────────────────────┘
                                           │ HTTP
                                  ui/streamlit_app.py (demo)
```

Main design choices:
- **One protocol over two backends.** CSV and PostgreSQL sit behind a single `IndicatorRepository` protocol, so the API starts on bundled fixtures and does not wait for the ETL.
- **One service layer.** REST and the LLM tools both go through `GeoService`, so the assistant sees exactly the data and errors the API returns.
- **Async only where there is I/O.** PostgreSQL, the LLM and SSE run async. CPU-bound training and prediction run in `to_thread`, and training is serialized by a lock that returns 409 while a run is in progress.
- **The served model is the evaluated model.** The registry stores the estimator fitted on the training split, so the reported test metrics describe the served artifact.

## Machine learning

- **Target.** Population growth from year t to t+1.
- **Features.** Only information from t, t-1 and t-2:
  - log population
  - current growth and lagged growth
  - sex ratio
  - three covariates
- **Leakage controls.**
  - Lags are joined on the key `(cvegeo, periodo-k)`, never with positional `shift()`, so a missing year yields NaN instead of mixing periods.
  - `periodo` is not used as a feature.
  - Scaling happens inside the Pipeline.
  - The last 3 target years are a temporal holdout.
  - Model selection uses expanding-window CV by year, on the training years only.
  - Tests check each of these properties.
- **Baselines.** Training mean, and persistence (g[t+1] = g[t]).
- **Candidates.** Ridge and HistGradientBoosting. Ridge was selected by CV MAE.
- **Reproducibility.** The panel and the trained model can be regenerated exactly. Regenerating the committed panel gives byte-identical files, and retraining gives the same data hash (`5c4f7e89`) and the same metrics.

Results on the **synthetic** panel (seed 2026; test years 2022-2024, 360 rows):

| Model | Test MAE (growth) | RMSE | R² |
|---|---|---|---|
| Training mean | 0.0169 | 0.0209 | −0.70 |
| Persistence | 0.00438 | 0.00542 | 0.885 |
| **Ridge (served)** | **0.00402** | **0.00498** | **0.903** |
| HistGradientBoosting | 0.00429 | 0.00536 | 0.888 |

How to read these numbers:
- Ridge beats the strongest baseline by 8.4%.
- The generator's irreducible noise puts the floor at about 0.0032 MAE.
- The generator's non-linear term is not recovered: boosting did not beat Ridge with 1,680 training rows. That is why the simpler model is served.

## Configuration

All settings are environment variables with the `GEODATAOPS_ML_` prefix; a `.env` file is also read.

| Variable | Default (local) | Meaning |
|---|---|---|
| `DATA_SOURCE` | `auto` | One of `auto`, `etl_csv`, `csv` (with `CSV_PATH`), `fixtures`, `postgres` |
| `ETL_CSV_PATH` | `../geodataops-etl/output/municipios_indicadores.csv` | ETL output read by `auto` / `etl_csv` |
| `FIXTURE_CSV_PATH` | `data/fixtures/municipios_indicadores_fixture.csv` | Synthetic fixtures (fallback) |
| `DATABASE_URL`, `POSTGRES_DATASET` | none, `jalisco` | libpq DSN and dataset label for `postgres` mode |
| `TRAINING_CSV_PATH` | `data/synthetic/panel_sintetico.csv` | Training data |
| `MODEL_DIR` | `artifacts/models` | Model registry |
| `ADMIN_TOKEN` | none | If set, `POST /modelos/entrenar` requires `X-Admin-Token` |
| `LLM_PROVIDER` | `none` | `anthropic` enables the assistant (paid API; not verified, see above) |
| `LLM_MODEL`, `LLM_EFFORT` | `claude-opus-5-5`, `low` | Model and effort |
| `LLM_TOTAL_TIMEOUT_S`, `LLM_REQUEST_TIMEOUT_S`, `LLM_TOOL_TIMEOUT_S`, `LLM_MAX_TOOL_ROUNDS` | `120`, `60`, `10`, `5` | Assistant limits |

**Data sources.**
- **`auto`.** Uses the ETL CSV if it exists and passes the contract, otherwise the fixtures. The choice is made **at startup**.
- **CSV modes.** The file is reloaded when its mtime or size changes. If a new version breaks the contract, the previous one is kept and `/salud` shows `fuente_datos.extra.error_recarga`.
- **`postgres`.** Every request reads current data from `analytics.municipal_indicators`.

## Run locally

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 if needed.

```bash
git clone https://github.com/gerardomrubalcava02/geodataops-ml-api.git && cd geodataops-ml-api
uv sync --frozen
uv run geodataops-ml train          # ~10 s; models are not committed
uv run geodataops-ml serve --port 8000
```

`uv run geodataops-ml train` is required because models are not committed. Without it, the API still starts: `/api/v1/salud` reports `"estado": "degradado"` and `"modelo_version": null`, and predictions return `503 modelo_no_disponible` with the command to run.

- OpenAPI docs: <http://127.0.0.1:8000/docs>
- Demo UI:

```bash
uv sync --frozen --extra ui
uv run streamlit run ui/streamlit_app.py
```

- Validate an ETL file against the contract:

```bash
uv run geodataops-ml validate path/to/municipios_indicadores.csv
```

On Windows, `serve` uses a `SelectorEventLoop` because async psycopg does not support the default Proactor loop. On Linux and macOS the default asyncio loop is used.

## Run with Docker

The image installs exact dependencies from `uv.lock` (`uv sync --frozen --no-dev`), runs as a non-root user (uid 10001) and listens on `0.0.0.0:8000`. **The model is trained during `docker build`** from the committed synthetic panel, so the image works without any volume or local state.

```bash
docker build -t geodataops-ml-api .
docker run --rm -p 8000:8000 geodataops-ml-api
```

There is no `../geodataops-etl` inside the container. The image sets `GEODATAOPS_ML_ETL_CSV_PATH=/data/etl/municipios_indicadores.csv`, so `auto` mode behaves as follows:

| Source | How |
|---|---|
| Fixtures (default) | Nothing mounted: `auto` falls back to the bundled synthetic fixtures |
| ETL CSV | Mount the file: `-v /path/to/municipios_indicadores.csv:/data/etl/municipios_indicadores.csv:ro` |
| PostgreSQL | `-e GEODATAOPS_ML_DATA_SOURCE=postgres -e GEODATAOPS_ML_DATABASE_URL="host=... dbname=... user=... password=..."` |

## Example requests and responses

These are the actual responses from `scripts/smoke_http.py` against the API with the real ETL sample mounted, trimmed for length. The same script runs against the container in CI.

**`GET /api/v1/salud`**, model loaded:

```json
{"estado": "ok", "version": "0.1.0",
 "fuente_datos": {"tipo": "etl_csv", "sintetico": false,
                  "sha256": "462082b0cc342a345ded87ce6532f2b5d0245eedc7a6d8c5e09ad88e0d8dbc9e", "extra": {"filas": 18}},
 "modelo_version": "20261122T025251Z-5c4f7e89", "llm_proveedor": "none", "llm_configurado": false}
```

**`GET /api/v1/observaciones?cvegeo=14039&indicador=poblacion_total`**, real census data:

```json
{"total": 2, "limit": 100, "offset": 0,
 "items": [
  {"cvegeo": "14039", "periodo": 2010, "indicador": "poblacion_total", "valor": 1495189.0, "unidad": "personas",
   "fuente": "IIEG con datos de INEGI (Censo de Población y Vivienda 2010)"},
  {"cvegeo": "14039", "periodo": 2020, "indicador": "poblacion_total", "valor": 1385629.0, "unidad": "personas",
   "fuente": "IIEG con datos de INEGI (Censo de Población y Vivienda 2020)"}]}
```

**`POST /api/v1/predicciones` `{"cvegeo": "99007", "periodo": 2023}`**, synthetic training panel:

```json
{"modelo_version": "20261122T025251Z-5c4f7e89", "algoritmo": "ridge",
 "datos_entrenamiento_sinteticos": true,
 "crecimiento_esperado": 0.0505, "periodo_base": 2023, "periodo_objetivo": 2024,
 "poblacion_base": 147115, "poblacion_estimada": 154543,
 "advertencia": "Modelo entrenado con datos demostrativos SINTÉTICOS: la predicción ilustra el flujo de ML y no es una proyección demográfica real."}
```

**`POST /api/v1/predicciones` `{"cvegeo": "14039", "periodo": 2020, "fuente_datos": "operativa"}`**, real sample, rejected:

```json
HTTP 422
{"error": {"codigo": "datos_insuficientes",
  "mensaje": "14039/2020: faltan datos para construir características: crecimiento, crecimiento_lag1, tasa_ocupacion, escolaridad_promedio, hogares_con_internet",
  "detalle": {"faltantes": ["crecimiento", "crecimiento_lag1", "tasa_ocupacion", "escolaridad_promedio", "hogares_con_internet"],
              "requisito": "poblaciones de t, t-1, t-2 y covariables de t"}}}
```

API field names and messages are in Spanish; the API's contract is kept as is. Every error has the shape `{"error": {"codigo", "mensaje", "detalle"}}`.

### Endpoints (`/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/salud` | Health, data source (type, synthetic flag, sha256), model version, LLM status |
| GET | `/municipios`, `/indicadores` | Coverage and catalog |
| GET | `/observaciones` | Filters `cvegeo`, `indicador` (repeatable), `periodo_desde`, `periodo_hasta`, `limit` ≤ 1000, `offset` |
| GET | `/modelos`, `/modelos/actual`, `/modelos/{version}` | Model versions, metrics and metadata |
| POST | `/modelos/entrenar` | Train, register and activate a model (401 without token if configured; 409 if already training) |
| POST | `/predicciones` | `{cvegeo, periodo, fuente_datos}` or `{caracteristicas}` |
| POST | `/chat`, `/chat/stream` | Assistant (JSON or SSE); 503 when not configured |

## Tests and CI

```bash
uv run ruff check .
uv run pytest -q -m "not postgres"
```

The test suite covers:
- the contract and its domain rules
- leakage, gap years and temporal splits
- the registry
- every endpoint and error path
- source fallback and CSV reload
- the LLM loop with a scripted provider

**CI** (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request:
1. **Lint and tests.** `uv sync --frozen`, `ruff check`, then `pytest -m "not postgres"`. The 2 PostgreSQL tests are **deselected, not passed**.
2. **Container.** Builds the image, checks it runs as non-root, then starts two containers: default fixtures, and the real ETL sample mounted from `tests/data/`. It runs `scripts/smoke_http.py` against each over real HTTP and waits for the Docker health check.

First verified CI run, on pull request #1: [run 45054174773](https://github.com/gerardomrubalcava02/geodataops-ml-api/actions/runs/45054174773).
- Lint: ruff clean. Tests: 57 passed, 3 skipped (need the sibling ETL repo), 2 deselected (`postgres`).
- Container: built on CPython 3.12.16 and runs as uid 10001. Docker health status reaches `healthy`.
- Smoke test: passed in fixtures mode and with the real ETL sample, including the 422.
- Cross-platform reproducibility: the model trained inside the image matched the local Windows run, with the same data hash `5c4f7e89` and the same test MAE.

PostgreSQL integration tests (manual, not in CI) need a PostGIS server where the user can create databases. They also need the sibling `geodataops-etl` checkout for `schema.sql`.

```bash
GEODATAOPS_ML_TEST_DATABASE_URL="host=127.0.0.1 port=5432 user=postgres dbname=postgres" uv run pytest -q tests/test_postgres.py
```

## Limitations

- The model is trained on synthetic data. With the real ETL sample, predictions return 422 on purpose.
- A model on real data is **not** a matter of waiting for more ETL output. It requires reframing the problem around census data, and that decision is still open. Open questions:
  - territorial comparability of municipalities between censuses
  - a leak-free baseline
  - spatial validation versus temporal forecasting
- The real Anthropic integration has not been verified (no key, no authorization to use a paid service). To try it: `GEODATAOPS_ML_LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...`.
- There are no user authentication, rate limits or CORS policy. A public deployment would need them.
- The CSV repository filters in memory, which is fine for thousands of rows. Use `postgres` mode for larger data.
- `joblib` artifacts are pickles. They are only loaded from the local registry and never accepted from clients.

## Collaboration log

Cross-reviews with the ETL agent are in `../geodataops-sync/` (`revisiones_para_etl.md`, `revisiones_para_ml.md`, `estado_ml.md`).

- Applied from the ETL review:
  - one-pass read for the hash and the data
  - pagination totals in SQL
  - domain validation before type conversion
  - CSV reload when the ETL publishes a new file
- Proposed to the ETL:
  - a population-triple check that is skipped when `fuente` changes
  - a Windows `replace` retry
  - reserving entity code `99` for synthetic data
