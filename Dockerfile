# GeoDataOps ML API: reproducible image built from uv.lock.
#
# The model is trained at build time from the versioned synthetic panel
# (data/synthetic/panel_sintetico.csv, fixed seeds), so the image is self-contained
# and does not depend on the state of the build machine.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 app
WORKDIR /app

# 1) Dependencies only (cached layer): exact versions from uv.lock, no dev group, no UI extra.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Project code and versioned data.
COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev && mkdir -p artifacts/models && chown -R app:app /app

USER app
ENV PATH="/app/.venv/bin:$PATH" \
    GEODATAOPS_ML_DATA_SOURCE=auto \
    GEODATAOPS_ML_ETL_CSV_PATH=/data/etl/municipios_indicadores.csv \
    GEODATAOPS_ML_FIXTURE_CSV_PATH=/app/data/fixtures/municipios_indicadores_fixture.csv \
    GEODATAOPS_ML_TRAINING_CSV_PATH=/app/data/synthetic/panel_sintetico.csv \
    GEODATAOPS_ML_MODEL_DIR=/app/artifacts/models

# 3) Train and register the model inside the image (about 10 s).
RUN geodataops-ml train > /app/artifacts/models/train_summary.json

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/salud', timeout=2).status == 200 else 1)"

CMD ["geodataops-ml", "serve", "--host", "0.0.0.0", "--port", "8000"]
