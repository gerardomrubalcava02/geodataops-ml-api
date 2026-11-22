# GeoDataOps ML API

> Versión en español (documentación de la fase 4). La referencia principal, actualizada en la iteración 0 con contenedor, CI y estado de verificación de cada afirmación, es el [README en inglés](README.md). Desde la iteración 0 el modelo no se versiona: ejecute `uv run geodataops-ml train` tras clonar, o use la imagen Docker, que lo entrena durante la construcción. Las 2 pruebas de PostgreSQL **no** se ejecutan en CI.

Backend de la plataforma **GeoDataOps**: consulta de indicadores municipales, entrenamiento y servicio de un modelo predictivo y un asistente LLM con *tool calling* que responde con los datos disponibles.

Se construyó en paralelo con [`geodataops-etl`](../geodataops-etl) (pipeline ETL y PostgreSQL/PostGIS). Ambos proyectos se integran mediante el contrato CSV v1 (`../geodataops-sync/contrato_datos.md`) y la vista `analytics.municipal_indicators`.

> **Aviso sobre los datos.** La muestra real que publica el ETL (Guadalajara, Tlaquepaque y Zapopan; censos 2010 y 2020; solo población) **no permite entrenar ni validar un modelo**. El modelo se entrena con un **panel DEMOSTRATIVO SINTÉTICO** (claves `99xxx`, fuente `DEMO sintético …`). Sus métricas demuestran el flujo de ML y **no describen población real**. La API, los metadatos del modelo y el asistente lo advierten en cada respuesta.

## Arquitectura

```
                 ┌──────────────── geodataops-etl ────────────────┐
                 │ output/municipios_indicadores.csv  (contrato v1)│
                 │ PostgreSQL/PostGIS analytics.municipal_indicators│
                 └───────────────┬─────────────────┬──────────────┘
                                 │ CSV (recarga)    │ SQL (psycopg async)
┌────────────────────────────────▼─────────────────▼──────────────────────────┐
│ data/   contract.py (validación v1) · repository.py (CsvRepository,         │
│         PostgresRepository) · sources.py (auto|etl_csv|csv|postgres)         │
│         synthetic.py (panel demostrativo)                                    │
│ ml/     features.py (sin fuga) · train.py (CV temporal, baselines)          │
│         registry.py (joblib + metadata.json versionados) · predict.py        │
│ services.py  GeoService: capa común para REST y herramientas del LLM        │
│ llm/    provider.py (LLMProvider · AnthropicProvider) · tools.py (Pydantic) │
│         agent.py (bucle de herramientas, timeouts, eventos)                  │
│ api/    FastAPI: routes.py · chat.py (JSON + SSE) · errors.py · schemas.py  │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ HTTP
                       ui/streamlit_app.py (demo)
```

Decisiones principales:

| Decisión | Motivo |
|---|---|
| Protocolo `IndicatorRepository` con implementaciones CSV y PostgreSQL | La API no depende de la fuente. Se arranca con fixtures sin esperar al ETL. |
| `GeoService` compartido por REST y el LLM | El asistente ve exactamente los mismos datos y errores que la API. No hay una segunda ruta de acceso a datos. |
| Async donde hay I/O (PostgreSQL, LLM, SSE) y `to_thread` para CPU (entrenar, predecir) | Entrenar o predecir no bloquea el event loop. El entrenamiento se serializa con un lock (409 si ya hay uno en curso). |
| Modelo persistido = modelo evaluado | Se guarda el ajuste con datos de entrenamiento, por lo que las métricas de prueba corresponden al artefacto servido. |
| SDK oficial de Anthropic tras una interfaz `LLMProvider` | El proveedor es configurable y sustituible. Las pruebas usan un proveedor guionizado sin red ni credenciales. |

## Instalación

Requisitos: [uv](https://docs.astral.sh/uv/). uv instala Python 3.12 si falta.

```powershell
cd geodataops-ml-api
uv sync --extra ui
```

```powershell
uv run geodataops-ml generate-synthetic
```

```powershell
uv run geodataops-ml train
```

```powershell
uv run geodataops-ml serve --port 8000
```

- Documentación OpenAPI: <http://127.0.0.1:8000/docs>
- Interfaz de demostración, en otra terminal:

```powershell
uv run streamlit run ui/streamlit_app.py
```

Validar un CSV del ETL contra el contrato:

```powershell
uv run geodataops-ml validate ..\geodataops-etl\output\municipios_indicadores.csv
```

### Configuración (variables `GEODATAOPS_ML_*` o archivo `.env`)

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `DATA_SOURCE` | `auto` | `auto` usa el CSV del ETL si existe y cumple el contrato; si no, los fixtures. También acepta `etl_csv`, `csv` (con `CSV_PATH`), `fixtures` o `postgres`. |
| `DATABASE_URL` / `POSTGRES_DATASET` | `None` / `jalisco` | DSN de libpq para el modo `postgres` |
| `TRAINING_CSV_PATH` | `data/synthetic/panel_sintetico.csv` | Conjunto de entrenamiento |
| `MODEL_DIR` | `artifacts/models` | Registro de modelos |
| `ADMIN_TOKEN` | `None` | Si se define, `POST /modelos/entrenar` exige `X-Admin-Token` |
| `LLM_PROVIDER` | `none` | `anthropic` activa el asistente |
| `LLM_API_KEY` | `None` | Si falta, el SDK usa `ANTHROPIC_API_KEY` o el perfil de `ant auth login` |
| `LLM_MODEL` / `LLM_EFFORT` | `claude-opus-5-5` / `low` | Modelo y esfuerzo |
| `LLM_TOTAL_TIMEOUT_S` / `LLM_REQUEST_TIMEOUT_S` / `LLM_TOOL_TIMEOUT_S` | `120` / `60` / `10` | Plazos |
| `LLM_MAX_TOOL_ROUNDS` | `5` | Rondas máximas de herramientas |

**Recarga de datos.** En modo CSV, el archivo se recarga cuando cambian su `mtime` o tamaño. Si la nueva versión incumple el contrato, se conserva la anterior y `/salud` muestra `fuente_datos.extra.error_recarga`. En modo `auto`, la elección entre ETL y fixtures se hace al arrancar: si el ETL publica por primera vez con la API en marcha, hay que reiniciarla. En modo `postgres`, cada consulta lee los datos vigentes.

## Endpoints (`/api/v1`)

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/salud` | Estado, fuente de datos (tipo, sintética, sha256), versión del modelo y estado del LLM |
| GET | `/municipios` · `/indicadores` | Cobertura y catálogo |
| GET | `/observaciones` | Filtros `cvegeo`, `indicador` (repetible), `periodo_desde`, `periodo_hasta`, `limit` ≤ 1000 y `offset` |
| GET | `/modelos` · `/modelos/actual` · `/modelos/{version}` | Versiones, métricas y metadatos |
| POST | `/modelos/entrenar` | Entrena, registra y activa una versión (201). 401 sin token y 409 si ya hay un entrenamiento en curso. |
| POST | `/predicciones` | `{cvegeo, periodo, fuente_datos}` o `{caracteristicas}` |
| POST | `/chat` · `/chat/stream` | Asistente (JSON o SSE). Responde 503 si no está configurado. |

Los errores siempre tienen la forma `{"error": {"codigo", "mensaje", "detalle"}}`. Códigos: `validacion` (422), `datos_insuficientes` (422), `no_encontrado` (404), `modelo_no_disponible` (503), `llm_no_configurado` (503), `tiempo_agotado` (504), `error_proveedor`/`limite_herramientas`/`rechazo_modelo` (502) y `error_interno` (500, sin traza).

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/api/v1/predicciones -H "content-type: application/json" -d '{\"cvegeo\":\"99007\",\"periodo\":2023}'
```

## Machine learning

**Pregunta.** Dado un municipio en el año t, ¿cuál será su crecimiento poblacional de t a t+1?

- **Características, solo t y anteriores:** `log_poblacion`, `crecimiento` (t-1→t), `crecimiento_lag1` (t-2→t-1), `razon_masculinidad`, `tasa_ocupacion`, `escolaridad_promedio` y `hogares_con_internet`.
- **Prevención de fugas:**
  - Los rezagos se unen por la clave `(cvegeo, periodo−k)`, no con `shift()`, así que un año faltante produce NaN en lugar de mezclar periodos. Hay una prueba que lo verifica.
  - `periodo` no se usa como característica.
  - El escalado se ajusta dentro del `Pipeline`.
  - Holdout temporal: los últimos 3 años objetivo no participan en la selección.
  - La CV es de ventana creciente por año: cada pliegue entrena solo con años anteriores.
  - Una prueba altera datos futuros y comprueba que las características de t no cambian.
- **Baselines:** media del entrenamiento y persistencia (g[t+1] = g[t]).
- **Candidatos:** `Ridge` (con `StandardScaler`) y `HistGradientBoostingRegressor`. Se selecciona por el MAE medio en CV.
- **Registro:** `artifacts/models/<version>/{model.joblib, metadata.json}`, con la versión `YYYYMMDDTHHMMSSZ-<sha8 de datos>`. Los metadatos incluyen la huella SHA-256 de los datos, la partición, la CV, las métricas de prueba de todos los modelos, la importancia por permutación y las versiones de Python y scikit-learn. `LATEST` se reemplaza de forma atómica. Solo se cargan versiones que cumplen el patrón, lo que evita rutas arbitrarias, y nunca se aceptan artefactos subidos por clientes (joblib usa pickle).

### Resultados (panel SINTÉTICO, semilla 2026; no son resultados de un estudio real)

120 municipios. Años objetivo 2008-2021 para entrenamiento (1,680 filas) y 2022-2024 para prueba (360 filas). Entrenamiento: unos 11 s.

| Modelo | CV MAE (media ± de) | Prueba MAE crecimiento | Prueba RMSE | Prueba R² | MAPE población |
|---|---|---|---|---|---|
| baseline_media | 0.0159 ± 0.0006 | 0.0169 | 0.0209 | −0.70 | 1.63 % |
| baseline_persistencia | 0.00435 ± 0.0003 | 0.00438 | 0.00542 | 0.885 | 0.43 % |
| **ridge (seleccionado)** | **0.00392 ± 0.0003** | **0.00402** | **0.00498** | **0.903** | **0.39 %** |
| hist_gradient_boosting | 0.00423 ± 0.0004 | 0.00429 | 0.00536 | 0.888 | 0.42 % |

Lectura honesta:

- Ridge mejora un **8.4 %** al mejor baseline (persistencia). La dinámica AR(1) del generador hace que la persistencia sea un rival fuerte.
- El ruido irreducible del generador (σ = 0.004) impone un piso de MAE de unos 0.0032. Ridge está cerca de ese piso.
- La importancia por permutación coloca el crecimiento actual muy por encima del resto. El término no lineal de `hogares_con_internet` no aporta: el boosting no logra explotarlo con 1,680 filas y este nivel de ruido. Por eso no se eligió un modelo más complejo.

## Asistente LLM

- `LLMProvider` define `stream_turn`, `assistant_message` y `tool_results_message`. `AnthropicProvider` usa `anthropic.AsyncAnthropic` con streaming y bucle de herramientas manual. Reenvía el `content` completo en cada turno y envía todos los resultados en un único mensaje. Usa `fallbacks="default"` (beta del servidor) y `effort` configurable. El SDK reintenta 429, 5xx y errores de red.
- Herramientas: `listar_indicadores`, `listar_municipios`, `consultar_indicadores`, `predecir_crecimiento` y `metricas_modelo`. El esquema JSON se deriva de modelos Pydantic con `extra="forbid"`. Antes de ejecutar se validan los argumentos. Un error de validación o de dominio vuelve al modelo como `is_error` para que se corrija, sin romper la conversación.
- Respuestas fundamentadas: el prompt de sistema exige que toda cifra provenga de una herramienta, que se cite la `fuente` y que se advierta cuando los datos son sintéticos. La respuesta final incluye la lista de herramientas usadas para auditarla.
- Controles: plazo total aplicado a cada espera, timeout por herramienta, límite de rondas, `refusal`, `max_tokens` con herramienta truncada y desconexión del cliente SSE.
- Eventos SSE: `tool_call`, `tool_result`, `text`, `done` y `error`.

**Verificación.** El bucle, la validación, los eventos SSE y los controles se prueban con un proveedor guionizado. **La llamada real a Claude no se verificó:** no hay API key en el entorno y no se usaron servicios de pago sin autorización. Para probarla:

```powershell
$env:GEODATAOPS_ML_LLM_PROVIDER = "anthropic"; $env:ANTHROPIC_API_KEY = "<su clave>"; uv run geodataops-ml serve
```

## Pruebas

```powershell
uv run pytest -q
```

Integración con PostgreSQL/PostGIS: usa el `schema.sql` del ETL en una base temporal. Requiere un servidor con PostGIS y permiso para crear bases:

```powershell
$env:GEODATAOPS_ML_TEST_DATABASE_URL = "host=127.0.0.1 port=55432 user=postgres dbname=postgres"; uv run pytest -q tests/test_postgres.py
```

Resultado actual: **61 aprobadas** (59 sin dependencias y 2 de PostgreSQL 17.12 + PostGIS 3.7.0). Ruff sin hallazgos.

| Archivo | Cubre |
|---|---|
| `test_contract.py` | Contrato v1: tipos, dominio (entidad, municipio 000, año, exponente, separadores, 2^53), BOM, duplicados, filas reportadas y compatibilidad con los archivos del ETL |
| `test_ml.py` | Ausencia de fuga, objetivo correcto, huecos de años, split y CV sin mirar al futuro, registro y versiones inseguras |
| `test_api.py` | Endpoints, filtros y paginación, errores estructurados, modos de predicción, 404/422/503, token de entrenamiento, fallback de fuente y recarga del CSV |
| `test_llm.py` | Esquemas desde Pydantic, datos idénticos a REST, argumentos inválidos, herramientas desconocidas, secuencia SSE, timeout total, límite de rondas, rechazo y truncamiento |
| `test_postgres.py` | Paridad PostgreSQL ↔ CSV sobre el esquema real del ETL |

**Nota para Windows.** psycopg async no funciona con el `ProactorEventLoop` predeterminado. `geodataops-ml serve` usa `SelectorEventLoop` y las pruebas fijan la misma política.

## Colaboración con el agente ETL

Resumen; el detalle está en `../geodataops-sync/`.

| Dirección | Recomendación | Resultado |
|---|---|---|
| ETL → ML | Calcular la huella y el DataFrame desde una sola lectura | Incorporada |
| ETL → ML | Conservar el total de paginación con un offset fuera de rango en PostgreSQL | Incorporada y verificada contra PostGIS |
| ETL → ML | Validar el dominio antes de convertir (`14000`, `0999`, `1e3`) | Incorporada, con límite de exactitud de 2^53 |
| ETL → ML | Hacer visibles las publicaciones posteriores al arranque | `CsvRepository` con recarga y documentación del modo `auto` |
| ML → ETL | Series anuales y covariables para modelar con datos reales | Retirada: un modelo con datos reales requiere replantear el problema con datos censales (ver Limitaciones) |
| ML → ETL | La validación de ternas se omite si cambia `fuente` (`_POPULATION_CONFLICT`) | Pendiente en ETL |
| ML → ETL | Reintentar `Path.replace` en Windows cuando hay lectores concurrentes | Pendiente en ETL |
| ML → ETL | Reservar la entidad `99` para datos sintéticos en el contrato | Pendiente en ETL |

## Competencias demostradas (para entrevistas)

- **Backend con Python:**
  - FastAPI con factory y lifespan, inyección de dependencias y Pydantic v2 para entrada, salida y configuración (`pydantic-settings`).
  - OpenAPI documentado y errores homogéneos.
  - Async real: psycopg async con pool, SSE y `to_thread` para trabajo de CPU.
  - Lock de concurrencia, `secrets.compare_digest`, protección contra *path traversal* en versiones y ausencia de credenciales en `/salud`.
- **Ingeniería de datos e integración:**
  - Contrato de datos validado de forma estricta con errores accionables.
  - Patrón repositorio y adaptador de fuentes, paridad CSV ↔ SQL probada contra el esquema real del otro equipo.
  - Recarga segura ante publicaciones atómicas.
- **Machine learning:**
  - Formulación de un objetivo sin fuga y validación temporal.
  - Baselines fuertes y selección por CV.
  - Interpretación honesta de las métricas (piso de ruido, por qué no usar un modelo más complejo), importancia por permutación y registro reproducible con huella de datos.
- **IA aplicada:**
  - Tool calling con validación de argumentos y respuestas fundamentadas.
  - Streaming SSE, timeouts, límites, manejo de rechazos y proveedor abstraído y testeable sin red.
- **Calidad:** 61 pruebas con pytest, entre ellas pruebas de propiedades de ML (fuga, huecos) y de integración real con PostgreSQL. Ruff, uv con lockfile, commits por fase y revisión cruzada documentada.

## Limitaciones conocidas

- El modelo se entrenó con datos sintéticos. Con la muestra real, `fuente_datos=operativa` responde 422 de forma intencional.
- Un modelo con datos reales **no** depende de que el ETL publique más datos. Requiere replantear el problema con datos censales y la decisión está pendiente. Preguntas abiertas:
  - comparabilidad territorial de los municipios entre censos
  - un baseline sin fuga de información
  - validación espacial frente a pronóstico temporal
- Llamada real a Claude no verificada (ver arriba).
- Sin autenticación de usuarios, salvo el token opcional para entrenar. Un despliegue público requeriría autenticación, límites de tasa y CORS.
- `FrameRepository` filtra en memoria. Es adecuado para miles de filas; para volúmenes grandes conviene el modo `postgres`.
