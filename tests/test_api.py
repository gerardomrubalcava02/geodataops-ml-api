from fastapi.testclient import TestClient

from geodataops_ml.api.app import create_app
from tests.conftest import make_settings


def test_health(client):
    r = client.get("/api/v1/salud")
    assert r.status_code == 200
    body = r.json()
    assert body["estado"] == "ok" and body["modelo_version"]
    assert body["fuente_datos"]["tipo"] == "csv" and body["fuente_datos"]["sintetico"] is True
    assert body["llm_configurado"] is False


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/api/v1/observaciones", "/api/v1/predicciones", "/api/v1/chat/stream", "/api/v1/modelos/actual"]:
        assert p in paths


def test_catalog_and_municipios(client):
    cat = {c["indicador"]: c for c in client.get("/api/v1/indicadores").json()}
    assert cat["poblacion_total"]["unidad"] == "personas" and cat["poblacion_total"]["municipios"] == 4
    mun = client.get("/api/v1/municipios").json()
    assert [m["cvegeo"] for m in mun] == ["99001", "99002", "99003", "99004"]
    assert mun[0]["periodo_min"] == 2016 and mun[0]["periodo_max"] == 2021


def test_observations_filters_and_pagination(client):
    r = client.get("/api/v1/observaciones", params={
        "cvegeo": "99002", "indicador": ["poblacion_total", "tasa_ocupacion"],
        "periodo_desde": 2017, "periodo_hasta": 2019, "limit": 4, "offset": 0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 6 and len(body["items"]) == 4
    assert {i["indicador"] for i in body["items"]} <= {"poblacion_total", "tasa_ocupacion"}
    assert all(2017 <= i["periodo"] <= 2019 for i in body["items"])
    # offset más allá del final: total se conserva (paridad con PostgreSQL)
    beyond = client.get("/api/v1/observaciones", params={"cvegeo": "99002", "offset": 10_000}).json()
    assert beyond["items"] == [] and beyond["total"] > 0


def test_validation_errors_are_structured(client):
    r = client.get("/api/v1/observaciones", params={"cvegeo": "123", "limit": 0})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["codigo"] == "validacion"
    assert {d["campo"] for d in err["detalle"]} == {"query.cvegeo", "query.limit"}


def test_predict_by_municipio_from_training_data(client):
    r = client.post("/api/v1/predicciones", json={"cvegeo": "99005", "periodo": 2015})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["periodo_objetivo"] == 2016 and b["datos_entrenamiento_sinteticos"] is True
    assert b["advertencia"] and abs(b["crecimiento_esperado"]) < 0.2
    assert b["poblacion_estimada"] == round(b["poblacion_base"] * (1 + b["crecimiento_esperado"]))


def test_predict_from_operational_source(client):
    r = client.post("/api/v1/predicciones", json={"cvegeo": "99001", "periodo": 2020, "fuente_datos": "operativa"})
    assert r.status_code == 200, r.text


def test_predict_by_features(client):
    feats = {"log_poblacion": 10.5, "crecimiento": 0.01, "crecimiento_lag1": 0.012, "razon_masculinidad": 96,
             "tasa_ocupacion": 58, "escolaridad_promedio": 9.5, "hogares_con_internet": 55}
    r = client.post("/api/v1/predicciones", json={"caracteristicas": feats})
    assert r.status_code == 200
    assert r.json()["cvegeo"] is None


def test_predict_errors(client):
    # modos mezclados / incompletos
    assert client.post("/api/v1/predicciones", json={"cvegeo": "99001"}).status_code == 422
    assert client.post("/api/v1/predicciones", json={}).status_code == 422
    # rangos físicamente absurdos
    bad = client.post("/api/v1/predicciones", json={"caracteristicas": {"log_poblacion": 99}})
    assert bad.status_code == 422
    # municipio inexistente
    r = client.post("/api/v1/predicciones", json={"cvegeo": "01001", "periodo": 2015})
    assert r.status_code == 404 and r.json()["error"]["codigo"] == "no_encontrado"
    # historial insuficiente (primer año: no hay t-1 ni t-2)
    r = client.post("/api/v1/predicciones", json={"cvegeo": "99001", "periodo": 2016, "fuente_datos": "operativa"})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["codigo"] == "datos_insuficientes" and "crecimiento" in err["detalle"]["faltantes"]


def test_real_etl_census_sample_cannot_feed_model(data_dir, trained_model_dir, tmp_path):
    """La muestra censal (2010 y 2020, solo población) no permite predecir: 422 explícito, no un número inventado."""
    csv = tmp_path / "censo.csv"
    csv.write_text(
        "cvegeo,periodo,indicador,valor,unidad,fuente\n"
        "14039,2010,poblacion_total,1495189,personas,INEGI\n"
        "14039,2020,poblacion_total,1385629,personas,INEGI\n", "utf-8")
    s = make_settings(data_dir, trained_model_dir, csv_path=csv)
    with TestClient(create_app(s)) as c:
        r = c.post("/api/v1/predicciones", json={"cvegeo": "14039", "periodo": 2020, "fuente_datos": "operativa"})
    assert r.status_code == 422


def test_model_endpoints(client):
    actual = client.get("/api/v1/modelos/actual").json()
    assert actual["version"] and actual["prueba"]
    assert client.get(f"/api/v1/modelos/{actual['version']}").status_code == 200
    assert client.get("/api/v1/modelos/..%2F..").status_code == 404
    assert client.get("/api/v1/modelos").json()[0]["version"] == actual["version"]


def test_no_model_returns_503(data_dir, tmp_path):
    with TestClient(create_app(make_settings(data_dir, tmp_path))) as c:
        assert c.get("/api/v1/salud").json()["estado"] == "degradado"
        r = c.post("/api/v1/predicciones", json={"cvegeo": "99001", "periodo": 2015})
        assert r.status_code == 503 and r.json()["error"]["codigo"] == "modelo_no_disponible"


def test_train_endpoint_requires_token_when_configured(data_dir, tmp_path):
    s = make_settings(data_dir, tmp_path, admin_token="s3cret")
    with TestClient(create_app(s)) as c:
        assert c.post("/api/v1/modelos/entrenar").status_code == 401
        assert c.post("/api/v1/modelos/entrenar", headers={"X-Admin-Token": "nope"}).status_code == 401
        r = c.post("/api/v1/modelos/entrenar", headers={"X-Admin-Token": "s3cret"})
        assert r.status_code == 201
        version = r.json()["version"]
        assert c.get("/api/v1/salud").json()["modelo_version"] == version
        assert c.post("/api/v1/predicciones", json={"cvegeo": "99001", "periodo": 2010}).status_code == 200


def test_auto_source_falls_back_to_fixtures(data_dir, trained_model_dir, tmp_path):
    s = make_settings(data_dir, trained_model_dir, data_source="auto",
                      etl_csv_path=tmp_path / "no_existe.csv", fixture_csv_path=data_dir / "fixture.csv")
    with TestClient(create_app(s)) as c:
        assert c.get("/api/v1/salud").json()["fuente_datos"]["tipo"] == "fixtures"


def test_auto_source_falls_back_when_etl_breaks_contract(data_dir, trained_model_dir, tmp_path):
    broken = tmp_path / "etl.csv"
    broken.write_text("cvegeo,valor\n1,2\n", "utf-8")
    s = make_settings(data_dir, trained_model_dir, data_source="auto",
                      etl_csv_path=broken, fixture_csv_path=data_dir / "fixture.csv")
    with TestClient(create_app(s)) as c:
        assert c.get("/api/v1/salud").json()["fuente_datos"]["tipo"] == "fixtures"


def test_csv_source_reloads_when_producer_replaces_file(data_dir, trained_model_dir, tmp_path):
    """Revisión ETL #4: una nueva publicación del CSV se sirve sin reiniciar; una inválida no."""
    header = "cvegeo,periodo,indicador,valor,unidad,fuente\n"
    csv = tmp_path / "etl.csv"
    csv.write_text(header + "14039,2010,poblacion_total,1495189,personas,INEGI\n", "utf-8")
    s = make_settings(data_dir, trained_model_dir, data_source="etl_csv", etl_csv_path=csv)
    with TestClient(create_app(s)) as c:
        assert c.get("/api/v1/observaciones").json()["total"] == 1
        tmp = tmp_path / "etl.tmp"
        tmp.write_text(header + "14039,2010,poblacion_total,1495189,personas,INEGI\n"
                       "14039,2020,poblacion_total,1385629,personas,INEGI\n", "utf-8")
        tmp.replace(csv)  # reemplazo atómico, como el ETL
        assert c.get("/api/v1/observaciones").json()["total"] == 2
        csv.write_text("cvegeo,valor\nroto,1\n", "utf-8")
        assert c.get("/api/v1/observaciones").json()["total"] == 2  # se conserva la versión válida
        extra = c.get("/api/v1/salud").json()["fuente_datos"]["extra"]
        assert "cabecera" in extra["error_recarga"]
