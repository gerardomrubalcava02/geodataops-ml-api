import numpy as np
import pandas as pd
import pytest

from geodataops_ml.data.contract import load_csv
from geodataops_ml.data.synthetic import generate_panel
from geodataops_ml.ml.features import (
    FEATURES,
    TARGET,
    InsufficientHistoryError,
    build_training_table,
    features_for,
)
from geodataops_ml.ml.registry import ModelNotFoundError, ModelRegistry
from geodataops_ml.ml.train import expanding_year_folds, temporal_split


@pytest.fixture(scope="module")
def panel():
    return generate_panel(n_municipios=6, periodo_inicio=2010, periodo_fin=2018, seed=5)


def test_features_do_not_use_future_data(panel):
    """Alterar datos de años posteriores a t no cambia las características de t (sin fuga)."""
    before = features_for(panel, "99001", 2014)
    tampered = panel.copy()
    future = tampered["periodo"] > 2014
    tampered.loc[future, "valor"] = tampered.loc[future, "valor"] * 3
    after = features_for(tampered, "99001", 2014)
    pd.testing.assert_frame_equal(before, after)


def test_target_is_next_year_growth(panel):
    table = build_training_table(panel)
    row = table[(table.cvegeo == "99002") & (table.periodo == 2013)].iloc[0]
    pop = panel[(panel.cvegeo == "99002") & (panel.indicador == "poblacion_total")].set_index("periodo")["valor"]
    assert row[TARGET] == pytest.approx(pop[2014] / pop[2013] - 1)
    assert row["crecimiento"] == pytest.approx(pop[2013] / pop[2012] - 1)
    assert row["periodo_objetivo"] == 2014


def test_gap_years_do_not_produce_lags(panel):
    """Sin el año 2013 no hay crecimiento para 2014 ni objetivo para 2012: no se mezclan periodos."""
    gapped = panel[~((panel.cvegeo == "99003") & (panel.periodo == 2013))]
    table = build_training_table(gapped)
    sub = table[table.cvegeo == "99003"]
    assert not sub["periodo"].isin([2012, 2013, 2014, 2015]).any()
    with pytest.raises(InsufficientHistoryError) as exc:
        features_for(gapped, "99003", 2014)
    assert "crecimiento" in exc.value.missing


def test_temporal_split_and_folds_never_look_ahead(panel):
    table = build_training_table(panel)
    tr, te = temporal_split(table, test_periodos=2)
    assert tr["periodo_objetivo"].max() < te["periodo_objetivo"].min()
    years = tr["periodo_objetivo"].reset_index(drop=True)
    for tr_idx, va_idx in expanding_year_folds(years, 3):
        assert years[tr_idx].max() < years[va_idx].min()


def test_trained_model_report_and_registry(trained_model_dir, data_dir):
    reg = ModelRegistry(trained_model_dir)
    bundle = reg.load()
    m = bundle.metadata
    assert m["datos"]["sintetico"] is True and "SINTÉTICOS" in m["datos"]["advertencia"]
    assert m["datos"]["sha256"] == load_csv(data_dir / "panel.csv").sha256
    assert m["caracteristicas"] == list(FEATURES)
    sel = m["modelo_seleccionado"]
    # En el panel sintético el modelo debe superar claramente a la media.
    assert m["prueba"][sel]["mae_crecimiento"] < m["prueba"]["baseline_media"]["mae_crecimiento"]
    expected = {"baseline_media", "baseline_persistencia", "ridge", "hist_gradient_boosting"}
    assert set(m["validacion_cruzada"]) == expected
    pred = bundle.model.predict(pd.DataFrame([dict.fromkeys(FEATURES, 0.0) | {"log_poblacion": 10.0,
                                              "razon_masculinidad": 96.0, "tasa_ocupacion": 55.0,
                                              "escolaridad_promedio": 9.0, "hogares_con_internet": 50.0}]))
    assert np.isfinite(pred).all()
    assert reg.list()[0]["version"] == bundle.version


@pytest.mark.parametrize("version", ["../etc", "latest", "20260101T000000Z-zzzzzzzz"])
def test_registry_rejects_unsafe_versions(trained_model_dir, version):
    with pytest.raises(ModelNotFoundError):
        ModelRegistry(trained_model_dir).load(version)


def test_registry_without_models(tmp_path):
    with pytest.raises(ModelNotFoundError):
        ModelRegistry(tmp_path).load()
