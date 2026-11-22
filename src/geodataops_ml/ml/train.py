"""Entrenamiento y evaluación con validación temporal.

Protocolo:
1. Holdout temporal: los últimos `test_periodos` años objetivo forman el conjunto de prueba.
   Nunca se usan para seleccionar modelo ni hiperparámetros.
2. Selección: validación cruzada de ventana creciente por año objetivo dentro de entrenamiento.
3. Comparación contra dos baselines: media del entrenamiento y persistencia (g[t+1] = g[t]).
4. Se persiste el modelo ajustado SOLO con entrenamiento, para que las métricas de prueba
   reportadas correspondan exactamente al artefacto servido.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geodataops_ml.ml.features import FEATURES, TARGET, build_training_table


class PersistenceBaseline(RegressorMixin, BaseEstimator):
    """Predice que el crecimiento del año siguiente será igual al del año actual."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> PersistenceBaseline:
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["crecimiento"].to_numpy(dtype=float)


def baselines() -> dict[str, BaseEstimator]:
    return {"baseline_media": DummyRegressor(strategy="mean"), "baseline_persistencia": PersistenceBaseline()}


def candidates(seed: int) -> dict[str, BaseEstimator]:
    return {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, random_state=seed,
        ),
    }


@dataclass(frozen=True)
class TrainConfig:
    test_periodos: int = 3
    cv_folds: int = 4
    seed: int = 2026


def temporal_split(table: pd.DataFrame, test_periodos: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = np.sort(table["periodo_objetivo"].unique())
    if len(years) < test_periodos + 3:
        raise ValueError(
            f"se requieren al menos {test_periodos + 3} años objetivo; hay {len(years)}"
        )
    cutoff = years[-test_periodos]
    return table[table["periodo_objetivo"] < cutoff], table[table["periodo_objetivo"] >= cutoff]


def expanding_year_folds(years: pd.Series, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cada pliegue valida un año objetivo y entrena con todos los anteriores."""
    uniq = np.sort(years.unique())
    n_folds = min(n_folds, len(uniq) - 2)
    folds = []
    for val_year in uniq[-n_folds:]:
        folds.append((np.where(years < val_year)[0], np.where(years == val_year)[0]))
    return folds


def regression_metrics(y: np.ndarray, pred: np.ndarray, pop: np.ndarray, pop_next: np.ndarray) -> dict:
    pop_pred = pop * (1 + pred)
    return {
        "mae_crecimiento": float(mean_absolute_error(y, pred)),
        "rmse_crecimiento": float(np.sqrt(mean_squared_error(y, pred))),
        "r2_crecimiento": float(r2_score(y, pred)),
        "mae_poblacion": float(np.mean(np.abs(pop_pred - pop_next))),
        "mape_poblacion_pct": float(100 * np.mean(np.abs(pop_pred - pop_next) / pop_next)),
    }


def _cv_mae(est: BaseEstimator, X: pd.DataFrame, y: pd.Series, folds) -> tuple[float, float]:
    scores = []
    for tr, va in folds:
        model = clone(est).fit(X.iloc[tr], y.iloc[tr])
        scores.append(mean_absolute_error(y.iloc[va], model.predict(X.iloc[va])))
    return float(np.mean(scores)), float(np.std(scores))


def _year_range(df: pd.DataFrame) -> list[int]:
    return [int(df["periodo_objetivo"].min()), int(df["periodo_objetivo"].max())]


@dataclass
class TrainResult:
    model: BaseEstimator
    model_name: str
    report: dict[str, Any]


def train(frame: pd.DataFrame, config: TrainConfig | None = None) -> TrainResult:
    config = config or TrainConfig()
    t0 = time.perf_counter()
    table = build_training_table(frame)
    train_df, test_df = temporal_split(table, config.test_periodos)
    feats = list(FEATURES)
    X_tr, y_tr = train_df[feats], train_df[TARGET]
    X_te, y_te = test_df[feats], test_df[TARGET]
    folds = expanding_year_folds(train_df["periodo_objetivo"].reset_index(drop=True), config.cv_folds)

    pool = {**baselines(), **candidates(config.seed)}
    cv = {}
    for name, est in pool.items():
        mean, std = _cv_mae(est, X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), folds)
        cv[name] = {"mae_crecimiento_media": mean, "mae_crecimiento_std": std}

    ml_names = list(candidates(config.seed))
    best = min(ml_names, key=lambda n: cv[n]["mae_crecimiento_media"])

    test_metrics, fitted = {}, {}
    for name, est in pool.items():
        fitted[name] = clone(est).fit(X_tr, y_tr)
        test_metrics[name] = regression_metrics(
            y_te.to_numpy(), fitted[name].predict(X_te),
            test_df["poblacion"].to_numpy(), test_df["poblacion_siguiente"].to_numpy(),
        )

    model = fitted[best]
    imp = permutation_importance(
        model, X_te, y_te, scoring="neg_mean_absolute_error", n_repeats=5, random_state=config.seed
    )
    importance = sorted(
        ({"caracteristica": f, "importancia_mae": float(m)} for f, m in zip(feats, imp.importances_mean, strict=True)),
        key=lambda d: -d["importancia_mae"],
    )
    best_baseline = min(baselines(), key=lambda n: test_metrics[n]["mae_crecimiento"])
    gain = 1 - test_metrics[best]["mae_crecimiento"] / test_metrics[best_baseline]["mae_crecimiento"]

    report = {
        "objetivo": TARGET,
        "caracteristicas": feats,
        "modelo_seleccionado": best,
        "criterio_seleccion": "menor MAE medio en CV temporal de ventana creciente (solo entrenamiento)",
        "particion": {
            "anios_objetivo_entrenamiento": _year_range(train_df),
            "anios_objetivo_prueba": _year_range(test_df),
            "filas_entrenamiento": len(train_df),
            "filas_prueba": len(test_df),
            "municipios": int(table["cvegeo"].nunique()),
            "pliegues_cv": len(folds),
        },
        "validacion_cruzada": cv,
        "prueba": test_metrics,
        "mejor_baseline_prueba": best_baseline,
        "mejora_mae_vs_mejor_baseline_pct": round(100 * gain, 2),
        "importancia_permutacion_prueba": importance,
        "segundos_entrenamiento": round(time.perf_counter() - t0, 2),
    }
    return TrainResult(model=model, model_name=best, report=report)
