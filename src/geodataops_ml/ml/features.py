"""Ingeniería de características para predecir el crecimiento poblacional del año siguiente.

Unidad de análisis: municipio i en el año t.
Objetivo: crecimiento_siguiente = P[t+1] / P[t] - 1.
Características: solo información observada en t o antes (t-1, t-2).

Prevención de fugas:
- Los rezagos se calculan uniendo por (cvegeo, periodo - k), no con shift() posicional,
  así un año faltante produce NaN en lugar de mezclar periodos no consecutivos.
- `periodo` no es característica: evita que el modelo memorice el calendario.
- El escalado se ajusta dentro del Pipeline de scikit-learn, solo con datos de entrenamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "crecimiento_siguiente"
COVARIATES = ("tasa_ocupacion", "escolaridad_promedio", "hogares_con_internet")
POPULATION = ("poblacion_total", "poblacion_femenina", "poblacion_masculina")
REQUIRED_INDICATORS = POPULATION + COVARIATES
FEATURES: tuple[str, ...] = (
    "log_poblacion",
    "crecimiento",
    "crecimiento_lag1",
    "razon_masculinidad",
    *COVARIATES,
)


class InsufficientHistoryError(LookupError):
    def __init__(self, cvegeo: str, periodo: int, missing: list[str]):
        self.cvegeo, self.periodo, self.missing = cvegeo, periodo, missing
        super().__init__(
            f"{cvegeo}/{periodo}: faltan datos para construir características: {', '.join(missing)}"
        )


def to_wide(frame: pd.DataFrame) -> pd.DataFrame:
    """Formato largo del contrato -> una fila por (cvegeo, periodo), una columna por indicador."""
    wide = frame.pivot_table(
        index=["cvegeo", "periodo"], columns="indicador", values="valor", aggfunc="first"
    )
    wide.columns.name = None
    return wide.sort_index()


def _lag(series: pd.Series, k: int) -> pd.Series:
    lagged = series.copy()
    lagged.index = pd.MultiIndex.from_arrays(
        [series.index.get_level_values(0), series.index.get_level_values(1) + k],
        names=series.index.names,
    )
    return lagged.reindex(series.index)


def _lead(series: pd.Series, k: int) -> pd.Series:
    return _lag(series, -k)


def build_features(wide: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_INDICATORS if c not in wide.columns]
    if missing:
        raise ValueError(f"faltan indicadores requeridos por el modelo: {missing}")
    p = wide["poblacion_total"].where(wide["poblacion_total"] > 0)
    p1, p2 = _lag(p, 1), _lag(p, 2)
    out = pd.DataFrame(index=wide.index)
    out["log_poblacion"] = np.log(p)
    out["crecimiento"] = p / p1 - 1
    out["crecimiento_lag1"] = p1 / p2 - 1
    fem = wide["poblacion_femenina"].where(wide["poblacion_femenina"] > 0)
    out["razon_masculinidad"] = 100 * wide["poblacion_masculina"] / fem
    for c in COVARIATES:
        out[c] = wide[c]
    return out.loc[:, list(FEATURES)]


def build_training_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Características en t + objetivo en t+1. Filas incompletas se descartan."""
    wide = to_wide(frame)
    table = build_features(wide)
    p = wide["poblacion_total"]
    table["poblacion"] = p
    table["poblacion_siguiente"] = _lead(p, 1)
    table[TARGET] = table["poblacion_siguiente"] / p - 1
    table = table.replace([np.inf, -np.inf], np.nan).dropna()
    table = table.reset_index()
    table["periodo_objetivo"] = table["periodo"] + 1
    return table


def features_for(frame: pd.DataFrame, cvegeo: str, periodo: int) -> pd.DataFrame:
    """Una fila de características para (cvegeo, periodo) o InsufficientHistoryError."""
    sub = frame[(frame["cvegeo"] == cvegeo) & frame["periodo"].between(periodo - 2, periodo)]
    if sub.empty:
        raise InsufficientHistoryError(cvegeo, periodo, ["sin observaciones en t-2..t"])
    wide = to_wide(sub)
    for col in REQUIRED_INDICATORS:
        if col not in wide.columns:
            wide[col] = np.nan
    feats = build_features(wide).replace([np.inf, -np.inf], np.nan)
    key = (cvegeo, periodo)
    if key not in feats.index:
        raise InsufficientHistoryError(cvegeo, periodo, [f"sin observaciones en {periodo}"])
    row = feats.loc[[key]]
    missing = [c for c in FEATURES if pd.isna(row.iloc[0][c])]
    if missing:
        raise InsufficientHistoryError(cvegeo, periodo, missing)
    return row.reset_index(drop=True)
