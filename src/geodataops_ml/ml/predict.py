"""Inferencia: características -> crecimiento esperado y población estimada en t+1."""

from __future__ import annotations

from typing import Any

import pandas as pd

from geodataops_ml.ml.features import FEATURES, features_for
from geodataops_ml.ml.registry import ModelBundle


def predict_from_features(bundle: ModelBundle, features: dict[str, float]) -> float:
    row = pd.DataFrame([{f: features[f] for f in FEATURES}])
    return float(bundle.model.predict(row)[0])


def predict_for_municipio(
    bundle: ModelBundle, frame: pd.DataFrame, cvegeo: str, periodo: int
) -> dict[str, Any]:
    row = features_for(frame, cvegeo, periodo)
    growth = float(bundle.model.predict(row)[0])
    pop = float(
        frame.loc[
            (frame["cvegeo"] == cvegeo) & (frame["periodo"] == periodo)
            & (frame["indicador"] == "poblacion_total"),
            "valor",
        ].iloc[0]
    )
    return {
        "cvegeo": cvegeo,
        "periodo_base": periodo,
        "periodo_objetivo": periodo + 1,
        "crecimiento_esperado": growth,
        "poblacion_base": int(pop),
        "poblacion_estimada": int(round(pop * (1 + growth))),
        "caracteristicas": {k: float(v) for k, v in row.iloc[0].items()},
    }
