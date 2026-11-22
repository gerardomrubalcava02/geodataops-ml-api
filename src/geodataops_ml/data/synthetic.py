"""Generador del panel DEMOSTRATIVO SINTÉTICO para entrenar el modelo.

Motivo: la muestra real del ETL (3 municipios x 2 censos) no permite entrenar ni
validar un modelo de forma defendible. Este panel reproduce el contrato v1 con
claves `99xxx` (fuera del rango INEGI 01-32) y `fuente` marcada como sintética.
Sus métricas demuestran el flujo de ML; NO describen población real.

Proceso generador (conocido, para poder razonar sobre lo que el modelo aprende):
  g[i,t] = 0.45 g[i,t-1] + 0.55 mu_i
           + 0.0009 (ocupacion - 58) + 0.0025 (escolaridad - 9)
           - 0.004 (ln P[i,t-1] - 11)                 # convergencia
           + 0.012 max(0, internet - 60) / 40         # no linealidad
           + ruido N(0, 0.004)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_SOURCE = "DEMO sintético (GeoDataOps ML, semilla {seed})"
SYNTHETIC_ENTITY = "99"


def generate_panel(
    n_municipios: int = 120,
    periodo_inicio: int = 2005,
    periodo_fin: int = 2024,
    seed: int = 2026,
) -> pd.DataFrame:
    if not 1 <= n_municipios <= 999:
        raise ValueError("n_municipios debe estar entre 1 y 999")
    if periodo_fin - periodo_inicio < 3:
        raise ValueError("se requieren al menos 4 periodos")

    rng = np.random.default_rng(seed)
    years = np.arange(periodo_inicio, periodo_fin + 1)
    fuente = SYNTHETIC_SOURCE.format(seed=seed)
    rows: list[tuple] = []

    for i in range(1, n_municipios + 1):
        cvegeo = f"{SYNTHETIC_ENTITY}{i:03d}"
        pop = float(np.exp(rng.normal(np.log(40_000), 1.1)))
        mu = rng.normal(0.012, 0.008)
        ocup_base = rng.uniform(45, 70)
        esc_base = rng.uniform(6.0, 10.5)
        net_mid = rng.uniform(2008, 2022)
        fem_share = rng.normal(0.51, 0.008)
        g_prev = mu

        for t, year in enumerate(years):
            ocupacion = float(np.clip(ocup_base + 0.15 * t + rng.normal(0, 1.2), 30, 85))
            escolaridad = float(np.clip(esc_base + 0.06 * t + rng.normal(0, 0.1), 3, 16))
            internet = float(100 / (1 + np.exp(-(year - net_mid) / 3)) + rng.normal(0, 2))
            internet = float(np.clip(internet, 0, 100))

            if t > 0:
                g = (
                    0.45 * g_prev
                    + 0.55 * mu
                    + 0.0009 * (ocupacion - 58)
                    + 0.0025 * (escolaridad - 9)
                    - 0.004 * (np.log(pop) - 11)
                    + 0.012 * max(0.0, internet - 60) / 40
                    + rng.normal(0, 0.004)
                )
                pop *= 1 + g
                g_prev = g

            total = int(round(pop))
            fem = int(round(total * float(np.clip(fem_share + rng.normal(0, 0.002), 0.45, 0.55))))
            rows += [
                (cvegeo, year, "poblacion_total", total, "personas"),
                (cvegeo, year, "poblacion_femenina", fem, "personas"),
                (cvegeo, year, "poblacion_masculina", total - fem, "personas"),
                (cvegeo, year, "tasa_ocupacion", round(ocupacion, 2), "porcentaje"),
                (cvegeo, year, "escolaridad_promedio", round(escolaridad, 2), "años"),
                (cvegeo, year, "hogares_con_internet", round(internet, 2), "porcentaje"),
            ]

    frame = pd.DataFrame(rows, columns=["cvegeo", "periodo", "indicador", "valor", "unidad"])
    frame["valor"] = frame["valor"].astype(float)
    frame["fuente"] = fuente
    return frame.sort_values(["cvegeo", "periodo", "indicador"]).reset_index(drop=True)
