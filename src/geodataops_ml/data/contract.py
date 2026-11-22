"""Validación del contrato CSV municipal v1 (geodataops-sync/contrato_datos.md).

El consumidor es estricto con lo que rompe la integración (columnas, claves, tipos)
y reporta los errores por fila para que el productor pueda corregirlos.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_VERSION = "v1"
COLUMNS: tuple[str, ...] = ("cvegeo", "periodo", "indicador", "valor", "unidad", "fuente")
KEY: list[str] = ["cvegeo", "periodo", "indicador"]
POPULATION_INDICATORS = frozenset({"poblacion_total", "poblacion_femenina", "poblacion_masculina"})
SYNTHETIC_MARKERS = ("sintético", "sintetico", "synthetic")

_CVEGEO_RE = r"^[0-9]{5}$"
_PERIODO_RE = r"^[0-9]{4}$"
_INDICADOR_RE = r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$"
_MAX_REPORTED_ERRORS = 20


class ContractError(ValueError):
    """El archivo no cumple el contrato. `errors` lista problemas legibles."""

    def __init__(self, source: str, errors: list[str]):
        self.source = source
        self.errors = errors
        super().__init__(f"{source}: {len(errors)} problema(s) de contrato: " + "; ".join(errors[:5]))


@dataclass(frozen=True)
class LoadedDataset:
    frame: pd.DataFrame
    source: str
    sha256: str
    is_synthetic: bool

    @property
    def rows(self) -> int:
        return len(self.frame)


def _row_errors(mask: pd.Series, message: str) -> list[str]:
    # +2: cabecera y numeración desde 1, para coincidir con el editor de texto.
    rows = (mask[mask].index[:_MAX_REPORTED_ERRORS] + 2).tolist()
    return [f"{message} (filas {rows})"] if rows else []


def validate_frame(raw: pd.DataFrame, source: str = "<frame>") -> pd.DataFrame:
    """Valida un DataFrame de texto y lo devuelve tipado. Lanza ContractError."""
    if tuple(raw.columns) != COLUMNS:
        raise ContractError(source, [f"cabecera {list(raw.columns)} != {list(COLUMNS)}"])

    df = raw.copy()
    errors: list[str] = []
    for col in COLUMNS:
        errors += _row_errors(df[col].isna() | (df[col].astype(str).str.strip() == ""), f"{col} vacío")
    if errors:
        raise ContractError(source, errors)

    df["cvegeo"] = df["cvegeo"].astype(str)
    errors += _row_errors(~df["cvegeo"].str.fullmatch(_CVEGEO_RE), "cvegeo no tiene 5 dígitos")
    errors += _row_errors(~df["periodo"].astype(str).str.fullmatch(_PERIODO_RE), "periodo no es YYYY")
    errors += _row_errors(~df["indicador"].astype(str).str.fullmatch(_INDICADOR_RE), "indicador inválido")

    valor = pd.to_numeric(df["valor"], errors="coerce")
    errors += _row_errors(valor.isna() | ~np.isfinite(valor), "valor no es un número finito")
    if errors:
        raise ContractError(source, errors)

    df["periodo"] = df["periodo"].astype(int)
    df["valor"] = valor.astype(float)

    pop = df["indicador"].isin(POPULATION_INDICATORS)
    bad_pop = pop & ((df["valor"] < 0) | (df["valor"] % 1 != 0) | (df["unidad"] != "personas"))
    errors += _row_errors(bad_pop, "población debe ser entero no negativo en 'personas'")
    errors += _row_errors(df.duplicated(KEY, keep=False), "clave (cvegeo, periodo, indicador) duplicada")
    if errors:
        raise ContractError(source, errors)

    return df.sort_values(KEY, kind="stable").reset_index(drop=True)


def is_synthetic(frame: pd.DataFrame) -> bool:
    fuente = frame["fuente"].str.lower()
    return bool(fuente.apply(lambda s: any(m in s for m in SYNTHETIC_MARKERS)).any())


def load_csv(path: Path) -> LoadedDataset:
    """Carga un CSV del contrato. `cvegeo` y `periodo` se leen como texto para no perder ceros."""
    content = path.read_bytes()
    if content.startswith(b"\xef\xbb\xbf"):
        raise ContractError(str(path), ["el archivo tiene BOM; el contrato exige UTF-8 sin BOM"])
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    raw = raw.replace("", np.nan)
    frame = validate_frame(raw, source=str(path))
    return LoadedDataset(
        frame=frame,
        source=str(path),
        sha256=hashlib.sha256(content).hexdigest(),
        is_synthetic=is_synthetic(frame),
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Escribe en el formato del contrato (orden, sin notación científica, sin BOM, LF)."""
    out = frame.loc[:, list(COLUMNS)].sort_values(KEY, kind="stable").copy()

    def fmt(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else np.format_float_positional(v, trim="-")

    out["valor"] = out["valor"].map(fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
