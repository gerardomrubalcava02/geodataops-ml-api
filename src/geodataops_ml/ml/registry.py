"""Registro local de modelos: artefacto joblib + metadatos JSON por versión.

Estructura:
    model_dir/<version>/model.joblib
    model_dir/<version>/metadata.json
    model_dir/LATEST            -> nombre de la versión activa

Seguridad: joblib usa pickle. Solo se cargan artefactos de `model_dir`, generados por
este proyecto; nunca se acepta un modelo subido por un cliente.
"""

from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn

from geodataops_ml import __version__
from geodataops_ml.data.contract import CONTRACT_VERSION, LoadedDataset
from geodataops_ml.ml.train import TrainResult

_VERSION_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")


class ModelNotFoundError(LookupError):
    pass


@dataclass
class ModelBundle:
    model: Any
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return self.metadata["version"]


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = root

    def save(self, result: TrainResult, dataset: LoadedDataset) -> ModelBundle:
        now = datetime.now(UTC)
        version = f"{now:%Y%m%dT%H%M%SZ}-{dataset.sha256[:8]}"
        target = self.root / version
        target.mkdir(parents=True, exist_ok=False)
        metadata = {
            "version": version,
            "creado_utc": now.isoformat(),
            "algoritmo": result.model_name,
            "datos": {
                "origen": Path(dataset.source).name,
                "sha256": dataset.sha256,
                "filas": dataset.rows,
                "contrato": CONTRACT_VERSION,
                "sintetico": dataset.is_synthetic,
                "advertencia": (
                    "Métricas obtenidas con datos demostrativos SINTÉTICOS; no describen "
                    "población real." if dataset.is_synthetic else None
                ),
            },
            "entorno": {
                "geodataops_ml": __version__,
                "scikit_learn": sklearn.__version__,
                "python": platform.python_version(),
            },
            **result.report,
        }
        joblib.dump(result.model, target / "model.joblib")
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), "utf-8")
        tmp = self.root / "LATEST.tmp"
        tmp.write_text(version, "utf-8")
        os.replace(tmp, self.root / "LATEST")  # cambio atómico de versión activa
        return ModelBundle(result.model, metadata)

    def latest_version(self) -> str | None:
        marker = self.root / "LATEST"
        if not marker.exists():
            return None
        return marker.read_text("utf-8").strip() or None

    def load(self, version: str | None = None) -> ModelBundle:
        version = version or self.latest_version()
        if version is None:
            raise ModelNotFoundError("no hay modelos entrenados")
        if not _VERSION_RE.fullmatch(version):  # evita rutas arbitrarias (../)
            raise ModelNotFoundError(f"versión inválida: {version!r}")
        path = self.root / version
        if not (path / "model.joblib").exists():
            raise ModelNotFoundError(f"no existe la versión {version}")
        metadata = json.loads((path / "metadata.json").read_text("utf-8"))
        return ModelBundle(joblib.load(path / "model.joblib"), metadata)

    def list(self) -> list[dict[str, Any]]:
        out = []
        if not self.root.exists():
            return out
        for meta in sorted(self.root.glob("*/metadata.json"), reverse=True):
            m = json.loads(meta.read_text("utf-8"))
            out.append({
                "version": m["version"],
                "creado_utc": m["creado_utc"],
                "algoritmo": m["algoritmo"],
                "sintetico": m["datos"]["sintetico"],
                "mae_crecimiento_prueba": m["prueba"][m["modelo_seleccionado"]]["mae_crecimiento"],
            })
        return out
