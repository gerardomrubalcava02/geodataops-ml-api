"""CLI: generar datos demostrativos, entrenar, validar CSV del ETL y servir la API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geodataops_ml.config import get_settings


def _generate(args: argparse.Namespace) -> int:
    from geodataops_ml.data.contract import write_csv
    from geodataops_ml.data.synthetic import generate_panel

    s = get_settings()
    panel = generate_panel(n_municipios=args.municipios, seed=args.seed)
    write_csv(panel, s.training_csv_path)
    fixture = generate_panel(n_municipios=5, periodo_inicio=2016, periodo_fin=2022, seed=7)
    write_csv(fixture, s.fixture_csv_path)
    print(f"panel: {s.training_csv_path} ({len(panel)} filas)")
    print(f"fixture: {s.fixture_csv_path} ({len(fixture)} filas)")
    return 0


def _train(args: argparse.Namespace) -> int:
    from geodataops_ml.data.contract import load_csv
    from geodataops_ml.ml.registry import ModelRegistry
    from geodataops_ml.ml.train import TrainConfig, train

    s = get_settings()
    path = Path(args.csv) if args.csv else s.training_csv_path
    dataset = load_csv(path)
    result = train(dataset.frame, TrainConfig(test_periodos=args.test_periodos, seed=args.seed))
    bundle = ModelRegistry(s.model_dir).save(result, dataset)
    m = bundle.metadata
    summary = {
        "version": m["version"],
        "algoritmo": m["algoritmo"],
        "sintetico": m["datos"]["sintetico"],
        "prueba": m["prueba"],
        "mejora_mae_vs_mejor_baseline_pct": m["mejora_mae_vs_mejor_baseline_pct"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _validate(args: argparse.Namespace) -> int:
    from geodataops_ml.data.contract import ContractError, load_csv

    try:
        ds = load_csv(Path(args.csv))
    except ContractError as exc:
        print(f"INVÁLIDO: {exc.source}")
        for e in exc.errors:
            print(f"  - {e}")
        return 1
    print(f"VÁLIDO: {ds.rows} filas, sintético={ds.is_synthetic}, sha256={ds.sha256[:12]}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import asyncio

    import uvicorn

    config = uvicorn.Config(
        "geodataops_ml.api.app:create_app", factory=True, host=args.host, port=args.port, loop="none"
    )
    # psycopg async no admite el ProactorEventLoop (predeterminado en Windows): se usa
    # SelectorEventLoop en todas las plataformas para que el modo PostgreSQL funcione.
    asyncio.run(uvicorn.Server(config).serve(), loop_factory=asyncio.SelectorEventLoop)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="geodataops-ml")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-synthetic", help="genera el panel sintético y los fixtures")
    g.add_argument("--municipios", type=int, default=120)
    g.add_argument("--seed", type=int, default=2026)
    g.set_defaults(func=_generate)

    t = sub.add_parser("train", help="entrena, evalúa y registra un modelo")
    t.add_argument("--csv", help="CSV del contrato v1 (por defecto, el panel sintético)")
    t.add_argument("--test-periodos", type=int, default=3)
    t.add_argument("--seed", type=int, default=2026)
    t.set_defaults(func=_train)

    v = sub.add_parser("validate", help="valida un CSV contra el contrato v1")
    v.add_argument("csv")
    v.set_defaults(func=_validate)

    sv = sub.add_parser("serve", help="inicia la API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_serve)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
