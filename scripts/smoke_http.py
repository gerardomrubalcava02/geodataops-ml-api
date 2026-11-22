"""HTTP smoke test against a running GeoDataOps ML API (container or local server).

Standard library only, so it runs on a bare CI runner:

    python scripts/smoke_http.py --base-url http://127.0.0.1:8000 --expect-source etl_csv

Checks:
  1. /api/v1/salud reports a loaded model and the expected data source.
  2. /api/v1/observaciones returns rows from that source.
  3. A prediction from the synthetic training panel is flagged as synthetic, with the warning.
  4. A case without enough history (real census sample, or first fixture year) still returns 422.
Exits non-zero on the first failed check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def request(base: str, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def show(label: str, method: str, path: str, status: int, payload: Any, body: Any = None) -> None:
    print(f"\n### {label}\n{method} {path}" + (f"  {json.dumps(body)}" if body else ""))
    print(f"-> {status}\n{json.dumps(payload, indent=2, ensure_ascii=False)[:1500]}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"\nFAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--expect-source", required=True, choices=["fixtures", "etl_csv", "csv", "postgres"])
    ap.add_argument("--wait", type=float, default=60, help="seconds to wait for the API to come up")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    deadline = time.monotonic() + args.wait
    while True:
        try:
            status, salud = request(base, "GET", "/api/v1/salud")
            break
        except (urllib.error.URLError, ConnectionError):
            if time.monotonic() > deadline:
                check(False, f"API not reachable at {base} after {args.wait}s")
            time.sleep(1)

    show("Health", "GET", "/api/v1/salud", status, salud)
    check(status == 200 and salud["estado"] == "ok", "health is ok")
    check(bool(salud["modelo_version"]), f"model loaded ({salud['modelo_version']})")
    check(salud["fuente_datos"]["tipo"] == args.expect_source, f"data source is {args.expect_source}")

    real = args.expect_source == "etl_csv"
    query = "cvegeo=14039&indicador=poblacion_total" if real else "limit=3"
    obs_path = f"/api/v1/observaciones?{query}"
    status, obs = request(base, "GET", obs_path)
    show("Observations", "GET", obs_path, status, obs)
    check(status == 200 and obs["total"] > 0 and obs["items"], "observations returned")
    if real:
        check(obs["fuente"]["sintetico"] is False, "ETL sample is flagged as real (not synthetic)")
        check({i["periodo"] for i in obs["items"]} == {2010, 2020}, "real sample has the 2010 and 2020 censuses")

    body = {"cvegeo": "99007", "periodo": 2023}
    status, pred = request(base, "POST", "/api/v1/predicciones", body)
    show("Prediction (synthetic training panel)", "POST", "/api/v1/predicciones", status, pred, body)
    check(status == 200, "prediction succeeds")
    check(pred["datos_entrenamiento_sinteticos"] is True, "datos_entrenamiento_sinteticos == true")
    check("SINTÉTICOS" in (pred.get("advertencia") or ""), "synthetic-data warning present")

    body = (
        {"cvegeo": "14039", "periodo": 2020, "fuente_datos": "operativa"}
        if real
        else {"cvegeo": "99001", "periodo": 2016, "fuente_datos": "operativa"}
    )
    status, err = request(base, "POST", "/api/v1/predicciones", body)
    show("Insufficient history is still rejected", "POST", "/api/v1/predicciones", status, err, body)
    check(status == 422 and err["error"]["codigo"] == "datos_insuficientes", "422 datos_insuficientes")

    print("\nSMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
