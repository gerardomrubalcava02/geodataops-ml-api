from pathlib import Path

import pytest

from geodataops_ml.config import DEFAULT_ETL_CSV, WORKSPACE_ROOT
from geodataops_ml.data.contract import COLUMNS, ContractError, load_csv

HEADER = ",".join(COLUMNS)


def _write(tmp_path: Path, body: str, name: str = "x.csv", bom: bool = False) -> Path:
    p = tmp_path / name
    p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + (HEADER + "\n" + body).encode("utf-8"))
    return p


def test_fixture_is_valid_and_synthetic(data_dir):
    ds = load_csv(data_dir / "fixture.csv")
    assert ds.is_synthetic and ds.rows > 0
    assert ds.frame["cvegeo"].dtype == object
    assert len(ds.sha256) == 64


def test_preserves_leading_zeros(tmp_path):
    ds = load_csv(_write(tmp_path, "01001,2020,poblacion_total,100,personas,INEGI\n"))
    assert ds.frame.loc[0, "cvegeo"] == "01001"
    assert not ds.is_synthetic


@pytest.mark.parametrize(
    "body,fragment",
    [
        ("1001,2020,poblacion_total,1,personas,X\n", "cvegeo"),
        ("01001,20,poblacion_total,1,personas,X\n", "periodo"),
        ("01001,2020,Poblacion,1,personas,X\n", "indicador"),
        ("01001,2020,poblacion_total,abc,personas,X\n", "valor"),
        ("01001,2020,poblacion_total,inf,personas,X\n", "valor"),
        # Revisión ETL #3: validar el texto antes de convertir.
        ("01001,2020,poblacion_total,1e3,personas,X\n", "valor"),
        ('01001,2020,tasa,"1,000",%,X\n', "valor"),
        ("01001,2020,tasa,9007199254740993,%,X\n", "2^53"),
        ("14000,2020,tasa,1,%,X\n", "cvegeo"),
        ("33001,2020,tasa,1,%,X\n", "cvegeo"),
        ("99001,2020,tasa,1,%,INEGI\n", "entidad 99"),
        ("01001,0999,tasa,1,%,X\n", "periodo"),
        ("01001,2020,poblacion_total,1.5,personas,X\n", "población"),
        ("01001,2020,poblacion_total,1,habitantes,X\n", "población"),
        ("01001,2020,poblacion_total,1,personas,\n", "fuente vacío"),
        ("01001,2020,tasa,1,%,X\n01001,2020,tasa,2,%,X\n", "duplicada"),
    ],
)
def test_rejects_contract_violations(tmp_path, body, fragment):
    with pytest.raises(ContractError) as exc:
        load_csv(_write(tmp_path, body))
    assert any(fragment in e for e in exc.value.errors), exc.value.errors


def test_rejects_bad_header_and_bom(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("cvegeo,valor\n01001,1\n", "utf-8")
    with pytest.raises(ContractError, match="cabecera"):
        load_csv(p)
    with pytest.raises(ContractError, match="BOM"):
        load_csv(_write(tmp_path, "01001,2020,x,1,u,f\n", bom=True))


def test_error_reports_row_numbers(tmp_path):
    body = "01001,2020,a,1,u,f\n01001,2020,b,zz,u,f\n"
    with pytest.raises(ContractError) as exc:
        load_csv(_write(tmp_path, body))
    assert "[3]" in exc.value.errors[0]  # línea 3 del archivo (cabecera = 1)


@pytest.mark.skipif(not DEFAULT_ETL_CSV.exists(), reason="el ETL aún no publica su salida")
def test_etl_published_output_is_compatible():
    ds = load_csv(DEFAULT_ETL_CSV)
    assert ds.rows > 0
    assert ds.frame["cvegeo"].str.fullmatch(r"\d{5}").all()


@pytest.mark.parametrize(
    "rel", ["geodataops-etl/data/demo/indicadores.csv", "geodataops-etl/data/jalisco/poblacion.csv"]
)
def test_etl_versioned_inputs_are_compatible(rel):
    path = WORKSPACE_ROOT / rel
    if not path.exists():
        pytest.skip(f"no existe {rel}")
    load_csv(path)
