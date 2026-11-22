"""Repositorios de indicadores. La API depende del protocolo, no de la fuente.

- FrameRepository: datos en memoria validados con el contrato.
- CsvRepository: FrameRepository que recarga el CSV cuando el productor lo reemplaza.
- PostgresRepository: lee la vista `analytics.municipal_indicators` del ETL con psycopg async.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from anyio import to_thread

from geodataops_ml.data.contract import COLUMNS, SYNTHETIC_MARKERS, ContractError, LoadedDataset, load_csv

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationQuery:
    cvegeo: str | None = None
    indicadores: tuple[str, ...] = ()
    periodo_desde: int | None = None
    periodo_hasta: int | None = None
    limit: int = 500
    offset: int = 0


@dataclass(frozen=True)
class SourceInfo:
    kind: str
    location: str
    is_synthetic: bool
    sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class IndicatorRepository(Protocol):
    info: SourceInfo

    async def observations(self, q: ObservationQuery) -> tuple[list[dict[str, Any]], int]: ...
    async def municipios(self) -> list[dict[str, Any]]: ...
    async def catalog(self) -> list[dict[str, Any]]: ...
    async def frame(self) -> pd.DataFrame: ...
    async def close(self) -> None: ...


class FrameRepository:
    """Repositorio en memoria. Las consultas son O(n) sobre unos miles de filas: suficiente
    para el MVP y sin I/O, por lo que no bloquea el event loop de forma apreciable."""

    def __init__(self, dataset: LoadedDataset, kind: str):
        self._kind = kind
        self._set(dataset)

    def _set(self, dataset: LoadedDataset, **extra: Any) -> None:
        self._df = dataset.frame
        self.info = SourceInfo(
            kind=self._kind,
            location=dataset.source,
            is_synthetic=dataset.is_synthetic,
            sha256=dataset.sha256,
            extra={"filas": dataset.rows, **extra},
        )

    async def _refresh(self) -> None:
        return None

    async def observations(self, q: ObservationQuery) -> tuple[list[dict[str, Any]], int]:
        await self._refresh()
        df = self._df
        mask = pd.Series(True, index=df.index)
        if q.cvegeo:
            mask &= df["cvegeo"] == q.cvegeo
        if q.indicadores:
            mask &= df["indicador"].isin(q.indicadores)
        if q.periodo_desde is not None:
            mask &= df["periodo"] >= q.periodo_desde
        if q.periodo_hasta is not None:
            mask &= df["periodo"] <= q.periodo_hasta
        hits = df[mask]
        page = hits.iloc[q.offset : q.offset + q.limit]
        return page.loc[:, list(COLUMNS)].to_dict(orient="records"), len(hits)

    async def municipios(self) -> list[dict[str, Any]]:
        await self._refresh()
        g = self._df.groupby("cvegeo")
        out = pd.DataFrame(
            {
                "periodo_min": g["periodo"].min(),
                "periodo_max": g["periodo"].max(),
                "n_periodos": g["periodo"].nunique(),
                "indicadores": g["indicador"].agg(lambda s: sorted(s.unique())),
            }
        ).reset_index()
        out["nombre"] = None
        return out.to_dict(orient="records")

    async def catalog(self) -> list[dict[str, Any]]:
        await self._refresh()
        g = self._df.groupby(["indicador", "unidad"])
        out = pd.DataFrame(
            {
                "observaciones": g.size(),
                "municipios": g["cvegeo"].nunique(),
                "periodo_min": g["periodo"].min(),
                "periodo_max": g["periodo"].max(),
            }
        ).reset_index()
        return out.to_dict(orient="records")

    async def frame(self) -> pd.DataFrame:
        await self._refresh()
        return self._df.copy()

    async def close(self) -> None:
        return None


class CsvRepository(FrameRepository):
    """Recarga el CSV cuando cambian su mtime o tamaño (revisión ETL #4).

    El ETL reemplaza el archivo atómicamente, así que un `stat()` por solicitud basta para
    detectar una publicación. Si la nueva versión incumple el contrato se conserva la
    anterior y el error queda visible en /salud (`fuente_datos.extra.error_recarga`).
    """

    def __init__(self, path: Path, kind: str):
        self._path = path
        self._sig = self._stat()
        self._lock = asyncio.Lock()
        super().__init__(load_csv(path), kind)

    def _stat(self) -> tuple[int, int] | None:
        try:
            st = os.stat(self._path)
        except FileNotFoundError:
            return None
        return st.st_mtime_ns, st.st_size

    async def _refresh(self) -> None:
        sig = self._stat()
        if sig is None or sig == self._sig:
            return
        async with self._lock:
            if sig == self._sig:
                return
            self._sig = sig
            try:
                dataset = await to_thread.run_sync(load_csv, self._path)
            except (ContractError, OSError) as exc:
                log.error("Recarga rechazada, se conserva la versión anterior: %s", exc)
                self.info.extra["error_recarga"] = str(exc)[:500]
                return
            self._set(dataset, recargado=True)
            log.info("CSV recargado: %s (%s filas)", self._path, dataset.rows)


class PostgresRepository:
    """Lee la capa analytics del ETL sin escribir en ella (solo SELECT)."""

    _BASE = "FROM analytics.municipal_indicators WHERE dataset = %(dataset)s"

    def __init__(self, pool: Any, dataset: str, info: SourceInfo):
        self._pool = pool
        self._dataset = dataset
        self.info = info

    @classmethod
    async def open(cls, dsn: str, dataset: str) -> PostgresRepository:
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        pool = AsyncConnectionPool(
            dsn, min_size=1, max_size=5, open=False, kwargs={"row_factory": dict_row}
        )
        await pool.open(wait=True, timeout=10)
        markers = [f"%{m}%" for m in SYNTHETIC_MARKERS]
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS n, bool_or(lower(fuente) LIKE ANY(%(m)s)) AS synth " + cls._BASE,
                {"dataset": dataset, "m": markers},
            )
            row = await cur.fetchone()
        if not row or row["n"] == 0:
            await pool.close()
            raise LookupError(f"el dataset '{dataset}' no tiene indicadores en PostgreSQL")
        from psycopg.conninfo import conninfo_to_dict

        params = conninfo_to_dict(dsn)  # solo host y base: nunca exponer credenciales
        info = SourceInfo(
            kind="postgres",
            location=f"analytics.municipal_indicators[dataset={dataset}]",
            is_synthetic=bool(row["synth"]),
            extra={"filas": row["n"], "host": params.get("host"), "dbname": params.get("dbname")},
        )
        return cls(pool, dataset, info)

    async def _fetch(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, {"dataset": self._dataset, **params})
            return await cur.fetchall()

    async def observations(self, q: ObservationQuery) -> tuple[list[dict[str, Any]], int]:
        where, params = [], {}
        if q.cvegeo:
            where.append("cvegeo = %(cvegeo)s")
            params["cvegeo"] = q.cvegeo
        if q.indicadores:
            where.append("indicador = ANY(%(indicadores)s)")
            params["indicadores"] = list(q.indicadores)
        if q.periodo_desde is not None:
            where.append("periodo >= %(desde)s")
            params["desde"] = q.periodo_desde
        if q.periodo_hasta is not None:
            where.append("periodo <= %(hasta)s")
            params["hasta"] = q.periodo_hasta
        cond = "".join(f" AND {w}" for w in where)
        # El total se calcula aparte: con count(*) OVER () una página vacía (offset > total)
        # devolvería 0 y rompería la paridad con FrameRepository (revisión ETL #2).
        async with self._pool.connection() as conn:
            base = {"dataset": self._dataset, **params}
            cur = await conn.execute(f"SELECT count(*)::int AS n {self._BASE}{cond}", base)
            total = (await cur.fetchone())["n"]
            cur = await conn.execute(
                "SELECT cvegeo, periodo::int AS periodo, indicador, valor::float8 AS valor,"
                f" unidad, fuente {self._BASE}{cond} ORDER BY cvegeo, periodo, indicador"
                " LIMIT %(limit)s OFFSET %(offset)s",
                {**base, "limit": q.limit, "offset": q.offset},
            )
            rows = await cur.fetchall()
        return rows, int(total)

    async def municipios(self) -> list[dict[str, Any]]:
        return await self._fetch(
            "SELECT cvegeo, min(municipio) AS nombre, min(periodo)::int AS periodo_min,"
            " max(periodo)::int AS periodo_max, count(DISTINCT periodo)::int AS n_periodos,"
            " array_agg(DISTINCT indicador ORDER BY indicador) AS indicadores "
            f"{self._BASE} GROUP BY cvegeo ORDER BY cvegeo",
            {},
        )

    async def catalog(self) -> list[dict[str, Any]]:
        return await self._fetch(
            "SELECT indicador, unidad, count(*)::int AS observaciones,"
            " count(DISTINCT cvegeo)::int AS municipios, min(periodo)::int AS periodo_min,"
            " max(periodo)::int AS periodo_max "
            f"{self._BASE} GROUP BY indicador, unidad ORDER BY indicador",
            {},
        )

    async def frame(self) -> pd.DataFrame:
        rows = await self._fetch(
            "SELECT cvegeo, periodo::int AS periodo, indicador, valor::float8 AS valor, unidad,"
            f" fuente {self._BASE} ORDER BY cvegeo, periodo, indicador",
            {},
        )
        return pd.DataFrame(rows, columns=list(COLUMNS))

    async def close(self) -> None:
        await self._pool.close()
