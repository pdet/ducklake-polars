"""Shared test fixtures for ducklake-polars tests."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pytest


def _get_backends():
    """Return pytest parameters for available backends."""
    backends = [pytest.param("sqlite", id="sqlite")]
    if os.environ.get("DUCKLAKE_PG_DSN"):
        backends.append(
            pytest.param("postgres", id="postgres", marks=pytest.mark.postgres)
        )
    return backends


@dataclass
class DuckLakeTestCatalog:
    """
    Helper for creating and populating DuckLake catalogs in tests.

    Usage:
        catalog = DuckLakeTestCatalog(tmp_path)
        catalog.execute("CREATE TABLE ducklake.test (a INTEGER)")
        catalog.execute("INSERT INTO ducklake.test VALUES (1)")
        catalog.close()  # Release the file lock

        # Now read with ducklake-polars
        result = read_ducklake(catalog.metadata_path, "test")
    """

    metadata_path: str
    data_path: str
    _con: duckdb.DuckDBPyConnection = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False)
    inline: bool = True
    backend: str = "sqlite"

    def __post_init__(self) -> None:
        os.makedirs(self.data_path, exist_ok=True)
        self._con = duckdb.connect()
        self._con.install_extension("ducklake")
        self._con.load_extension("ducklake")

        if self.backend == "sqlite":
            self._con.install_extension("sqlite_scanner")
            self._con.load_extension("sqlite_scanner")
            attach_source = f"ducklake:sqlite:{self.metadata_path}"
        else:
            self._cleanup_postgres_tables()
            attach_source = f"ducklake:postgres:{self.metadata_path}"

        inline_opt = "" if self.inline else ", DATA_INLINING_ROW_LIMIT 0"
        self._con.execute(
            f"""
            ATTACH '{attach_source}' AS ducklake
                (DATA_PATH '{self.data_path}'{inline_opt})
            """
        )

    def _cleanup_postgres_tables(self) -> None:
        """Drop all tables from the PostgreSQL database for test isolation."""
        import psycopg2

        con = psycopg2.connect(self.metadata_path)
        try:
            con.autocommit = True
            cur = con.cursor()
            try:
                cur.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
                tables = [row[0] for row in cur.fetchall()]
                for table in tables:
                    safe = table.replace('"', '""')
                    cur.execute(f'DROP TABLE IF EXISTS "{safe}" CASCADE')
            finally:
                cur.close()
        finally:
            con.close()

    def execute(self, sql: str, params: list[Any] | None = None) -> Any:
        if params is not None:
            return self._con.execute(sql, params)
        return self._con.execute(sql)

    def fetchone(self, sql: str) -> Any:
        return self._con.execute(sql).fetchone()

    def fetchall(self, sql: str) -> list[Any]:
        return self._con.execute(sql).fetchall()

    def query_metadata(self, sql: str, params: list[Any] | None = None) -> Any:
        """
        Query the underlying metadata database directly.

        Use this after ``close()`` to read catalog metadata without going
        through DuckDB. Handles placeholder translation for PostgreSQL.
        """
        if self.backend == "sqlite":
            import sqlite3

            con = sqlite3.connect(self.metadata_path)
            try:
                return con.execute(sql, params or []).fetchone()
            finally:
                con.close()
        else:
            import psycopg2

            con = psycopg2.connect(self.metadata_path)
            try:
                con.autocommit = True
                cur = con.cursor()
                try:
                    cur.execute(sql.replace("?", "%s"), params or [])
                    return cur.fetchone()
                finally:
                    cur.close()
            finally:
                con.close()

    def close(self) -> None:
        """Close the DuckDB connection, releasing the file lock."""
        if not self._closed:
            self._con.close()
            self._closed = True

    def __enter__(self) -> "DuckLakeTestCatalog":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


@pytest.fixture(params=_get_backends())
def ducklake_catalog(request, tmp_path):
    """
    Create a DuckLake catalog with data inlining disabled.

    Parametrized over available backends (always SQLite; PostgreSQL when
    ``DUCKLAKE_PG_DSN`` is set). The caller MUST call ``.close()`` before
    reading with ducklake-polars to release the file lock.
    """
    backend = request.param

    if backend == "sqlite":
        metadata_path = str(tmp_path / "test.ducklake")
    else:
        metadata_path = os.environ["DUCKLAKE_PG_DSN"]

    data_path = str(tmp_path / "data")

    catalog = DuckLakeTestCatalog(
        metadata_path=metadata_path,
        data_path=data_path,
        inline=False,
        backend=backend,
    )

    yield catalog

    catalog.close()
    if backend == "postgres":
        catalog._cleanup_postgres_tables()


@pytest.fixture(params=_get_backends())
def ducklake_catalog_inline(request, tmp_path):
    """
    Create a DuckLake catalog with data inlining enabled (default).

    Parametrized over available backends (always SQLite; PostgreSQL when
    ``DUCKLAKE_PG_DSN`` is set).
    """
    backend = request.param

    if backend == "sqlite":
        metadata_path = str(tmp_path / "test.ducklake")
    else:
        metadata_path = os.environ["DUCKLAKE_PG_DSN"]

    data_path = str(tmp_path / "data")

    catalog = DuckLakeTestCatalog(
        metadata_path=metadata_path,
        data_path=data_path,
        inline=True,
        backend=backend,
    )

    yield catalog

    catalog.close()
    if backend == "postgres":
        catalog._cleanup_postgres_tables()
