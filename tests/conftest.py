"""Shared test fixtures for ducklake-dataframe tests."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import duckdb
import polars as pl
import pytest


def _get_backends():
    """Return pytest parameters for available backends."""
    backends = [
        pytest.param("sqlite", id="sqlite"),
        pytest.param("duckdb", id="duckdb"),
    ]
    if os.environ.get("DUCKLAKE_PG_DSN"):
        backends.append(
            pytest.param("postgres", id="postgres", marks=pytest.mark.postgres)
        )
    return backends


def _metadata_path_for(backend: str, tmp_path: Any, stem: str = "test") -> str:
    """Return a per-backend metadata path."""
    if backend == "sqlite":
        return str(tmp_path / f"{stem}.ducklake")
    if backend == "duckdb":
        return str(tmp_path / f"{stem}.duckdb")
    return os.environ["DUCKLAKE_PG_DSN"]


def _attach_source_for(backend: str, metadata_path: str) -> str:
    """Return the DuckDB ATTACH source string for the given backend."""
    if backend == "sqlite":
        return f"ducklake:sqlite:{metadata_path}"
    if backend == "duckdb":
        return f"ducklake:duckdb:{metadata_path}"
    return f"ducklake:postgres:{metadata_path}"


@dataclass
class DuckLakeTestCatalog:
    """
    Helper for creating and populating DuckLake catalogs in tests.

    Usage:
        catalog = DuckLakeTestCatalog(tmp_path)
        catalog.execute("CREATE TABLE ducklake.test (a INTEGER)")
        catalog.execute("INSERT INTO ducklake.test VALUES (1)")
        catalog.close()  # Release the file lock

        # Now read with ducklake-dataframe
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
        elif self.backend == "duckdb":
            attach_source = f"ducklake:duckdb:{self.metadata_path}"
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
        elif self.backend == "duckdb":
            con = duckdb.connect(self.metadata_path, read_only=True)
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

    def attach_source(self) -> str:
        """Return the DuckDB ATTACH source string for this catalog's backend."""
        return _attach_source_for(self.backend, self.metadata_path)

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
    reading with ducklake-dataframe to release the file lock.
    """
    backend = request.param
    metadata_path = _metadata_path_for(backend, tmp_path)
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


@pytest.fixture
def ducklake_catalog_sqlite(tmp_path):
    """
    Create a DuckLake catalog with SQLite backend only.

    Used for tests that deadlock on Postgres (e.g., many sequential
    DuckDB write operations on a Postgres-backed catalog).
    """
    metadata_path = str(tmp_path / "test.ducklake")
    data_path = str(tmp_path / "data")

    catalog = DuckLakeTestCatalog(
        metadata_path=metadata_path,
        data_path=data_path,
        inline=False,
        backend="sqlite",
    )

    yield catalog

    catalog.close()


@pytest.fixture(params=_get_backends())
def ducklake_catalog_inline(request, tmp_path):
    """
    Create a DuckLake catalog with data inlining enabled (default).

    Parametrized over available backends (always SQLite; PostgreSQL when
    ``DUCKLAKE_PG_DSN`` is set).
    """
    backend = request.param
    metadata_path = _metadata_path_for(backend, tmp_path)
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


# -----------------------------------------------------------------------
# Write-test helpers
# -----------------------------------------------------------------------


@dataclass
class WriteCatalogHelper:
    """
    Lightweight helper for write-path tests.

    Provides ``metadata_path``, ``data_path``, ``backend`` and methods for
    querying the metadata database and reading back with DuckDB. Created via
    :func:`make_write_catalog` factory fixture.
    """

    metadata_path: str
    data_path: str
    backend: str  # "sqlite", "duckdb", or "postgres"
    inline_limit: int = 0

    def attach_source(self) -> str:
        """Return the DuckDB ATTACH source string for this catalog's backend."""
        return _attach_source_for(self.backend, self.metadata_path)

    # -- metadata queries ------------------------------------------------

    def query_one(self, sql: str, params: list[Any] | None = None) -> Any:
        """Execute SQL and return one row (fetchone)."""
        if self.backend == "sqlite":
            import sqlite3

            con = sqlite3.connect(self.metadata_path)
            try:
                return con.execute(sql, params or []).fetchone()
            finally:
                con.close()
        elif self.backend == "duckdb":
            con = duckdb.connect(self.metadata_path, read_only=True)
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

    def query_all(self, sql: str, params: list[Any] | None = None) -> list[Any]:
        """Execute SQL and return all rows (fetchall)."""
        if self.backend == "sqlite":
            import sqlite3

            con = sqlite3.connect(self.metadata_path)
            try:
                return con.execute(sql, params or []).fetchall()
            finally:
                con.close()
        elif self.backend == "duckdb":
            con = duckdb.connect(self.metadata_path, read_only=True)
            try:
                return con.execute(sql, params or []).fetchall()
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
                    return cur.fetchall()
                finally:
                    cur.close()
            finally:
                con.close()

    # -- DuckDB interop --------------------------------------------------

    def read_with_duckdb(
        self, table_name: str, *, inline_limit: int | None = None
    ) -> pl.DataFrame:
        """Read a table back using DuckDB's DuckLake extension."""
        limit = inline_limit if inline_limit is not None else self.inline_limit
        source = _attach_source_for(self.backend, self.metadata_path)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{self.data_path}', DATA_INLINING_ROW_LIMIT {limit})"
        )
        cursor = con.execute(f'SELECT * FROM ducklake."{table_name}"')
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        con.close()
        if not rows:
            return pl.DataFrame({c: [] for c in columns})
        data = {c: [r[i] for r in rows] for i, c in enumerate(columns)}
        return pl.DataFrame(data)

    def read_with_duckdb_schema(
        self, table_name: str, schema_name: str, *, inline_limit: int | None = None
    ) -> pl.DataFrame:
        """Read a table in a non-default schema using DuckDB."""
        limit = inline_limit if inline_limit is not None else self.inline_limit
        source = _attach_source_for(self.backend, self.metadata_path)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{self.data_path}', DATA_INLINING_ROW_LIMIT {limit})"
        )
        safe_schema = schema_name.replace('"', '""')
        safe_table = table_name.replace('"', '""')
        cursor = con.execute(
            f'SELECT * FROM ducklake."{safe_schema}"."{safe_table}"'
        )
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        con.close()
        if not rows:
            return pl.DataFrame({c: [] for c in columns})
        data = {c: [r[i] for r in rows] for i, c in enumerate(columns)}
        return pl.DataFrame(data)

    # -- PostgreSQL cleanup -----------------------------------------------

    def cleanup(self) -> None:
        """Clean up PostgreSQL tables if applicable."""
        if self.backend == "postgres":
            import psycopg2

            con = psycopg2.connect(self.metadata_path)
            try:
                con.autocommit = True
                cur = con.cursor()
                try:
                    cur.execute(
                        "SELECT tablename FROM pg_tables "
                        "WHERE schemaname = 'public'"
                    )
                    tables = [row[0] for row in cur.fetchall()]
                    for table in tables:
                        safe = table.replace('"', '""')
                        cur.execute(f'DROP TABLE IF EXISTS "{safe}" CASCADE')
                finally:
                    cur.close()
            finally:
                con.close()

    # -- PRAGMA / table_info helper (SQLite PRAGMA → PG info_schema) -----

    def get_table_columns(self, table_name: str) -> list[tuple[str, str]]:
        """Return [(column_name, column_type)] for a raw SQL table.

        Uses ``PRAGMA table_info`` on SQLite and ``information_schema``
        on PostgreSQL.
        """
        if self.backend == "sqlite":
            rows = self.query_all(f'PRAGMA table_info("{table_name}")')
            return [(r[1], r[2]) for r in rows]
        rows = self.query_all(
            "SELECT column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table_name],
        )
        return [(r[0], r[1]) for r in rows]


def _create_write_catalog(
    tmp_path: Any,
    backend: str,
    inline: bool = False,
    inline_limit: int = 20,
) -> WriteCatalogHelper:
    """Create a DuckLake catalog via DuckDB and return a WriteCatalogHelper."""
    metadata_path = _metadata_path_for(backend, tmp_path, stem="write_test")

    data_path = str(tmp_path / "data")
    os.makedirs(data_path, exist_ok=True)

    helper = WriteCatalogHelper(
        metadata_path=metadata_path,
        data_path=data_path,
        backend=backend,
        inline_limit=inline_limit if inline else 0,
    )

    # Clean up first for PostgreSQL
    helper.cleanup()

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")

    attach_source = _attach_source_for(backend, metadata_path)

    inline_opt = (
        f", DATA_INLINING_ROW_LIMIT {inline_limit}"
        if inline
        else ", DATA_INLINING_ROW_LIMIT 0"
    )
    con.execute(
        f"ATTACH '{attach_source}' AS ducklake "
        f"(DATA_PATH '{data_path}'{inline_opt})"
    )
    con.close()

    return helper


@pytest.fixture(params=_get_backends())
def make_write_catalog(request, tmp_path):
    """
    Factory fixture for write-path tests.

    Yields a callable ``make(inline=False, inline_limit=20)`` that returns
    a :class:`WriteCatalogHelper`. Parametrized over backends.
    """
    backend = request.param
    created: list[WriteCatalogHelper] = []

    def factory(inline: bool = False, inline_limit: int = 20) -> WriteCatalogHelper:
        helper = _create_write_catalog(tmp_path, backend, inline, inline_limit)
        created.append(helper)
        return helper

    yield factory

    for h in created:
        h.cleanup()
