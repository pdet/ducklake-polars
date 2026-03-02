"""Backend adapters for DuckLake metadata catalog connections."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass
class SQLiteBackend:
    """SQLite metadata backend."""

    path: str
    placeholder: str = "?"

    def connect(self) -> sqlite3.Connection:
        """Open a read-only SQLite connection."""
        abs_path = os.path.abspath(self.path)
        return sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)

    def connect_writable(self) -> sqlite3.Connection:
        """Open a read-write SQLite connection."""
        abs_path = os.path.abspath(self.path)
        return sqlite3.connect(abs_path)

    def is_table_not_found(self, exc: BaseException) -> bool:
        """Check if an exception indicates a missing table."""
        return isinstance(exc, sqlite3.OperationalError) and "no such table" in str(exc)

    def table_exists(self, con: Any, table_name: str) -> bool:
        """Check if a table exists in the SQLite database."""
        row = con.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            [table_name],
        ).fetchone()
        return row is not None and row[0] > 0


class _PsycopgConnectionWrapper:
    """
    Wraps a psycopg2 connection to provide a sqlite3-compatible interface.

    psycopg2 connections require ``cursor().execute()`` while sqlite3
    connections support ``connection.execute()`` directly.  This wrapper
    bridges the difference so ``_catalog.py`` can use the same calling
    convention for both backends.

    A single cursor is reused across calls.  This is safe because
    ``_catalog.py`` always consumes results (via ``fetchone``/``fetchall``)
    before the next ``execute`` call, and connections are single-threaded.
    """

    def __init__(self, con: Any) -> None:
        self._con = con
        self._cur = con.cursor()

    def execute(self, sql: str, params: Any = None) -> Any:
        try:
            if params is None:
                self._cur.execute(sql)
            else:
                self._cur.execute(sql, params)
        except Exception:
            # Auto-rollback on postgres to prevent InFailedSqlTransaction cascade
            try:
                self._con.rollback()
            except Exception:
                pass
            raise
        return self._cur

    def commit(self) -> None:
        self._con.commit()

    def rollback(self) -> None:
        self._con.rollback()

    def close(self) -> None:
        self._cur.close()
        self._con.close()


@dataclass
class PostgreSQLBackend:
    """PostgreSQL metadata backend."""

    connection_string: str
    placeholder: str = "%s"

    def _import_psycopg2(self) -> Any:
        """Import psycopg2, raising a clear error if not installed."""
        try:
            import psycopg2

            return psycopg2
        except ImportError:
            msg = (
                "psycopg2 is required for PostgreSQL catalog backends. "
                "Install it with: pip install ducklake-dataframe[postgres]"
            )
            raise ImportError(msg) from None

    def connect(self) -> Any:
        """Open a read-only PostgreSQL connection via psycopg2."""
        psycopg2 = self._import_psycopg2()
        con = psycopg2.connect(self.connection_string)
        con.set_session(readonly=True, autocommit=True)
        return _PsycopgConnectionWrapper(con)

    def connect_writable(self) -> Any:
        """Open a read-write PostgreSQL connection via psycopg2."""
        psycopg2 = self._import_psycopg2()
        con = psycopg2.connect(self.connection_string)
        # autocommit=False (default) — caller manages transactions via commit()
        return _PsycopgConnectionWrapper(con)

    def is_table_not_found(self, exc: BaseException) -> bool:
        """Check if an exception indicates a missing table (pgcode 42P01)."""
        try:
            import psycopg2
        except ImportError:
            return False
        return isinstance(exc, psycopg2.ProgrammingError) and getattr(exc, "pgcode", None) == "42P01"

    def table_exists(self, con: Any, table_name: str) -> bool:
        """Check if a table exists in the Postgres database."""
        row = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = %s",
            [table_name],
        ).fetchone()
        return row is not None and row[0] > 0


class _DuckDBConnectionWrapper:
    """
    Wraps a duckdb.DuckDBPyConnection to provide a sqlite3-compatible interface.

    DuckDB connections use ``conn.execute(sql).fetchall()`` while sqlite3
    uses ``cursor = conn.execute(sql); cursor.fetchall()``.  This wrapper
    bridges the difference.
    """

    def __init__(self, con: Any) -> None:
        self._con = con
        self._last_result: Any = None

    def execute(self, sql: str, params: Any = None) -> "_DuckDBConnectionWrapper":
        if params is None:
            self._last_result = self._con.execute(sql)
        else:
            self._last_result = self._con.execute(sql, list(params) if isinstance(params, tuple) else params)
        return self

    def fetchone(self) -> Any:
        if self._last_result is None:
            return None
        row = self._last_result.fetchone()
        return row

    def fetchall(self) -> list:
        if self._last_result is None:
            return []
        return self._last_result.fetchall()

    @property
    def description(self) -> Any:
        if self._last_result is None:
            return None
        return self._last_result.description

    @property
    def lastrowid(self) -> int | None:
        # DuckDB doesn't support lastrowid; use RETURNING or sequences
        return None

    def commit(self) -> None:
        # DuckDB autocommits by default
        pass

    def rollback(self) -> None:
        try:
            self._con.execute("ROLLBACK")
        except Exception:
            pass

    def close(self) -> None:
        self._con.close()


@dataclass
class DuckDBBackend:
    """DuckDB metadata backend."""

    path: str
    placeholder: str = "?"

    def _import_duckdb(self) -> Any:
        """Import duckdb, raising a clear error if not installed."""
        try:
            import duckdb

            return duckdb
        except ImportError:
            msg = (
                "duckdb is required for DuckDB catalog backends. "
                "Install it with: pip install duckdb"
            )
            raise ImportError(msg) from None

    def connect(self) -> Any:
        """Open a read-only DuckDB connection."""
        duckdb = self._import_duckdb()
        con = duckdb.connect(self.path, read_only=True)
        return _DuckDBConnectionWrapper(con)

    def connect_writable(self) -> Any:
        """Open a read-write DuckDB connection."""
        duckdb = self._import_duckdb()
        con = duckdb.connect(self.path)
        return _DuckDBConnectionWrapper(con)

    def is_table_not_found(self, exc: BaseException) -> bool:
        """Check if an exception indicates a missing table."""
        try:
            import duckdb
        except ImportError:
            return False
        return isinstance(exc, duckdb.CatalogException) and "does not exist" in str(exc)

    def table_exists(self, con: Any, table_name: str) -> bool:
        """Check if a table exists in the DuckDB database."""
        row = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return row is not None and row[0] > 0


def create_backend(path: str) -> SQLiteBackend | PostgreSQLBackend | DuckDBBackend:
    """
    Auto-detect the backend type from the connection string.

    PostgreSQL is detected when the path starts with ``postgresql://``,
    ``postgres://``, or contains ``host=`` or ``dbname=`` (libpq key-value
    format).  Everything else is treated as a SQLite file path.
    """
    lower = path.strip().lower()
    if (
        lower.startswith("postgresql://")
        or lower.startswith("postgres://")
        or "host=" in lower
        or "dbname=" in lower
    ):
        return PostgreSQLBackend(connection_string=path.strip())
    if lower.endswith(".duckdb") or lower.startswith("duckdb:"):
        clean = path.strip()
        if clean.lower().startswith("duckdb:"):
            clean = clean[7:]
        return DuckDBBackend(path=clean)
    return SQLiteBackend(path=path)
