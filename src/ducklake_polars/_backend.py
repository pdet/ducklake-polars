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
        if params is None:
            self._cur.execute(sql)
        else:
            self._cur.execute(sql, params)
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
                "Install it with: pip install ducklake-polars[postgres]"
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


def create_backend(path: str) -> SQLiteBackend | PostgreSQLBackend:
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
    return SQLiteBackend(path=path)
