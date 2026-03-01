"""Backend adapters — re-exports from ducklake_core."""

from ducklake_core._backend import (  # noqa: F401
    DuckDBBackend,
    PostgreSQLBackend,
    SQLiteBackend,
    _DuckDBConnectionWrapper,
    _PsycopgConnectionWrapper,
    create_backend,
)

__all__ = [
    "DuckDBBackend",
    "PostgreSQLBackend",
    "SQLiteBackend",
    "_DuckDBConnectionWrapper",
    "_PsycopgConnectionWrapper",
    "create_backend",
]
