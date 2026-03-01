"""Backend adapters — re-exports from ducklake_core."""

from ducklake_core._backend import (  # noqa: F401
    PostgreSQLBackend,
    SQLiteBackend,
    _PsycopgConnectionWrapper,
    create_backend,
)

__all__ = [
    "PostgreSQLBackend",
    "SQLiteBackend",
    "_PsycopgConnectionWrapper",
    "create_backend",
]
