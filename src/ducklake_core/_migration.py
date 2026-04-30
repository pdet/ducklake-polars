"""Catalog version migration to DuckLake 1.0.

Ports the migration SQL from ``ducklake/src/storage/ducklake_metadata_manager.cpp``
to plain SQL that runs on both SQLite and PostgreSQL.

Only 0.3 → 0.4 → 1.0 is implemented. Catalogs older than 0.3 must first be
opened with the DuckDB ``ducklake`` extension.
"""

from __future__ import annotations

import re
from typing import Any

# Schema-level DDL — uses ``IF NOT EXISTS`` / ``IF EXISTS`` so each migration
# is idempotent (matches the C++ ``allow_failures=true`` mode).

_MIGRATE_V03_DDL: list[str] = [
    # Macros
    """CREATE TABLE IF NOT EXISTS ducklake_macro (
        schema_id BIGINT, macro_id BIGINT, macro_name VARCHAR,
        begin_snapshot BIGINT, end_snapshot BIGINT
    )""",
    """CREATE TABLE IF NOT EXISTS ducklake_macro_impl (
        macro_id BIGINT, impl_id BIGINT, dialect VARCHAR,
        sql VARCHAR, type VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS ducklake_macro_parameters (
        macro_id BIGINT, impl_id BIGINT, column_id BIGINT,
        parameter_name VARCHAR, parameter_type VARCHAR,
        default_value VARCHAR, default_value_type VARCHAR
    )""",
    # Sort
    """CREATE TABLE IF NOT EXISTS ducklake_sort_info (
        sort_id BIGINT, table_id BIGINT,
        begin_snapshot BIGINT, end_snapshot BIGINT
    )""",
    """CREATE TABLE IF NOT EXISTS ducklake_sort_expression (
        sort_id BIGINT, table_id BIGINT, sort_key_index BIGINT,
        expression VARCHAR, dialect VARCHAR,
        sort_direction VARCHAR, null_order VARCHAR
    )""",
    # Variant stats
    """CREATE TABLE IF NOT EXISTS ducklake_file_variant_stats (
        data_file_id BIGINT, table_id BIGINT, column_id BIGINT,
        variant_path VARCHAR, shredded_type VARCHAR,
        column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT,
        min_value VARCHAR, max_value VARCHAR,
        contains_nan BOOLEAN, extra_stats VARCHAR
    )""",
]


def _column_exists(con: Any, table: str, column: str, *, backend: str) -> bool:
    if backend == "sqlite":
        rows = con.execute(f'PRAGMA table_info("{table}")').fetchall()
        return any(r[1] == column for r in rows)
    rows = con.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s",
        [table, column],
    ).fetchall()
    return bool(rows)


def _add_column_if_missing(
    con: Any,
    table: str,
    column: str,
    coltype: str,
    *,
    backend: str,
    default: str | None = None,
) -> None:
    if _column_exists(con, table, column, backend=backend):
        return
    default_clause = f" DEFAULT {default}" if default is not None else ""
    con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {coltype}{default_clause}')


def _drop_column_if_present(con: Any, table: str, column: str, *, backend: str) -> None:
    if not _column_exists(con, table, column, backend=backend):
        return
    con.execute(f'ALTER TABLE "{table}" DROP COLUMN "{column}"')


_PARTIAL_MAX_RE = re.compile(r"partial_max:(\d+)")


def migrate_v03_to_v04(con: Any, *, backend: str) -> None:
    """Bring a v0.3 catalog up to v0.4.

    Mirrors ``MigrateV03`` in ``ducklake_metadata_manager.cpp`` (with
    ``allow_failures=true``).
    """
    for ddl in _MIGRATE_V03_DDL:
        con.execute(ddl)

    _add_column_if_missing(
        con, "ducklake_column", "default_value_type", "VARCHAR",
        backend=backend, default="'literal'",
    )
    con.execute(
        "UPDATE ducklake_column SET default_value_type = 'literal' "
        "WHERE default_value_type IS NULL"
    )
    _add_column_if_missing(
        con, "ducklake_column", "default_value_dialect", "VARCHAR",
        backend=backend,
    )
    _add_column_if_missing(
        con, "ducklake_schema_versions", "table_id", "BIGINT",
        backend=backend,
    )
    _add_column_if_missing(
        con, "ducklake_data_file", "partial_max", "BIGINT", backend=backend,
    )
    _add_column_if_missing(
        con, "ducklake_delete_file", "partial_max", "BIGINT", backend=backend,
    )

    # Backfill ``partial_max`` from the legacy ``partial_file_info`` text column,
    # then drop the legacy column.
    if _column_exists(con, "ducklake_data_file", "partial_file_info", backend=backend):
        rows = con.execute(
            "SELECT data_file_id, partial_file_info "
            "FROM ducklake_data_file "
            "WHERE partial_file_info IS NOT NULL "
            "AND partial_file_info LIKE '%partial_max:%'"
        ).fetchall()
        ph = "?" if backend == "sqlite" else "%s"
        for data_file_id, info in rows:
            m = _PARTIAL_MAX_RE.search(info or "")
            if m is None:
                continue
            con.execute(
                f"UPDATE ducklake_data_file SET partial_max = {ph} "
                f"WHERE data_file_id = {ph}",
                [int(m.group(1)), data_file_id],
            )
        _drop_column_if_present(
            con, "ducklake_data_file", "partial_file_info", backend=backend,
        )

    # Migrate ducklake_schema_versions.table_id (per-table tracking).
    # For each pre-1.0 row with table_id IS NULL, create one row per table
    # whose lifetime overlapped that snapshot range.
    con.execute(
        "INSERT INTO ducklake_schema_versions (table_id, begin_snapshot, schema_version) "
        "SELECT t.table_id, sv.begin_snapshot, sv.schema_version "
        "FROM ducklake_schema_versions sv "
        "JOIN ducklake_table t "
        "  ON sv.begin_snapshot >= t.begin_snapshot "
        "  AND sv.begin_snapshot <= COALESCE(t.end_snapshot, sv.begin_snapshot) "
        "WHERE sv.table_id IS NULL"
    )
    con.execute("DELETE FROM ducklake_schema_versions WHERE table_id IS NULL")

    # Stamp the new version
    ph = "?" if backend == "sqlite" else "%s"
    con.execute(
        f"UPDATE ducklake_metadata SET value = {ph} WHERE key = 'version'",
        ["0.4"],
    )


def migrate_v04_to_v10(con: Any, *, backend: str) -> None:
    """Bring a v0.4 catalog up to v1.0 (just a version-row bump)."""
    ph = "?" if backend == "sqlite" else "%s"
    con.execute(
        f"UPDATE ducklake_metadata SET value = {ph} WHERE key = 'version'",
        ["1.0"],
    )


def migrate_catalog(metadata_path: str) -> str:
    """Migrate the catalog at ``metadata_path`` to DuckLake 1.0 in-place.

    Returns the new version string. Idempotent — calling on a 1.0 catalog
    is a no-op.
    """
    from ducklake_core._backend import create_backend

    backend = create_backend(metadata_path)
    backend_kind = "sqlite" if backend.placeholder == "?" else "postgres"

    # Read current version with a read-only connection
    ro = backend.connect()
    try:
        row = ro.execute(
            "SELECT value FROM ducklake_metadata WHERE key = 'version'"
        ).fetchone()
    finally:
        ro.close()
    if row is None:
        raise ValueError(
            "No version found in ducklake_metadata — is this a valid DuckLake catalog?"
        )
    current_version = row[0]
    if current_version == "1.0":
        return "1.0"

    write_con = backend.connect_writable()
    try:
        new_version = migrate_to_latest(
            write_con, current_version=current_version, backend=backend_kind,
        )
        write_con.commit()
        return new_version
    except Exception:
        try:
            write_con.rollback()
        except Exception:
            pass
        raise
    finally:
        write_con.close()


def migrate_to_latest(con: Any, *, current_version: str, backend: str) -> str:
    """Apply migrations until the catalog is at version 1.0.

    Returns the new version string.

    Only ``0.3`` and ``0.4`` are accepted as starting points. Use the DuckDB
    ducklake extension to migrate older catalogs first.
    """
    if current_version == "1.0":
        return "1.0"
    if current_version == "0.4":
        migrate_v04_to_v10(con, backend=backend)
        return "1.0"
    if current_version == "0.3":
        migrate_v03_to_v04(con, backend=backend)
        migrate_v04_to_v10(con, backend=backend)
        return "1.0"
    raise ValueError(
        f"Cannot auto-migrate catalog version '{current_version}' to 1.0. "
        f"Use the DuckDB ducklake extension to migrate older versions first, "
        f"or open the catalog with that extension once."
    )
