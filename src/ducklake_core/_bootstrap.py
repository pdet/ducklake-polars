"""Bootstrap an empty DuckLake catalog with the required schema tables."""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# DDL for the DuckLake catalog tables (matches DuckDB v1.0 catalog layout —
# identical schema to v0.4; only the ``ducklake_metadata.version`` value differs).
# ---------------------------------------------------------------------------

_CATALOG_DDL: list[str] = [
    """CREATE TABLE ducklake_metadata(
        "key" VARCHAR NOT NULL,
        "value" VARCHAR NOT NULL,
        "scope" VARCHAR,
        scope_id BIGINT
    )""",
    """CREATE TABLE ducklake_snapshot(
        snapshot_id BIGINT PRIMARY KEY,
        snapshot_time TIMESTAMPTZ,
        schema_version BIGINT,
        next_catalog_id BIGINT,
        next_file_id BIGINT
    )""",
    """CREATE TABLE ducklake_snapshot_changes(
        snapshot_id BIGINT PRIMARY KEY,
        changes_made VARCHAR,
        author VARCHAR,
        commit_message VARCHAR,
        commit_extra_info VARCHAR
    )""",
    """CREATE TABLE ducklake_schema(
        schema_id BIGINT PRIMARY KEY,
        schema_uuid UUID,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        schema_name VARCHAR,
        path VARCHAR,
        path_is_relative BOOLEAN
    )""",
    """CREATE TABLE ducklake_schema_versions(
        begin_snapshot BIGINT,
        schema_version BIGINT,
        table_id BIGINT
    )""",
    """CREATE TABLE ducklake_table(
        table_id BIGINT,
        table_uuid UUID,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        schema_id BIGINT,
        table_name VARCHAR,
        path VARCHAR,
        path_is_relative BOOLEAN
    )""",
    """CREATE TABLE ducklake_column(
        column_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        table_id BIGINT,
        column_order BIGINT,
        column_name VARCHAR,
        column_type VARCHAR,
        initial_default VARCHAR,
        default_value VARCHAR,
        nulls_allowed BOOLEAN,
        parent_column BIGINT,
        default_value_type VARCHAR,
        default_value_dialect VARCHAR
    )""",
    """CREATE TABLE ducklake_data_file(
        data_file_id BIGINT PRIMARY KEY,
        table_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        file_order BIGINT,
        path VARCHAR,
        path_is_relative BOOLEAN,
        file_format VARCHAR,
        record_count BIGINT,
        file_size_bytes BIGINT,
        footer_size BIGINT,
        row_id_start BIGINT,
        partition_id BIGINT,
        encryption_key VARCHAR,
        mapping_id BIGINT,
        partial_max BIGINT
    )""",
    """CREATE TABLE ducklake_delete_file(
        delete_file_id BIGINT PRIMARY KEY,
        table_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        data_file_id BIGINT,
        path VARCHAR,
        path_is_relative BOOLEAN,
        format VARCHAR,
        delete_count BIGINT,
        file_size_bytes BIGINT,
        footer_size BIGINT,
        encryption_key VARCHAR,
        partial_max BIGINT
    )""",
    """CREATE TABLE ducklake_file_column_stats(
        data_file_id BIGINT,
        table_id BIGINT,
        column_id BIGINT,
        column_size_bytes BIGINT,
        value_count BIGINT,
        null_count BIGINT,
        min_value VARCHAR,
        max_value VARCHAR,
        contains_nan BOOLEAN,
        extra_stats VARCHAR
    )""",
    """CREATE TABLE ducklake_file_partition_value(
        data_file_id BIGINT,
        table_id BIGINT,
        partition_key_index BIGINT,
        partition_value VARCHAR
    )""",
    """CREATE TABLE ducklake_file_variant_stats(
        data_file_id BIGINT,
        table_id BIGINT,
        column_id BIGINT,
        variant_path VARCHAR,
        shredded_type VARCHAR,
        column_size_bytes BIGINT,
        value_count BIGINT,
        null_count BIGINT,
        min_value VARCHAR,
        max_value VARCHAR,
        contains_nan BOOLEAN,
        extra_stats VARCHAR
    )""",
    """CREATE TABLE ducklake_files_scheduled_for_deletion(
        data_file_id BIGINT,
        path VARCHAR,
        path_is_relative BOOLEAN,
        schedule_start TIMESTAMPTZ
    )""",
    """CREATE TABLE ducklake_inlined_data_tables(
        table_id BIGINT,
        table_name VARCHAR,
        schema_version BIGINT
    )""",
    """CREATE TABLE ducklake_column_mapping(
        mapping_id BIGINT,
        table_id BIGINT,
        "type" VARCHAR
    )""",
    """CREATE TABLE ducklake_name_mapping(
        mapping_id BIGINT,
        column_id BIGINT,
        source_name VARCHAR,
        target_field_id BIGINT,
        parent_column BIGINT,
        is_partition BOOLEAN
    )""",
    """CREATE TABLE ducklake_partition_column(
        partition_id BIGINT,
        table_id BIGINT,
        partition_key_index BIGINT,
        column_id BIGINT,
        "transform" VARCHAR
    )""",
    """CREATE TABLE ducklake_partition_info(
        partition_id BIGINT,
        table_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT
    )""",
    """CREATE TABLE ducklake_table_stats(
        table_id BIGINT,
        record_count BIGINT,
        next_row_id BIGINT,
        file_size_bytes BIGINT
    )""",
    """CREATE TABLE ducklake_table_column_stats(
        table_id BIGINT,
        column_id BIGINT,
        contains_null BIGINT,
        contains_nan BIGINT,
        min_value VARCHAR,
        max_value VARCHAR,
        extra_stats VARCHAR
    )""",
    """CREATE TABLE ducklake_tag(
        object_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        "key" VARCHAR,
        "value" VARCHAR
    )""",
    """CREATE TABLE ducklake_column_tag(
        table_id BIGINT,
        column_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        "key" VARCHAR,
        "value" VARCHAR
    )""",
    """CREATE TABLE ducklake_view(
        view_id BIGINT,
        view_uuid UUID,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        schema_id BIGINT,
        view_name VARCHAR,
        dialect VARCHAR,
        "sql" VARCHAR,
        column_aliases VARCHAR
    )""",
    """CREATE TABLE ducklake_sort_info(
        sort_id BIGINT,
        table_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT
    )""",
    """CREATE TABLE ducklake_sort_expression(
        sort_id BIGINT,
        table_id BIGINT,
        sort_key_index BIGINT,
        expression VARCHAR,
        dialect VARCHAR,
        sort_direction VARCHAR,
        null_order VARCHAR
    )""",
    """CREATE TABLE ducklake_macro(
        schema_id BIGINT,
        macro_id BIGINT,
        macro_name VARCHAR,
        begin_snapshot BIGINT,
        end_snapshot BIGINT
    )""",
    """CREATE TABLE ducklake_macro_impl(
        macro_id BIGINT,
        impl_id BIGINT,
        dialect VARCHAR,
        "sql" VARCHAR,
        "type" VARCHAR
    )""",
    """CREATE TABLE ducklake_macro_parameters(
        macro_id BIGINT,
        impl_id BIGINT,
        column_id BIGINT,
        parameter_name VARCHAR,
        parameter_type VARCHAR,
        default_value VARCHAR,
        default_value_type VARCHAR
    )""",
]


def _needs_bootstrap(path: str) -> bool:
    """Return True if the SQLite file needs catalog bootstrapping.

    A catalog needs bootstrapping when the file does not exist, is empty
    (0 bytes), or exists but doesn't contain the ``ducklake_metadata``
    table.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return True
    if os.path.getsize(abs_path) == 0:
        return True
    # File exists and has content — check for the marker table.
    # Use a regular connection (not URI ``mode=ro``): a WAL-mode catalog needs
    # to open the ``-shm`` / ``-wal`` files, which read-only URI mode forbids.
    # If the probe errors with a transient/concurrent-access message
    # (e.g. another process — DuckDB's ducklake extension — has the file
    # attached and SQLite momentarily returns "disk I/O error" or
    # "database is locked"), default to "no bootstrap needed": the file
    # already has content, so we should not risk re-bootstrapping over a
    # populated catalog. Other OperationalErrors (genuinely corrupt header,
    # truncated file, unreadable database) propagate so the caller surfaces
    # the real error rather than silently overwriting state.
    _TRANSIENT_FRAGMENTS = ("disk i/o error", "database is locked")
    try:
        con = sqlite3.connect(abs_path, timeout=30)
        try:
            row = con.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type='table' AND name='ducklake_metadata'"
            ).fetchone()
            return row is None or row[0] == 0
        finally:
            con.close()
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if any(frag in msg for frag in _TRANSIENT_FRAGMENTS):
            return False
        raise


_INITIAL_INSERTS_TEMPLATE = (
    ("ducklake_metadata", "(?, ?, NULL, NULL)", ("version", "1.0")),
    ("ducklake_metadata", "(?, ?, NULL, NULL)", ("created_by", "ducklake-dataframe")),
    ("ducklake_metadata", "(?, ?, NULL, NULL)", ("encrypted", "false")),
)


def _seed_catalog(
    execute,
    *,
    data_path: str,
    placeholder: str = "?",
) -> None:
    """Seed an initialised catalog (tables already created) with snapshot 0.

    *execute* is a callable taking ``(sql, params)``. *placeholder* is
    ``?`` for SQLite and ``%s`` for PostgreSQL.
    """
    p = placeholder
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00")
    schema_uuid = str(uuid.uuid4())

    for table, _values, params in _INITIAL_INSERTS_TEMPLATE:
        execute(
            f"INSERT INTO {table} VALUES ({p}, {p}, NULL, NULL)", params,
        )
    execute(
        f"INSERT INTO ducklake_metadata VALUES ({p}, {p}, NULL, NULL)",
        ("data_path", data_path),
    )
    execute(
        f"INSERT INTO ducklake_snapshot VALUES ({p}, {p}, {p}, {p}, {p})",
        (0, now, 0, 1, 0),
    )
    execute(
        "INSERT INTO ducklake_schema VALUES "
        f"({p}, {p}, {p}, NULL, {p}, {p}, {p})",
        (0, schema_uuid, 0, "main", "main/", True),
    )
    # ducklake_snapshot_changes seed row for snapshot 0: matches the
    # row C++ writes at metadata_manager.cpp:182. Without this, our
    # change-feed and DuckDB diagnostics report a missing entry.
    execute(
        "INSERT INTO ducklake_snapshot_changes "
        f"(snapshot_id, changes_made, author, commit_message, commit_extra_info) "
        f"VALUES ({p}, {p}, NULL, NULL, NULL)",
        (0, 'created_schema:"main"'),
    )
    # Note: no seed row in ducklake_schema_versions. v1.0 requires
    # ``table_id`` to be non-NULL on every row; the writer inserts a row
    # per-table at creation time. (See MigrateV03 in
    # ducklake_metadata_manager.cpp, which deletes pre-1.0 NULL-table_id
    # entries.)


def bootstrap_catalog(path: str, *, data_path: str | None = None) -> None:
    """Create all DuckLake catalog tables and seed initial data.

    Parameters
    ----------
    path
        Path to the SQLite catalog file (``.ducklake``).
    data_path
        Storage path for Parquet data files.  If *None*, defaults to a
        ``data/`` directory next to the catalog file.

    This is idempotent — calling it on an already-bootstrapped catalog
    is a no-op.
    """
    if not _needs_bootstrap(path):
        return

    abs_path = os.path.abspath(path)

    # Ensure parent directory exists
    parent_dir = os.path.dirname(abs_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Resolve data_path
    if data_path is None:
        data_path = os.path.join(parent_dir, "data") if parent_dir else "data"
    data_path = os.path.abspath(data_path)

    # Ensure data_path ends with / to match DuckDB convention
    if not data_path.endswith("/"):
        data_path = data_path + "/"

    # Ensure data directory exists
    os.makedirs(data_path, exist_ok=True)

    con = sqlite3.connect(abs_path)
    try:
        # WAL mode allows concurrent readers alongside one writer; without it,
        # a reader colliding with a write transaction can surface as "disk I/O
        # error" on macOS. Persisted in the file header — one-shot at bootstrap.
        con.execute("PRAGMA journal_mode=WAL")
        # Create all catalog tables
        for ddl in _CATALOG_DDL:
            con.execute(ddl)

        def _exec(sql, params):
            con.execute(sql, params)

        _seed_catalog(_exec, data_path=data_path, placeholder="?")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _needs_bootstrap_postgres(con) -> bool:
    """Return True if a Postgres catalog is missing the marker table."""
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'ducklake_metadata' LIMIT 1"
        )
        return cur.fetchone() is None
    finally:
        cur.close()


def bootstrap_catalog_postgres(
    connection_string: str, *, data_path: str | None = None,
) -> None:
    """Bootstrap a PostgreSQL DuckLake catalog.

    Creates all DuckLake catalog tables in the target schema (default
    ``public``) and seeds the snapshot 0 / metadata rows. Idempotent.

    The PG-native types ``BOOLEAN``, ``TIMESTAMPTZ``, and ``UUID`` round
    through cleanly; the writer code reads ``bool(value)`` /
    ``str(value)`` so values from either backend are interpreted
    consistently.
    """
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "psycopg2 is required for PostgreSQL catalog bootstrap"
        ) from exc

    if data_path is None:
        # No file-system anchor for PG — caller must provide.
        raise ValueError(
            "bootstrap_catalog_postgres requires an explicit data_path"
        )
    data_path = os.path.abspath(data_path)
    if not data_path.endswith("/"):
        data_path = data_path + "/"
    os.makedirs(data_path, exist_ok=True)

    con = psycopg2.connect(connection_string)
    try:
        if not _needs_bootstrap_postgres(con):
            return
        cur = con.cursor()
        try:
            for ddl in _CATALOG_DDL:
                cur.execute(ddl)

            def _exec(sql, params):
                cur.execute(sql, params)

            _seed_catalog(_exec, data_path=data_path, placeholder="%s")
            con.commit()
        finally:
            cur.close()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
