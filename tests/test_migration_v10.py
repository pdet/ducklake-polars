"""Migration tests for catalog auto-upgrade to DuckLake 1.0."""

from __future__ import annotations

import os
import sqlite3

import duckdb
import pytest

from ducklake_core._catalog import DuckLakeCatalogReader
from ducklake_core._migration import (
    migrate_catalog,
    migrate_to_latest,
    migrate_v03_to_v04,
    migrate_v04_to_v10,
)
from ducklake_core._exceptions import CatalogVersionError

# Re-use the v0.3 / v0.4 catalog factories from the existing migration test.
from tests.test_migration import _create_v03_catalog, _create_v04_catalog


# ------------------------------------------------------------------
# 0.4 → 1.0 (version bump only)
# ------------------------------------------------------------------


class TestV04ToV10:
    def test_version_bumped(self, tmp_path):
        db_path = str(tmp_path / "v04.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)
        _create_v04_catalog(db_path, data_path)

        new_version = migrate_catalog(db_path)
        assert new_version == "1.0"

        con = sqlite3.connect(db_path)
        version = con.execute(
            "SELECT value FROM ducklake_metadata WHERE key = 'version'"
        ).fetchone()
        con.close()
        assert version == ("1.0",)

    def test_idempotent(self, tmp_path):
        db_path = str(tmp_path / "v04.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)
        _create_v04_catalog(db_path, data_path)

        migrate_catalog(db_path)
        # Second call is a no-op
        assert migrate_catalog(db_path) == "1.0"


# ------------------------------------------------------------------
# 0.3 → 1.0 (full migration)
# ------------------------------------------------------------------


def _create_v03_legacy_catalog(db_path: str, data_path: str) -> None:
    """Create a v0.3 catalog whose schema is missing the 0.4 columns.

    The ``_create_v03_catalog`` helper already adds 0.4 columns (because the
    fixture writes the unified DDL); this variant produces a *strict* 0.3
    schema so we can verify the migration adds the missing columns.
    """
    con = sqlite3.connect(db_path)
    con.executescript(f"""
        CREATE TABLE ducklake_metadata (
            key TEXT NOT NULL, value TEXT NOT NULL,
            scope TEXT, scope_id BIGINT
        );
        INSERT INTO ducklake_metadata VALUES ('version', '0.3', NULL, NULL);
        INSERT INTO ducklake_metadata VALUES ('data_path', '{data_path}', NULL, NULL);

        CREATE TABLE ducklake_snapshot (
            snapshot_id BIGINT PRIMARY KEY, snapshot_time TEXT,
            schema_version BIGINT DEFAULT 0,
            next_catalog_id BIGINT DEFAULT 0,
            next_file_id BIGINT DEFAULT 0
        );
        INSERT INTO ducklake_snapshot VALUES (0, '2024-01-01 00:00:00', 1, 2, 0);

        CREATE TABLE ducklake_snapshot_changes (
            snapshot_id BIGINT PRIMARY KEY, changes_made TEXT,
            author TEXT, commit_message TEXT, commit_extra_info TEXT
        );

        CREATE TABLE ducklake_schema (
            schema_id BIGINT PRIMARY KEY, schema_uuid TEXT,
            begin_snapshot BIGINT DEFAULT 0, end_snapshot BIGINT,
            schema_name TEXT, path TEXT DEFAULT '',
            path_is_relative BIGINT DEFAULT 1
        );
        INSERT INTO ducklake_schema VALUES (0, 'schema-0', 0, NULL, 'main', '', 1);

        CREATE TABLE ducklake_table (
            table_id BIGINT, table_uuid TEXT,
            begin_snapshot BIGINT DEFAULT 0, end_snapshot BIGINT,
            schema_id BIGINT, table_name TEXT,
            path TEXT DEFAULT '', path_is_relative BIGINT DEFAULT 1
        );
        INSERT INTO ducklake_table VALUES (0, 'table-0', 0, NULL, 0, 't', '', 1);

        CREATE TABLE ducklake_column (
            column_id BIGINT, begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT, table_id BIGINT, column_order BIGINT,
            column_name TEXT, column_type TEXT,
            initial_default TEXT, default_value TEXT,
            nulls_allowed BIGINT DEFAULT 1, parent_column BIGINT
        );
        INSERT INTO ducklake_column VALUES
            (0, 0, NULL, 0, 0, 'a', 'int32', NULL, NULL, 1, NULL);

        -- v0.3 ducklake_data_file: has partial_file_info, no partial_max
        CREATE TABLE ducklake_data_file (
            data_file_id BIGINT PRIMARY KEY, table_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0, end_snapshot BIGINT,
            file_order BIGINT, path TEXT,
            path_is_relative BIGINT DEFAULT 1, file_format TEXT,
            record_count BIGINT, file_size_bytes BIGINT DEFAULT 0,
            footer_size BIGINT, row_id_start BIGINT DEFAULT 0,
            partition_id BIGINT, encryption_key TEXT,
            mapping_id BIGINT, partial_file_info TEXT
        );
        -- One row with partial_max:7 in partial_file_info
        INSERT INTO ducklake_data_file VALUES
            (1, 0, 0, NULL, 0, 'f.parquet', 1, 'parquet', 100, 1024, 100, 0,
             NULL, NULL, NULL, 'partial_max:7;other:x');

        CREATE TABLE ducklake_delete_file (
            delete_file_id BIGINT PRIMARY KEY, table_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0, end_snapshot BIGINT,
            data_file_id BIGINT, path TEXT,
            path_is_relative BIGINT DEFAULT 1, format TEXT,
            delete_count BIGINT DEFAULT 0, file_size_bytes BIGINT,
            footer_size BIGINT, encryption_key TEXT
        );

        CREATE TABLE ducklake_file_column_stats (
            data_file_id BIGINT, table_id BIGINT, column_id BIGINT,
            column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT,
            min_value TEXT, max_value TEXT,
            contains_nan BIGINT, extra_stats TEXT
        );

        CREATE TABLE ducklake_inlined_data_tables (
            table_id BIGINT, table_name TEXT, schema_version BIGINT DEFAULT 0
        );
        CREATE TABLE ducklake_tag (
            object_id BIGINT, begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT, key TEXT, value TEXT
        );
        CREATE TABLE ducklake_column_tag (
            table_id BIGINT, column_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0, end_snapshot BIGINT,
            key TEXT, value TEXT
        );
        CREATE TABLE ducklake_partition_info (
            partition_id BIGINT, table_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0, end_snapshot BIGINT
        );
        CREATE TABLE ducklake_partition_column (
            partition_id BIGINT, table_id BIGINT, partition_key_index BIGINT,
            column_id BIGINT, transform TEXT
        );
        CREATE TABLE ducklake_file_partition_value (
            data_file_id BIGINT, table_id BIGINT,
            partition_key_index BIGINT, partition_value TEXT
        );
        CREATE TABLE ducklake_column_mapping (
            mapping_id BIGINT, table_id BIGINT, type TEXT
        );
        CREATE TABLE ducklake_name_mapping (
            mapping_id BIGINT, column_id BIGINT, source_name TEXT,
            target_field_id BIGINT, parent_column BIGINT, is_partition BIGINT
        );

        -- v0.3 schema_versions: no table_id column
        CREATE TABLE ducklake_schema_versions (
            begin_snapshot BIGINT, schema_version BIGINT
        );
        INSERT INTO ducklake_schema_versions VALUES (0, 1);
    """)
    con.close()


class TestV03ToV10:
    def test_full_migration_path(self, tmp_path):
        db_path = str(tmp_path / "v03.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)
        _create_v03_legacy_catalog(db_path, data_path)

        new_version = migrate_catalog(db_path)
        assert new_version == "1.0"

        con = sqlite3.connect(db_path)
        try:
            version = con.execute(
                "SELECT value FROM ducklake_metadata WHERE key = 'version'"
            ).fetchone()
            assert version == ("1.0",)

            # New columns added to ducklake_column
            col_cols = [r[1] for r in con.execute(
                "PRAGMA table_info(ducklake_column)"
            ).fetchall()]
            assert "default_value_type" in col_cols
            assert "default_value_dialect" in col_cols

            # New columns on ducklake_data_file: partial_max present, partial_file_info gone
            df_cols = [r[1] for r in con.execute(
                "PRAGMA table_info(ducklake_data_file)"
            ).fetchall()]
            assert "partial_max" in df_cols
            assert "partial_file_info" not in df_cols

            # partial_max backfilled from partial_file_info
            partial = con.execute(
                "SELECT data_file_id, partial_max FROM ducklake_data_file"
            ).fetchone()
            assert partial == (1, 7)

            # ducklake_delete_file gained partial_max
            del_cols = [r[1] for r in con.execute(
                "PRAGMA table_info(ducklake_delete_file)"
            ).fetchall()]
            assert "partial_max" in del_cols

            # ducklake_schema_versions has table_id, NULL rows replaced by per-table ones
            sv_cols = [r[1] for r in con.execute(
                "PRAGMA table_info(ducklake_schema_versions)"
            ).fetchall()]
            assert "table_id" in sv_cols
            null_count = con.execute(
                "SELECT COUNT(*) FROM ducklake_schema_versions WHERE table_id IS NULL"
            ).fetchone()[0]
            assert null_count == 0
            sv_rows = con.execute(
                "SELECT begin_snapshot, schema_version, table_id "
                "FROM ducklake_schema_versions"
            ).fetchall()
            assert (0, 1, 0) in sv_rows

            # New tables created
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            for t in [
                "ducklake_macro", "ducklake_macro_impl", "ducklake_macro_parameters",
                "ducklake_sort_info", "ducklake_sort_expression",
                "ducklake_file_variant_stats",
            ]:
                assert t in tables
        finally:
            con.close()

    def test_reader_with_automatic_migration(self, tmp_path):
        """DuckLakeCatalogReader auto-migrates when automatic_migration=True."""
        db_path = str(tmp_path / "v03.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)
        _create_v03_legacy_catalog(db_path, data_path)

        # Tag the catalog as a pre-1.0 unsupported version (e.g. simulated 0.2)
        con = sqlite3.connect(db_path)
        # Keep version=0.3 — it is supported, but auto-migration should still
        # bump it to 1.0 when requested.
        con.close()

        with DuckLakeCatalogReader(
            db_path,
            data_path_override=data_path,
            automatic_migration=True,
        ) as reader:
            snap = reader.get_current_snapshot()
            assert snap.snapshot_id == 0
            assert reader._catalog_version == "1.0"

        # Catalog is now physically 1.0 on disk.
        con = sqlite3.connect(db_path)
        version = con.execute(
            "SELECT value FROM ducklake_metadata WHERE key = 'version'"
        ).fetchone()
        con.close()
        assert version == ("1.0",)


# ------------------------------------------------------------------
# Unsupported version handling
# ------------------------------------------------------------------


def test_unsupported_version_without_flag(tmp_path):
    """An unsupported version raises CatalogVersionError without the flag."""
    db_path = str(tmp_path / "bad.ducklake")
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE ducklake_metadata (key TEXT NOT NULL, value TEXT NOT NULL)")
    con.execute("INSERT INTO ducklake_metadata VALUES ('version', '0.1')")
    con.commit()
    con.close()

    with pytest.raises(CatalogVersionError, match="Unsupported"):
        with DuckLakeCatalogReader(db_path) as reader:
            reader.get_current_snapshot()


def test_unsupported_version_with_flag_still_raises_for_pre_03(tmp_path):
    """Auto-migration only handles 0.3+; 0.1/0.2 still raise."""
    db_path = str(tmp_path / "bad.ducklake")
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE ducklake_metadata (key TEXT NOT NULL, value TEXT NOT NULL)")
    con.execute("INSERT INTO ducklake_metadata VALUES ('version', '0.2')")
    con.commit()
    con.close()

    with pytest.raises(ValueError, match="Cannot auto-migrate"):
        with DuckLakeCatalogReader(
            db_path, automatic_migration=True,
        ) as reader:
            reader.get_current_snapshot()


# ------------------------------------------------------------------
# DuckDB roundtrip (when DuckDB ≥ 1.5 is available)
# ------------------------------------------------------------------


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_migrated_catalog_readable_by_duckdb(tmp_path):
    """A 0.3 catalog migrated to 1.0 via our code is readable by DuckDB."""
    db_path = str(tmp_path / "migrated.ducklake")
    data_path = str(tmp_path / "data")
    os.makedirs(data_path, exist_ok=True)
    _create_v03_legacy_catalog(db_path, data_path)

    migrate_catalog(db_path)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{db_path}' AS d "
        f"(DATA_PATH '{data_path}', OVERRIDE_DATA_PATH true)"
    )
    # The table from the legacy catalog is visible
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_catalog = 'd'"
    ).fetchall()
    con.close()
    assert ("t",) in tables
