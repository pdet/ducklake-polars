"""Version compatibility / migration tests for DuckLake catalog.

Covers:
- Reading a v0.3-style catalog (without ducklake_snapshot_changes table)
- Reading a v0.4-style catalog (with ducklake_snapshot_changes + partial_max on delete files)
- Graceful handling when expected tables don't exist
"""

from __future__ import annotations

import os
import sqlite3

import polars as pl
import pytest

from ducklake_polars import read_ducklake, scan_ducklake
from ducklake_core._catalog import DuckLakeCatalogReader, SUPPORTED_DUCKLAKE_VERSIONS


# ------------------------------------------------------------------
# Helpers: create minimal catalog schemas via raw SQLite
# ------------------------------------------------------------------


def _create_v03_catalog(db_path: str, data_path: str) -> None:
    """Create a minimal v0.3-style DuckLake catalog via raw SQL.

    v0.3 catalogs do NOT have:
    - ducklake_snapshot_changes table
    - partial_max column on ducklake_delete_file
    Column names match the actual DuckLake DDL: path/path_is_relative on
    both ducklake_table and ducklake_schema, etc.
    """
    con = sqlite3.connect(db_path)
    con.executescript(f"""
        CREATE TABLE ducklake_metadata (
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            scope TEXT,
            scope_id BIGINT
        );
        INSERT INTO ducklake_metadata VALUES ('version', '0.3', NULL, NULL);
        INSERT INTO ducklake_metadata VALUES ('data_path', '{data_path}', NULL, NULL);

        CREATE TABLE ducklake_snapshot (
            snapshot_id BIGINT PRIMARY KEY,
            snapshot_time TEXT,
            schema_version BIGINT DEFAULT 0,
            next_catalog_id BIGINT DEFAULT 0,
            next_file_id BIGINT DEFAULT 0
        );

        CREATE TABLE ducklake_schema (
            schema_id BIGINT PRIMARY KEY,
            schema_uuid TEXT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            schema_name TEXT,
            path TEXT DEFAULT '',
            path_is_relative BIGINT DEFAULT 1
        );
        INSERT INTO ducklake_schema VALUES (0, 'schema-uuid-0', 0, NULL, 'main', '', 1);

        CREATE TABLE ducklake_table (
            table_id BIGINT,
            table_uuid TEXT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            schema_id BIGINT,
            table_name TEXT,
            path TEXT DEFAULT '',
            path_is_relative BIGINT DEFAULT 1
        );

        CREATE TABLE ducklake_column (
            column_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            table_id BIGINT,
            column_order BIGINT,
            column_name TEXT,
            column_type TEXT,
            initial_default TEXT,
            default_value TEXT,
            nulls_allowed BIGINT DEFAULT 1,
            parent_column BIGINT,
            default_value_type TEXT,
            default_value_dialect TEXT
        );

        CREATE TABLE ducklake_data_file (
            data_file_id BIGINT PRIMARY KEY,
            table_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            file_order BIGINT,
            path TEXT,
            path_is_relative BIGINT DEFAULT 1,
            file_format TEXT,
            record_count BIGINT,
            file_size_bytes BIGINT DEFAULT 0,
            footer_size BIGINT,
            row_id_start BIGINT DEFAULT 0,
            partition_id BIGINT,
            encryption_key TEXT,
            mapping_id BIGINT
        );

        CREATE TABLE ducklake_delete_file (
            delete_file_id BIGINT PRIMARY KEY,
            table_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            data_file_id BIGINT,
            path TEXT,
            path_is_relative BIGINT DEFAULT 1,
            format TEXT,
            delete_count BIGINT DEFAULT 0,
            file_size_bytes BIGINT,
            footer_size BIGINT,
            encryption_key TEXT
        );

        CREATE TABLE ducklake_file_column_stats (
            data_file_id BIGINT,
            table_id BIGINT,
            column_id BIGINT,
            column_size_bytes BIGINT,
            value_count BIGINT,
            null_count BIGINT,
            min_value TEXT,
            max_value TEXT,
            contains_nan BIGINT,
            extra_stats TEXT
        );

        CREATE TABLE ducklake_inlined_data_tables (
            table_id BIGINT,
            table_name TEXT,
            schema_version BIGINT DEFAULT 0
        );

        CREATE TABLE ducklake_tag (
            object_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            key TEXT,
            value TEXT
        );

        CREATE TABLE ducklake_column_tag (
            table_id BIGINT,
            column_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            key TEXT,
            value TEXT
        );

        CREATE TABLE ducklake_partition_info (
            partition_id BIGINT,
            table_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT
        );

        CREATE TABLE ducklake_file_partition_value (
            data_file_id BIGINT,
            table_id BIGINT,
            partition_key_index BIGINT,
            partition_value TEXT
        );

        CREATE TABLE ducklake_column_mapping (
            mapping_id BIGINT,
            table_id BIGINT,
            type TEXT
        );

        CREATE TABLE ducklake_name_mapping (
            mapping_id BIGINT,
            column_id BIGINT,
            source_name TEXT,
            target_field_id BIGINT,
            parent_column BIGINT,
            is_partition BIGINT
        );

        CREATE TABLE ducklake_schema_versions (
            begin_snapshot BIGINT,
            schema_version BIGINT,
            table_id BIGINT
        );

        -- Insert initial snapshot
        INSERT INTO ducklake_snapshot VALUES (0, '2024-01-01 00:00:00', 1, 2, 0);

        -- Create a test table
        INSERT INTO ducklake_table VALUES (0, 'table-uuid-0', 0, NULL, 0, 'legacy_table', '', 1);

        -- Add columns (id=0: a INTEGER, id=1: b VARCHAR)
        INSERT INTO ducklake_column VALUES (0, 0, NULL, 0, 0, 'a', 'INTEGER', NULL, NULL, 1, NULL, NULL, NULL);
        INSERT INTO ducklake_column VALUES (1, 0, NULL, 0, 1, 'b', 'VARCHAR', NULL, NULL, 1, NULL, NULL, NULL);

        -- Schema versions entry
        INSERT INTO ducklake_schema_versions VALUES (0, 1, 0);
    """)
    con.close()


def _create_v04_catalog(db_path: str, data_path: str) -> None:
    """Create a minimal v0.4-style DuckLake catalog via raw SQL.

    v0.4 catalogs add:
    - ducklake_snapshot_changes table
    - partial_max column on ducklake_delete_file
    """
    con = sqlite3.connect(db_path)
    con.executescript(f"""
        CREATE TABLE ducklake_metadata (
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            scope TEXT,
            scope_id BIGINT
        );
        INSERT INTO ducklake_metadata VALUES ('version', '0.4', NULL, NULL);
        INSERT INTO ducklake_metadata VALUES ('data_path', '{data_path}', NULL, NULL);

        CREATE TABLE ducklake_snapshot (
            snapshot_id BIGINT PRIMARY KEY,
            snapshot_time TEXT,
            schema_version BIGINT DEFAULT 0,
            next_catalog_id BIGINT DEFAULT 0,
            next_file_id BIGINT DEFAULT 0
        );

        CREATE TABLE ducklake_snapshot_changes (
            snapshot_id BIGINT PRIMARY KEY,
            changes_made TEXT,
            author TEXT,
            commit_message TEXT,
            commit_extra_info TEXT
        );

        CREATE TABLE ducklake_schema (
            schema_id BIGINT PRIMARY KEY,
            schema_uuid TEXT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            schema_name TEXT,
            path TEXT DEFAULT '',
            path_is_relative BIGINT DEFAULT 1
        );
        INSERT INTO ducklake_schema VALUES (0, 'schema-uuid-0', 0, NULL, 'main', '', 1);

        CREATE TABLE ducklake_table (
            table_id BIGINT,
            table_uuid TEXT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            schema_id BIGINT,
            table_name TEXT,
            path TEXT DEFAULT '',
            path_is_relative BIGINT DEFAULT 1
        );

        CREATE TABLE ducklake_column (
            column_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            table_id BIGINT,
            column_order BIGINT,
            column_name TEXT,
            column_type TEXT,
            initial_default TEXT,
            default_value TEXT,
            nulls_allowed BIGINT DEFAULT 1,
            parent_column BIGINT,
            default_value_type TEXT,
            default_value_dialect TEXT
        );

        CREATE TABLE ducklake_data_file (
            data_file_id BIGINT PRIMARY KEY,
            table_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            file_order BIGINT,
            path TEXT,
            path_is_relative BIGINT DEFAULT 1,
            file_format TEXT,
            record_count BIGINT,
            file_size_bytes BIGINT DEFAULT 0,
            footer_size BIGINT,
            row_id_start BIGINT DEFAULT 0,
            partition_id BIGINT,
            encryption_key TEXT,
            mapping_id BIGINT,
            partial_max BIGINT
        );

        CREATE TABLE ducklake_delete_file (
            delete_file_id BIGINT PRIMARY KEY,
            table_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            data_file_id BIGINT,
            path TEXT,
            path_is_relative BIGINT DEFAULT 1,
            format TEXT,
            delete_count BIGINT DEFAULT 0,
            file_size_bytes BIGINT,
            footer_size BIGINT,
            encryption_key TEXT,
            partial_max BIGINT
        );

        CREATE TABLE ducklake_file_column_stats (
            data_file_id BIGINT,
            table_id BIGINT,
            column_id BIGINT,
            column_size_bytes BIGINT,
            value_count BIGINT,
            null_count BIGINT,
            min_value TEXT,
            max_value TEXT,
            contains_nan BIGINT,
            extra_stats TEXT
        );

        CREATE TABLE ducklake_inlined_data_tables (
            table_id BIGINT,
            table_name TEXT,
            schema_version BIGINT DEFAULT 0
        );

        CREATE TABLE ducklake_tag (
            object_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            key TEXT,
            value TEXT
        );

        CREATE TABLE ducklake_column_tag (
            table_id BIGINT,
            column_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT,
            key TEXT,
            value TEXT
        );

        CREATE TABLE ducklake_partition_info (
            partition_id BIGINT,
            table_id BIGINT,
            begin_snapshot BIGINT DEFAULT 0,
            end_snapshot BIGINT
        );

        CREATE TABLE ducklake_file_partition_value (
            data_file_id BIGINT,
            table_id BIGINT,
            partition_key_index BIGINT,
            partition_value TEXT
        );

        CREATE TABLE ducklake_column_mapping (
            mapping_id BIGINT,
            table_id BIGINT,
            type TEXT
        );

        CREATE TABLE ducklake_name_mapping (
            mapping_id BIGINT,
            column_id BIGINT,
            source_name TEXT,
            target_field_id BIGINT,
            parent_column BIGINT,
            is_partition BIGINT
        );

        CREATE TABLE ducklake_schema_versions (
            begin_snapshot BIGINT,
            schema_version BIGINT,
            table_id BIGINT
        );

        -- Insert initial snapshot
        INSERT INTO ducklake_snapshot VALUES (0, '2024-01-01 00:00:00', 1, 2, 0);

        -- Create a test table
        INSERT INTO ducklake_table VALUES (0, 'table-uuid-0', 0, NULL, 0, 'modern_table', '', 1);

        -- Add columns
        INSERT INTO ducklake_column VALUES (0, 0, NULL, 0, 0, 'id', 'INTEGER', NULL, NULL, 1, NULL, NULL, NULL);
        INSERT INTO ducklake_column VALUES (1, 0, NULL, 0, 1, 'name', 'VARCHAR', NULL, NULL, 1, NULL, NULL, NULL);

        -- Snapshot changes record
        INSERT INTO ducklake_snapshot_changes VALUES (0, 'create_table:"main"."modern_table"', NULL, NULL, NULL);

        -- Schema versions entry
        INSERT INTO ducklake_schema_versions VALUES (0, 1, 0);
    """)
    con.close()


# ------------------------------------------------------------------
# v0.3 catalog compatibility
# ------------------------------------------------------------------


class TestV03Catalog:
    """Test reading a v0.3-style catalog (no snapshot_changes table)."""

    def test_read_v03_catalog_metadata(self, tmp_path):
        """Can open and read metadata from a v0.3 catalog."""
        db_path = str(tmp_path / "v03.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        _create_v03_catalog(db_path, data_path)

        with DuckLakeCatalogReader(db_path, data_path_override=data_path) as reader:
            snap = reader.get_current_snapshot()
            assert snap.snapshot_id == 0

            table = reader.get_table("legacy_table", "main", snap.snapshot_id)
            assert table.table_name == "legacy_table"

            cols = reader.get_columns(table.table_id, snap.snapshot_id)
            col_names = [c.column_name for c in cols]
            assert "a" in col_names
            assert "b" in col_names

    def test_read_v03_catalog_empty_table(self, tmp_path):
        """Reading an empty v0.3 table returns empty DataFrame."""
        db_path = str(tmp_path / "v03.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        _create_v03_catalog(db_path, data_path)

        result = read_ducklake(db_path, "legacy_table", data_path=data_path)
        assert result.shape[0] == 0
        assert "a" in result.columns
        assert "b" in result.columns

    def test_v03_no_snapshot_changes_table(self, tmp_path):
        """v0.3 catalog has no ducklake_snapshot_changes — should not error."""
        db_path = str(tmp_path / "v03.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        _create_v03_catalog(db_path, data_path)

        # Verify ducklake_snapshot_changes doesn't exist
        con = sqlite3.connect(db_path)
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        con.close()
        assert "ducklake_snapshot_changes" not in tables

        # Reading should still work fine
        result = read_ducklake(db_path, "legacy_table", data_path=data_path)
        assert result.shape[0] == 0


# ------------------------------------------------------------------
# v0.4 catalog compatibility
# ------------------------------------------------------------------


class TestV04Catalog:
    """Test reading a v0.4-style catalog."""

    def test_read_v04_catalog_metadata(self, tmp_path):
        """Can open and read metadata from a v0.4 catalog."""
        db_path = str(tmp_path / "v04.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        _create_v04_catalog(db_path, data_path)

        with DuckLakeCatalogReader(db_path, data_path_override=data_path) as reader:
            snap = reader.get_current_snapshot()
            assert snap.snapshot_id == 0

            table = reader.get_table("modern_table", "main", snap.snapshot_id)
            assert table.table_name == "modern_table"

            cols = reader.get_columns(table.table_id, snap.snapshot_id)
            col_names = [c.column_name for c in cols]
            assert "id" in col_names
            assert "name" in col_names

    def test_read_v04_catalog_empty_table(self, tmp_path):
        """Reading an empty v0.4 table returns empty DataFrame."""
        db_path = str(tmp_path / "v04.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        _create_v04_catalog(db_path, data_path)

        result = read_ducklake(db_path, "modern_table", data_path=data_path)
        assert result.shape[0] == 0
        assert "id" in result.columns
        assert "name" in result.columns

    def test_v04_has_snapshot_changes_table(self, tmp_path):
        """v0.4 catalog has ducklake_snapshot_changes table."""
        db_path = str(tmp_path / "v04.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        _create_v04_catalog(db_path, data_path)

        con = sqlite3.connect(db_path)
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        con.close()
        assert "ducklake_snapshot_changes" in tables


# ------------------------------------------------------------------
# Unsupported versions
# ------------------------------------------------------------------


class TestUnsupportedVersion:
    """Test graceful handling of unsupported catalog versions."""

    def test_unsupported_version_raises(self, tmp_path):
        """Opening a catalog with an unsupported version raises ValueError."""
        db_path = str(tmp_path / "bad.ducklake")
        con = sqlite3.connect(db_path)
        con.execute(
            "CREATE TABLE ducklake_metadata (key TEXT NOT NULL, value TEXT NOT NULL)"
        )
        con.execute("INSERT INTO ducklake_metadata VALUES ('version', '0.1')")
        con.commit()
        con.close()

        with pytest.raises(ValueError, match="Unsupported DuckLake catalog version"):
            with DuckLakeCatalogReader(db_path) as reader:
                reader.get_current_snapshot()

    def test_missing_version_raises(self, tmp_path):
        """Opening a catalog with no version entry raises ValueError."""
        db_path = str(tmp_path / "noversion.ducklake")
        con = sqlite3.connect(db_path)
        con.execute(
            "CREATE TABLE ducklake_metadata (key TEXT NOT NULL, value TEXT NOT NULL)"
        )
        con.execute("INSERT INTO ducklake_metadata VALUES ('data_path', '/tmp')")
        con.commit()
        con.close()

        with pytest.raises(ValueError, match="No version found"):
            with DuckLakeCatalogReader(db_path) as reader:
                reader.get_current_snapshot()


# ------------------------------------------------------------------
# Supported versions constant
# ------------------------------------------------------------------


def test_supported_versions_include_expected():
    """SUPPORTED_DUCKLAKE_VERSIONS includes 0.3, 0.4, and 1.0."""
    assert "0.3" in SUPPORTED_DUCKLAKE_VERSIONS
    assert "0.4" in SUPPORTED_DUCKLAKE_VERSIONS
    assert "1.0" in SUPPORTED_DUCKLAKE_VERSIONS


# ------------------------------------------------------------------
# v1.0 catalog compatibility
# ------------------------------------------------------------------


def _create_v10_catalog(db_path: str, data_path: str) -> None:
    """Create a minimal v1.0-style DuckLake catalog via raw SQL.

    The v1.0 schema is identical to v0.4; the only on-disk change is
    ``ducklake_metadata.value = '1.0'``. (See MigrateV04 in
    ducklake_metadata_manager.cpp.)
    """
    _create_v04_catalog(db_path, data_path)
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "UPDATE ducklake_metadata SET value = '1.0' WHERE key = 'version'"
        )
        con.execute(
            "UPDATE ducklake_table SET table_name = 'modern10_table' "
            "WHERE table_name = 'modern_table'"
        )
        con.commit()
    finally:
        con.close()


class TestV10Catalog:
    """Test reading a v1.0-style catalog."""

    def test_read_v10_catalog_metadata(self, tmp_path):
        """Can open and read metadata from a v1.0 catalog."""
        db_path = str(tmp_path / "v10.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        _create_v10_catalog(db_path, data_path)

        with DuckLakeCatalogReader(db_path, data_path_override=data_path) as reader:
            snap = reader.get_current_snapshot()
            assert snap.snapshot_id == 0

            table = reader.get_table("modern10_table", "main", snap.snapshot_id)
            assert table.table_name == "modern10_table"

            cols = reader.get_columns(table.table_id, snap.snapshot_id)
            col_names = [c.column_name for c in cols]
            assert "id" in col_names
            assert "name" in col_names

    def test_read_v10_catalog_empty_table(self, tmp_path):
        """Reading an empty v1.0 table returns empty DataFrame."""
        db_path = str(tmp_path / "v10.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        _create_v10_catalog(db_path, data_path)

        result = read_ducklake(db_path, "modern10_table", data_path=data_path)
        assert result.shape[0] == 0
        assert "id" in result.columns
        assert "name" in result.columns

    def test_read_v10_catalog_version_recognized(self, tmp_path):
        """v1.0 version string is accepted (no CatalogVersionError)."""
        db_path = str(tmp_path / "v10.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        _create_v10_catalog(db_path, data_path)

        with DuckLakeCatalogReader(db_path, data_path_override=data_path) as reader:
            reader.get_current_snapshot()
            assert reader._catalog_version == "1.0"
