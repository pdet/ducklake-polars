"""Tests for custom DuckLake error hierarchy and user-facing messages."""

from __future__ import annotations

import os
import sqlite3

import polars as pl
import pyarrow as pa
import pytest

from ducklake_core import (
    CatalogVersionError,
    DuckLakeError,
    SchemaNotFoundError,
    TableNotFoundError,
)
from ducklake_core._catalog import DuckLakeCatalogReader
from ducklake_core._writer import DuckLakeCatalogWriter
from ducklake_polars import read_ducklake


# ------------------------------------------------------------------
# Exception hierarchy
# ------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify exception class relationships."""

    def test_base_exception(self):
        assert issubclass(DuckLakeError, Exception)

    def test_table_not_found_is_ducklake_error(self):
        assert issubclass(TableNotFoundError, DuckLakeError)

    def test_schema_not_found_is_ducklake_error(self):
        assert issubclass(SchemaNotFoundError, DuckLakeError)

    def test_catalog_version_error_is_ducklake_error(self):
        assert issubclass(CatalogVersionError, DuckLakeError)

    def test_catch_all_ducklake_errors(self):
        """All specific errors are catchable via DuckLakeError."""
        for exc_cls in (TableNotFoundError, SchemaNotFoundError, CatalogVersionError):
            with pytest.raises(DuckLakeError):
                raise exc_cls("test")


# ------------------------------------------------------------------
# TableNotFoundError
# ------------------------------------------------------------------


class TestTableNotFoundError:
    """Reading a non-existent table should raise TableNotFoundError."""

    def test_read_nonexistent_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.real_table (x INTEGER)")
        cat.execute("INSERT INTO ducklake.real_table VALUES (1)")
        cat.close()

        with pytest.raises(TableNotFoundError, match="ghost_table"):
            read_ducklake(cat.metadata_path, "ghost_table")

    def test_catalog_reader_get_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.close()

        with DuckLakeCatalogReader(cat.metadata_path) as reader:
            snap = reader.get_current_snapshot()
            with pytest.raises(TableNotFoundError, match="no_such_table"):
                reader.get_table("no_such_table", "main", snap.snapshot_id)

    def test_writer_table_not_found(self, ducklake_catalog):
        """Writer operations on missing tables should raise TableNotFoundError."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.close()

        writer = DuckLakeCatalogWriter(cat.metadata_path, data_path_override=cat.data_path)
        with pytest.raises(TableNotFoundError, match="missing_table"):
            writer.insert_data(
                pa.table({"a": [1]}),
                "missing_table",
                schema_name="main",
            )
        writer.close()

    def test_error_message_is_helpful(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.close()

        with DuckLakeCatalogReader(cat.metadata_path) as reader:
            snap = reader.get_current_snapshot()
            try:
                reader.get_table("nope", "main", snap.snapshot_id)
            except TableNotFoundError as e:
                msg = str(e)
                assert "nope" in msg
                assert "main" in msg
                assert "snapshot" in msg.lower() or str(snap.snapshot_id) in msg


# ------------------------------------------------------------------
# SchemaNotFoundError
# ------------------------------------------------------------------


class TestSchemaNotFoundError:
    """Reading from a non-existent schema should raise SchemaNotFoundError."""

    def test_writer_resolve_schema_not_found(self, ducklake_catalog):
        """_resolve_schema_info raises SchemaNotFoundError for missing schemas."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.close()

        writer = DuckLakeCatalogWriter(cat.metadata_path, data_path_override=cat.data_path)
        with pytest.raises(SchemaNotFoundError, match="fake_schema"):
            writer.create_table(
                "new_table",
                pa.schema([pa.field("a", pa.int32())]),
                schema_name="fake_schema",
            )
        writer.close()

    def test_insert_bad_schema_raises_ducklake_error(self, ducklake_catalog):
        """insert_data with bad schema raises a DuckLakeError (table+schema joined lookup)."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.close()

        writer = DuckLakeCatalogWriter(cat.metadata_path, data_path_override=cat.data_path)
        with pytest.raises(DuckLakeError, match="fake_schema"):
            writer.insert_data(
                pa.table({"a": [1]}),
                "t",
                schema_name="fake_schema",
            )
        writer.close()


# ------------------------------------------------------------------
# CatalogVersionError
# ------------------------------------------------------------------


class TestCatalogVersionError:
    """Unsupported catalog versions should raise CatalogVersionError."""

    def test_unsupported_version(self, tmp_path):
        db_path = str(tmp_path / "bad.ducklake")
        con = sqlite3.connect(db_path)
        con.execute(
            "CREATE TABLE ducklake_metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        con.execute(
            "INSERT INTO ducklake_metadata VALUES ('version', '99.0')"
        )
        con.commit()
        con.close()

        with pytest.raises(CatalogVersionError, match="99.0"):
            with DuckLakeCatalogReader(db_path) as reader:
                reader.get_current_snapshot()

    def test_missing_version(self, tmp_path):
        db_path = str(tmp_path / "empty.ducklake")
        con = sqlite3.connect(db_path)
        con.execute(
            "CREATE TABLE ducklake_metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        con.commit()
        con.close()

        with pytest.raises(CatalogVersionError, match="No version found"):
            with DuckLakeCatalogReader(db_path) as reader:
                reader.get_current_snapshot()


# ------------------------------------------------------------------
# Invalid connection / misc
# ------------------------------------------------------------------


class TestInvalidConnection:
    """Invalid connection strings should raise appropriate errors."""

    def test_nonexistent_file(self, tmp_path):
        bad_path = str(tmp_path / "does_not_exist.ducklake")
        # Opening a non-existent SQLite file will create an empty DB,
        # which will fail on metadata query
        with pytest.raises(Exception):
            with DuckLakeCatalogReader(bad_path) as reader:
                reader.get_current_snapshot()

    def test_write_to_nonexistent_table(self, ducklake_catalog):
        """Writing to a table that doesn't exist should fail clearly."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.existing (a INTEGER)")
        cat.close()

        writer = DuckLakeCatalogWriter(cat.metadata_path, data_path_override=cat.data_path)
        with pytest.raises((TableNotFoundError, SchemaNotFoundError, DuckLakeError)):
            writer.insert_data(
                pa.table({"a": [1]}),
                "nonexistent",
                schema_name="main",
            )
        writer.close()
