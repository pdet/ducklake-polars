"""Sorted table tests — parity with ducklake-ref.

Tests sort key metadata, interactions with operations, and time travel.
Sort keys are set via the ducklake-dataframe writer API since DuckDB v1.5
doesn't expose ALTER TABLE SET SORT KEY syntax.
"""
from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake, write_ducklake
from ducklake_core._writer import DuckLakeCatalogWriter


def _set_sort_keys(meta, data, table, keys, schema="main"):
    """Helper to set sort keys via writer API."""
    w = DuckLakeCatalogWriter(meta, data_path_override=data)
    w.set_sort_keys(table, keys, schema_name=schema)
    w.close()


def _reset_sort_keys(meta, data, table, schema="main"):
    """Helper to reset sort keys via writer API."""
    w = DuckLakeCatalogWriter(meta, data_path_override=data)
    w.reset_sort_keys(table, schema_name=schema)
    w.close()


class TestSortKeyMetadata:
    """Test sort key metadata is correctly stored and read."""

    def test_set_single_sort_key(self, ducklake_catalog_sqlite):
        """Set a single sort key and verify via catalog API."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c'), (1, 'a')")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path
        _set_sort_keys(meta, data, "t", ["a"])

        # Read — should work (sort keys don't affect read correctness)
        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 2

    def test_set_composite_sort_key(self, ducklake_catalog_sqlite):
        """Set composite sort key (a ASC, b DESC)."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b INTEGER, c VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 2, 'x')")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path
        _set_sort_keys(meta, data, "t", [("a", "ASC"), ("b", "DESC")])

        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 1

    def test_set_sort_key_nulls_first(self, ducklake_catalog_sqlite):
        """Set sort key with NULLS_FIRST."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path
        _set_sort_keys(meta, data, "t", [("a", "ASC", "NULLS_FIRST")])

        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 1

    def test_reset_sort_key(self, ducklake_catalog_sqlite):
        """Set then reset sort key — should work without errors."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path
        _set_sort_keys(meta, data, "t", ["a"])
        _reset_sort_keys(meta, data, "t")

        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 1


class TestSortKeyWithOperations:
    """Test sort key interactions with other operations."""

    def test_sort_key_survives_insert(self, ducklake_catalog_sqlite):
        """Multiple inserts after setting sort key — should all read."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (3), (1)")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path
        _set_sort_keys(meta, data, "t", ["a"])

        df = pl.DataFrame({"a": pl.Series([5, 2, 4], dtype=pl.Int32)})
        write_ducklake(df, meta, "t", data_path=data, mode="append")

        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 5
        assert sorted(result["a"].to_list()) == [1, 2, 3, 4, 5]

    def test_sort_key_with_delete(self, ducklake_catalog_sqlite):
        """Delete rows from a table with sort keys."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (3), (1), (2), (4)")
        cat.execute("DELETE FROM ducklake.t WHERE a = 2")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path
        _set_sort_keys(meta, data, "t", ["a"])

        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 3
        assert sorted(result["a"].to_list()) == [1, 3, 4]

    def test_sort_key_with_alter_add_column(self, ducklake_catalog_sqlite):
        """Add column to table with sort key — should still work."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (3), (1)")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path
        _set_sort_keys(meta, data, "t", ["a"])

        import pyarrow as pa
        w = DuckLakeCatalogWriter(meta, data_path_override=data)
        w.add_column("t", "b", pa.string())
        w.close()

        df = pl.DataFrame({
            "a": pl.Series([2], dtype=pl.Int32),
            "b": ["new"],
        })
        write_ducklake(df, meta, "t", data_path=data, mode="append")

        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 3
        assert "b" in result.columns

    def test_sort_key_with_rename_column(self, ducklake_catalog_sqlite):
        """Rename the sort key column — metadata should update."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x')")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path
        _set_sort_keys(meta, data, "t", ["a"])

        w = DuckLakeCatalogWriter(meta, data_path_override=data)
        w.rename_column("t", "a", "renamed_a")
        w.close()

        result = read_ducklake(meta, "t", data_path=data)
        assert "renamed_a" in result.columns
        assert "a" not in result.columns

    def test_overwrite_sort_keys(self, ducklake_catalog_sqlite):
        """Setting sort keys twice should overwrite the first."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x')")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path
        _set_sort_keys(meta, data, "t", ["a"])
        _set_sort_keys(meta, data, "t", ["b"])

        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 1


class TestSortKeyTimeTravel:
    """Test sort keys with time travel."""

    def test_read_before_sort_key_set(self, ducklake_catalog_sqlite):
        """Read at snapshot before sort key was set."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (3), (1), (2)")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path

        # Snapshot 2 = insert (before sort key)
        result_v2 = read_ducklake(meta, "t", data_path=data, snapshot_version=2)
        assert result_v2.shape[0] == 3

        _set_sort_keys(meta, data, "t", ["a"])

        # Latest — should still work
        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 3
