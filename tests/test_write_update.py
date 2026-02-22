"""Tests for ducklake-polars UPDATE support."""

from __future__ import annotations

import os
import sqlite3

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    read_ducklake,
    scan_ducklake,
    update_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(tmp_path):
    """Create a DuckLake catalog via DuckDB and return (metadata_path, data_path)."""
    metadata_path = str(tmp_path / "update_test.ducklake")
    data_path = str(tmp_path / "data")
    os.makedirs(data_path, exist_ok=True)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
        f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.close()
    return metadata_path, data_path


def _read_with_duckdb(metadata_path, data_path, table_name):
    """Read a table with DuckDB's DuckLake extension."""
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
        f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
    )
    cursor = con.execute(f'SELECT * FROM ducklake."{table_name}"')
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    con.close()
    if not rows:
        return pl.DataFrame({c: [] for c in columns})
    data = {c: [r[i] for r in rows] for i, c in enumerate(columns)}
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Basic update operations
# ---------------------------------------------------------------------------


class TestUpdateBasic:
    """Test basic update operations."""

    def test_update_single_column_literal(self, tmp_path):
        """Update a single column to a literal value."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["one", "two", "three", "four", "five"],
        })
        write_ducklake(df, metadata_path, "test", mode="error")

        updated = update_ducklake(
            metadata_path, "test", {"b": "UPDATED"}, pl.col("a") > 3
        )
        assert updated == 2

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4, 5]
        assert result["b"].to_list() == ["one", "two", "three", "UPDATED", "UPDATED"]

    def test_update_multiple_columns(self, tmp_path):
        """Update multiple columns at once."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [10.0, 20.0, 30.0],
        })
        write_ducklake(df, metadata_path, "test", mode="error")

        updated = update_ducklake(
            metadata_path, "test",
            {"b": "NEW", "c": 99.0},
            pl.col("a") == 2,
        )
        assert updated == 1

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["b"].to_list() == ["x", "NEW", "z"]
        assert result["c"].to_list() == [10.0, 99.0, 30.0]

    def test_update_with_expression(self, tmp_path):
        """Update using a Polars expression (computed value)."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["one", "two", "three", "four", "five"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        write_ducklake(df, metadata_path, "test", mode="error")

        updated = update_ducklake(
            metadata_path, "test",
            {"c": pl.col("c") + 100.0, "b": "UPDATED"},
            pl.col("a") >= 3,
        )
        assert updated == 3

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["c"].to_list() == [1.0, 2.0, 103.0, 104.0, 105.0]
        assert result["b"].to_list() == ["one", "two", "UPDATED", "UPDATED", "UPDATED"]

    def test_update_no_matching_rows(self, tmp_path):
        """Update with no matching rows creates no snapshot."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        snap_before = con.execute("SELECT COUNT(*) FROM ducklake_snapshot").fetchone()[0]
        con.close()

        updated = update_ducklake(
            metadata_path, "test", {"a": 99}, pl.col("a") > 100
        )
        assert updated == 0

        con = sqlite3.connect(metadata_path)
        snap_after = con.execute("SELECT COUNT(*) FROM ducklake_snapshot").fetchone()[0]
        con.close()
        assert snap_after == snap_before

    def test_update_all_rows(self, tmp_path):
        """Update all rows."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        updated = update_ducklake(
            metadata_path, "test", {"b": "ALL"}, pl.lit(True)
        )
        assert updated == 3

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["ALL", "ALL", "ALL"]


# ---------------------------------------------------------------------------
# Multi-file updates
# ---------------------------------------------------------------------------


class TestUpdateMultiFile:
    """Test updates spanning multiple data files."""

    def test_update_across_multiple_files(self, tmp_path):
        """Two inserts (two files), update rows from both."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pl.DataFrame({"a": [4, 5, 6], "b": ["p", "q", "r"]})
        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        # Update even numbers from both files
        updated = update_ducklake(
            metadata_path, "test", {"b": "EVEN"}, pl.col("a") % 2 == 0
        )
        assert updated == 3

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["b"].to_list() == ["x", "EVEN", "z", "EVEN", "q", "EVEN"]

    def test_update_one_of_two_files(self, tmp_path):
        """Two files, update only affects one."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pl.DataFrame({"a": [100, 200], "b": ["p", "q"]})
        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        updated = update_ducklake(
            metadata_path, "test", {"b": "BIG"}, pl.col("a") >= 100
        )
        assert updated == 2

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["b"].to_list() == ["x", "y", "z", "BIG", "BIG"]


# ---------------------------------------------------------------------------
# Time travel
# ---------------------------------------------------------------------------


class TestUpdateTimeTravel:
    """Test time travel with updates."""

    def test_time_travel_before_after_update(self, tmp_path):
        """Read at snapshot before and after update."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        snap_before = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
        con.close()

        update_ducklake(metadata_path, "test", {"b": "NEW"}, pl.col("a") == 2)

        # Latest: updated
        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["b"].to_list() == ["x", "NEW", "z"]

        # Before update: original
        result_old = read_ducklake(
            metadata_path, "test", snapshot_version=snap_before
        ).sort("a")
        assert result_old["b"].to_list() == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# DuckDB interop: write with polars, update with polars, read with DuckDB
# ---------------------------------------------------------------------------


class TestUpdateDuckDBInterop:
    """Verify updated catalogs are readable by DuckDB."""

    def test_polars_update_duckdb_reads(self, tmp_path):
        """Write + update with ducklake-polars, read with DuckDB."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        update_ducklake(metadata_path, "test", {"b": "UPDATED"}, pl.col("a") > 1)

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        pdf = pdf.sort("a")
        assert pdf["a"].to_list() == [1, 2, 3]
        assert pdf["b"].to_list() == ["x", "UPDATED", "UPDATED"]

    def test_duckdb_write_polars_update_duckdb_reads(self, tmp_path):
        """DuckDB creates data, polars updates, DuckDB reads result."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'one'), (2, 'two'), (3, 'three')"
        )
        con.close()

        updated = update_ducklake(
            metadata_path, "test", {"b": "CHANGED"}, pl.col("a") >= 2
        )
        assert updated == 2

        # Read with ducklake-polars
        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["b"].to_list() == ["one", "CHANGED", "CHANGED"]

        # Read with DuckDB
        pdf = _read_with_duckdb(metadata_path, data_path, "test").sort("a")
        assert pdf["b"].to_list() == ["one", "CHANGED", "CHANGED"]

    def test_update_with_expression_duckdb_reads(self, tmp_path):
        """Expression-based update readable by DuckDB."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "val": [10.0, 20.0, 30.0]})
        write_ducklake(df, metadata_path, "test", mode="error")

        update_ducklake(
            metadata_path, "test",
            {"val": pl.col("val") * 2},
            pl.col("a") >= 2,
        )

        pdf = _read_with_duckdb(metadata_path, data_path, "test").sort("a")
        assert pdf["val"].to_list() == [10.0, 40.0, 60.0]


# ---------------------------------------------------------------------------
# Metadata verification
# ---------------------------------------------------------------------------


class TestUpdateMetadata:
    """Verify update metadata is correctly stored."""

    def test_snapshot_changes_has_both_ops(self, tmp_path):
        """Check that changes_made has both inserted and deleted."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        update_ducklake(metadata_path, "test", {"a": 99}, pl.col("a") == 2)

        con = sqlite3.connect(metadata_path)
        row = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        con.close()

        assert row is not None
        assert "inserted_into_table" in row[0]
        assert "deleted_from_table" in row[0]

    def test_update_creates_delete_and_data_file(self, tmp_path):
        """One delete file and one new data file should be created."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        update_ducklake(metadata_path, "test", {"b": "NEW"}, pl.col("a") == 2)

        con = sqlite3.connect(metadata_path)

        # Should have 2 data files total (original + updated)
        data_files = con.execute(
            "SELECT COUNT(*) FROM ducklake_data_file"
        ).fetchone()[0]
        assert data_files == 2

        # Should have 1 delete file
        delete_files = con.execute(
            "SELECT COUNT(*) FROM ducklake_delete_file"
        ).fetchone()[0]
        assert delete_files == 1

        # New data file should have 1 row (the updated row)
        new_data = con.execute(
            "SELECT record_count FROM ducklake_data_file "
            "ORDER BY data_file_id DESC LIMIT 1"
        ).fetchone()
        assert new_data[0] == 1

        con.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestUpdateEdgeCases:
    """Edge cases for the update path."""

    def test_update_preserves_unmatched_rows(self, tmp_path):
        """Rows not matching the predicate are unchanged."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, metadata_path, "test", mode="error")

        update_ducklake(metadata_path, "test", {"a": 0}, pl.col("a") == 3)

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [0, 1, 2, 4, 5]

    def test_update_then_read_columns(self, tmp_path):
        """Column selection works after update."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        write_ducklake(df, metadata_path, "test", mode="error")

        update_ducklake(metadata_path, "test", {"b": "NEW"}, pl.col("a") == 1)

        result = read_ducklake(metadata_path, "test", columns=["a", "b"]).sort("a")
        assert result.columns == ["a", "b"]
        assert result["b"].to_list() == ["NEW", "y"]

    def test_update_with_null_value(self, tmp_path):
        """Update a column to NULL."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })
        write_ducklake(df, metadata_path, "test", mode="error")

        update_ducklake(
            metadata_path, "test",
            {"b": pl.lit(None, dtype=pl.String)},
            pl.col("a") == 2,
        )

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["b"].to_list() == ["x", None, "z"]

    def test_update_then_delete(self, tmp_path):
        """Update followed by delete."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, metadata_path, "test", mode="error")

        update_ducklake(metadata_path, "test", {"a": 99}, pl.col("a") == 3)

        from ducklake_polars import delete_ducklake
        delete_ducklake(metadata_path, "test", pl.col("a") > 50)

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 4, 5]

    def test_scan_after_update(self, tmp_path):
        """scan_ducklake with filter after update."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": list(range(10)), "b": ["x"] * 10})
        write_ducklake(df, metadata_path, "test", mode="error")

        update_ducklake(
            metadata_path, "test", {"b": "UPDATED"}, pl.col("a") >= 5
        )

        result = (
            scan_ducklake(metadata_path, "test")
            .filter(pl.col("b") == "UPDATED")
            .collect()
            .sort("a")
        )
        assert result["a"].to_list() == [5, 6, 7, 8, 9]
