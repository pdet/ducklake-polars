"""Tests for ducklake-polars ALTER TABLE support (ADD/DROP COLUMN)."""

from __future__ import annotations

import os
import sqlite3

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(tmp_path):
    """Create a DuckLake catalog via DuckDB and return (metadata_path, data_path)."""
    metadata_path = str(tmp_path / "alter_test.ducklake")
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
    result = con.execute(f'SELECT * FROM ducklake."{table_name}"').pl()
    con.close()
    return result


# ---------------------------------------------------------------------------
# ADD COLUMN: basic operations
# ---------------------------------------------------------------------------


class TestAddColumn:
    """Test ALTER TABLE ADD COLUMN."""

    def test_add_column_no_default(self, tmp_path):
        """Add a column without a default value."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_add_column(metadata_path, "test", "b", pl.Float64())

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]
        # Existing rows should have NULL for the new column
        assert result["b"].to_list() == [None, None, None]

    def test_add_column_with_default(self, tmp_path):
        """Add a column with a default value (stored in metadata)."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_add_column(
            metadata_path, "test", "d", pl.Int64(), default=42
        )

        # Metadata stores the default
        con = sqlite3.connect(metadata_path)
        row = con.execute(
            "SELECT initial_default, default_value FROM ducklake_column "
            "WHERE column_name = 'd'"
        ).fetchone()
        con.close()
        assert row[0] == "42"
        assert row[1] == "42"

    def test_add_column_then_insert(self, tmp_path):
        """Insert data after adding a column."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_add_column(metadata_path, "test", "b", pl.String())

        # Insert new row with both columns
        new_row = pl.DataFrame({"a": [3], "b": ["hello"]})
        write_ducklake(new_row, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == [None, None, "hello"]

    def test_add_column_metadata_correct(self, tmp_path):
        """Verify metadata: schema_version, column row, schema_versions."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        sv_before = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        con.close()

        alter_ducklake_add_column(metadata_path, "test", "b", pl.Int32())

        con = sqlite3.connect(metadata_path)
        # Schema version incremented
        sv_after = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        assert sv_after == sv_before + 1

        # New column row exists
        col = con.execute(
            "SELECT column_name, column_type FROM ducklake_column "
            "WHERE column_name = 'b'"
        ).fetchone()
        assert col is not None
        assert col[0] == "b"
        assert col[1] == "int32"

        # Changes recorded
        change = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        assert change is not None
        assert "altered_table" in change[0]

        # Schema version recorded
        sv_row = con.execute(
            "SELECT schema_version FROM ducklake_schema_versions "
            "ORDER BY begin_snapshot DESC LIMIT 1"
        ).fetchone()
        assert sv_row[0] == sv_after

        con.close()

    def test_add_multiple_columns(self, tmp_path):
        """Add multiple columns sequentially."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_add_column(metadata_path, "test", "b", pl.String())
        alter_ducklake_add_column(metadata_path, "test", "c", pl.Float64())
        alter_ducklake_add_column(metadata_path, "test", "d", pl.Boolean())

        result = read_ducklake(metadata_path, "test")
        assert result.columns == ["a", "b", "c", "d"]
        assert result.schema == {
            "a": pl.Int64, "b": pl.String, "c": pl.Float64, "d": pl.Boolean,
        }

    def test_add_column_duplicate_raises(self, tmp_path):
        """Adding a column that already exists raises."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="already exists"):
            alter_ducklake_add_column(metadata_path, "test", "a", pl.Int32())

    def test_add_column_nonexistent_table_raises(self, tmp_path):
        """Adding a column to nonexistent table raises."""
        metadata_path, data_path = _make_catalog(tmp_path)

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_add_column(
                metadata_path, "missing", "x", pl.Int32()
            )

    def test_add_column_duckdb_interop(self, tmp_path):
        """DuckDB can read after polars adds a column."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_add_column(metadata_path, "test", "b", pl.String())

        new_row = pl.DataFrame({"a": [3], "b": ["new"]})
        write_ducklake(new_row, metadata_path, "test", mode="append")

        pdf = _read_with_duckdb(metadata_path, data_path, "test").sort("a")
        assert pdf["a"].to_list() == [1, 2, 3]
        assert pdf["b"].to_list() == [None, None, "new"]

    def test_duckdb_add_column_polars_reads(self, tmp_path):
        """DuckDB adds a column, polars reads correctly."""
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
        con.execute("CREATE TABLE ducklake.test (a INTEGER)")
        con.execute("INSERT INTO ducklake.test VALUES (1), (2)")
        con.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        con.execute("INSERT INTO ducklake.test VALUES (3, 'hello')")
        con.close()

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == [None, None, "hello"]


# ---------------------------------------------------------------------------
# DROP COLUMN: basic operations
# ---------------------------------------------------------------------------


class TestDropColumn:
    """Test ALTER TABLE DROP COLUMN."""

    def test_drop_column(self, tmp_path):
        """Drop a column from a table."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10, 20, 30]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_drop_column(metadata_path, "test", "c")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "z"]

    def test_drop_column_then_insert(self, tmp_path):
        """Insert data after dropping a column."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_drop_column(metadata_path, "test", "c")

        new_row = pl.DataFrame({"a": [3], "b": ["z"]})
        write_ducklake(new_row, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "z"]

    def test_drop_column_metadata_correct(self, tmp_path):
        """Verify metadata: end_snapshot set, schema_version bumped."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        sv_before = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        con.close()

        alter_ducklake_drop_column(metadata_path, "test", "b")

        con = sqlite3.connect(metadata_path)
        # Schema version incremented
        sv_after = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        assert sv_after == sv_before + 1

        # Column has end_snapshot set
        col = con.execute(
            "SELECT end_snapshot FROM ducklake_column "
            "WHERE column_name = 'b'"
        ).fetchone()
        assert col is not None
        assert col[0] is not None  # end_snapshot is set

        # Changes recorded
        change = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        assert "altered_table" in change[0]

        con.close()

    def test_drop_column_nonexistent_raises(self, tmp_path):
        """Dropping a column that doesn't exist raises."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_drop_column(metadata_path, "test", "missing")

    def test_drop_column_nonexistent_table_raises(self, tmp_path):
        """Dropping from a nonexistent table raises."""
        metadata_path, data_path = _make_catalog(tmp_path)

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_drop_column(metadata_path, "missing", "col")

    def test_drop_column_time_travel(self, tmp_path):
        """Time travel sees the column before it was dropped."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        snap_before = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
        con.close()

        alter_ducklake_drop_column(metadata_path, "test", "b")

        # Latest: column b is gone
        result = read_ducklake(metadata_path, "test")
        assert result.columns == ["a"]

        # Before drop: column b is visible
        result_old = read_ducklake(
            metadata_path, "test", snapshot_version=snap_before
        )
        assert result_old.columns == ["a", "b"]
        assert result_old.sort("a")["b"].to_list() == ["x", "y"]

    def test_drop_column_duckdb_interop(self, tmp_path):
        """DuckDB can read after polars drops a column."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_drop_column(metadata_path, "test", "c")

        pdf = _read_with_duckdb(metadata_path, data_path, "test").sort("a")
        assert "c" not in pdf.columns
        assert pdf["a"].to_list() == [1, 2]
        assert pdf["b"].to_list() == ["x", "y"]

    def test_duckdb_drop_column_polars_reads(self, tmp_path):
        """DuckDB drops a column, polars reads correctly."""
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
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c FLOAT)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 1.0), (2, 'world', 2.0)")
        con.execute("ALTER TABLE ducklake.test DROP COLUMN c")
        con.execute("INSERT INTO ducklake.test VALUES (3, 'new')")
        con.close()

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["hello", "world", "new"]


# ---------------------------------------------------------------------------
# Combined ADD + DROP operations
# ---------------------------------------------------------------------------


class TestAddDropCombined:
    """Test combining ADD and DROP operations."""

    def test_add_then_drop(self, tmp_path):
        """Add a column then drop it."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_add_column(metadata_path, "test", "b", pl.String())
        alter_ducklake_drop_column(metadata_path, "test", "b")

        result = read_ducklake(metadata_path, "test")
        assert result.columns == ["a"]

    def test_drop_then_add_same_name(self, tmp_path):
        """Drop a column then add a new one with the same name."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_drop_column(metadata_path, "test", "b")
        alter_ducklake_add_column(metadata_path, "test", "b", pl.Int32())

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2]
        # Old data files have string "b", new column is Int32 → NULLs
        assert result["b"].to_list() == [None, None]

    def test_add_drop_add_insert(self, tmp_path):
        """Add column, drop it, add another, then insert."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_add_column(metadata_path, "test", "temp", pl.String())
        alter_ducklake_drop_column(metadata_path, "test", "temp")
        alter_ducklake_add_column(metadata_path, "test", "final", pl.Float64())

        new_row = pl.DataFrame({"a": [2], "final": [42.0]})
        write_ducklake(new_row, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result.columns == ["a", "final"]
        assert result["a"].to_list() == [1, 2]
        assert result["final"].to_list() == [None, 42.0]


# ---------------------------------------------------------------------------
# ALTER + UPDATE combined
# ---------------------------------------------------------------------------


class TestAlterAndUpdate:
    """Test ALTER TABLE combined with UPDATE/DELETE."""

    def test_add_column_then_update(self, tmp_path):
        """Add column, insert data, then update."""
        from ducklake_polars import update_ducklake

        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_add_column(metadata_path, "test", "b", pl.String())

        new_data = pl.DataFrame({"a": [4, 5], "b": ["four", "five"]})
        write_ducklake(new_data, metadata_path, "test", mode="append")

        update_ducklake(
            metadata_path, "test", {"b": "updated"}, pl.col("a") >= 4
        )

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4, 5]
        assert result["b"].to_list() == [None, None, None, "updated", "updated"]

    def test_drop_column_then_update(self, tmp_path):
        """Drop column, then update remaining columns."""
        from ducklake_polars import update_ducklake

        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10, 20, 30]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_drop_column(metadata_path, "test", "c")

        update_ducklake(
            metadata_path, "test", {"b": "NEW"}, pl.col("a") == 2
        )

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["b"].to_list() == ["x", "NEW", "z"]
