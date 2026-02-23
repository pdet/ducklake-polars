"""Tests for ducklake-polars ALTER TABLE support (ADD/DROP COLUMN)."""

from __future__ import annotations

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
# ADD COLUMN: basic operations
# ---------------------------------------------------------------------------


class TestAddColumn:
    """Test ALTER TABLE ADD COLUMN."""

    def test_add_column_no_default(self, make_write_catalog):
        """Add a column without a default value."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", pl.Float64())

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]
        # Existing rows should have NULL for the new column
        assert result["b"].to_list() == [None, None, None]

    def test_add_column_with_default(self, make_write_catalog):
        """Add a column with a default value (stored in metadata)."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(
            cat.metadata_path, "test", "d", pl.Int64(), default=42
        )

        # Metadata stores the default
        row = cat.query_one(
            "SELECT initial_default, default_value FROM ducklake_column "
            "WHERE column_name = 'd'"
        )
        assert row[0] == "42"
        assert row[1] == "42"

    def test_add_column_then_insert(self, make_write_catalog):
        """Insert data after adding a column."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", pl.String())

        # Insert new row with both columns
        new_row = pl.DataFrame({"a": [3], "b": ["hello"]})
        write_ducklake(new_row, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == [None, None, "hello"]

    def test_add_column_metadata_correct(self, make_write_catalog):
        """Verify metadata: schema_version, column row, schema_versions."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        sv_before = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]

        alter_ducklake_add_column(cat.metadata_path, "test", "b", pl.Int32())

        # Schema version incremented
        sv_after = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        assert sv_after == sv_before + 1

        # New column row exists
        col = cat.query_one(
            "SELECT column_name, column_type FROM ducklake_column "
            "WHERE column_name = 'b'"
        )
        assert col is not None
        assert col[0] == "b"
        assert col[1] == "int32"

        # Changes recorded
        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert change is not None
        assert "altered_table" in change[0]

        # Schema version recorded
        sv_row = cat.query_one(
            "SELECT schema_version FROM ducklake_schema_versions "
            "ORDER BY begin_snapshot DESC LIMIT 1"
        )
        assert sv_row[0] == sv_after

    def test_add_multiple_columns(self, make_write_catalog):
        """Add multiple columns sequentially."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", pl.String())
        alter_ducklake_add_column(cat.metadata_path, "test", "c", pl.Float64())
        alter_ducklake_add_column(cat.metadata_path, "test", "d", pl.Boolean())

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a", "b", "c", "d"]
        assert result.schema == {
            "a": pl.Int64, "b": pl.String, "c": pl.Float64, "d": pl.Boolean,
        }

    def test_add_column_duplicate_raises(self, make_write_catalog):
        """Adding a column that already exists raises."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="already exists"):
            alter_ducklake_add_column(cat.metadata_path, "test", "a", pl.Int32())

    def test_add_column_nonexistent_table_raises(self, make_write_catalog):
        """Adding a column to nonexistent table raises."""
        cat = make_write_catalog()

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_add_column(
                cat.metadata_path, "missing", "x", pl.Int32()
            )

    def test_add_column_duckdb_interop(self, make_write_catalog):
        """DuckDB can read after polars adds a column."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", pl.String())

        new_row = pl.DataFrame({"a": [3], "b": ["new"]})
        write_ducklake(new_row, cat.metadata_path, "test", mode="append")

        pdf = cat.read_with_duckdb("test").sort("a")
        assert pdf["a"].to_list() == [1, 2, 3]
        assert pdf["b"].to_list() == [None, None, "new"]

    def test_duckdb_add_column_polars_reads(self, make_write_catalog):
        """DuckDB adds a column, polars reads correctly."""
        cat = make_write_catalog()

        if cat.backend == "sqlite":
            attach_source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            attach_source = f"ducklake:postgres:{cat.metadata_path}"

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{attach_source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER)")
        con.execute("INSERT INTO ducklake.test VALUES (1), (2)")
        con.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        con.execute("INSERT INTO ducklake.test VALUES (3, 'hello')")
        con.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == [None, None, "hello"]


# ---------------------------------------------------------------------------
# DROP COLUMN: basic operations
# ---------------------------------------------------------------------------


class TestDropColumn:
    """Test ALTER TABLE DROP COLUMN."""

    def test_drop_column(self, make_write_catalog):
        """Drop a column from a table."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10, 20, 30]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_drop_column(cat.metadata_path, "test", "c")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "z"]

    def test_drop_column_then_insert(self, make_write_catalog):
        """Insert data after dropping a column."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_drop_column(cat.metadata_path, "test", "c")

        new_row = pl.DataFrame({"a": [3], "b": ["z"]})
        write_ducklake(new_row, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "z"]

    def test_drop_column_metadata_correct(self, make_write_catalog):
        """Verify metadata: end_snapshot set, schema_version bumped."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        sv_before = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]

        alter_ducklake_drop_column(cat.metadata_path, "test", "b")

        # Schema version incremented
        sv_after = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        assert sv_after == sv_before + 1

        # Column has end_snapshot set
        col = cat.query_one(
            "SELECT end_snapshot FROM ducklake_column "
            "WHERE column_name = 'b'"
        )
        assert col is not None
        assert col[0] is not None  # end_snapshot is set

        # Changes recorded
        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert "altered_table" in change[0]

    def test_drop_column_nonexistent_raises(self, make_write_catalog):
        """Dropping a column that doesn't exist raises."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_drop_column(cat.metadata_path, "test", "missing")

    def test_drop_column_nonexistent_table_raises(self, make_write_catalog):
        """Dropping from a nonexistent table raises."""
        cat = make_write_catalog()

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_drop_column(cat.metadata_path, "missing", "col")

    def test_drop_column_time_travel(self, make_write_catalog):
        """Time travel sees the column before it was dropped."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        snap_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        alter_ducklake_drop_column(cat.metadata_path, "test", "b")

        # Latest: column b is gone
        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a"]

        # Before drop: column b is visible
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        )
        assert result_old.columns == ["a", "b"]
        assert result_old.sort("a")["b"].to_list() == ["x", "y"]

    def test_drop_column_duckdb_interop(self, make_write_catalog):
        """DuckDB can read after polars drops a column."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_drop_column(cat.metadata_path, "test", "c")

        pdf = cat.read_with_duckdb("test").sort("a")
        assert "c" not in pdf.columns
        assert pdf["a"].to_list() == [1, 2]
        assert pdf["b"].to_list() == ["x", "y"]

    def test_duckdb_drop_column_polars_reads(self, make_write_catalog):
        """DuckDB drops a column, polars reads correctly."""
        cat = make_write_catalog()

        if cat.backend == "sqlite":
            attach_source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            attach_source = f"ducklake:postgres:{cat.metadata_path}"

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{attach_source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c FLOAT)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 1.0), (2, 'world', 2.0)")
        con.execute("ALTER TABLE ducklake.test DROP COLUMN c")
        con.execute("INSERT INTO ducklake.test VALUES (3, 'new')")
        con.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["hello", "world", "new"]


# ---------------------------------------------------------------------------
# Combined ADD + DROP operations
# ---------------------------------------------------------------------------


class TestAddDropCombined:
    """Test combining ADD and DROP operations."""

    def test_add_then_drop(self, make_write_catalog):
        """Add a column then drop it."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", pl.String())
        alter_ducklake_drop_column(cat.metadata_path, "test", "b")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a"]

    def test_drop_then_add_same_name(self, make_write_catalog):
        """Drop a column then add a new one with the same name."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_drop_column(cat.metadata_path, "test", "b")
        alter_ducklake_add_column(cat.metadata_path, "test", "b", pl.Int32())

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2]
        # Old data files have string "b", new column is Int32 → NULLs
        assert result["b"].to_list() == [None, None]

    def test_add_drop_add_insert(self, make_write_catalog):
        """Add column, drop it, add another, then insert."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "temp", pl.String())
        alter_ducklake_drop_column(cat.metadata_path, "test", "temp")
        alter_ducklake_add_column(cat.metadata_path, "test", "final", pl.Float64())

        new_row = pl.DataFrame({"a": [2], "final": [42.0]})
        write_ducklake(new_row, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "final"]
        assert result["a"].to_list() == [1, 2]
        assert result["final"].to_list() == [None, 42.0]


# ---------------------------------------------------------------------------
# ALTER + UPDATE combined
# ---------------------------------------------------------------------------


class TestAlterAndUpdate:
    """Test ALTER TABLE combined with UPDATE/DELETE."""

    def test_add_column_then_update(self, make_write_catalog):
        """Add column, insert data, then update."""
        from ducklake_polars import update_ducklake

        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", pl.String())

        new_data = pl.DataFrame({"a": [4, 5], "b": ["four", "five"]})
        write_ducklake(new_data, cat.metadata_path, "test", mode="append")

        update_ducklake(
            cat.metadata_path, "test", {"b": "updated"}, pl.col("a") >= 4
        )

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4, 5]
        assert result["b"].to_list() == [None, None, None, "updated", "updated"]

    def test_drop_column_then_update(self, make_write_catalog):
        """Drop column, then update remaining columns."""
        from ducklake_polars import update_ducklake

        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10, 20, 30]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_drop_column(cat.metadata_path, "test", "c")

        update_ducklake(
            cat.metadata_path, "test", {"b": "NEW"}, pl.col("a") == 2
        )

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["b"].to_list() == ["x", "NEW", "z"]
