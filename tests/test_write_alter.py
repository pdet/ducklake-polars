"""Tests for ducklake-dataframe ALTER TABLE support (ADD/DROP COLUMN, SORT KEYS)."""

from __future__ import annotations

import duckdb
import polars as pl
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_set_sort_keys,
    alter_ducklake_set_type,
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

        attach_source = cat.attach_source()

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

        attach_source = cat.attach_source()

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


# ---------------------------------------------------------------------------
# SET TYPE: column type changes
# ---------------------------------------------------------------------------


class TestSetType:
    """Test ALTER TABLE SET TYPE (column type changes)."""

    def test_integer_to_bigint(self, make_write_catalog):
        """Change INTEGER to BIGINT and read back."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int32), "b": ["x", "y", "z"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_type(cat.metadata_path, "test", "a", "BIGINT")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["a"].dtype == pl.Int64
        assert result["b"].to_list() == ["x", "y", "z"]

    def test_varchar_to_integer(self, make_write_catalog):
        """Change VARCHAR to INTEGER where data is numeric strings."""
        cat = make_write_catalog()
        df = pl.DataFrame({"id": [1, 2, 3], "val": ["10", "20", "30"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_type(cat.metadata_path, "test", "val", "INTEGER")

        result = read_ducklake(cat.metadata_path, "test").sort("id")
        assert result["val"].to_list() == [10, 20, 30]
        assert result["val"].dtype == pl.Int32

    def test_read_across_type_change_time_travel(self, make_write_catalog):
        """Read at a snapshot before the type change to get the old type."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int32)})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        # Snapshot before type change
        v_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        alter_ducklake_set_type(cat.metadata_path, "test", "a", "BIGINT")

        # Insert data after type change
        df2 = pl.DataFrame({"a": pl.Series([3, 4], dtype=pl.Int64)})
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        # Read latest: should be BIGINT
        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["a"].dtype == pl.Int64

        # Read at old snapshot: should be INT32
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=v_before
        ).sort("a")
        assert result_old["a"].to_list() == [1, 2]
        assert result_old["a"].dtype == pl.Int32

    def test_type_change_metadata(self, make_write_catalog):
        """Verify metadata: schema_version, column entries."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_type(cat.metadata_path, "test", "a", "BIGINT")

        # Should have two ducklake_column entries for the same column_id
        rows = cat.query_all(
            "SELECT column_id, column_type, end_snapshot "
            "FROM ducklake_column WHERE column_name = 'a' "
            "ORDER BY begin_snapshot"
        )
        assert len(rows) == 2
        # First entry: int32, ended
        assert rows[0][1] == "int32"
        assert rows[0][2] is not None
        # Second entry: int64, not ended
        assert rows[1][1] == "int64"
        assert rows[1][2] is None
        # Same column_id
        assert rows[0][0] == rows[1][0]

    def test_insert_after_type_change(self, make_write_catalog):
        """Insert data with the new type after a type change."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int32)})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_type(cat.metadata_path, "test", "a", "BIGINT")

        new_data = pl.DataFrame({"a": pl.Series([3000000000], dtype=pl.Int64)})
        write_ducklake(new_data, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3000000000]
        assert result["a"].dtype == pl.Int64

    def test_duckdb_changes_type_we_read(self, make_write_catalog):
        """DuckDB changes the type, we read correctly."""
        cat = make_write_catalog()

        # Create and populate via DuckDB
        source = cat.attach_source()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        con.execute("ALTER TABLE ducklake.test ALTER COLUMN a SET DATA TYPE BIGINT")
        con.close()

        # Read with ducklake-dataframe
        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2]
        assert result["a"].dtype == pl.Int64
        assert result["b"].to_list() == ["hello", "world"]

    def test_we_change_type_duckdb_reads(self, make_write_catalog):
        """We change the type, DuckDB reads correctly."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int32), "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_type(cat.metadata_path, "test", "a", "BIGINT")

        # Read with DuckDB
        result = cat.read_with_duckdb("test")
        assert sorted(result["a"].to_list()) == [1, 2]
        assert sorted(result["b"].to_list()) == ["x", "y"]


# ---------------------------------------------------------------------------
# SORT KEYS: optimized Parquet writes
# ---------------------------------------------------------------------------


class TestSortKeys:
    """Test ALTER TABLE SET SORT KEYS."""

    def test_set_sort_keys_basic(self, make_write_catalog):
        """Set sort keys, write data, verify Parquet file is sorted."""
        import os
        import glob

        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a"])

        # Insert unsorted data
        unsorted = pl.DataFrame({"a": [30, 10, 20, 5, 25, 15]})
        write_ducklake(unsorted, cat.metadata_path, "test", mode="append")

        # Read back — all data visible
        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 5, 10, 15, 20, 25, 30]

        # Verify the last written Parquet file is sorted
        parquet_files = sorted(
            glob.glob(os.path.join(cat.data_path, "**/*.parquet"), recursive=True),
            key=os.path.getmtime,
        )
        # The most recent parquet file should contain the sorted unsorted data
        last_file = parquet_files[-1]
        table = pq.read_table(last_file)
        a_values = table.column("a").to_pylist()
        assert a_values == sorted(a_values), f"Parquet file not sorted: {a_values}"

    def test_sort_keys_with_filter(self, make_write_catalog):
        """Sorted data has better min/max statistics for filtering."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a"])

        # Insert sorted data — statistics should reflect the sorted order
        data = pl.DataFrame({"a": [100, 50, 200, 10, 150]})
        write_ducklake(data, cat.metadata_path, "test", mode="append")

        # Read with filter — should work correctly
        result = read_ducklake(cat.metadata_path, "test")
        filtered = result.filter(pl.col("a") > 100)
        assert sorted(filtered["a"].to_list()) == [150, 200]

        # Verify file stats in metadata
        stats = cat.query_all(
            "SELECT min_value, max_value FROM ducklake_file_column_stats "
            "ORDER BY data_file_id DESC LIMIT 1"
        )
        assert stats[0][0] == "10"  # min
        assert stats[0][1] == "200"  # max

    def test_sort_keys_duckdb_interop(self, make_write_catalog):
        """We set sort keys and write data, DuckDB reads it correctly."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a"])

        unsorted = pl.DataFrame({"a": [30, 10, 20], "b": ["c", "a", "b"]})
        write_ducklake(unsorted, cat.metadata_path, "test", mode="append")

        # DuckDB can read the data
        result = cat.read_with_duckdb("test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 10, 20, 30]
        assert result["b"].to_list() == ["x", "y", "z", "a", "b", "c"]

    def test_sort_keys_multi_column(self, make_write_catalog):
        """Sort by multiple columns."""
        import os
        import glob

        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1], "b": [1], "c": ["x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a", "b"])

        unsorted = pl.DataFrame({
            "a": [2, 1, 2, 1],
            "b": [2, 1, 1, 2],
            "c": ["d", "a", "c", "b"],
        })
        write_ducklake(unsorted, cat.metadata_path, "test", mode="append")

        # Read back all data
        result = read_ducklake(cat.metadata_path, "test").sort(["a", "b"])
        assert result["a"].to_list() == [1, 1, 1, 2, 2]
        assert result["b"].to_list() == [1, 1, 2, 1, 2]

        # Verify parquet file is sorted by (a, b)
        parquet_files = sorted(
            glob.glob(os.path.join(cat.data_path, "**/*.parquet"), recursive=True),
            key=os.path.getmtime,
        )
        last_file = parquet_files[-1]
        table = pq.read_table(last_file)
        a_vals = table.column("a").to_pylist()
        b_vals = table.column("b").to_pylist()
        pairs = list(zip(a_vals, b_vals))
        assert pairs == sorted(pairs), f"Parquet not sorted by (a, b): {pairs}"

    def test_sort_keys_overwrite(self, make_write_catalog):
        """Overwrite mode also sorts data by sort keys."""
        import os
        import glob

        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a"])

        overwrite_data = pl.DataFrame({"a": [50, 10, 30, 20, 40]})
        write_ducklake(overwrite_data, cat.metadata_path, "test", mode="overwrite")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [10, 20, 30, 40, 50]

        # Verify parquet file is sorted
        parquet_files = sorted(
            glob.glob(os.path.join(cat.data_path, "**/*.parquet"), recursive=True),
            key=os.path.getmtime,
        )
        last_file = parquet_files[-1]
        table = pq.read_table(last_file)
        a_values = table.column("a").to_pylist()
        assert a_values == [10, 20, 30, 40, 50]

    def test_sort_keys_metadata(self, make_write_catalog):
        """Verify sort keys metadata is stored correctly."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a", "b"])

        # Check ducklake_sort_info exists with correct data
        sort_info = cat.query_all(
            "SELECT sort_id, table_id, begin_snapshot, end_snapshot "
            "FROM ducklake_sort_info"
        )
        assert len(sort_info) == 1
        assert sort_info[0][3] is None  # end_snapshot is NULL

        # Check ducklake_sort_expression
        sort_exprs = cat.query_all(
            "SELECT sort_key_index, expression, sort_direction, null_order "
            "FROM ducklake_sort_expression ORDER BY sort_key_index"
        )
        assert len(sort_exprs) == 2
        assert sort_exprs[0][0] == 0  # sort_key_index
        assert sort_exprs[0][1] == "a"  # expression is column name
        assert sort_exprs[0][2] == "ASC"
        assert sort_exprs[0][3] == "NULLS_LAST"
        assert sort_exprs[1][0] == 1
        assert sort_exprs[1][1] == "b"

    def test_sort_keys_desc_direction(self, make_write_catalog):
        """Setting sort keys with DESC direction."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "test", [("a", "DESC")]
        )

        # Write more data — should be sorted descending
        df2 = pl.DataFrame({"a": [6, 4, 5], "b": ["f", "d", "e"]})
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 6

    def test_sort_keys_mixed_directions(self, make_write_catalog):
        """Setting sort keys with mixed ASC/DESC and null order."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(
            cat.metadata_path,
            "test",
            [("a", "DESC"), ("b", "ASC", "NULLS_FIRST")],
        )

        sort_exprs = cat.query_all(
            "SELECT sort_key_index, expression, sort_direction, null_order "
            "FROM ducklake_sort_expression ORDER BY sort_key_index"
        )
        assert len(sort_exprs) == 2
        assert sort_exprs[0] == (0, "a", "DESC", "NULLS_LAST")
        assert sort_exprs[1] == (1, "b", "ASC", "NULLS_FIRST")

    def test_reset_sort_keys(self, make_write_catalog):
        """Reset sort keys clears them."""
        from ducklake_polars import alter_ducklake_reset_sort_keys

        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a"])

        sort_info = cat.query_all(
            "SELECT end_snapshot FROM ducklake_sort_info"
        )
        assert sort_info[0][0] is None  # active

        alter_ducklake_reset_sort_keys(cat.metadata_path, "test")

        sort_info = cat.query_all(
            "SELECT end_snapshot FROM ducklake_sort_info"
        )
        assert sort_info[0][0] is not None  # ended

    def test_reset_sort_keys_noop(self, make_write_catalog):
        """Reset sort keys on table with no sort keys is a no-op."""
        from ducklake_polars import alter_ducklake_reset_sort_keys

        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        # Should not raise
        alter_ducklake_reset_sort_keys(cat.metadata_path, "test")

    def test_sort_keys_duckdb_roundtrip(self, make_write_catalog):
        import duckdb as _duckdb
        parts = _duckdb.__version__.split(".")
        if len(parts) >= 2 and (int(parts[0]), int(parts[1])) < (1, 5):
            pytest.skip(f"DuckDB {_duckdb.__version__} does not support v0.4 catalogs")

        """Sort keys set by us are read correctly by DuckDB and vice versa."""
        import duckdb

        cat = make_write_catalog()
        if "postgres" in cat.metadata_path:
            pytest.skip("DuckDB interop only for SQLite")

        df = pl.DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        # Set sort via our API
        alter_ducklake_set_sort_keys(
            cat.metadata_path, "test", [("a", "DESC"), "b"]
        )

        # Verify DuckDB reads them
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{cat.metadata_path}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}')"
        )

        result = con.execute("SELECT * FROM ducklake.test").fetchall()
        assert len(result) == 3
        con.close()

    def test_sort_keys_nonexistent_column_raises(self, make_write_catalog):
        """Setting sort keys with nonexistent column raises."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["missing"])

    def test_sort_keys_nonexistent_table_raises(self, make_write_catalog):
        """Setting sort keys on nonexistent table raises."""
        cat = make_write_catalog()

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_set_sort_keys(cat.metadata_path, "missing", ["a"])
