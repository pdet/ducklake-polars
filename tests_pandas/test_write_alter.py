"""Tests for ducklake-pandas ALTER TABLE support (ADD/DROP COLUMN, SORT KEYS)."""

from __future__ import annotations

import duckdb
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from tests_pandas.helpers import assert_list_equal
from ducklake_pandas import (
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_set_sort_keys,
    alter_ducklake_set_type,
    read_ducklake,
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
        df = pd.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", "DOUBLE")

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 2, 3]
        # Existing rows should have NULL (NaN for float) for the new column
        import math
        assert all(math.isnan(x) for x in result["b"].tolist())

    def test_add_column_with_default(self, make_write_catalog):
        """Add a column with a default value (stored in metadata)."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(
            cat.metadata_path, "test", "d", "BIGINT", default=42
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
        df = pd.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", "VARCHAR")

        # Insert new row with both columns
        new_row = pd.DataFrame({"a": [3], "b": ["hello"]})
        write_ducklake(new_row, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert_list_equal(result["b"].tolist(), [None, None, "hello"])

    def test_add_column_metadata_correct(self, make_write_catalog):
        """Verify metadata: schema_version, column row, schema_versions."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        sv_before = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]

        alter_ducklake_add_column(cat.metadata_path, "test", "b", "INTEGER")

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
        df = pd.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", "VARCHAR")
        alter_ducklake_add_column(cat.metadata_path, "test", "c", "DOUBLE")
        alter_ducklake_add_column(cat.metadata_path, "test", "d", "BOOLEAN")

        result = read_ducklake(cat.metadata_path, "test")
        assert list(result.columns) == ["a", "b", "c", "d"]

    def test_add_column_duplicate_raises(self, make_write_catalog):
        """Adding a column that already exists raises."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="already exists"):
            alter_ducklake_add_column(cat.metadata_path, "test", "a", "INTEGER")

    def test_add_column_nonexistent_table_raises(self, make_write_catalog):
        """Adding a column to nonexistent table raises."""
        cat = make_write_catalog()

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_add_column(
                cat.metadata_path, "missing", "x", "INTEGER"
            )

    def test_add_column_duckdb_interop(self, make_write_catalog):
        """DuckDB can read after pandas adds a column."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", "VARCHAR")

        new_row = pd.DataFrame({"a": [3], "b": ["new"]})
        write_ducklake(new_row, cat.metadata_path, "test", mode="append")

        pdf = cat.read_with_duckdb("test").sort_values("a").reset_index(drop=True)
        assert pdf["a"].tolist() == [1, 2, 3]
        assert_list_equal(pdf["b"].tolist(), [None, None, "new"])

    def test_duckdb_add_column_pandas_reads(self, make_write_catalog):
        """DuckDB adds a column, pandas reads correctly."""
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

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert_list_equal(result["b"].tolist(), [None, None, "hello"])


# ---------------------------------------------------------------------------
# DROP COLUMN: basic operations
# ---------------------------------------------------------------------------


class TestDropColumn:
    """Test ALTER TABLE DROP COLUMN."""

    def test_drop_column(self, make_write_catalog):
        """Drop a column from a table."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10, 20, 30]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_drop_column(cat.metadata_path, "test", "c")

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["x", "y", "z"]

    def test_drop_column_then_insert(self, make_write_catalog):
        """Insert data after dropping a column."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_drop_column(cat.metadata_path, "test", "c")

        new_row = pd.DataFrame({"a": [3], "b": ["z"]})
        write_ducklake(new_row, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["x", "y", "z"]

    def test_drop_column_metadata_correct(self, make_write_catalog):
        """Verify metadata: end_snapshot set, schema_version bumped."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1], "b": ["x"]})
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
        df = pd.DataFrame({"a": [1]})
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
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        snap_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        alter_ducklake_drop_column(cat.metadata_path, "test", "b")

        # Latest: column b is gone
        result = read_ducklake(cat.metadata_path, "test")
        assert list(result.columns) == ["a"]

        # Before drop: column b is visible
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        )
        assert list(result_old.columns) == ["a", "b"]
        assert result_old.sort_values("a")["b"].tolist() == ["x", "y"]

    def test_drop_column_duckdb_interop(self, make_write_catalog):
        """DuckDB can read after pandas drops a column."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_drop_column(cat.metadata_path, "test", "c")

        pdf = cat.read_with_duckdb("test").sort_values("a").reset_index(drop=True)
        assert "c" not in pdf.columns
        assert pdf["a"].tolist() == [1, 2]
        assert pdf["b"].tolist() == ["x", "y"]

    def test_duckdb_drop_column_pandas_reads(self, make_write_catalog):
        """DuckDB drops a column, pandas reads correctly."""
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

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["hello", "world", "new"]


# ---------------------------------------------------------------------------
# Combined ADD + DROP operations
# ---------------------------------------------------------------------------


class TestAddDropCombined:
    """Test combining ADD and DROP operations."""

    def test_add_then_drop(self, make_write_catalog):
        """Add a column then drop it."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", "VARCHAR")
        alter_ducklake_drop_column(cat.metadata_path, "test", "b")

        result = read_ducklake(cat.metadata_path, "test")
        assert list(result.columns) == ["a"]

    def test_drop_then_add_same_name(self, make_write_catalog):
        """Drop a column then add a new one with the same name."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_drop_column(cat.metadata_path, "test", "b")
        alter_ducklake_add_column(cat.metadata_path, "test", "b", "INTEGER")

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 2]
        # Old data files have string "b", new column is Int32 → NULLs (NaN in pandas)
        assert all(pd.isna(x) for x in result["b"].tolist())

    def test_add_drop_add_insert(self, make_write_catalog):
        """Add column, drop it, add another, then insert."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "temp", "VARCHAR")
        alter_ducklake_drop_column(cat.metadata_path, "test", "temp")
        alter_ducklake_add_column(cat.metadata_path, "test", "final", "DOUBLE")

        new_row = pd.DataFrame({"a": [2], "final": [42.0]})
        write_ducklake(new_row, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert list(result.columns) == ["a", "final"]
        assert result["a"].tolist() == [1, 2]
        import math
        final_list = result["final"].tolist()
        assert math.isnan(final_list[0])
        assert final_list[1] == 42.0


# ---------------------------------------------------------------------------
# ALTER + UPDATE combined
# ---------------------------------------------------------------------------


class TestAlterAndUpdate:
    """Test ALTER TABLE combined with UPDATE/DELETE."""

    def test_add_column_then_update(self, make_write_catalog):
        """Add column, insert data, then update."""
        from ducklake_pandas import update_ducklake

        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_add_column(cat.metadata_path, "test", "b", "VARCHAR")

        new_data = pd.DataFrame({"a": [4, 5], "b": ["four", "five"]})
        write_ducklake(new_data, cat.metadata_path, "test", mode="append")

        update_ducklake(
            cat.metadata_path, "test", {"b": "updated"}, lambda df: df["a"] >= 4
        )

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3, 4, 5]
        assert_list_equal(result["b"].tolist(), [None, None, None, "updated", "updated"])

    def test_drop_column_then_update(self, make_write_catalog):
        """Drop column, then update remaining columns."""
        from ducklake_pandas import update_ducklake

        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10, 20, 30]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_drop_column(cat.metadata_path, "test", "c")

        update_ducklake(
            cat.metadata_path, "test", {"b": "NEW"}, lambda df: df["a"] == 2
        )

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert list(result.columns) == ["a", "b"]
        assert result["b"].tolist() == ["x", "NEW", "z"]


# ---------------------------------------------------------------------------
# SET TYPE: column type changes
# ---------------------------------------------------------------------------


class TestSetType:
    """Test ALTER TABLE SET TYPE (column type changes)."""

    def test_integer_to_bigint(self, make_write_catalog):
        """Change INTEGER to BIGINT and read back."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int32), "b": ["x", "y", "z"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_type(cat.metadata_path, "test", "a", "BIGINT")

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["x", "y", "z"]

    def test_varchar_to_integer(self, make_write_catalog):
        """Change VARCHAR to INTEGER where data is numeric strings."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["10", "20", "30"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_type(cat.metadata_path, "test", "val", "INTEGER")

        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["val"].tolist() == [10, 20, 30]

    def test_read_across_type_change_time_travel(self, make_write_catalog):
        """Read at a snapshot before the type change to get the old type."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": np.array([1, 2], dtype=np.int32)})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        # Snapshot before type change
        v_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        alter_ducklake_set_type(cat.metadata_path, "test", "a", "BIGINT")

        # Insert data after type change
        df2 = pd.DataFrame({"a": np.array([3, 4], dtype=np.int64)})
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        # Read latest: should have all values
        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3, 4]

        # Read at old snapshot: should only have old values
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=v_before
        ).sort_values("a").reset_index(drop=True)
        assert result_old["a"].tolist() == [1, 2]

    def test_duckdb_changes_type_we_read(self, make_write_catalog):
        """DuckDB changes the type, we read correctly."""
        cat = make_write_catalog()

        # Create and populate via DuckDB
        if cat.backend == "sqlite":
            source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            source = f"ducklake:postgres:{cat.metadata_path}"

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

        # Read with ducklake-pandas
        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == ["hello", "world"]

    def test_we_change_type_duckdb_reads(self, make_write_catalog):
        """We change the type, DuckDB reads correctly."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": np.array([1, 2], dtype=np.int32), "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_type(cat.metadata_path, "test", "a", "BIGINT")

        # Read with DuckDB
        result = cat.read_with_duckdb("test")
        assert sorted(result["a"].tolist()) == [1, 2]
        assert sorted(result["b"].tolist()) == ["x", "y"]


# ---------------------------------------------------------------------------
# SORT KEYS: optimized Parquet writes
# ---------------------------------------------------------------------------


class TestSortKeys:
    """Test ALTER TABLE SET SORT KEYS."""

    def test_set_sort_keys_basic(self, make_write_catalog):
        """Set sort keys, write data, verify Parquet file is sorted."""
        import os
        import glob
        import pyarrow.parquet as pq

        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a"])

        # Insert unsorted data
        unsorted = pd.DataFrame({"a": [30, 10, 20, 5, 25, 15]})
        write_ducklake(unsorted, cat.metadata_path, "test", mode="append")

        # Read back — all data visible
        result = read_ducklake(cat.metadata_path, "test").sort_values("a")
        assert result["a"].tolist() == [1, 2, 3, 5, 10, 15, 20, 25, 30]

        # Verify the last written Parquet file is sorted
        parquet_files = sorted(
            glob.glob(os.path.join(cat.data_path, "**/*.parquet"), recursive=True),
            key=os.path.getmtime,
        )
        last_file = parquet_files[-1]
        table = pq.read_table(last_file)
        a_values = table.column("a").to_pylist()
        assert a_values == sorted(a_values), f"Parquet file not sorted: {a_values}"

    def test_sort_keys_with_filter(self, make_write_catalog):
        """Sorted data has better min/max statistics for filtering."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a"])

        data = pd.DataFrame({"a": [100, 50, 200, 10, 150]})
        write_ducklake(data, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test")
        filtered = result[result["a"] > 100]
        assert sorted(filtered["a"].tolist()) == [150, 200]

        # Verify file stats
        stats = cat.query_all(
            "SELECT min_value, max_value FROM ducklake_file_column_stats "
            "ORDER BY data_file_id DESC LIMIT 1"
        )
        assert stats[0][0] == "10"
        assert stats[0][1] == "200"

    def test_sort_keys_duckdb_interop(self, make_write_catalog):
        """We set sort keys and write data, DuckDB reads it correctly."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a"])

        unsorted = pd.DataFrame({"a": [30, 10, 20], "b": ["c", "a", "b"]})
        write_ducklake(unsorted, cat.metadata_path, "test", mode="append")

        result = cat.read_with_duckdb("test").sort_values("a")
        assert result["a"].tolist() == [1, 2, 3, 10, 20, 30]
        assert result["b"].tolist() == ["x", "y", "z", "a", "b", "c"]

    def test_sort_keys_multi_column(self, make_write_catalog):
        """Sort by multiple columns."""
        import os
        import glob
        import pyarrow.parquet as pq

        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1], "b": [1], "c": ["x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(cat.metadata_path, "test", ["a", "b"])

        unsorted = pd.DataFrame({
            "a": [2, 1, 2, 1],
            "b": [2, 1, 1, 2],
            "c": ["d", "a", "c", "b"],
        })
        write_ducklake(unsorted, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort_values(["a", "b"])
        assert result["a"].tolist() == [1, 1, 1, 2, 2]
        assert result["b"].tolist() == [1, 1, 2, 1, 2]

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

    def test_sort_keys_desc_direction(self, make_write_catalog):
        """Sort keys with DESC direction."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "test", [("a", "DESC")]
        )

        df2 = pd.DataFrame({"a": [6, 4, 5], "b": ["f", "d", "e"]})
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 6

    def test_sort_keys_mixed_directions(self, make_write_catalog):
        """Sort keys with mixed ASC/DESC and null order."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
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
        from ducklake_pandas import alter_ducklake_reset_sort_keys

        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1], "b": ["x"]})
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
