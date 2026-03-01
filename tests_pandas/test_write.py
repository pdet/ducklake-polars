"""Tests for ducklake-pandas write path."""

from __future__ import annotations

import os

import duckdb
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from ducklake_pandas import (
    create_ducklake_table,
    read_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# CREATE TABLE
# ---------------------------------------------------------------------------


class TestCreateTable:
    """Test create_ducklake_table."""

    def test_create_simple_table(self, make_write_catalog):
        cat = make_write_catalog()
        schema = {"a": "int32", "b": "varchar", "c": "float64"}

        create_ducklake_table(cat.metadata_path, "test", schema)

        # Verify table exists and is empty
        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (0, 3)
        assert list(result.columns) == ["a", "b", "c"]

    def test_create_table_already_exists(self, make_write_catalog):
        cat = make_write_catalog()
        schema = {"a": "int32"}

        create_ducklake_table(cat.metadata_path, "test", schema)

        with pytest.raises(ValueError, match="already exists"):
            create_ducklake_table(cat.metadata_path, "test", schema)

    def test_create_table_metadata_correct(self, make_write_catalog):
        cat = make_write_catalog()
        schema = {"a": "int32", "b": "varchar"}

        create_ducklake_table(cat.metadata_path, "test", schema)

        # Table exists
        row = cat.query_one(
            "SELECT table_name, path FROM ducklake_table WHERE table_name = 'test'"
        )
        assert row is not None
        assert row[0] == "test"
        assert row[1] == "test/"

        # Columns exist
        rows = cat.query_all(
            "SELECT column_name, column_type FROM ducklake_column "
            "WHERE table_id = (SELECT table_id FROM ducklake_table WHERE table_name = 'test') "
            "ORDER BY column_order"
        )
        assert rows == [("a", "int32"), ("b", "varchar")]

        # Snapshot changes recorded
        row = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row is not None
        assert "created_table" in row[0]
        assert "test" in row[0]

        # Schema version incremented
        versions = cat.query_one(
            "SELECT schema_version FROM ducklake_schema_versions "
            "ORDER BY begin_snapshot DESC LIMIT 1"
        )
        assert versions is not None
        assert versions[0] > 0

    def test_create_table_duckdb_interop(self, make_write_catalog):
        """DuckDB can read a table created by ducklake-pandas."""
        cat = make_write_catalog()
        schema = {"a": "int32", "b": "varchar"}

        create_ducklake_table(cat.metadata_path, "test", schema)

        # Write some data so DuckDB has something to verify
        df = pd.DataFrame({"a": [1, 2], "b": ["hello", "world"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        # Read with DuckDB
        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 2
        assert sorted(pdf["a"].tolist()) == [1, 2]

    def test_create_multiple_tables(self, make_write_catalog):
        cat = make_write_catalog()

        create_ducklake_table(cat.metadata_path, "t1", {"a": "int32"})
        create_ducklake_table(cat.metadata_path, "t2", {"x": "varchar", "y": "float64"})

        r1 = read_ducklake(cat.metadata_path, "t1")
        r2 = read_ducklake(cat.metadata_path, "t2")
        assert r1.shape == (0, 1)
        assert r2.shape == (0, 2)


# ---------------------------------------------------------------------------
# INSERT (write_ducklake)
# ---------------------------------------------------------------------------


class TestWriteDucklake:
    """Test write_ducklake with various modes."""

    def test_write_mode_error_new_table(self, make_write_catalog):
        """mode='error' creates a new table and inserts data."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        df = df.sort_values("a").reset_index(drop=True)
        assert_frame_equal(result, df, check_dtype=False)

    def test_write_mode_error_existing_table_raises(self, make_write_catalog):
        """mode='error' raises if table already exists."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="already exists"):
            write_ducklake(df, cat.metadata_path, "test", mode="error")

    def test_write_mode_append_new_table(self, make_write_catalog):
        """mode='append' creates table if it doesn't exist."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2]})

        write_ducklake(df, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == [1, 2]

    def test_write_mode_append_existing(self, make_write_catalog):
        """mode='append' appends data to existing table."""
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pd.DataFrame({"a": [3, 4], "b": ["z", "w"]})

        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 4
        assert sorted(result["a"].tolist()) == [1, 2, 3, 4]

    def test_write_mode_overwrite_new_table(self, make_write_catalog):
        """mode='overwrite' creates table if it doesn't exist."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [10, 20]})

        write_ducklake(df, cat.metadata_path, "test", mode="overwrite")

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == [10, 20]

    def test_write_mode_overwrite_replaces(self, make_write_catalog):
        """mode='overwrite' replaces all existing data."""
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pd.DataFrame({"a": [10, 20], "b": ["new1", "new2"]})

        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="overwrite")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 2
        assert sorted(result["a"].tolist()) == [10, 20]

    def test_write_invalid_mode_raises(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})

        with pytest.raises(ValueError, match="Invalid write mode"):
            write_ducklake(df, cat.metadata_path, "test", mode="invalid")

    def test_write_empty_df_mode_error(self, make_write_catalog):
        """mode='error' with empty DataFrame creates table but no data file."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": pd.array([], dtype="int32"), "b": pd.Series([], dtype="string")})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0
        assert len(result.columns) == 2


# ---------------------------------------------------------------------------
# DuckDB interop: write with ducklake-pandas, read with DuckDB
# ---------------------------------------------------------------------------


class TestDuckDBInterop:
    """Verify that catalogs written by ducklake-pandas are readable by DuckDB."""

    def test_basic_interop(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["hello", "world", "test"]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 3
        assert sorted(pdf["a"].tolist()) == [1, 2, 3]
        assert sorted(pdf["b"].tolist()) == ["hello", "test", "world"]

    def test_interop_multiple_inserts(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pd.DataFrame({"x": [10, 20]})
        df2 = pd.DataFrame({"x": [30]})

        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        pdf = cat.read_with_duckdb("test")
        assert sorted(pdf["x"].tolist()) == [10, 20, 30]

    def test_interop_after_overwrite(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pd.DataFrame({"v": [1, 2, 3]})
        df2 = pd.DataFrame({"v": [99]})

        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="overwrite")

        pdf = cat.read_with_duckdb("test")
        assert pdf["v"].tolist() == [99]

    def test_interop_various_types(self, make_write_catalog):
        """Verify interop with various column types."""
        cat = make_write_catalog()
        df = pd.DataFrame({
            "int_col": pd.array([1, 2], dtype="int32"),
            "bigint_col": pd.array([100, 200], dtype="int64"),
            "float_col": [1.5, 2.5],
            "bool_col": [True, False],
            "str_col": ["hello", "world"],
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 2
        assert sorted(pdf["int_col"].tolist()) == [1, 2]
        assert sorted(pdf["bigint_col"].tolist()) == [100, 200]

    def test_interop_null_values(self, make_write_catalog):
        """Verify interop with NULL values."""
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": pd.array([1, pd.NA, 3], dtype="Int32"),
            "b": ["hello", None, "world"],
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 3

    def test_interop_date_types(self, make_write_catalog):
        """Verify interop with date/datetime types."""
        from datetime import date, datetime

        cat = make_write_catalog()
        df = pd.DataFrame({
            "d": [date(2025, 1, 1), date(2025, 6, 15)],
            "ts": [datetime(2025, 1, 1, 12, 0), datetime(2025, 6, 15, 18, 30)],
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# Round-trip: write with ducklake-pandas, read with ducklake-pandas
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Write and read with ducklake-pandas (no DuckDB in the loop)."""

    def test_roundtrip_basic(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "value": [1.1, 2.2, 3.3],
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")
        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("id").reset_index(drop=True)
        df = df.sort_values("id").reset_index(drop=True)
        assert_frame_equal(result, df, check_dtype=False)

    def test_roundtrip_append_then_read(self, make_write_catalog):
        cat = make_write_catalog()

        for i in range(5):
            df = pd.DataFrame({"batch": [i], "val": [i * 10]})
            write_ducklake(df, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 5
        assert sorted(result["batch"].tolist()) == [0, 1, 2, 3, 4]

    def test_roundtrip_overwrite(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [100]})

        write_ducklake(df1, cat.metadata_path, "test", mode="error")
        write_ducklake(df2, cat.metadata_path, "test", mode="overwrite")

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].tolist() == [100]


# ---------------------------------------------------------------------------
# Column statistics
# ---------------------------------------------------------------------------


class TestColumnStats:
    """Verify column statistics are correctly computed and stored."""

    def test_stats_registered(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": pd.array([1, 5, 3], dtype="int32"),
            "b": ["x", "z", "y"],
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        # Check file column stats directly
        stats = cat.query_all(
            "SELECT column_id, null_count, min_value, max_value "
            "FROM ducklake_file_column_stats "
            "ORDER BY column_id"
        )

        # Should have stats for both columns
        assert len(stats) >= 2

        # Find stats for column 'a' (int32, column_id=1)
        a_stats = [s for s in stats if s[0] == 1]
        assert len(a_stats) == 1
        assert a_stats[0][1] == 0  # null_count
        assert a_stats[0][2] == "1"  # min
        assert a_stats[0][3] == "5"  # max

        # Find stats for column 'b' (varchar, column_id=2)
        b_stats = [s for s in stats if s[0] == 2]
        assert len(b_stats) == 1
        assert b_stats[0][2] == "x"  # min
        assert b_stats[0][3] == "z"  # max

    def test_stats_with_nulls(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": pd.array([1, pd.NA, 3], dtype="Int32"),
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        stats = cat.query_one(
            "SELECT null_count, min_value, max_value "
            "FROM ducklake_file_column_stats WHERE column_id = 1"
        )

        assert stats is not None
        assert stats[0] == 1  # null_count
        assert stats[1] == "1"  # min
        assert stats[2] == "3"  # max

    def test_table_stats_updated(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3]})

        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        row = cat.query_one(
            "SELECT record_count, next_row_id FROM ducklake_table_stats"
        )

        assert row is not None
        assert row[0] == 3  # total records
        assert row[1] == 3  # next_row_id

    def test_table_column_stats_aggregated(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": pd.array([1, 5], dtype="int32")})
        df2 = pd.DataFrame({"a": pd.array([3, 10], dtype="int32")})

        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        row = cat.query_one(
            "SELECT min_value, max_value FROM ducklake_table_column_stats WHERE column_id = 1"
        )

        assert row is not None
        assert row[0] == "1"  # global min
        assert row[1] == "10"  # global max


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestWriteEdgeCases:
    """Test edge cases in the write path."""

    def test_write_single_row(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [42]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].tolist() == [42]

    def test_write_many_columns(self, make_write_catalog):
        cat = make_write_catalog()
        data = {f"col_{i}": [i] for i in range(20)}
        df = pd.DataFrame(data)

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 20)
        for i in range(20):
            assert result[f"col_{i}"].tolist() == [i]

    def test_write_boolean_stats(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"flag": [True, False, True]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        stats = cat.query_one(
            "SELECT min_value, max_value FROM ducklake_file_column_stats WHERE column_id = 1"
        )

        assert stats is not None
        assert stats[0] == "false"
        assert stats[1] == "true"

    def test_snapshot_progression(self, make_write_catalog):
        """Each write creates a new snapshot."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        snap_count_1 = cat.query_one("SELECT COUNT(*) FROM ducklake_snapshot")[0]

        write_ducklake(pd.DataFrame({"a": [2]}), cat.metadata_path, "test", mode="append")
        snap_count_2 = cat.query_one("SELECT COUNT(*) FROM ducklake_snapshot")[0]

        assert snap_count_2 > snap_count_1

    def test_parquet_file_naming(self, make_write_catalog):
        """Written Parquet files follow ducklake-{uuid7}.parquet naming."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        row = cat.query_one("SELECT path FROM ducklake_data_file LIMIT 1")

        assert row is not None
        assert row[0].startswith("ducklake-")
        assert row[0].endswith(".parquet")

    def test_time_travel_after_write(self, make_write_catalog):
        """Time travel works correctly after writes."""
        cat = make_write_catalog()

        df1 = pd.DataFrame({"a": [1, 2]})
        write_ducklake(df1, cat.metadata_path, "test", mode="error")

        # Get snapshot after first write
        snap_after_create = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        df2 = pd.DataFrame({"a": [3, 4]})
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        # Read at latest: should have 4 rows
        result_latest = read_ducklake(cat.metadata_path, "test")
        assert result_latest.shape[0] == 4

        # Read at snapshot after first write: should have 2 rows
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_after_create
        )
        assert result_old.shape[0] == 2
        assert sorted(result_old["a"].tolist()) == [1, 2]


# ---------------------------------------------------------------------------
# DuckDB writes, ducklake-pandas reads (existing capability, regression)
# ---------------------------------------------------------------------------


class TestDuckDBWritePandasRead:
    """Verify existing read path still works with DuckDB-created catalogs."""

    def test_duckdb_write_pandas_read(self, tmp_path):
        """Standard flow: DuckDB creates data, ducklake-pandas reads."""
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
        con.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        con.close()

        result = read_ducklake(metadata_path, "test")
        assert result.shape == (2, 2)
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == ["hello", "world"]
