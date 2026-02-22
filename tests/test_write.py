"""Tests for ducklake-polars write path (Phase 5a + 5b)."""

from __future__ import annotations

import os
import sqlite3

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    create_ducklake_table,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)
from ducklake_polars._schema import polars_type_to_duckdb


# ---------------------------------------------------------------------------
# Phase 5a: Reverse type mapping
# ---------------------------------------------------------------------------


class TestPolarsTypeToDuckdb:
    """Test polars_type_to_duckdb reverse mapping."""

    def test_integer_types(self):
        assert polars_type_to_duckdb(pl.Int8()) == "int8"
        assert polars_type_to_duckdb(pl.Int16()) == "int16"
        assert polars_type_to_duckdb(pl.Int32()) == "int32"
        assert polars_type_to_duckdb(pl.Int64()) == "int64"

    def test_unsigned_integer_types(self):
        assert polars_type_to_duckdb(pl.UInt8()) == "uint8"
        assert polars_type_to_duckdb(pl.UInt16()) == "uint16"
        assert polars_type_to_duckdb(pl.UInt32()) == "uint32"
        assert polars_type_to_duckdb(pl.UInt64()) == "uint64"

    def test_float_types(self):
        assert polars_type_to_duckdb(pl.Float32()) == "float32"
        assert polars_type_to_duckdb(pl.Float64()) == "float64"

    def test_string_and_binary(self):
        assert polars_type_to_duckdb(pl.String()) == "varchar"
        assert polars_type_to_duckdb(pl.Binary()) == "blob"

    def test_boolean(self):
        assert polars_type_to_duckdb(pl.Boolean()) == "boolean"

    def test_date_and_time(self):
        assert polars_type_to_duckdb(pl.Date()) == "date"
        assert polars_type_to_duckdb(pl.Time()) == "time"

    def test_datetime_variants(self):
        assert polars_type_to_duckdb(pl.Datetime("us")) == "timestamp"
        assert polars_type_to_duckdb(pl.Datetime("ms")) == "timestamp_ms"
        assert polars_type_to_duckdb(pl.Datetime("ns")) == "timestamp_ns"
        assert polars_type_to_duckdb(pl.Datetime("us", "UTC")) == "timestamp with time zone"

    def test_decimal(self):
        assert polars_type_to_duckdb(pl.Decimal(18, 3)) == "decimal(18,3)"
        assert polars_type_to_duckdb(pl.Decimal(10, 0)) == "decimal(10,0)"

    def test_compound_types(self):
        assert polars_type_to_duckdb(pl.List(pl.Int32())) == "list"
        assert polars_type_to_duckdb(pl.Struct({"a": pl.Int32()})) == "struct"

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Cannot map Polars type"):
            polars_type_to_duckdb(pl.Null())


# ---------------------------------------------------------------------------
# Helpers: create DuckLake catalog fixtures for write tests
# ---------------------------------------------------------------------------


def _make_catalog(tmp_path, inline=False):
    """Create a DuckLake catalog via DuckDB and return (metadata_path, data_path)."""
    metadata_path = str(tmp_path / "write_test.ducklake")
    data_path = str(tmp_path / "data")
    os.makedirs(data_path, exist_ok=True)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    inline_opt = "" if inline else ", DATA_INLINING_ROW_LIMIT 0"
    con.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
        f"(DATA_PATH '{data_path}'{inline_opt})"
    )
    con.close()
    return metadata_path, data_path


def _read_with_duckdb(metadata_path, data_path, table_name):
    """Read a table with DuckDB's DuckLake extension (interop verification)."""
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
# Phase 5b: CREATE TABLE
# ---------------------------------------------------------------------------


class TestCreateTable:
    """Test create_ducklake_table."""

    def test_create_simple_table(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        schema = {"a": pl.Int32(), "b": pl.String(), "c": pl.Float64()}

        create_ducklake_table(metadata_path, "test", schema)

        # Verify table exists and is empty
        result = read_ducklake(metadata_path, "test")
        assert result.shape == (0, 3)
        assert result.schema == {"a": pl.Int32, "b": pl.String, "c": pl.Float64}

    def test_create_table_already_exists(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        schema = {"a": pl.Int32()}

        create_ducklake_table(metadata_path, "test", schema)

        with pytest.raises(ValueError, match="already exists"):
            create_ducklake_table(metadata_path, "test", schema)

    def test_create_table_with_polars_schema(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        schema = pl.Schema({"x": pl.Int64, "y": pl.Boolean})

        create_ducklake_table(metadata_path, "test", schema)

        result = read_ducklake(metadata_path, "test")
        assert result.schema == {"x": pl.Int64, "y": pl.Boolean}

    def test_create_table_metadata_correct(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        schema = {"a": pl.Int32(), "b": pl.String()}

        create_ducklake_table(metadata_path, "test", schema)

        # Check metadata directly
        con = sqlite3.connect(metadata_path)

        # Table exists
        row = con.execute(
            "SELECT table_name, path FROM ducklake_table WHERE table_name = 'test'"
        ).fetchone()
        assert row is not None
        assert row[0] == "test"
        assert row[1] == "test/"

        # Columns exist
        rows = con.execute(
            "SELECT column_name, column_type FROM ducklake_column "
            "WHERE table_id = (SELECT table_id FROM ducklake_table WHERE table_name = 'test') "
            "ORDER BY column_order"
        ).fetchall()
        assert rows == [("a", "int32"), ("b", "varchar")]

        # Snapshot changes recorded
        row = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert "created_table" in row[0]
        assert "test" in row[0]

        # Schema version incremented
        versions = con.execute(
            "SELECT schema_version FROM ducklake_schema_versions "
            "ORDER BY begin_snapshot DESC LIMIT 1"
        ).fetchone()
        assert versions is not None
        assert versions[0] > 0

        con.close()

    def test_create_table_duckdb_interop(self, tmp_path):
        """DuckDB can read a table created by ducklake-polars."""
        metadata_path, data_path = _make_catalog(tmp_path)
        schema = {"a": pl.Int32(), "b": pl.String()}

        create_ducklake_table(metadata_path, "test", schema)

        # Write some data so DuckDB has something to verify
        df = pl.DataFrame({"a": [1, 2], "b": ["hello", "world"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        # Read with DuckDB
        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert len(pdf) == 2
        assert sorted(pdf["a"].to_list()) == [1, 2]

    def test_create_multiple_tables(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)

        create_ducklake_table(metadata_path, "t1", {"a": pl.Int32()})
        create_ducklake_table(metadata_path, "t2", {"x": pl.String(), "y": pl.Float64()})

        r1 = read_ducklake(metadata_path, "t1")
        r2 = read_ducklake(metadata_path, "t2")
        assert r1.shape == (0, 1)
        assert r2.shape == (0, 2)


# ---------------------------------------------------------------------------
# Phase 5b: INSERT (write_ducklake)
# ---------------------------------------------------------------------------


class TestWriteDucklake:
    """Test write_ducklake with various modes."""

    def test_write_mode_error_new_table(self, tmp_path):
        """mode='error' creates a new table and inserts data."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        write_ducklake(df, metadata_path, "test", mode="error")

        result = read_ducklake(metadata_path, "test")
        assert_frame_equal(result.sort("a"), df.sort("a"))

    def test_write_mode_error_existing_table_raises(self, tmp_path):
        """mode='error' raises if table already exists."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})

        write_ducklake(df, metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="already exists"):
            write_ducklake(df, metadata_path, "test", mode="error")

    def test_write_mode_append_new_table(self, tmp_path):
        """mode='append' creates table if it doesn't exist."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2]})

        write_ducklake(df, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test")
        assert_frame_equal(result.sort("a"), df.sort("a"))

    def test_write_mode_append_existing(self, tmp_path):
        """mode='append' appends data to existing table."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pl.DataFrame({"a": [3, 4], "b": ["z", "w"]})

        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test")
        assert result.shape[0] == 4
        assert sorted(result["a"].to_list()) == [1, 2, 3, 4]

    def test_write_mode_overwrite_new_table(self, tmp_path):
        """mode='overwrite' creates table if it doesn't exist."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [10, 20]})

        write_ducklake(df, metadata_path, "test", mode="overwrite")

        result = read_ducklake(metadata_path, "test")
        assert_frame_equal(result.sort("a"), df.sort("a"))

    def test_write_mode_overwrite_replaces(self, tmp_path):
        """mode='overwrite' replaces all existing data."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pl.DataFrame({"a": [10, 20], "b": ["new1", "new2"]})

        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="overwrite")

        result = read_ducklake(metadata_path, "test")
        assert result.shape[0] == 2
        assert sorted(result["a"].to_list()) == [10, 20]

    def test_write_invalid_mode_raises(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})

        with pytest.raises(ValueError, match="Invalid write mode"):
            write_ducklake(df, metadata_path, "test", mode="invalid")

    def test_write_empty_df_mode_error(self, tmp_path):
        """mode='error' with empty DataFrame creates table but no data file."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int32), "b": pl.Series([], dtype=pl.String)})

        write_ducklake(df, metadata_path, "test", mode="error")

        result = read_ducklake(metadata_path, "test")
        assert result.shape == (0, 2)
        assert result.schema == {"a": pl.Int32, "b": pl.String}


# ---------------------------------------------------------------------------
# DuckDB interop: write with ducklake-polars, read with DuckDB
# ---------------------------------------------------------------------------


class TestDuckDBInterop:
    """Verify that catalogs written by ducklake-polars are readable by DuckDB."""

    def test_basic_interop(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["hello", "world", "test"]})

        write_ducklake(df, metadata_path, "test", mode="error")

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert len(pdf) == 3
        assert sorted(pdf["a"].to_list()) == [1, 2, 3]
        assert sorted(pdf["b"].to_list()) == ["hello", "test", "world"]

    def test_interop_multiple_inserts(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"x": [10, 20]})
        df2 = pl.DataFrame({"x": [30]})

        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert sorted(pdf["x"].to_list()) == [10, 20, 30]

    def test_interop_after_overwrite(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"v": [1, 2, 3]})
        df2 = pl.DataFrame({"v": [99]})

        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="overwrite")

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert pdf["v"].to_list() == [99]

    def test_interop_various_types(self, tmp_path):
        """Verify interop with various column types."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "int_col": pl.Series([1, 2], dtype=pl.Int32),
            "bigint_col": pl.Series([100, 200], dtype=pl.Int64),
            "float_col": pl.Series([1.5, 2.5], dtype=pl.Float64),
            "bool_col": [True, False],
            "str_col": ["hello", "world"],
        })

        write_ducklake(df, metadata_path, "test", mode="error")

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert len(pdf) == 2
        assert sorted(pdf["int_col"].to_list()) == [1, 2]
        assert sorted(pdf["bigint_col"].to_list()) == [100, 200]

    def test_interop_null_values(self, tmp_path):
        """Verify interop with NULL values."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "a": pl.Series([1, None, 3], dtype=pl.Int32),
            "b": ["hello", None, "world"],
        })

        write_ducklake(df, metadata_path, "test", mode="error")

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert len(pdf) == 3

        # Verify null handling
        result = read_ducklake(metadata_path, "test")
        assert result.sort("a", nulls_last=True)["a"].to_list() == [1, 3, None]

    def test_interop_date_types(self, tmp_path):
        """Verify interop with date/datetime types."""
        from datetime import date, datetime

        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "d": [date(2025, 1, 1), date(2025, 6, 15)],
            "ts": [datetime(2025, 1, 1, 12, 0), datetime(2025, 6, 15, 18, 30)],
        })

        write_ducklake(df, metadata_path, "test", mode="error")

        result = read_ducklake(metadata_path, "test")
        assert result.shape == (2, 2)
        assert result.sort("d")["d"].to_list() == [date(2025, 1, 1), date(2025, 6, 15)]


# ---------------------------------------------------------------------------
# Round-trip: write with ducklake-polars, read with ducklake-polars
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Write and read with ducklake-polars (no DuckDB in the loop)."""

    def test_roundtrip_basic(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "value": [1.1, 2.2, 3.3],
        })

        write_ducklake(df, metadata_path, "test", mode="error")
        result = read_ducklake(metadata_path, "test")
        assert_frame_equal(result.sort("id"), df.sort("id"))

    def test_roundtrip_with_scan(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"x": list(range(100))})

        write_ducklake(df, metadata_path, "test", mode="error")
        result = scan_ducklake(metadata_path, "test").filter(pl.col("x") > 95).collect()
        assert sorted(result["x"].to_list()) == [96, 97, 98, 99]

    def test_roundtrip_append_then_read(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)

        for i in range(5):
            df = pl.DataFrame({"batch": [i], "val": [i * 10]})
            write_ducklake(df, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test")
        assert result.shape[0] == 5
        assert sorted(result["batch"].to_list()) == [0, 1, 2, 3, 4]

    def test_roundtrip_overwrite(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [100]})

        write_ducklake(df1, metadata_path, "test", mode="error")
        write_ducklake(df2, metadata_path, "test", mode="overwrite")

        result = read_ducklake(metadata_path, "test")
        assert result["a"].to_list() == [100]


# ---------------------------------------------------------------------------
# Column statistics
# ---------------------------------------------------------------------------


class TestColumnStats:
    """Verify column statistics are correctly computed and stored."""

    def test_stats_registered(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "a": pl.Series([1, 5, 3], dtype=pl.Int32),
            "b": ["x", "z", "y"],
        })

        write_ducklake(df, metadata_path, "test", mode="error")

        # Check file column stats directly
        con = sqlite3.connect(metadata_path)
        stats = con.execute(
            "SELECT column_id, null_count, min_value, max_value "
            "FROM ducklake_file_column_stats "
            "ORDER BY column_id"
        ).fetchall()
        con.close()

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

    def test_stats_with_nulls(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "a": pl.Series([1, None, 3], dtype=pl.Int32),
        })

        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        stats = con.execute(
            "SELECT null_count, min_value, max_value "
            "FROM ducklake_file_column_stats WHERE column_id = 1"
        ).fetchone()
        con.close()

        assert stats is not None
        assert stats[0] == 1  # null_count
        assert stats[1] == "1"  # min
        assert stats[2] == "3"  # max

    def test_table_stats_updated(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3]})

        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        con = sqlite3.connect(metadata_path)
        row = con.execute(
            "SELECT record_count, next_row_id FROM ducklake_table_stats"
        ).fetchone()
        con.close()

        assert row is not None
        assert row[0] == 3  # total records
        assert row[1] == 3  # next_row_id

    def test_table_column_stats_aggregated(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": pl.Series([1, 5], dtype=pl.Int32)})
        df2 = pl.DataFrame({"a": pl.Series([3, 10], dtype=pl.Int32)})

        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        con = sqlite3.connect(metadata_path)
        row = con.execute(
            "SELECT min_value, max_value FROM ducklake_table_column_stats WHERE column_id = 1"
        ).fetchone()
        con.close()

        assert row is not None
        assert row[0] == "1"  # global min
        assert row[1] == "10"  # global max


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestWriteEdgeCases:
    """Test edge cases in the write path."""

    def test_write_single_row(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [42]})

        write_ducklake(df, metadata_path, "test", mode="error")

        result = read_ducklake(metadata_path, "test")
        assert result["a"].to_list() == [42]

    def test_write_many_columns(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        data = {f"col_{i}": [i] for i in range(20)}
        df = pl.DataFrame(data)

        write_ducklake(df, metadata_path, "test", mode="error")

        result = read_ducklake(metadata_path, "test")
        assert result.shape == (1, 20)
        for i in range(20):
            assert result[f"col_{i}"].to_list() == [i]

    def test_write_boolean_stats(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"flag": [True, False, True]})

        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        stats = con.execute(
            "SELECT min_value, max_value FROM ducklake_file_column_stats WHERE column_id = 1"
        ).fetchone()
        con.close()

        assert stats is not None
        assert stats[0] == "false"
        assert stats[1] == "true"

    def test_snapshot_progression(self, tmp_path):
        """Each write creates a new snapshot."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})

        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        snap_count_1 = con.execute("SELECT COUNT(*) FROM ducklake_snapshot").fetchone()[0]

        write_ducklake(pl.DataFrame({"a": [2]}), metadata_path, "test", mode="append")
        snap_count_2 = con.execute("SELECT COUNT(*) FROM ducklake_snapshot").fetchone()[0]
        con.close()

        # mode='error' creates 2 snapshots (create_table + insert)
        # mode='append' creates 1 more snapshot
        assert snap_count_2 > snap_count_1

    def test_parquet_file_naming(self, tmp_path):
        """Written Parquet files follow ducklake-{uuid7}.parquet naming."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})

        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        row = con.execute("SELECT path FROM ducklake_data_file LIMIT 1").fetchone()
        con.close()

        assert row is not None
        assert row[0].startswith("ducklake-")
        assert row[0].endswith(".parquet")

    def test_time_travel_after_write(self, tmp_path):
        """Time travel works correctly after writes."""
        metadata_path, data_path = _make_catalog(tmp_path)

        df1 = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df1, metadata_path, "test", mode="error")

        # Get snapshot after first write
        con = sqlite3.connect(metadata_path)
        snap_after_create = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
        con.close()

        df2 = pl.DataFrame({"a": [3, 4]})
        write_ducklake(df2, metadata_path, "test", mode="append")

        # Read at latest: should have 4 rows
        result_latest = read_ducklake(metadata_path, "test")
        assert result_latest.shape[0] == 4

        # Read at snapshot after first write: should have 2 rows
        result_old = read_ducklake(
            metadata_path, "test", snapshot_version=snap_after_create
        )
        assert result_old.shape[0] == 2
        assert sorted(result_old["a"].to_list()) == [1, 2]


# ---------------------------------------------------------------------------
# DuckDB writes, ducklake-polars reads (existing capability, regression)
# ---------------------------------------------------------------------------


class TestDuckDBWritePolarsRead:
    """Verify existing read path still works with DuckDB-created catalogs."""

    def test_duckdb_write_polars_read(self, tmp_path):
        """Standard flow: DuckDB creates data, ducklake-polars reads."""
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
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == ["hello", "world"]
