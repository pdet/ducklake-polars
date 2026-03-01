"""Tests for data inlining in the write path."""

from __future__ import annotations

import duckdb
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from tests_pandas.helpers import assert_list_equal
from ducklake_pandas import (
    delete_ducklake,
    read_ducklake,
    update_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INLINE_LIMIT = 1000


# ---------------------------------------------------------------------------
# Basic inlined insert
# ---------------------------------------------------------------------------


class TestInlinedInsert:
    """Test that small inserts go into inlined data tables."""

    def test_basic_inlined_insert(self, make_write_catalog):
        """Small insert below threshold is inlined — no Parquet files written."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Read back with ducklake-pandas
        result = read_ducklake(cat.metadata_path, "test")
        assert_frame_equal(result.sort_values("a").reset_index(drop=True), df.sort_values("a").reset_index(drop=True))

        # Verify data is inlined — no Parquet data files
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file"
        )
        assert row[0] == 0

        # Verify inlined data table exists
        inlined = cat.query_all(
            "SELECT table_name FROM ducklake_inlined_data_tables"
        )
        assert len(inlined) >= 1

        # Verify inlined data has correct row count
        tbl_name = inlined[0][0]
        safe = tbl_name.replace('"', '""')
        row = cat.query_one(
            f'SELECT COUNT(*) FROM "{safe}" WHERE end_snapshot IS NULL'
        )
        assert row[0] == 3

    def test_inlined_insert_roundtrip(self, make_write_catalog):
        """Write inlined, read back and filter."""
        cat = make_write_catalog()
        df = pd.DataFrame({"x": [10, 20, 30], "y": [1.1, 2.2, 3.3]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        result = read_ducklake(cat.metadata_path, "test")
        result = result[result["x"] > 15]
        result = result.sort_values("x").reset_index(drop=True)
        assert result["x"].tolist() == [20, 30]
        assert result["y"].tolist() == [2.2, 3.3]

    def test_inlined_multiple_inserts(self, make_write_catalog):
        """Multiple inlined inserts accumulate correctly."""
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        write_ducklake(
            df1, cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )
        write_ducklake(
            df2, cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == [1, 2, 3, 4]

    def test_inlined_various_types(self, make_write_catalog):
        """Verify inlining with various column types."""
        cat = make_write_catalog()
        df = pd.DataFrame({
            "int_col": pd.array([1, 2], dtype="object"),
            "bigint_col": pd.array([100, 200], dtype="object"),
            "float_col": pd.array([1.5, 2.5], dtype="object"),
            "bool_col": [True, False],
            "str_col": ["hello", "world"],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 5)
        assert sorted(result["int_col"].tolist()) == [1, 2]
        assert sorted(result["str_col"].tolist()) == ["hello", "world"]
        assert sorted(result["bool_col"].tolist()) == [False, True]


# ---------------------------------------------------------------------------
# DuckDB interop: DuckDB reads inlined data written by ducklake-pandas
# ---------------------------------------------------------------------------


class TestInlinedDuckDBInterop:
    """DuckDB can read inlined data written by ducklake-pandas."""

    def test_duckdb_reads_inlined(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        pdf = cat.read_with_duckdb("test", inline_limit=INLINE_LIMIT)
        assert len(pdf) == 3
        assert sorted(pdf["a"].tolist()) == [1, 2, 3]
        assert sorted(pdf["b"].tolist()) == ["x", "y", "z"]

    def test_duckdb_reads_after_multiple_inlined_inserts(self, make_write_catalog):
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"v": [10, 20]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )
        write_ducklake(
            pd.DataFrame({"v": [30]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        pdf = cat.read_with_duckdb("test", inline_limit=INLINE_LIMIT)
        assert sorted(pdf["v"].tolist()) == [10, 20, 30]


# ---------------------------------------------------------------------------
# Reverse interop: DuckDB writes inlined data, ducklake-pandas reads
# ---------------------------------------------------------------------------


class TestDuckDBInlinedPandasRead:
    """ducklake-pandas reads inlined data written by DuckDB."""

    def test_duckdb_inline_pandas_read(self, make_write_catalog):
        cat = make_write_catalog(inline=True, inline_limit=1000)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        if cat.backend == "sqlite":
            attach_source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            attach_source = f"ducklake:postgres:{cat.metadata_path}"
        con.execute(
            f"ATTACH '{attach_source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 1000)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute(
            "INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world'), (3, 'test')"
        )
        con.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (3, 2)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["hello", "world", "test"]


# ---------------------------------------------------------------------------
# Delete on inlined data
# ---------------------------------------------------------------------------


class TestInlinedDelete:
    """Test deleting from inlined data sets end_snapshot."""

    def test_delete_inlined_rows(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        deleted = delete_ducklake(
            cat.metadata_path, "test", lambda df: df["a"] == 3,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert deleted == 1

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 4, 5]
        assert result["b"].tolist() == ["a", "b", "d", "e"]

    def test_delete_multiple_inlined_rows(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        deleted = delete_ducklake(
            cat.metadata_path, "test", lambda df: df["a"] > 3,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert deleted == 2

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == [1, 2, 3]

    def test_delete_all_inlined_rows(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        deleted = delete_ducklake(
            cat.metadata_path, "test", True,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert deleted == 3

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0

    def test_delete_no_match_returns_zero(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        deleted = delete_ducklake(
            cat.metadata_path, "test", lambda df: df["a"] > 100,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert deleted == 0

    def test_delete_inlined_duckdb_interop(self, make_write_catalog):
        """DuckDB can read after inlined delete by ducklake-pandas."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )
        delete_ducklake(
            cat.metadata_path, "test", lambda df: df["a"] == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        pdf = cat.read_with_duckdb("test", inline_limit=INLINE_LIMIT)
        assert len(pdf) == 2
        assert sorted(pdf["a"].tolist()) == [1, 3]


# ---------------------------------------------------------------------------
# Threshold: insert exceeding limit goes to Parquet
# ---------------------------------------------------------------------------


class TestInlineThreshold:
    """Test that inserts exceeding the threshold go to Parquet."""

    def test_exceed_threshold_goes_to_parquet(self, make_write_catalog):
        """Insert exceeding limit writes to Parquet, not inline."""
        cat = make_write_catalog()
        small_df = pd.DataFrame({"a": list(range(5))})

        # First insert: inlined (5 rows < 10 limit)
        write_ducklake(
            small_df, cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=10,
        )

        # Verify inlined
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        assert row[0] == 0

        # Second insert: 6 more rows, total would be 11 > 10 limit
        big_df = pd.DataFrame({"a": list(range(100, 106))})
        write_ducklake(
            big_df, cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=10,
        )

        # Now there should be a Parquet file
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        assert row[0] == 1

        # Both inlined and Parquet rows should be readable
        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 11
        expected = list(range(5)) + list(range(100, 106))
        assert sorted(result["a"].tolist()) == sorted(expected)

    def test_disabled_inlining(self, make_write_catalog):
        """data_inlining_row_limit=0 means no inlining."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=0,
        )

        # Should have a Parquet file
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file"
        )
        assert row[0] == 1


# ---------------------------------------------------------------------------
# Time travel with inlined data
# ---------------------------------------------------------------------------


class TestInlinedTimeTravel:
    """Test time travel with inlined data."""

    def test_time_travel_inlined(self, make_write_catalog):
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1, 2]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Get snapshot after first insert
        row = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )
        snap_v1 = row[0]

        write_ducklake(
            pd.DataFrame({"a": [3, 4]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Latest: 4 rows
        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 4

        # At v1: 2 rows
        result_v1 = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_v1
        )
        assert result_v1.shape[0] == 2
        assert sorted(result_v1["a"].tolist()) == [1, 2]

    def test_time_travel_after_inlined_delete(self, make_write_catalog):
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1, 2, 3]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        row = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )
        snap_before_delete = row[0]

        delete_ducklake(
            cat.metadata_path, "test", lambda df: df["a"] == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        # Current: 2 rows
        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == [1, 3]

        # Before delete: 3 rows
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before_delete,
        )
        assert sorted(result_old["a"].tolist()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Overwrite with inlined data
# ---------------------------------------------------------------------------


class TestInlinedOverwrite:
    """Test overwrite mode with inlined data."""

    def test_overwrite_inlined(self, make_write_catalog):
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1, 2, 3]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )
        write_ducklake(
            pd.DataFrame({"a": [99]}), cat.metadata_path, "test",
            mode="overwrite", data_inlining_row_limit=INLINE_LIMIT,
        )

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].tolist() == [99]

    def test_overwrite_inlined_duckdb_interop(self, make_write_catalog):
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1, 2, 3]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )
        write_ducklake(
            pd.DataFrame({"a": [42]}), cat.metadata_path, "test",
            mode="overwrite", data_inlining_row_limit=INLINE_LIMIT,
        )

        pdf = cat.read_with_duckdb("test", inline_limit=INLINE_LIMIT)
        assert pdf["a"].tolist() == [42]


# ---------------------------------------------------------------------------
# Update on inlined data
# ---------------------------------------------------------------------------


class TestInlinedUpdate:
    """Test UPDATE on tables with inlined data."""

    def test_update_inlined_rows(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        updated = update_ducklake(
            cat.metadata_path, "test",
            {"b": "updated"},
            lambda df: df["a"] == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert updated == 1

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["x", "updated", "z"]

    def test_update_inlined_duckdb_interop(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["old", "old", "old"]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        update_ducklake(
            cat.metadata_path, "test",
            {"b": "new"},
            lambda df: df["a"] <= 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        pdf = cat.read_with_duckdb("test", inline_limit=INLINE_LIMIT)
        pdf_sorted = pdf.sort_values("a").reset_index(drop=True)
        assert pdf_sorted["a"].tolist() == [1, 2, 3]
        assert pdf_sorted["b"].tolist() == ["new", "new", "old"]


# ---------------------------------------------------------------------------
# Null values with inlined data
# ---------------------------------------------------------------------------


class TestInlinedNulls:
    """Test NULL handling in inlined data."""

    def test_inlined_nulls(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": pd.array([1, None, 3], dtype="object"),
            "b": ["hello", None, "world"],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a", na_position="last").reset_index(drop=True)
        import math
        a_list = result["a"].tolist()
        assert a_list[0] == 1.0
        assert a_list[1] == 3.0
        assert math.isnan(a_list[2])
        assert_list_equal(result["b"].tolist(), ["hello", "world", None])


# ---------------------------------------------------------------------------
# Metadata correctness
# ---------------------------------------------------------------------------


class TestInlinedMetadata:
    """Test that metadata tables are correctly populated."""

    def test_inlined_data_table_registration(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"x": [1, 2]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Check ducklake_inlined_data_tables
        row = cat.query_one(
            "SELECT table_id, table_name, schema_version "
            "FROM ducklake_inlined_data_tables"
        )
        assert row is not None
        table_id = row[0]
        tbl_name = row[1]
        schema_ver = row[2]

        # Table name follows the convention
        assert tbl_name == f"ducklake_inlined_data_{table_id}_{schema_ver}"

        # Check inlined data table has correct structure
        cols = cat.get_table_columns(tbl_name)
        col_names = [c[0] for c in cols]
        assert "row_id" in col_names
        assert "begin_snapshot" in col_names
        assert "end_snapshot" in col_names
        assert "x" in col_names

    def test_table_stats_updated(self, make_write_catalog):
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1, 2]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )
        write_ducklake(
            pd.DataFrame({"a": [3]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        row = cat.query_one(
            "SELECT record_count, next_row_id FROM ducklake_table_stats"
        )

        assert row is not None
        assert row[0] == 3  # total records
        assert row[1] == 3  # next_row_id

    def test_snapshot_changes_recorded(self, make_write_catalog):
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1]}), cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        changes = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )

        assert changes is not None
        assert "inserted_into_table" in changes[0]

    def test_row_ids_sequential(self, make_write_catalog):
        """Row IDs in inlined data are sequential across inserts."""
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [10, 20]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )
        write_ducklake(
            pd.DataFrame({"a": [30]}), cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Find the inlined data table
        row = cat.query_one(
            "SELECT table_name FROM ducklake_inlined_data_tables LIMIT 1"
        )
        tbl_name = row[0]
        safe = tbl_name.replace('"', '""')
        rows = cat.query_all(
            f'SELECT row_id FROM "{safe}" ORDER BY row_id'
        )

        row_ids = [r[0] for r in rows]
        assert row_ids == [0, 1, 2]
