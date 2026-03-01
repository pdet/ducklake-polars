"""Tests for ducklake-pandas UPDATE support."""

from __future__ import annotations

import duckdb
import pandas as pd
import numpy as np

from tests_pandas.helpers import assert_list_equal
from ducklake_pandas import (
    read_ducklake,
    update_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Basic update operations
# ---------------------------------------------------------------------------


class TestUpdateBasic:
    """Test basic update operations."""

    def test_update_single_column_literal(self, make_write_catalog):
        """Update a single column to a literal value."""
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["one", "two", "three", "four", "five"],
        })
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        updated = update_ducklake(
            cat.metadata_path, "test", {"b": "UPDATED"}, lambda df: df["a"] > 3
        )
        assert updated == 2

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3, 4, 5]
        assert result["b"].tolist() == ["one", "two", "three", "UPDATED", "UPDATED"]

    def test_update_multiple_columns(self, make_write_catalog):
        """Update multiple columns at once."""
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [10.0, 20.0, 30.0],
        })
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        updated = update_ducklake(
            cat.metadata_path, "test",
            {"b": "NEW", "c": 99.0},
            lambda df: df["a"] == 2,
        )
        assert updated == 1

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["b"].tolist() == ["x", "NEW", "z"]
        assert result["c"].tolist() == [10.0, 99.0, 30.0]

    def test_update_with_expression(self, make_write_catalog):
        """Update using a callable (computed value)."""
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["one", "two", "three", "four", "five"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        updated = update_ducklake(
            cat.metadata_path, "test",
            {"c": lambda df: df["c"] + 100.0, "b": "UPDATED"},
            lambda df: df["a"] >= 3,
        )
        assert updated == 3

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["c"].tolist() == [1.0, 2.0, 103.0, 104.0, 105.0]
        assert result["b"].tolist() == ["one", "two", "UPDATED", "UPDATED", "UPDATED"]

    def test_update_no_matching_rows(self, make_write_catalog):
        """Update with no matching rows creates no snapshot."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        row = cat.query_one("SELECT COUNT(*) FROM ducklake_snapshot")
        snap_before = row[0]

        updated = update_ducklake(
            cat.metadata_path, "test", {"a": 99}, lambda df: df["a"] > 100
        )
        assert updated == 0

        row = cat.query_one("SELECT COUNT(*) FROM ducklake_snapshot")
        snap_after = row[0]
        assert snap_after == snap_before

    def test_update_all_rows(self, make_write_catalog):
        """Update all rows."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        updated = update_ducklake(
            cat.metadata_path, "test", {"b": "ALL"}, lambda df: pd.Series([True] * len(df), index=df.index)
        )
        assert updated == 3

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["ALL", "ALL", "ALL"]


# ---------------------------------------------------------------------------
# Multi-file updates
# ---------------------------------------------------------------------------


class TestUpdateMultiFile:
    """Test updates spanning multiple data files."""

    def test_update_across_multiple_files(self, make_write_catalog):
        """Two inserts (two files), update rows from both."""
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pd.DataFrame({"a": [4, 5, 6], "b": ["p", "q", "r"]})
        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        # Update even numbers from both files
        updated = update_ducklake(
            cat.metadata_path, "test", {"b": "EVEN"}, lambda df: df["a"] % 2 == 0
        )
        assert updated == 3

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["b"].tolist() == ["x", "EVEN", "z", "EVEN", "q", "EVEN"]

    def test_update_one_of_two_files(self, make_write_catalog):
        """Two files, update only affects one."""
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pd.DataFrame({"a": [100, 200], "b": ["p", "q"]})
        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        updated = update_ducklake(
            cat.metadata_path, "test", {"b": "BIG"}, lambda df: df["a"] >= 100
        )
        assert updated == 2

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["b"].tolist() == ["x", "y", "z", "BIG", "BIG"]


# ---------------------------------------------------------------------------
# Time travel
# ---------------------------------------------------------------------------


class TestUpdateTimeTravel:
    """Test time travel with updates."""

    def test_time_travel_before_after_update(self, make_write_catalog):
        """Read at snapshot before and after update."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        row = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )
        snap_before = row[0]

        update_ducklake(cat.metadata_path, "test", {"b": "NEW"}, lambda df: df["a"] == 2)

        # Latest: updated
        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["b"].tolist() == ["x", "NEW", "z"]

        # Before update: original
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        ).sort_values("a").reset_index(drop=True)
        assert result_old["b"].tolist() == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# DuckDB interop
# ---------------------------------------------------------------------------


class TestUpdateDuckDBInterop:
    """Verify updated catalogs are readable by DuckDB."""

    def test_pandas_update_duckdb_reads(self, make_write_catalog):
        """Write + update with ducklake-pandas, read with DuckDB."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        update_ducklake(cat.metadata_path, "test", {"b": "UPDATED"}, lambda df: df["a"] > 1)

        pdf = cat.read_with_duckdb("test")
        pdf = pdf.sort_values("a").reset_index(drop=True)
        assert pdf["a"].tolist() == [1, 2, 3]
        assert pdf["b"].tolist() == ["x", "UPDATED", "UPDATED"]

    def test_duckdb_write_pandas_update_duckdb_reads(self, make_write_catalog):
        """DuckDB creates data, pandas updates, DuckDB reads result."""
        cat = make_write_catalog()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        if cat.backend == "sqlite":
            attach_source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            attach_source = f"ducklake:postgres:{cat.metadata_path}"
        con.execute(
            f"ATTACH '{attach_source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'one'), (2, 'two'), (3, 'three')"
        )
        con.close()

        updated = update_ducklake(
            cat.metadata_path, "test", {"b": "CHANGED"}, lambda df: df["a"] >= 2
        )
        assert updated == 2

        # Read with ducklake-pandas
        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["b"].tolist() == ["one", "CHANGED", "CHANGED"]

        # Read with DuckDB
        pdf = cat.read_with_duckdb("test").sort_values("a").reset_index(drop=True)
        assert pdf["b"].tolist() == ["one", "CHANGED", "CHANGED"]

    def test_update_with_expression_duckdb_reads(self, make_write_catalog):
        """Expression-based update readable by DuckDB."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "val": [10.0, 20.0, 30.0]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        update_ducklake(
            cat.metadata_path, "test",
            {"val": lambda df: df["val"] * 2},
            lambda df: df["a"] >= 2,
        )

        pdf = cat.read_with_duckdb("test").sort_values("a").reset_index(drop=True)
        assert pdf["val"].tolist() == [10.0, 40.0, 60.0]


# ---------------------------------------------------------------------------
# Metadata verification
# ---------------------------------------------------------------------------


class TestUpdateMetadata:
    """Verify update metadata is correctly stored."""

    def test_snapshot_changes_has_both_ops(self, make_write_catalog):
        """Check that changes_made has both inserted and deleted."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        update_ducklake(cat.metadata_path, "test", {"a": 99}, lambda df: df["a"] == 2)

        row = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )

        assert row is not None
        assert "inserted_into_table" in row[0]
        assert "deleted_from_table" in row[0]

    def test_update_creates_delete_and_data_file(self, make_write_catalog):
        """One delete file and one new data file should be created."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        update_ducklake(cat.metadata_path, "test", {"b": "NEW"}, lambda df: df["a"] == 2)

        # Should have 2 data files total (original + updated)
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file"
        )
        assert row[0] == 2

        # Should have 1 delete file
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_delete_file"
        )
        assert row[0] == 1

        # New data file should have 1 row (the updated row)
        row = cat.query_one(
            "SELECT record_count FROM ducklake_data_file "
            "ORDER BY data_file_id DESC LIMIT 1"
        )
        assert row[0] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestUpdateEdgeCases:
    """Edge cases for the update path."""

    def test_update_preserves_unmatched_rows(self, make_write_catalog):
        """Rows not matching the predicate are unchanged."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        update_ducklake(cat.metadata_path, "test", {"a": 0}, lambda df: df["a"] == 3)

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [0, 1, 2, 4, 5]

    def test_update_then_read_columns(self, make_write_catalog):
        """Column selection works after update."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        update_ducklake(cat.metadata_path, "test", {"b": "NEW"}, lambda df: df["a"] == 1)

        result = read_ducklake(cat.metadata_path, "test", columns=["a", "b"]).sort_values("a").reset_index(drop=True)
        assert list(result.columns) == ["a", "b"]
        assert result["b"].tolist() == ["NEW", "y"]

    def test_update_with_null_value(self, make_write_catalog):
        """Update a column to NULL."""
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        update_ducklake(
            cat.metadata_path, "test",
            {"b": None},
            lambda df: df["a"] == 2,
        )

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert_list_equal(result["b"].tolist(), ["x", None, "z"])

    def test_update_then_delete(self, make_write_catalog):
        """Update followed by delete."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        update_ducklake(cat.metadata_path, "test", {"a": 99}, lambda df: df["a"] == 3)

        from ducklake_pandas import delete_ducklake
        delete_ducklake(cat.metadata_path, "test", lambda df: df["a"] > 50)

        result = read_ducklake(cat.metadata_path, "test").sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 4, 5]
