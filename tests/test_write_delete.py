"""Tests for ducklake-dataframe DELETE support (Phase 5e)."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    delete_ducklake,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Basic delete operations
# ---------------------------------------------------------------------------


class TestDeleteBasic:
    """Test basic delete operations."""

    def test_delete_some_rows(self, make_write_catalog):
        """Delete a subset of rows, verify remaining."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["one", "two", "three", "four", "five"],
        })
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        deleted = delete_ducklake(cat.metadata_path, "test", pl.col("a") > 3)
        assert deleted == 2

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        assert sorted(result["a"].to_list()) == [1, 2, 3]

    def test_delete_single_row(self, make_write_catalog):
        """Delete exactly one row."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [10, 20, 30]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        deleted = delete_ducklake(cat.metadata_path, "test", pl.col("a") == 20)
        assert deleted == 1

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [10, 30]

    def test_delete_all_rows(self, make_write_catalog):
        """Delete all rows from a table."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        deleted = delete_ducklake(cat.metadata_path, "test", pl.lit(True))
        assert deleted == 3

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0
        assert result.schema == {"a": pl.Int64}

    def test_delete_no_matching_rows(self, make_write_catalog):
        """Delete with no matching rows creates no snapshot."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        # Count snapshots before
        snap_before = cat.query_one("SELECT COUNT(*) FROM ducklake_snapshot")[0]

        deleted = delete_ducklake(cat.metadata_path, "test", pl.col("a") > 100)
        assert deleted == 0

        # No new snapshot
        snap_after = cat.query_one("SELECT COUNT(*) FROM ducklake_snapshot")[0]
        assert snap_after == snap_before

        # Data unchanged
        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Multiple deletes
# ---------------------------------------------------------------------------


class TestDeleteMultiple:
    """Test multiple sequential deletes."""

    def test_two_sequential_deletes(self, make_write_catalog):
        """Two deletes narrowing down data."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": list(range(10))})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.col("a") < 3)
        delete_ducklake(cat.metadata_path, "test", pl.col("a") >= 7)

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [3, 4, 5, 6]

    def test_three_sequential_deletes(self, make_write_catalog):
        """Three deletes."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": list(range(20))})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.col("a") < 5)
        delete_ducklake(cat.metadata_path, "test", pl.col("a") >= 15)
        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 10)

        result = read_ducklake(cat.metadata_path, "test")
        expected = [x for x in range(5, 15) if x != 10]
        assert sorted(result["a"].to_list()) == expected


# ---------------------------------------------------------------------------
# Multi-file deletes
# ---------------------------------------------------------------------------


class TestDeleteMultiFile:
    """Test deletes spanning multiple data files."""

    def test_delete_from_multiple_files(self, make_write_catalog):
        """Two inserts (two files), delete from both."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        df2 = pl.DataFrame({"a": [6, 7, 8, 9, 10]})
        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        # Delete even numbers from both files
        deleted = delete_ducklake(cat.metadata_path, "test", pl.col("a") % 2 == 0)
        assert deleted == 5

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 3, 5, 7, 9]

    def test_delete_from_one_of_two_files(self, make_write_catalog):
        """Two files, delete only affects one."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [100, 200, 300]})
        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        deleted = delete_ducklake(cat.metadata_path, "test", pl.col("a") < 10)
        assert deleted == 3

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [100, 200, 300]

    def test_delete_all_from_multiple_files(self, make_write_catalog):
        """Delete all rows from multiple files."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})
        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        deleted = delete_ducklake(cat.metadata_path, "test", pl.lit(True))
        assert deleted == 4

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# Delete + append
# ---------------------------------------------------------------------------


class TestDeleteThenAppend:
    """Test delete followed by append."""

    def test_delete_then_append(self, make_write_catalog):
        """Delete some rows then append new ones."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.col("a") <= 2)

        new_data = pl.DataFrame({"a": [10, 20]})
        write_ducklake(new_data, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [3, 4, 5, 10, 20]

    def test_append_then_delete(self, make_write_catalog):
        """Append new data then delete from both old and new."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df1, cat.metadata_path, "test", mode="error")

        df2 = pl.DataFrame({"a": [4, 5, 6]})
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        # Delete from both files
        deleted = delete_ducklake(cat.metadata_path, "test", pl.col("a") % 2 == 0)
        assert deleted == 3  # 2, 4, 6

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 3, 5]


# ---------------------------------------------------------------------------
# Time travel
# ---------------------------------------------------------------------------


class TestDeleteTimeTravel:
    """Test time travel with deletes."""

    def test_time_travel_before_delete(self, make_write_catalog):
        """Read at snapshot before delete sees all rows."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        # Get snapshot after insert
        snap_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        delete_ducklake(cat.metadata_path, "test", pl.col("a") > 3)

        # Latest: 3 rows
        result_latest = read_ducklake(cat.metadata_path, "test")
        assert sorted(result_latest["a"].to_list()) == [1, 2, 3]

        # Before delete: 5 rows
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        )
        assert sorted(result_old["a"].to_list()) == [1, 2, 3, 4, 5]

    def test_time_travel_between_deletes(self, make_write_catalog):
        """Multiple deletes, read at each intermediate version."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": list(range(10))})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        v0 = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")[0]

        delete_ducklake(cat.metadata_path, "test", pl.col("a") < 3)

        v1 = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")[0]

        delete_ducklake(cat.metadata_path, "test", pl.col("a") >= 7)

        # Latest: [3, 4, 5, 6]
        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [3, 4, 5, 6]

        # v1: [3, 4, 5, 6, 7, 8, 9]
        result_v1 = read_ducklake(cat.metadata_path, "test", snapshot_version=v1)
        assert sorted(result_v1["a"].to_list()) == [3, 4, 5, 6, 7, 8, 9]

        # v0: all 10
        result_v0 = read_ducklake(cat.metadata_path, "test", snapshot_version=v0)
        assert sorted(result_v0["a"].to_list()) == list(range(10))


# ---------------------------------------------------------------------------
# DuckDB interop: write with polars, delete with polars, read with DuckDB
# ---------------------------------------------------------------------------


class TestDeleteDuckDBInterop:
    """Verify delete files are readable by DuckDB."""

    def test_basic_delete_duckdb_reads(self, make_write_catalog):
        """Write + delete with ducklake-dataframe, read with DuckDB."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.col("a") > 3)

        pdf = cat.read_with_duckdb("test")
        assert sorted(pdf["a"].to_list()) == [1, 2, 3]

    def test_delete_all_duckdb_reads(self, make_write_catalog):
        """Delete all rows, DuckDB should see empty table."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.lit(True))

        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 0

    def test_multi_file_delete_duckdb_reads(self, make_write_catalog):
        """Multi-file delete interop."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [4, 5, 6]})
        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        delete_ducklake(cat.metadata_path, "test", pl.col("a") % 2 == 0)

        pdf = cat.read_with_duckdb("test")
        assert sorted(pdf["a"].to_list()) == [1, 3, 5]

    def test_delete_then_append_duckdb_reads(self, make_write_catalog):
        """Delete + append, verify DuckDB reads correctly."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 2)

        new_data = pl.DataFrame({"a": [10]})
        write_ducklake(new_data, cat.metadata_path, "test", mode="append")

        pdf = cat.read_with_duckdb("test")
        assert sorted(pdf["a"].to_list()) == [1, 3, 10]


# ---------------------------------------------------------------------------
# DuckDB writes, ducklake-dataframe deletes, both read
# ---------------------------------------------------------------------------


class TestDuckDBWritePolarsDelete:
    """DuckDB creates the data, ducklake-dataframe deletes, both read."""

    def test_duckdb_write_polars_delete(self, make_write_catalog):
        """Write with DuckDB, delete with ducklake-dataframe, read with both."""
        cat = make_write_catalog()

        # Write data with DuckDB
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
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four'), (5, 'five')"
        )
        con.close()

        # Delete with ducklake-dataframe
        deleted = delete_ducklake(cat.metadata_path, "test", pl.col("a") > 3)
        assert deleted == 2

        # Read with ducklake-dataframe
        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 2, 3]

        # Read with DuckDB
        pdf = cat.read_with_duckdb("test")
        assert sorted(pdf["a"].to_list()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Metadata verification
# ---------------------------------------------------------------------------


class TestDeleteMetadata:
    """Verify delete file metadata is correctly stored."""

    def test_delete_file_registered(self, make_write_catalog):
        """Check ducklake_delete_file entries."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.col("a") > 3)

        rows = cat.query_all(
            "SELECT delete_file_id, data_file_id, path, format, delete_count "
            "FROM ducklake_delete_file"
        )

        assert len(rows) == 1
        row = rows[0]
        assert row[2].startswith("ducklake-")
        assert row[2].endswith("-delete.parquet")
        assert row[3] == "parquet"
        assert row[4] == 2  # deleted 2 rows (a=4, a=5)

    def test_snapshot_changes_recorded(self, make_write_catalog):
        """Check snapshot_changes has deleted_from_table entry."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 2)

        row = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )

        assert row is not None
        assert "deleted_from_table" in row[0]

    def test_delete_file_naming(self, make_write_catalog):
        """Delete file follows ducklake-{uuid}-delete.parquet naming."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 1)

        row = cat.query_one("SELECT path FROM ducklake_delete_file LIMIT 1")

        assert row is not None
        path = row[0]
        assert path.startswith("ducklake-")
        assert "-delete.parquet" in path

    def test_delete_file_id_from_shared_counter(self, make_write_catalog):
        """delete_file_id is allocated from the same next_file_id counter."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        # After write, next_file_id should be 1 (data_file_id=0 was used)
        snap = cat.query_one(
            "SELECT next_file_id FROM ducklake_snapshot ORDER BY snapshot_id DESC LIMIT 1"
        )
        file_id_before = snap[0]

        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 2)

        del_row = cat.query_one(
            "SELECT delete_file_id FROM ducklake_delete_file"
        )
        snap_after = cat.query_one(
            "SELECT next_file_id FROM ducklake_snapshot ORDER BY snapshot_id DESC LIMIT 1"
        )

        assert del_row[0] == file_id_before  # delete_file_id = previous next_file_id
        assert snap_after[0] == file_id_before + 1  # counter incremented

    def test_multi_file_delete_creates_multiple_delete_files(self, make_write_catalog):
        """Each affected data file gets its own delete file."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [4, 5, 6]})
        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        # Delete even numbers from both files
        delete_ducklake(cat.metadata_path, "test", pl.col("a") % 2 == 0)

        del_rows = cat.query_all(
            "SELECT delete_file_id, data_file_id, delete_count "
            "FROM ducklake_delete_file ORDER BY delete_file_id"
        )

        assert len(del_rows) == 2  # one per affected data file
        # Each file has one even number deleted
        total_deleted = sum(r[2] for r in del_rows)
        assert total_deleted == 3  # 2, 4, 6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDeleteEdgeCases:
    """Edge cases for the delete path."""

    def test_delete_from_empty_table(self, make_write_catalog):
        """Delete from empty table returns 0."""
        cat = make_write_catalog()
        schema = {"a": pl.Int32()}
        from ducklake_polars import create_ducklake_table
        create_ducklake_table(cat.metadata_path, "test", schema)

        deleted = delete_ducklake(cat.metadata_path, "test", pl.col("a") > 0)
        assert deleted == 0

    def test_delete_with_string_predicate(self, make_write_catalog):
        """Delete using string column predicate."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "name": ["alice", "bob", "charlie"],
            "age": [30, 25, 35],
        })
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        deleted = delete_ducklake(cat.metadata_path, "test", pl.col("name") == "bob")
        assert deleted == 1

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["name"].to_list()) == ["alice", "charlie"]

    def test_delete_with_compound_predicate(self, make_write_catalog):
        """Delete using AND/OR predicates."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["x", "y", "x", "y", "x"],
        })
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        # Delete where a > 2 AND b == "x"
        deleted = delete_ducklake(
            cat.metadata_path, "test",
            (pl.col("a") > 2) & (pl.col("b") == "x"),
        )
        assert deleted == 2  # a=3,b=x and a=5,b=x

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 2, 4]

    def test_delete_preserves_schema(self, make_write_catalog):
        """After deleting all rows, schema is preserved."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "x": pl.Series([1, 2], dtype=pl.Int32),
            "y": ["a", "b"],
            "z": [1.0, 2.0],
        })
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.lit(True))

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (0, 3)
        assert result.schema == {"x": pl.Int32, "y": pl.String, "z": pl.Float64}

    def test_scan_after_delete(self, make_write_catalog):
        """scan_ducklake with filter after delete."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": list(range(100))})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(cat.metadata_path, "test", pl.col("a") < 50)

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("a") >= 75)
            .collect()
        )
        assert sorted(result["a"].to_list()) == list(range(75, 100))
