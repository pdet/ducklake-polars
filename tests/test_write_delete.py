"""Tests for ducklake-polars DELETE support (Phase 5e)."""

from __future__ import annotations

import os
import sqlite3

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
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(tmp_path):
    """Create a DuckLake catalog via DuckDB and return (metadata_path, data_path)."""
    metadata_path = str(tmp_path / "delete_test.ducklake")
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
    """Read a table with DuckDB's DuckLake extension (interop verification)."""
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
# Basic delete operations
# ---------------------------------------------------------------------------


class TestDeleteBasic:
    """Test basic delete operations."""

    def test_delete_some_rows(self, tmp_path):
        """Delete a subset of rows, verify remaining."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["one", "two", "three", "four", "five"],
        })
        write_ducklake(df, metadata_path, "test", mode="error")

        deleted = delete_ducklake(metadata_path, "test", pl.col("a") > 3)
        assert deleted == 2

        result = read_ducklake(metadata_path, "test")
        assert result.shape[0] == 3
        assert sorted(result["a"].to_list()) == [1, 2, 3]

    def test_delete_single_row(self, tmp_path):
        """Delete exactly one row."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [10, 20, 30]})
        write_ducklake(df, metadata_path, "test", mode="error")

        deleted = delete_ducklake(metadata_path, "test", pl.col("a") == 20)
        assert deleted == 1

        result = read_ducklake(metadata_path, "test")
        assert sorted(result["a"].to_list()) == [10, 30]

    def test_delete_all_rows(self, tmp_path):
        """Delete all rows from a table."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        deleted = delete_ducklake(metadata_path, "test", pl.lit(True))
        assert deleted == 3

        result = read_ducklake(metadata_path, "test")
        assert result.shape[0] == 0
        assert result.schema == {"a": pl.Int64}

    def test_delete_no_matching_rows(self, tmp_path):
        """Delete with no matching rows creates no snapshot."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        # Count snapshots before
        con = sqlite3.connect(metadata_path)
        snap_before = con.execute("SELECT COUNT(*) FROM ducklake_snapshot").fetchone()[0]
        con.close()

        deleted = delete_ducklake(metadata_path, "test", pl.col("a") > 100)
        assert deleted == 0

        # No new snapshot
        con = sqlite3.connect(metadata_path)
        snap_after = con.execute("SELECT COUNT(*) FROM ducklake_snapshot").fetchone()[0]
        con.close()
        assert snap_after == snap_before

        # Data unchanged
        result = read_ducklake(metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Multiple deletes
# ---------------------------------------------------------------------------


class TestDeleteMultiple:
    """Test multiple sequential deletes."""

    def test_two_sequential_deletes(self, tmp_path):
        """Two deletes narrowing down data."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": list(range(10))})
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.col("a") < 3)
        delete_ducklake(metadata_path, "test", pl.col("a") >= 7)

        result = read_ducklake(metadata_path, "test")
        assert sorted(result["a"].to_list()) == [3, 4, 5, 6]

    def test_three_sequential_deletes(self, tmp_path):
        """Three deletes."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": list(range(20))})
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.col("a") < 5)
        delete_ducklake(metadata_path, "test", pl.col("a") >= 15)
        delete_ducklake(metadata_path, "test", pl.col("a") == 10)

        result = read_ducklake(metadata_path, "test")
        expected = [x for x in range(5, 15) if x != 10]
        assert sorted(result["a"].to_list()) == expected


# ---------------------------------------------------------------------------
# Multi-file deletes
# ---------------------------------------------------------------------------


class TestDeleteMultiFile:
    """Test deletes spanning multiple data files."""

    def test_delete_from_multiple_files(self, tmp_path):
        """Two inserts (two files), delete from both."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        df2 = pl.DataFrame({"a": [6, 7, 8, 9, 10]})
        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        # Delete even numbers from both files
        deleted = delete_ducklake(metadata_path, "test", pl.col("a") % 2 == 0)
        assert deleted == 5

        result = read_ducklake(metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 3, 5, 7, 9]

    def test_delete_from_one_of_two_files(self, tmp_path):
        """Two files, delete only affects one."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [100, 200, 300]})
        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        deleted = delete_ducklake(metadata_path, "test", pl.col("a") < 10)
        assert deleted == 3

        result = read_ducklake(metadata_path, "test")
        assert sorted(result["a"].to_list()) == [100, 200, 300]

    def test_delete_all_from_multiple_files(self, tmp_path):
        """Delete all rows from multiple files."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})
        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        deleted = delete_ducklake(metadata_path, "test", pl.lit(True))
        assert deleted == 4

        result = read_ducklake(metadata_path, "test")
        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# Delete + append
# ---------------------------------------------------------------------------


class TestDeleteThenAppend:
    """Test delete followed by append."""

    def test_delete_then_append(self, tmp_path):
        """Delete some rows then append new ones."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.col("a") <= 2)

        new_data = pl.DataFrame({"a": [10, 20]})
        write_ducklake(new_data, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test")
        assert sorted(result["a"].to_list()) == [3, 4, 5, 10, 20]

    def test_append_then_delete(self, tmp_path):
        """Append new data then delete from both old and new."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df1, metadata_path, "test", mode="error")

        df2 = pl.DataFrame({"a": [4, 5, 6]})
        write_ducklake(df2, metadata_path, "test", mode="append")

        # Delete from both files
        deleted = delete_ducklake(metadata_path, "test", pl.col("a") % 2 == 0)
        assert deleted == 3  # 2, 4, 6

        result = read_ducklake(metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 3, 5]


# ---------------------------------------------------------------------------
# Time travel
# ---------------------------------------------------------------------------


class TestDeleteTimeTravel:
    """Test time travel with deletes."""

    def test_time_travel_before_delete(self, tmp_path):
        """Read at snapshot before delete sees all rows."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, metadata_path, "test", mode="error")

        # Get snapshot after insert
        con = sqlite3.connect(metadata_path)
        snap_before = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
        con.close()

        delete_ducklake(metadata_path, "test", pl.col("a") > 3)

        # Latest: 3 rows
        result_latest = read_ducklake(metadata_path, "test")
        assert sorted(result_latest["a"].to_list()) == [1, 2, 3]

        # Before delete: 5 rows
        result_old = read_ducklake(
            metadata_path, "test", snapshot_version=snap_before
        )
        assert sorted(result_old["a"].to_list()) == [1, 2, 3, 4, 5]

    def test_time_travel_between_deletes(self, tmp_path):
        """Multiple deletes, read at each intermediate version."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": list(range(10))})
        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        v0 = con.execute("SELECT MAX(snapshot_id) FROM ducklake_snapshot").fetchone()[0]
        con.close()

        delete_ducklake(metadata_path, "test", pl.col("a") < 3)

        con = sqlite3.connect(metadata_path)
        v1 = con.execute("SELECT MAX(snapshot_id) FROM ducklake_snapshot").fetchone()[0]
        con.close()

        delete_ducklake(metadata_path, "test", pl.col("a") >= 7)

        # Latest: [3, 4, 5, 6]
        result = read_ducklake(metadata_path, "test")
        assert sorted(result["a"].to_list()) == [3, 4, 5, 6]

        # v1: [3, 4, 5, 6, 7, 8, 9]
        result_v1 = read_ducklake(metadata_path, "test", snapshot_version=v1)
        assert sorted(result_v1["a"].to_list()) == [3, 4, 5, 6, 7, 8, 9]

        # v0: all 10
        result_v0 = read_ducklake(metadata_path, "test", snapshot_version=v0)
        assert sorted(result_v0["a"].to_list()) == list(range(10))


# ---------------------------------------------------------------------------
# DuckDB interop: write with polars, delete with polars, read with DuckDB
# ---------------------------------------------------------------------------


class TestDeleteDuckDBInterop:
    """Verify delete files are readable by DuckDB."""

    def test_basic_delete_duckdb_reads(self, tmp_path):
        """Write + delete with ducklake-polars, read with DuckDB."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.col("a") > 3)

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert sorted(pdf["a"].to_list()) == [1, 2, 3]

    def test_delete_all_duckdb_reads(self, tmp_path):
        """Delete all rows, DuckDB should see empty table."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.lit(True))

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert len(pdf) == 0

    def test_multi_file_delete_duckdb_reads(self, tmp_path):
        """Multi-file delete interop."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [4, 5, 6]})
        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        delete_ducklake(metadata_path, "test", pl.col("a") % 2 == 0)

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert sorted(pdf["a"].to_list()) == [1, 3, 5]

    def test_delete_then_append_duckdb_reads(self, tmp_path):
        """Delete + append, verify DuckDB reads correctly."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.col("a") == 2)

        new_data = pl.DataFrame({"a": [10]})
        write_ducklake(new_data, metadata_path, "test", mode="append")

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert sorted(pdf["a"].to_list()) == [1, 3, 10]


# ---------------------------------------------------------------------------
# DuckDB writes, ducklake-polars deletes, both read
# ---------------------------------------------------------------------------


class TestDuckDBWritePolarsDelete:
    """DuckDB creates the data, ducklake-polars deletes, both read."""

    def test_duckdb_write_polars_delete(self, tmp_path):
        """Write with DuckDB, delete with ducklake-polars, read with both."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        # Write data with DuckDB
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
            "(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four'), (5, 'five')"
        )
        con.close()

        # Delete with ducklake-polars
        deleted = delete_ducklake(metadata_path, "test", pl.col("a") > 3)
        assert deleted == 2

        # Read with ducklake-polars
        result = read_ducklake(metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 2, 3]

        # Read with DuckDB
        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert sorted(pdf["a"].to_list()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Metadata verification
# ---------------------------------------------------------------------------


class TestDeleteMetadata:
    """Verify delete file metadata is correctly stored."""

    def test_delete_file_registered(self, tmp_path):
        """Check ducklake_delete_file entries."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.col("a") > 3)

        con = sqlite3.connect(metadata_path)
        rows = con.execute(
            "SELECT delete_file_id, data_file_id, path, format, delete_count "
            "FROM ducklake_delete_file"
        ).fetchall()
        con.close()

        assert len(rows) == 1
        row = rows[0]
        assert row[2].startswith("ducklake-")
        assert row[2].endswith("-delete.parquet")
        assert row[3] == "parquet"
        assert row[4] == 2  # deleted 2 rows (a=4, a=5)

    def test_snapshot_changes_recorded(self, tmp_path):
        """Check snapshot_changes has deleted_from_table entry."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.col("a") == 2)

        con = sqlite3.connect(metadata_path)
        row = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        con.close()

        assert row is not None
        assert "deleted_from_table" in row[0]

    def test_delete_file_naming(self, tmp_path):
        """Delete file follows ducklake-{uuid}-delete.parquet naming."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.col("a") == 1)

        con = sqlite3.connect(metadata_path)
        row = con.execute("SELECT path FROM ducklake_delete_file LIMIT 1").fetchone()
        con.close()

        assert row is not None
        path = row[0]
        assert path.startswith("ducklake-")
        assert "-delete.parquet" in path

    def test_delete_file_id_from_shared_counter(self, tmp_path):
        """delete_file_id is allocated from the same next_file_id counter."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        # After write, next_file_id should be 1 (data_file_id=0 was used)
        con = sqlite3.connect(metadata_path)
        snap = con.execute(
            "SELECT next_file_id FROM ducklake_snapshot ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        file_id_before = snap[0]
        con.close()

        delete_ducklake(metadata_path, "test", pl.col("a") == 2)

        con = sqlite3.connect(metadata_path)
        del_row = con.execute(
            "SELECT delete_file_id FROM ducklake_delete_file"
        ).fetchone()
        snap_after = con.execute(
            "SELECT next_file_id FROM ducklake_snapshot ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        con.close()

        assert del_row[0] == file_id_before  # delete_file_id = previous next_file_id
        assert snap_after[0] == file_id_before + 1  # counter incremented

    def test_multi_file_delete_creates_multiple_delete_files(self, tmp_path):
        """Each affected data file gets its own delete file."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [4, 5, 6]})
        write_ducklake(df1, metadata_path, "test", mode="append")
        write_ducklake(df2, metadata_path, "test", mode="append")

        # Delete even numbers from both files
        delete_ducklake(metadata_path, "test", pl.col("a") % 2 == 0)

        con = sqlite3.connect(metadata_path)
        del_rows = con.execute(
            "SELECT delete_file_id, data_file_id, delete_count "
            "FROM ducklake_delete_file ORDER BY delete_file_id"
        ).fetchall()
        con.close()

        assert len(del_rows) == 2  # one per affected data file
        # Each file has one even number deleted
        total_deleted = sum(r[2] for r in del_rows)
        assert total_deleted == 3  # 2, 4, 6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDeleteEdgeCases:
    """Edge cases for the delete path."""

    def test_delete_from_empty_table(self, tmp_path):
        """Delete from empty table returns 0."""
        metadata_path, data_path = _make_catalog(tmp_path)
        schema = {"a": pl.Int32()}
        from ducklake_polars import create_ducklake_table
        create_ducklake_table(metadata_path, "test", schema)

        deleted = delete_ducklake(metadata_path, "test", pl.col("a") > 0)
        assert deleted == 0

    def test_delete_with_string_predicate(self, tmp_path):
        """Delete using string column predicate."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "name": ["alice", "bob", "charlie"],
            "age": [30, 25, 35],
        })
        write_ducklake(df, metadata_path, "test", mode="error")

        deleted = delete_ducklake(metadata_path, "test", pl.col("name") == "bob")
        assert deleted == 1

        result = read_ducklake(metadata_path, "test")
        assert sorted(result["name"].to_list()) == ["alice", "charlie"]

    def test_delete_with_compound_predicate(self, tmp_path):
        """Delete using AND/OR predicates."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["x", "y", "x", "y", "x"],
        })
        write_ducklake(df, metadata_path, "test", mode="error")

        # Delete where a > 2 AND b == "x"
        deleted = delete_ducklake(
            metadata_path, "test",
            (pl.col("a") > 2) & (pl.col("b") == "x"),
        )
        assert deleted == 2  # a=3,b=x and a=5,b=x

        result = read_ducklake(metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 2, 4]

    def test_delete_preserves_schema(self, tmp_path):
        """After deleting all rows, schema is preserved."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "x": pl.Series([1, 2], dtype=pl.Int32),
            "y": ["a", "b"],
            "z": [1.0, 2.0],
        })
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.lit(True))

        result = read_ducklake(metadata_path, "test")
        assert result.shape == (0, 3)
        assert result.schema == {"x": pl.Int32, "y": pl.String, "z": pl.Float64}

    def test_scan_after_delete(self, tmp_path):
        """scan_ducklake with filter after delete."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": list(range(100))})
        write_ducklake(df, metadata_path, "test", mode="error")

        delete_ducklake(metadata_path, "test", pl.col("a") < 50)

        result = (
            scan_ducklake(metadata_path, "test")
            .filter(pl.col("a") >= 75)
            .collect()
        )
        assert sorted(result["a"].to_list()) == list(range(75, 100))
