"""Tests for add_files — registering existing Parquet files into DuckLake.

Covers the scenarios from ducklake-ref's 31 add_files tests:
- Basic single/multi file registration
- Schema validation and mismatch detection
- Mixed workflows: add_files + normal inserts
- Delete from added files
- Compaction (rewrite_data_files) after add_files
- Duplicate detection (same file twice)
- DuckDB interop — read files added by ducklake-dataframe
- Stats / filter pushdown on added files
- Time travel: read at snapshots before/after add_files
- Empty Parquet files
- Large file registration
- Various data types
- Author/commit message metadata
"""

from __future__ import annotations

import os

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    add_files_ducklake,
    create_ducklake_table,
    delete_ducklake,
    list_snapshots,
    read_ducklake,
    rewrite_data_files_ducklake,
    scan_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet_file(path: str, df: pl.DataFrame) -> str:
    """Write a Polars DataFrame to a Parquet file and return its path."""
    arrow_table = df.to_arrow()
    pq.write_table(arrow_table, path)
    return path


def _write_arrow_parquet(path: str, table: pa.Table) -> str:
    """Write a PyArrow table directly to a Parquet file."""
    pq.write_table(table, path)
    return path


def _base_dir(catalog) -> str:
    """Return the base directory for writing external Parquet files."""
    if catalog.backend == "sqlite":
        return os.path.dirname(catalog.metadata_path)
    return catalog.data_path


# ---------------------------------------------------------------------------
# Basic add_files tests
# ---------------------------------------------------------------------------


class TestAddFilesBasic:
    """Core add_files functionality."""

    def test_add_single_file(self, ducklake_catalog):
        """Add a single Parquet file to an existing table."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT, b VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        parquet_path = _write_parquet_file(os.path.join(base, "single.parquet"), df)

        add_files_ducklake(
            catalog.metadata_path,
            "t1",
            [parquet_path],
            data_path=catalog.data_path,
        )

        result = read_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        )
        assert len(result) == 3
        assert sorted(result["a"].to_list()) == [1, 2, 3]

    def test_add_multiple_files(self, ducklake_catalog):
        """Add multiple Parquet files at once."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (x BIGINT, y DOUBLE)")
        catalog.close()

        base = _base_dir(catalog)
        df1 = pl.DataFrame({"x": [1, 2], "y": [1.0, 2.0]})
        df2 = pl.DataFrame({"x": [3, 4], "y": [3.0, 4.0]})
        path1 = _write_parquet_file(os.path.join(base, "part1.parquet"), df1)
        path2 = _write_parquet_file(os.path.join(base, "part2.parquet"), df2)

        add_files_ducklake(
            catalog.metadata_path,
            "t1",
            [path1, path2],
            data_path=catalog.data_path,
        )

        result = read_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        )
        assert len(result) == 4
        assert sorted(result["x"].to_list()) == [1, 2, 3, 4]

    def test_read_after_add_files(self, ducklake_catalog):
        """Data is correctly visible after adding files."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (id BIGINT, name VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"id": [10, 20, 30], "name": ["alice", "bob", "carol"]})
        parquet_path = _write_parquet_file(os.path.join(base, "people.parquet"), df)

        add_files_ducklake(
            catalog.metadata_path,
            "t1",
            [parquet_path],
            data_path=catalog.data_path,
        )

        result = read_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        )
        expected = df.sort("id")
        actual = result.sort("id")
        assert_frame_equal(actual, expected)

    def test_add_files_returns_snapshot_id(self, ducklake_catalog):
        """add_files returns a valid snapshot id."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"v": [1]})
        path = _write_parquet_file(os.path.join(base, "ret.parquet"), df)

        snap = add_files_ducklake(
            catalog.metadata_path,
            "t1",
            [path],
            data_path=catalog.data_path,
        )

        assert isinstance(snap, int)
        assert snap > 0

    def test_add_files_empty_list_raises(self, ducklake_catalog):
        """Passing an empty file list raises ValueError."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        with pytest.raises(ValueError, match="file_paths must not be empty"):
            add_files_ducklake(
                catalog.metadata_path,
                "t1",
                [],
                data_path=catalog.data_path,
            )

    def test_add_files_nonexistent_table_raises(self, ducklake_catalog):
        """Adding files to a nonexistent table raises an error."""
        catalog = ducklake_catalog
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"a": [1]})
        path = _write_parquet_file(os.path.join(base, "noent.parquet"), df)

        with pytest.raises(Exception):
            add_files_ducklake(
                catalog.metadata_path,
                "no_such_table",
                [path],
                data_path=catalog.data_path,
            )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestAddFilesSchemaValidation:
    """Schema mismatch detection."""

    def test_schema_mismatch_wrong_columns(self, ducklake_catalog):
        """File with different column names is rejected."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT, b VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        df_bad = pl.DataFrame({"a": [1], "c": [2]})
        bad_path = _write_parquet_file(os.path.join(base, "bad.parquet"), df_bad)

        with pytest.raises(ValueError, match="Schema mismatch"):
            add_files_ducklake(
                catalog.metadata_path,
                "t1",
                [bad_path],
                data_path=catalog.data_path,
            )

    def test_schema_mismatch_extra_column(self, ducklake_catalog):
        """File with extra columns is rejected."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df_extra = pl.DataFrame({"a": [1], "b": [2]})
        bad_path = _write_parquet_file(os.path.join(base, "extra.parquet"), df_extra)

        with pytest.raises(ValueError, match="Schema mismatch"):
            add_files_ducklake(
                catalog.metadata_path,
                "t1",
                [bad_path],
                data_path=catalog.data_path,
            )

    def test_schema_mismatch_missing_column(self, ducklake_catalog):
        """File with fewer columns than the table is rejected."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT, b VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        df_less = pl.DataFrame({"a": [1]})
        bad_path = _write_parquet_file(os.path.join(base, "fewer.parquet"), df_less)

        with pytest.raises(ValueError, match="Schema mismatch"):
            add_files_ducklake(
                catalog.metadata_path,
                "t1",
                [bad_path],
                data_path=catalog.data_path,
            )

    def test_schema_valid_same_types(self, ducklake_catalog):
        """File with matching schema is accepted."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT, b VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        df_ok = pl.DataFrame({"a": [42], "b": ["hello"]})
        ok_path = _write_parquet_file(os.path.join(base, "ok.parquet"), df_ok)

        add_files_ducklake(
            catalog.metadata_path,
            "t1",
            [ok_path],
            data_path=catalog.data_path,
        )

        result = read_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        )
        assert len(result) == 1
        assert result["a"][0] == 42


# ---------------------------------------------------------------------------
# Mixed workflows: add_files + normal inserts
# ---------------------------------------------------------------------------


class TestAddFilesMixedWorkflows:
    """Combining add_files with normal write operations."""

    def test_add_files_then_insert(self, ducklake_catalog):
        """Adding files then inserting more data -- both visible."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df_file = pl.DataFrame({"v": [100, 200]})
        parquet_path = _write_parquet_file(os.path.join(base, "added.parquet"), df_file)
        add_files_ducklake(
            catalog.metadata_path, "t1", [parquet_path], data_path=catalog.data_path,
        )

        df_insert = pl.DataFrame({"v": [300, 400]})
        write_ducklake(
            df_insert, catalog.metadata_path, "t1",
            mode="append", data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 4
        assert sorted(result["v"].to_list()) == [100, 200, 300, 400]

    def test_insert_then_add_files(self, ducklake_catalog):
        """Insert data first, then add files -- both visible."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.execute("INSERT INTO ducklake.t1 VALUES (1), (2)")
        catalog.close()

        base = _base_dir(catalog)
        df_file = pl.DataFrame({"v": [3, 4]})
        parquet_path = _write_parquet_file(os.path.join(base, "more.parquet"), df_file)
        add_files_ducklake(
            catalog.metadata_path, "t1", [parquet_path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 4
        assert sorted(result["v"].to_list()) == [1, 2, 3, 4]

    def test_multiple_add_files_calls(self, ducklake_catalog):
        """Calling add_files multiple times accumulates data."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (k BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        for i in range(3):
            df = pl.DataFrame({"k": [i * 10 + j for j in range(5)]})
            path = _write_parquet_file(os.path.join(base, f"batch_{i}.parquet"), df)
            add_files_ducklake(
                catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
            )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 15

    def test_mixed_insert_add_insert(self, ducklake_catalog):
        """Interleaved insert and add_files operations."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.execute("INSERT INTO ducklake.t1 VALUES (1)")
        catalog.close()

        base = _base_dir(catalog)
        df_ext = pl.DataFrame({"v": [2]})
        path = _write_parquet_file(os.path.join(base, "ext.parquet"), df_ext)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        df_ins = pl.DataFrame({"v": [3]})
        write_ducklake(
            df_ins, catalog.metadata_path, "t1",
            mode="append", data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert sorted(result["v"].to_list()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Delete from added files
# ---------------------------------------------------------------------------


class TestAddFilesDelete:
    """Deleting rows from data added via add_files."""

    def test_delete_from_added_file(self, ducklake_catalog):
        """Delete specific rows from a file registered via add_files."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (id BIGINT, val VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"id": [1, 2, 3, 4, 5], "val": ["a", "b", "c", "d", "e"]})
        path = _write_parquet_file(os.path.join(base, "to_delete.parquet"), df)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        deleted = delete_ducklake(
            catalog.metadata_path, "t1", pl.col("id") > 3, data_path=catalog.data_path,
        )
        assert deleted == 2

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 3
        assert sorted(result["id"].to_list()) == [1, 2, 3]

    def test_delete_all_from_added_file(self, ducklake_catalog):
        """Delete all rows from a file added via add_files."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"v": [10, 20, 30]})
        path = _write_parquet_file(os.path.join(base, "all_del.parquet"), df)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        deleted = delete_ducklake(
            catalog.metadata_path, "t1", pl.col("v") >= 0, data_path=catalog.data_path,
        )
        assert deleted == 3

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 0

    def test_delete_from_mixed_sources(self, ducklake_catalog):
        """Delete rows across both inserted data and added files."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.execute("INSERT INTO ducklake.t1 VALUES (1), (2), (3)")
        catalog.close()

        base = _base_dir(catalog)
        df_ext = pl.DataFrame({"v": [4, 5, 6]})
        path = _write_parquet_file(os.path.join(base, "ext.parquet"), df_ext)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        deleted = delete_ducklake(
            catalog.metadata_path, "t1", (pl.col("v") % 2) == 0, data_path=catalog.data_path,
        )
        assert deleted == 3  # 2, 4, 6

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert sorted(result["v"].to_list()) == [1, 3, 5]


# ---------------------------------------------------------------------------
# Compaction (rewrite_data_files) after add_files
# ---------------------------------------------------------------------------


class TestAddFilesCompaction:
    """Rewriting data files after add_files registration."""

    def test_compaction_after_add_files(self, ducklake_catalog):
        """Rewrite data files merges added files into the catalog-managed path."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT, b VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        for i in range(5):
            df = pl.DataFrame({"a": [i], "b": [f"val_{i}"]})
            path = _write_parquet_file(os.path.join(base, f"small_{i}.parquet"), df)
            add_files_ducklake(
                catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
            )

        result_before = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result_before) == 5

        snap = rewrite_data_files_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path,
        )
        assert snap > 0

        result_after = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result_after) == 5
        assert sorted(result_after["a"].to_list()) == [0, 1, 2, 3, 4]

    def test_compaction_after_add_files_and_delete(self, ducklake_catalog):
        """Compact after add_files + delete -- deleted rows are gone."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (id BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"id": list(range(10))})
        path = _write_parquet_file(os.path.join(base, "data.parquet"), df)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        delete_ducklake(
            catalog.metadata_path, "t1", pl.col("id") >= 5, data_path=catalog.data_path,
        )

        rewrite_data_files_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 5
        assert sorted(result["id"].to_list()) == [0, 1, 2, 3, 4]

    def test_compaction_mixed_added_and_inserted(self, ducklake_catalog):
        """Compact a table with both added files and normal inserts."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.execute("INSERT INTO ducklake.t1 VALUES (1), (2)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"v": [3, 4]})
        path = _write_parquet_file(os.path.join(base, "ext.parquet"), df)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        rewrite_data_files_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 4
        assert sorted(result["v"].to_list()) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# DuckDB interop
# ---------------------------------------------------------------------------


class TestAddFilesInterop:
    """DuckDB can read files added by ducklake-dataframe."""

    def test_duckdb_reads_added_files(self, ducklake_catalog):
        """DuckDB can read data registered via add_files_ducklake."""
        import duckdb

        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT, b VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"a": [10, 20], "b": ["foo", "bar"]})
        path = _write_parquet_file(os.path.join(base, "interop.parquet"), df)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        if catalog.backend == "sqlite":
            con.install_extension("sqlite_scanner")
            con.load_extension("sqlite_scanner")
            source = f"ducklake:sqlite:{catalog.metadata_path}"
        else:
            source = f"ducklake:postgres:{catalog.metadata_path}"

        con.execute(
            f"ATTACH \'{source}\' AS ducklake "
            f"(DATA_PATH \'{catalog.data_path}\', DATA_INLINING_ROW_LIMIT 0)"
        )
        rows = con.execute("SELECT * FROM ducklake.t1 ORDER BY a").fetchall()
        con.close()

        assert len(rows) == 2
        assert rows[0] == (10, "foo")
        assert rows[1] == (20, "bar")

    def test_duckdb_reads_mixed_added_and_inserted(self, ducklake_catalog):
        """DuckDB reads both add_files data and normally inserted data."""
        import duckdb

        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.execute("INSERT INTO ducklake.t1 VALUES (1)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"v": [2]})
        path = _write_parquet_file(os.path.join(base, "mix.parquet"), df)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        if catalog.backend == "sqlite":
            con.install_extension("sqlite_scanner")
            con.load_extension("sqlite_scanner")
            source = f"ducklake:sqlite:{catalog.metadata_path}"
        else:
            source = f"ducklake:postgres:{catalog.metadata_path}"

        con.execute(
            f"ATTACH \'{source}\' AS ducklake "
            f"(DATA_PATH \'{catalog.data_path}\', DATA_INLINING_ROW_LIMIT 0)"
        )
        rows = con.execute("SELECT v FROM ducklake.t1 ORDER BY v").fetchall()
        con.close()

        assert [r[0] for r in rows] == [1, 2]


# ---------------------------------------------------------------------------
# Stats / filter pushdown on added files
# ---------------------------------------------------------------------------


class TestAddFilesStats:
    """Filter pushdown works on files registered via add_files."""

    def test_filter_on_added_file(self, ducklake_catalog):
        """A filter on a scan over added files works correctly."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (id BIGINT, label VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({
            "id": list(range(100)),
            "label": [f"item_{i}" for i in range(100)],
        })
        path = _write_parquet_file(os.path.join(base, "stats.parquet"), df)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = scan_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        ).filter(pl.col("id") == 42).collect()

        assert len(result) == 1
        assert result["id"][0] == 42
        assert result["label"][0] == "item_42"

    def test_filter_across_multiple_added_files(self, ducklake_catalog):
        """Filter pushdown across multiple added files with non-overlapping ranges."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (n BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df1 = pl.DataFrame({"n": list(range(100))})
        df2 = pl.DataFrame({"n": list(range(1000, 1100))})
        p1 = _write_parquet_file(os.path.join(base, "range_a.parquet"), df1)
        p2 = _write_parquet_file(os.path.join(base, "range_b.parquet"), df2)

        add_files_ducklake(
            catalog.metadata_path, "t1", [p1, p2], data_path=catalog.data_path,
        )

        result = scan_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        ).filter(pl.col("n") == 1050).collect()

        assert len(result) == 1
        assert result["n"][0] == 1050

    def test_filter_range_on_added_files(self, ducklake_catalog):
        """Range filter on added files."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (val BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"val": list(range(50))})
        path = _write_parquet_file(os.path.join(base, "range_filt.parquet"), df)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = scan_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        ).filter((pl.col("val") >= 10) & (pl.col("val") < 20)).collect()

        assert len(result) == 10
        assert sorted(result["val"].to_list()) == list(range(10, 20))


# ---------------------------------------------------------------------------
# Time travel
# ---------------------------------------------------------------------------


class TestAddFilesTimeTravel:
    """Time travel reads at snapshots before/after add_files."""

    def test_time_travel_before_add_files(self, ducklake_catalog):
        """Reading at a snapshot before add_files shows no added data."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.execute("INSERT INTO ducklake.t1 VALUES (1)")
        catalog.close()

        snaps_before = list_snapshots(
            catalog.metadata_path, data_path=catalog.data_path
        )
        snap_before = max(s["snapshot_id"] for s in snaps_before)

        base = _base_dir(catalog)
        df = pl.DataFrame({"v": [99]})
        path = _write_parquet_file(os.path.join(base, "tt.parquet"), df)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result_latest = read_ducklake(
            catalog.metadata_path, "t1", data_path=catalog.data_path
        )
        assert len(result_latest) == 2

        result_old = read_ducklake(
            catalog.metadata_path, "t1",
            snapshot_version=snap_before, data_path=catalog.data_path,
        )
        assert len(result_old) == 1
        assert result_old["v"][0] == 1

    def test_time_travel_after_add_files(self, ducklake_catalog):
        """Reading at the add_files snapshot shows the added data."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"v": [42]})
        path = _write_parquet_file(os.path.join(base, "tt2.parquet"), df)
        snap = add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(
            catalog.metadata_path, "t1",
            snapshot_version=snap, data_path=catalog.data_path,
        )
        assert len(result) == 1
        assert result["v"][0] == 42

    def test_time_travel_multiple_add_files(self, ducklake_catalog):
        """Each add_files creates a new snapshot -- time travel to each."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        snapshots = []
        for i in range(3):
            df = pl.DataFrame({"v": [i * 100]})
            path = _write_parquet_file(os.path.join(base, f"tt_{i}.parquet"), df)
            s = add_files_ducklake(
                catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
            )
            snapshots.append(s)

        r1 = read_ducklake(
            catalog.metadata_path, "t1",
            snapshot_version=snapshots[0], data_path=catalog.data_path,
        )
        assert len(r1) == 1

        r2 = read_ducklake(
            catalog.metadata_path, "t1",
            snapshot_version=snapshots[1], data_path=catalog.data_path,
        )
        assert len(r2) == 2

        r3 = read_ducklake(
            catalog.metadata_path, "t1",
            snapshot_version=snapshots[2], data_path=catalog.data_path,
        )
        assert len(r3) == 3


# ---------------------------------------------------------------------------
# Empty files
# ---------------------------------------------------------------------------


class TestAddFilesEmpty:
    """Edge cases with empty Parquet files."""

    def test_add_empty_parquet_file(self, ducklake_catalog):
        """Adding an empty Parquet file (0 rows) is handled gracefully."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT, b VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        empty_table = pa.table({
            "a": pa.array([], type=pa.int64()),
            "b": pa.array([], type=pa.string()),
        })
        path = _write_arrow_parquet(os.path.join(base, "empty.parquet"), empty_table)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 0

    def test_add_empty_then_normal_data(self, ducklake_catalog):
        """Adding an empty file then inserting data -- inserts are visible."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        empty_table = pa.table({"v": pa.array([], type=pa.int64())})
        path = _write_arrow_parquet(os.path.join(base, "empty.parquet"), empty_table)
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        df = pl.DataFrame({"v": [1, 2, 3]})
        write_ducklake(
            df, catalog.metadata_path, "t1",
            mode="append", data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Large file registration
# ---------------------------------------------------------------------------


class TestAddFilesLarge:
    """Registering files with many rows."""

    def test_add_large_file(self, ducklake_catalog):
        """Add a Parquet file with 10k rows."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (id BIGINT, value DOUBLE)")
        catalog.close()

        base = _base_dir(catalog)
        n = 10_000
        df = pl.DataFrame({
            "id": list(range(n)),
            "value": [float(i) * 0.1 for i in range(n)],
        })
        path = _write_parquet_file(os.path.join(base, "big.parquet"), df)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == n

    def test_add_many_small_files(self, ducklake_catalog):
        """Add 20 small files in one call."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (k BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        paths = []
        for i in range(20):
            df = pl.DataFrame({"k": [i]})
            p = _write_parquet_file(os.path.join(base, f"s_{i}.parquet"), df)
            paths.append(p)

        add_files_ducklake(
            catalog.metadata_path, "t1", paths, data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 20
        assert sorted(result["k"].to_list()) == list(range(20))


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


class TestAddFilesDuplicates:
    """Adding the same file twice -- no dedup, just accumulates."""

    def test_add_same_file_twice(self, ducklake_catalog):
        """Adding the same file path twice registers it twice (rows doubled)."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"v": [1, 2, 3]})
        path = _write_parquet_file(os.path.join(base, "dupe.parquet"), df)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )
        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 6

    def test_add_same_file_in_one_call(self, ducklake_catalog):
        """Passing the same path twice in one call registers it twice."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"v": [10]})
        path = _write_parquet_file(os.path.join(base, "dup2.parquet"), df)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path, path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Various data types
# ---------------------------------------------------------------------------


class TestAddFilesTypes:
    """Adding files with different data types."""

    def test_integer_types(self, ducklake_catalog):
        """Add file with various integer columns."""
        catalog = ducklake_catalog
        catalog.execute(
            "CREATE TABLE ducklake.t1 ("
            "  i8 TINYINT, i16 SMALLINT, i32 INTEGER, i64 BIGINT"
            ")"
        )
        catalog.close()

        base = _base_dir(catalog)
        tbl = pa.table({
            "i8": pa.array([1, 2], type=pa.int8()),
            "i16": pa.array([100, 200], type=pa.int16()),
            "i32": pa.array([1000, 2000], type=pa.int32()),
            "i64": pa.array([10000, 20000], type=pa.int64()),
        })
        path = _write_arrow_parquet(os.path.join(base, "ints.parquet"), tbl)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 2

    def test_float_and_double(self, ducklake_catalog):
        """Add file with float and double columns."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (f FLOAT, d DOUBLE)")
        catalog.close()

        base = _base_dir(catalog)
        tbl = pa.table({
            "f": pa.array([1.5, 2.5], type=pa.float32()),
            "d": pa.array([3.14, 2.72], type=pa.float64()),
        })
        path = _write_arrow_parquet(os.path.join(base, "floats.parquet"), tbl)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 2

    def test_date_and_timestamp(self, ducklake_catalog):
        """Add file with date and timestamp columns."""
        from datetime import date, datetime

        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (dt DATE, ts TIMESTAMP)")
        catalog.close()

        base = _base_dir(catalog)
        tbl = pa.table({
            "dt": pa.array([date(2024, 1, 1), date(2024, 6, 15)], type=pa.date32()),
            "ts": pa.array([
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 6, 15, 18, 30, 0),
            ], type=pa.timestamp("us")),
        })
        path = _write_arrow_parquet(os.path.join(base, "temporal.parquet"), tbl)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 2

    def test_varchar_with_special_chars(self, ducklake_catalog):
        """Add file with varchar data including special characters."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (s VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"s": ["hello", "world", "duck", "line\nbreak", ""]})
        path = _write_parquet_file(os.path.join(base, "strings.parquet"), df)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 5

    def test_boolean_type(self, ducklake_catalog):
        """Add file with boolean column."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (flag BOOLEAN)")
        catalog.close()

        base = _base_dir(catalog)
        tbl = pa.table({"flag": pa.array([True, False, True], type=pa.bool_())})
        path = _write_arrow_parquet(os.path.join(base, "bools.parquet"), tbl)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 3

    def test_nullable_data(self, ducklake_catalog):
        """Add file with null values."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (a BIGINT, b VARCHAR)")
        catalog.close()

        base = _base_dir(catalog)
        tbl = pa.table({
            "a": pa.array([1, None, 3], type=pa.int64()),
            "b": pa.array(["x", None, "z"], type=pa.string()),
        })
        path = _write_arrow_parquet(os.path.join(base, "nulls.parquet"), tbl)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 3
        assert result["a"].null_count() == 1


# ---------------------------------------------------------------------------
# Author / commit message tracking
# ---------------------------------------------------------------------------


class TestAddFilesMetadata:
    """Snapshot metadata (author, commit_message) for add_files."""

    def test_add_files_with_author(self, ducklake_catalog):
        """add_files records author in the snapshot changes table."""
        catalog = ducklake_catalog
        catalog.execute("CREATE TABLE ducklake.t1 (v BIGINT)")
        catalog.close()

        base = _base_dir(catalog)
        df = pl.DataFrame({"v": [1]})
        path = _write_parquet_file(os.path.join(base, "auth.parquet"), df)

        snap = add_files_ducklake(
            catalog.metadata_path,
            "t1",
            [path],
            data_path=catalog.data_path,
            author="test_bot",
            commit_message="added external data",
        )

        # Query the metadata directly for author/commit_message
        row = catalog.query_metadata(
            "SELECT author, commit_message FROM ducklake_snapshot_changes "
            "WHERE snapshot_id = ?",
            [snap],
        )
        assert row is not None
        assert row[0] == "test_bot"
        assert row[1] == "added external data"


# ---------------------------------------------------------------------------
# create_ducklake_table + add_files (no DuckDB table creation)
# ---------------------------------------------------------------------------


class TestAddFilesWithCreateTable:
    """Using create_ducklake_table (pure dataframe API) then add_files."""

    def test_create_table_then_add_files(self, ducklake_catalog):
        """Create table via API, then add external files."""
        catalog = ducklake_catalog
        catalog.close()

        schema = {"id": pl.Int64, "name": pl.Utf8}
        create_ducklake_table(
            catalog.metadata_path, "t1", schema, data_path=catalog.data_path,
        )

        base = _base_dir(catalog)
        tbl = pa.table({
            "id": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["alice", "bob"], type=pa.string()),
        })
        path = _write_arrow_parquet(os.path.join(base, "api_created.parquet"), tbl)

        add_files_ducklake(
            catalog.metadata_path, "t1", [path], data_path=catalog.data_path,
        )

        result = read_ducklake(catalog.metadata_path, "t1", data_path=catalog.data_path)
        assert len(result) == 2
