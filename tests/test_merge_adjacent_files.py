"""Tests for merge_adjacent_files (partial_max compaction)."""

from __future__ import annotations

import os
import sqlite3

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from ducklake_polars import (
    merge_adjacent_files_ducklake,
    read_ducklake,
    write_ducklake,
)


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


# ------------------------------------------------------------------
# Basic merge behaviour
# ------------------------------------------------------------------


class TestMergeAdjacentFiles:
    def test_no_op_with_single_file(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, path, "t", mode="error", data_path=data_path)

        new_snap = merge_adjacent_files_ducklake(
            path, "t", data_path=data_path,
        )
        assert new_snap == -1

    def test_two_files_merged_into_one(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})
        write_ducklake(df1, path, "t", mode="error", data_path=data_path)
        write_ducklake(df2, path, "t", mode="append", data_path=data_path)

        # Two files exist
        con = sqlite3.connect(path)
        try:
            n_files_before = con.execute(
                "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
            ).fetchone()[0]
        finally:
            con.close()
        assert n_files_before == 2

        new_snap = merge_adjacent_files_ducklake(path, "t", data_path=data_path)
        assert new_snap > 0

        con = sqlite3.connect(path)
        try:
            active = con.execute(
                "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
            ).fetchone()[0]
            partial_rows = con.execute(
                "SELECT data_file_id, partial_max, begin_snapshot "
                "FROM ducklake_data_file WHERE end_snapshot IS NULL"
            ).fetchall()
        finally:
            con.close()

        assert active == 1
        # The merged file has partial_max set to the highest source begin_snapshot
        # and begin_snapshot set to the lowest.
        merged = partial_rows[0]
        assert merged[1] is not None  # partial_max
        assert merged[2] >= 0  # begin_snapshot

    def test_merged_data_readable_at_latest(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})
        write_ducklake(df1, path, "t", mode="error", data_path=data_path)
        write_ducklake(df2, path, "t", mode="append", data_path=data_path)

        merge_adjacent_files_ducklake(path, "t", data_path=data_path)

        result = read_ducklake(path, "t")
        assert sorted(result["a"].to_list()) == [1, 2, 3, 4]

    def test_time_travel_filters_by_partial_max(self, tmp_path):
        """Reading at an older snapshot must return only that snapshot's rows."""
        from ducklake_polars import scan_ducklake

        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})
        write_ducklake(df1, path, "t", mode="error", data_path=data_path)

        # Capture snapshot 1 (after first insert + post-create)
        con = sqlite3.connect(path)
        try:
            snap_after_first = con.execute(
                "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
            ).fetchone()[0]
        finally:
            con.close()

        write_ducklake(df2, path, "t", mode="append", data_path=data_path)
        merge_adjacent_files_ducklake(path, "t", data_path=data_path)

        # Reading at the snapshot taken before the second insert should only
        # return df1's rows, even though they live inside the merged file.
        result_old = scan_ducklake(
            path, "t", snapshot_version=snap_after_first,
        ).collect()
        assert sorted(result_old["a"].to_list()) == [1, 2]

        # Latest read returns everything.
        result_latest = read_ducklake(path, "t")
        assert sorted(result_latest["a"].to_list()) == [1, 2, 3, 4]

    def test_files_with_active_deletes_skipped(self, tmp_path):
        """Files with active deletes are not merged; rewrite first."""
        from ducklake_polars._writer import DuckLakeCatalogWriter

        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [4, 5, 6]})
        write_ducklake(df1, path, "t", mode="error", data_path=data_path)
        write_ducklake(df2, path, "t", mode="append", data_path=data_path)

        # Delete a row from the first file
        with DuckLakeCatalogWriter(path, data_path_override=data_path) as w:
            w.delete_data(pl.col("a") == 2, "t")

        # Merge should skip the file with deletes (only one merge candidate left)
        new_snap = merge_adjacent_files_ducklake(path, "t", data_path=data_path)
        assert new_snap == -1

    def test_min_file_size_excludes_files(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})
        write_ducklake(df1, path, "t", mode="error", data_path=data_path)
        write_ducklake(df2, path, "t", mode="append", data_path=data_path)

        # Both files are tiny; min_file_size=10MB excludes both → no merge
        new_snap = merge_adjacent_files_ducklake(
            path, "t", data_path=data_path,
            min_file_size=10 * 1024 * 1024,
        )
        assert new_snap == -1


# ------------------------------------------------------------------
# DuckDB roundtrip
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_reads_partial_data_file(tmp_path):
    """DuckDB sees the merged file as a partial data file and time-travels correctly."""
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")

    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [3, 4]})
    write_ducklake(df1, path, "t", mode="error", data_path=data_path)
    write_ducklake(df2, path, "t", mode="append", data_path=data_path)
    merge_adjacent_files_ducklake(path, "t", data_path=data_path)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{path}' AS d (DATA_PATH '{data_path}')"
    )
    rows = con.execute("SELECT a FROM d.t ORDER BY a").fetchall()
    con.close()
    assert [r[0] for r in rows] == [1, 2, 3, 4]
