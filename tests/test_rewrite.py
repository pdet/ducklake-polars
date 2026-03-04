"""Tests for rewrite_data_files — compaction / maintenance operation."""

from __future__ import annotations

import os

import polars as pl
import pytest

from ducklake_polars import (
    delete_ducklake,
    read_ducklake,
    rewrite_data_files_ducklake,
    write_ducklake,
)


def _count_parquet_files(data_path: str) -> int:
    """Count all .parquet files under a directory (recursive)."""
    count = 0
    for dirpath, _dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith(".parquet"):
                count += 1
    return count


class TestInsertDeleteRewrite:
    """Insert, delete some rows, rewrite → same data, fewer files."""

    def test_delete_then_rewrite(self, make_write_catalog):
        cat = make_write_catalog()

        # Insert several batches → multiple data files
        for i in range(5):
            df = pl.DataFrame({"id": [i * 10 + j for j in range(10)]})
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        files_before = _count_parquet_files(cat.data_path)
        assert files_before >= 5

        # Delete some rows
        delete_ducklake(cat.metadata_path, "t", pl.col("id") >= 40,
                        data_path=cat.data_path)

        data_before = read_ducklake(cat.metadata_path, "t",
                                    data_path=cat.data_path)
        assert sorted(data_before["id"].to_list()) == list(range(40))

        # Rewrite
        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap > 0

        data_after = read_ducklake(cat.metadata_path, "t",
                                   data_path=cat.data_path)
        assert sorted(data_after["id"].to_list()) == list(range(40))


class TestMultipleSmallInserts:
    """Multiple small inserts, rewrite → consolidated."""

    def test_consolidate_small_files(self, make_write_catalog):
        cat = make_write_catalog()

        df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df1, cat.metadata_path, "small", mode="error",
                       data_path=cat.data_path)

        for i in range(3, 11, 2):
            df = pl.DataFrame({"a": [i, i + 1], "b": [f"v{i}", f"v{i+1}"]})
            write_ducklake(df, cat.metadata_path, "small", mode="append",
                           data_path=cat.data_path)

        files_before = _count_parquet_files(cat.data_path)
        assert files_before >= 5

        data_before = read_ducklake(cat.metadata_path, "small",
                                    data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "small", data_path=cat.data_path,
        )

        data_after = read_ducklake(cat.metadata_path, "small",
                                   data_path=cat.data_path)

        assert sorted(data_before["a"].to_list()) == sorted(data_after["a"].to_list())
        assert sorted(data_before["b"].to_list()) == sorted(data_after["b"].to_list())

    def test_single_file_noop(self, make_write_catalog):
        """One file, no deletes → returns -1 (no rewrite needed)."""
        cat = make_write_catalog()

        df = pl.DataFrame({"x": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "one", mode="error",
                       data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "one", data_path=cat.data_path,
        )
        assert snap == -1


class TestRewritePreservesColumnStats:
    """Rewrite preserves column statistics (min/max, null info)."""

    def test_stats_preserved(self, make_write_catalog):
        cat = make_write_catalog()

        df1 = pl.DataFrame({"v": [10, 20]})
        write_ducklake(df1, cat.metadata_path, "stats", mode="error",
                       data_path=cat.data_path)

        df2 = pl.DataFrame({"v": [5, 30]})
        write_ducklake(df2, cat.metadata_path, "stats", mode="append",
                       data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "stats", data_path=cat.data_path,
        )

        # Read back column stats from catalog
        row = cat.query_one(
            "SELECT min_value, max_value FROM ducklake_table_column_stats "
            "WHERE table_id = (SELECT table_id FROM ducklake_table "
            "WHERE table_name = 'stats' AND end_snapshot IS NULL)"
        )
        assert row is not None
        assert str(row[0]) == "5"
        assert str(row[1]) == "30"

    def test_preserves_multiple_types(self, make_write_catalog):
        cat = make_write_catalog()

        df1 = pl.DataFrame({
            "i": [1, 2],
            "f": [1.5, 2.5],
            "s": ["alpha", "beta"],
        })
        write_ducklake(df1, cat.metadata_path, "types", mode="error",
                       data_path=cat.data_path)

        df2 = pl.DataFrame({
            "i": [3, 4],
            "f": [0.5, 4.5],
            "s": ["gamma", "delta"],
        })
        write_ducklake(df2, cat.metadata_path, "types", mode="append",
                       data_path=cat.data_path)

        data_before = read_ducklake(cat.metadata_path, "types",
                                    data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "types", data_path=cat.data_path,
        )

        data_after = read_ducklake(cat.metadata_path, "types",
                                   data_path=cat.data_path)

        assert sorted(data_before["i"].to_list()) == sorted(data_after["i"].to_list())
        assert sorted(data_before["f"].to_list()) == sorted(data_after["f"].to_list())
        assert sorted(data_before["s"].to_list()) == sorted(data_after["s"].to_list())
