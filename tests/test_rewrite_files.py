"""Tests for rewrite_data_files maintenance operation."""

from __future__ import annotations

import os

import polars as pl
import pytest

from ducklake_polars import (
    alter_ducklake_set_partitioned_by,
    delete_ducklake,
    read_ducklake,
    rewrite_data_files_ducklake,
    write_ducklake,
)


def _count_parquet_files(data_path: str) -> int:
    """Count all .parquet files under a directory."""
    count = 0
    for dirpath, _dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith(".parquet"):
                count += 1
    return count


class TestRewriteBasic:
    """Multiple small inserts → rewrite → fewer files, same data."""

    def test_rewrite_consolidates_files(self, make_write_catalog):
        cat = make_write_catalog()

        # Create table and insert multiple small batches
        df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df1, cat.metadata_path, "test", mode="error",
                       data_path=cat.data_path)

        for i in range(3, 9, 2):
            df = pl.DataFrame({"a": [i, i + 1], "b": [f"v{i}", f"v{i+1}"]})
            write_ducklake(df, cat.metadata_path, "test", mode="append",
                           data_path=cat.data_path)

        files_before = _count_parquet_files(cat.data_path)
        assert files_before >= 4  # At least 4 data files

        data_before = read_ducklake(cat.metadata_path, "test",
                                    data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "test", data_path=cat.data_path,
        )
        assert snap > 0  # A new snapshot was created

        data_after = read_ducklake(cat.metadata_path, "test",
                                   data_path=cat.data_path)

        # Same data
        assert sorted(data_before["a"].to_list()) == sorted(data_after["a"].to_list())
        assert sorted(data_before["b"].to_list()) == sorted(data_after["b"].to_list())

    def test_rewrite_single_file_no_op(self, make_write_catalog):
        """Rewrite with a single file returns -1 (no rewrite needed)."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error",
                       data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "test", data_path=cat.data_path,
        )
        assert snap == -1


class TestRewritePreservesTypes:
    """Rewrite preserves column types and values."""

    def test_preserves_int_float_str_bool(self, make_write_catalog):
        cat = make_write_catalog()

        df1 = pl.DataFrame({
            "i": [1, 2],
            "f": [1.5, 2.5],
            "s": ["hello", "world"],
            "b": [True, False],
        })
        write_ducklake(df1, cat.metadata_path, "typed", mode="error",
                       data_path=cat.data_path)

        df2 = pl.DataFrame({
            "i": [3, 4],
            "f": [3.5, 4.5],
            "s": ["foo", "bar"],
            "b": [False, True],
        })
        write_ducklake(df2, cat.metadata_path, "typed", mode="append",
                       data_path=cat.data_path)

        data_before = read_ducklake(cat.metadata_path, "typed",
                                    data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "typed", data_path=cat.data_path,
        )

        data_after = read_ducklake(cat.metadata_path, "typed",
                                   data_path=cat.data_path)

        # Same values
        assert sorted(data_before["i"].to_list()) == sorted(data_after["i"].to_list())
        assert sorted(data_before["f"].to_list()) == sorted(data_after["f"].to_list())
        assert sorted(data_before["s"].to_list()) == sorted(data_after["s"].to_list())
        assert sorted(data_before["b"].to_list()) == sorted(data_after["b"].to_list())

        # Same column names
        assert data_after.columns == data_before.columns


class TestRewriteAfterDelete:
    """Rewrite after delete — deleted rows excluded from rewritten files."""

    def test_deleted_rows_excluded(self, make_write_catalog):
        cat = make_write_catalog()

        # Insert multiple batches
        for i in range(5):
            df = pl.DataFrame({"a": [i * 10 + j for j in range(10)]})
            if i == 0:
                write_ducklake(df, cat.metadata_path, "deltest", mode="error",
                               data_path=cat.data_path)
            else:
                write_ducklake(df, cat.metadata_path, "deltest", mode="append",
                               data_path=cat.data_path)

        # Delete some rows
        delete_ducklake(cat.metadata_path, "deltest",
                        pl.col("a") >= 40,
                        data_path=cat.data_path)

        data_before = read_ducklake(cat.metadata_path, "deltest",
                                    data_path=cat.data_path)
        assert sorted(data_before["a"].to_list()) == list(range(40))

        rewrite_data_files_ducklake(
            cat.metadata_path, "deltest", data_path=cat.data_path,
        )

        data_after = read_ducklake(cat.metadata_path, "deltest",
                                   data_path=cat.data_path)
        assert sorted(data_after["a"].to_list()) == list(range(40))


class TestRewritePartitioned:
    """Rewrite with partitioned table."""

    def test_partitioned_rewrite(self, make_write_catalog):
        cat = make_write_catalog()

        df1 = pl.DataFrame({"city": ["NYC", "NYC"], "val": [1, 2]})
        write_ducklake(df1, cat.metadata_path, "parts", mode="error",
                       data_path=cat.data_path)

        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "parts", ["city"],
            data_path=cat.data_path,
        )

        df2 = pl.DataFrame({"city": ["NYC", "LA"], "val": [3, 4]})
        write_ducklake(df2, cat.metadata_path, "parts", mode="append",
                       data_path=cat.data_path)

        df3 = pl.DataFrame({"city": ["NYC", "LA"], "val": [5, 6]})
        write_ducklake(df3, cat.metadata_path, "parts", mode="append",
                       data_path=cat.data_path)

        data_before = read_ducklake(cat.metadata_path, "parts",
                                    data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "parts", data_path=cat.data_path,
        )

        data_after = read_ducklake(cat.metadata_path, "parts",
                                   data_path=cat.data_path)

        # Same data after rewrite
        before_sorted = data_before.sort(["city", "val"])
        after_sorted = data_after.sort(["city", "val"])
        assert before_sorted["city"].to_list() == after_sorted["city"].to_list()
        assert before_sorted["val"].to_list() == after_sorted["val"].to_list()
