"""Advanced compaction tests for ducklake-ref test parity.

Covers gaps identified in TEST_PARITY.md:
- Compaction after alter table (add/drop/rename column)
- Compaction with partitioned tables
- Multiple compactions in sequence
- Rewrite after mixed DML operations
- Idempotent rewrite behavior
"""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import (
    alter_ducklake_set_partitioned_by,
    read_ducklake,
    write_ducklake,
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    delete_ducklake,
    update_ducklake,
    rewrite_data_files_ducklake,
)


class TestCompactionAfterAlterTable:
    """Verify rewrite_data_files works correctly after schema changes."""

    def test_rewrite_after_add_column(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df1, cat.metadata_path, "test", data_path=cat.data_path)

        alter_ducklake_add_column(cat.metadata_path, "test", "c", pl.Int64)

        df2 = pl.DataFrame({"a": [4, 5], "b": ["w", "v"], "c": [10, 20]})
        write_ducklake(df2, cat.metadata_path, "test", data_path=cat.data_path, mode="append")

        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 5
        sorted_result = result.sort("a")
        assert sorted_result["c"].to_list() == [None, None, None, 10, 20]

    def test_rewrite_after_drop_column(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        write_ducklake(df1, cat.metadata_path, "test", data_path=cat.data_path)

        alter_ducklake_drop_column(cat.metadata_path, "test", "c")

        df2 = pl.DataFrame({"a": [3, 4], "b": ["z", "w"]})
        write_ducklake(df2, cat.metadata_path, "test", data_path=cat.data_path, mode="append")

        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 4
        assert set(result.columns) == {"a", "b"}

    def test_rewrite_after_rename_column(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pl.DataFrame({"old_name": [1, 2, 3]})
        write_ducklake(df1, cat.metadata_path, "test", data_path=cat.data_path)

        alter_ducklake_rename_column(cat.metadata_path, "test", "old_name", "new_name")

        df2 = pl.DataFrame({"new_name": [4, 5]})
        write_ducklake(df2, cat.metadata_path, "test", data_path=cat.data_path, mode="append")

        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 5
        assert result.columns == ["new_name"]
        assert sorted(result["new_name"].to_list()) == [1, 2, 3, 4, 5]


class TestCompactionWithPartitions:
    """Verify rewrite_data_files with partitioned tables."""

    def test_rewrite_partitioned_table(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pl.DataFrame({"region": ["us", "us", "eu"], "value": [1, 2, 3]})
        write_ducklake(df1, cat.metadata_path, "test", data_path=cat.data_path)
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["region"])

        df2 = pl.DataFrame({"region": ["us", "eu"], "value": [4, 5]})
        write_ducklake(df2, cat.metadata_path, "test",
                       data_path=cat.data_path, mode="append")

        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 5
        us_rows = result.filter(pl.col("region") == "us")
        eu_rows = result.filter(pl.col("region") == "eu")
        assert us_rows.shape[0] == 3
        assert eu_rows.shape[0] == 2

    def test_rewrite_after_partition_delete(self, make_write_catalog):
        cat = make_write_catalog()
        df = pl.DataFrame({"region": ["us", "us", "eu", "eu"], "value": [1, 2, 3, 4]})
        write_ducklake(df, cat.metadata_path, "test", data_path=cat.data_path)
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["region"])

        delete_ducklake(cat.metadata_path, "test",
                        predicate=pl.col("region") == "us",
                        data_path=cat.data_path)

        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert set(result["region"].to_list()) == {"eu"}


class TestMultipleCompactions:
    """Multiple rewrite cycles in sequence."""

    def test_two_rewrites(self, make_write_catalog):
        cat = make_write_catalog()

        # First batch + rewrite
        df = pl.DataFrame({"a": list(range(15))})
        write_ducklake(df, cat.metadata_path, "test", data_path=cat.data_path)
        for i in range(2):
            df_extra = pl.DataFrame({"a": [100 + i * 5 + j for j in range(5)]})
            write_ducklake(df_extra, cat.metadata_path, "test",
                           data_path=cat.data_path, mode="append")
        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)

        # Second batch + rewrite
        for i in range(2):
            df_extra = pl.DataFrame({"a": [200 + i * 5 + j for j in range(5)]})
            write_ducklake(df_extra, cat.metadata_path, "test",
                           data_path=cat.data_path, mode="append")
        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 35

    def test_rewrite_after_delete_then_insert(self, make_write_catalog):
        cat = make_write_catalog()
        df = pl.DataFrame({"id": list(range(20)), "val": [f"v{i}" for i in range(20)]})
        write_ducklake(df, cat.metadata_path, "test", data_path=cat.data_path)

        delete_ducklake(cat.metadata_path, "test",
                        predicate=pl.col("id") >= 10,
                        data_path=cat.data_path)

        df2 = pl.DataFrame({"id": list(range(20, 30)), "val": [f"v{i}" for i in range(20, 30)]})
        write_ducklake(df2, cat.metadata_path, "test",
                       data_path=cat.data_path, mode="append")

        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 20
        ids = sorted(result["id"].to_list())
        assert ids == list(range(10)) + list(range(20, 30))


class TestCompactionWithMixedDML:
    """Rewrite after interleaved insert/update/delete operations."""

    def test_insert_update_delete_rewrite(self, make_write_catalog):
        cat = make_write_catalog()

        df = pl.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
        write_ducklake(df, cat.metadata_path, "test", data_path=cat.data_path)

        update_ducklake(cat.metadata_path, "test",
                        predicate=pl.col("id") <= 2,
                        updates={"name": pl.lit("updated")},
                        data_path=cat.data_path)

        delete_ducklake(cat.metadata_path, "test",
                        predicate=pl.col("id") == 5,
                        data_path=cat.data_path)

        df2 = pl.DataFrame({"id": [6, 7], "name": ["f", "g"]})
        write_ducklake(df2, cat.metadata_path, "test",
                       data_path=cat.data_path, mode="append")

        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 6
        sorted_result = result.sort("id")
        assert sorted_result["name"].to_list() == ["updated", "updated", "c", "d", "f", "g"]

    def test_overwrite_then_rewrite(self, make_write_catalog):
        cat = make_write_catalog()

        df1 = pl.DataFrame({"x": [1, 2, 3]})
        write_ducklake(df1, cat.metadata_path, "test", data_path=cat.data_path)

        df2 = pl.DataFrame({"x": [10, 20, 30, 40]})
        write_ducklake(df2, cat.metadata_path, "test",
                       data_path=cat.data_path, mode="overwrite")

        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert sorted(result["x"].to_list()) == [10, 20, 30, 40]


class TestCompactionIdempotent:
    """Rewriting already-compacted data should be a no-op or safe."""

    def test_rewrite_single_file_noop(self, make_write_catalog):
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", data_path=cat.data_path)

        result = rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result == -1  # No rewrite needed

    def test_double_rewrite_stable(self, make_write_catalog):
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", data_path=cat.data_path)
        for i in range(2, 6):
            df_extra = pl.DataFrame({"a": [i]})
            write_ducklake(df_extra, cat.metadata_path, "test",
                           data_path=cat.data_path, mode="append")

        rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        result_after_second = rewrite_data_files_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result_after_second == -1  # Already compacted

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert sorted(result["a"].to_list()) == list(range(1, 6))
