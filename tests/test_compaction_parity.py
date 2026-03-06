"""Compaction test parity — close the gap with ducklake-ref.

ducklake-ref has 27 compaction tests. Existing DF tests cover basic
rewrite, partitions, schema evolution, and idempotence. This module fills
the remaining gaps:

  - Cleanup after compaction (expire + vacuum removes old compacted files)
  - Compaction preserving statistics (column stats correct after rewrite)
  - Multi-compaction (compact -> insert -> compact again, complex sequences)
  - Author / commit_message on rewrite_data_files
  - Compaction after complex schema evolution (multiple ALTERs)
  - Compaction with many partitions (many small files across many partitions)
  - File count reduction verification
  - Compaction with NULL-heavy data
  - Compaction after overwrite mode
  - Compaction with empty result (all rows deleted)
"""

from __future__ import annotations

import os

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    alter_ducklake_set_partitioned_by,
    alter_ducklake_set_sort_keys,
    delete_ducklake,
    expire_snapshots,
    read_ducklake,
    rewrite_data_files_ducklake,
    update_ducklake,
    vacuum_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_parquet_files(data_path: str) -> int:
    """Count all .parquet files under a directory."""
    count = 0
    for dirpath, _dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith(".parquet"):
                count += 1
    return count


def _write_batches(
    cat, table: str, n_batches: int, rows_per_batch: int = 10
) -> pl.DataFrame:
    """Write *n_batches* small appends and return the combined expected data."""
    frames = []
    for i in range(n_batches):
        df = pl.DataFrame({
            "id": [i * rows_per_batch + j for j in range(rows_per_batch)],
            "val": [f"v{i * rows_per_batch + j}" for j in range(rows_per_batch)],
        })
        mode = "error" if i == 0 else "append"
        write_ducklake(df, cat.metadata_path, table, mode=mode,
                       data_path=cat.data_path)
        frames.append(df)
    return pl.concat(frames)


# ---------------------------------------------------------------------------
# 1. Cleanup after compaction — expire + vacuum removes old compacted files
# ---------------------------------------------------------------------------


class TestCleanupAfterCompaction:
    """Verify that expire + vacuum after rewrite removes orphaned files."""

    def test_vacuum_removes_old_files_after_compaction(self, make_write_catalog):
        """Write many files -> compact -> expire -> vacuum -> old files deleted."""
        cat = make_write_catalog()
        _write_batches(cat, "t", n_batches=5, rows_per_batch=10)

        files_before = _count_parquet_files(cat.data_path)
        assert files_before >= 5

        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap > 0

        # Files still on disk (old + new)
        files_after_compact = _count_parquet_files(cat.data_path)
        assert files_after_compact >= files_before  # new file added, old still present

        # Expire all but latest snapshot
        expired = expire_snapshots(cat.metadata_path, keep_last_n=1,
                                   data_path=cat.data_path)
        assert expired > 0

        # Vacuum removes unreferenced files
        vacuumed = vacuum_ducklake(cat.metadata_path, data_path=cat.data_path)
        assert vacuumed >= files_before  # all old files should be removed

        # Only the consolidated file(s) remain
        files_final = _count_parquet_files(cat.data_path)
        assert files_final == 1

        # Data intact
        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 50

    def test_vacuum_after_compaction_with_deletes(self, make_write_catalog):
        """Delete + compact -> expire -> vacuum removes data files AND delete files."""
        cat = make_write_catalog()
        _write_batches(cat, "t", n_batches=5, rows_per_batch=10)

        # Delete some rows (creates delete files)
        delete_ducklake(cat.metadata_path, "t", pl.col("id") < 10,
                        data_path=cat.data_path)

        files_before_compact = _count_parquet_files(cat.data_path)
        assert files_before_compact >= 6  # 5 data + at least 1 delete

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        expire_snapshots(cat.metadata_path, keep_last_n=1,
                         data_path=cat.data_path)
        vacuumed = vacuum_ducklake(cat.metadata_path, data_path=cat.data_path)
        assert vacuumed >= files_before_compact

        files_final = _count_parquet_files(cat.data_path)
        assert files_final == 1

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 40
        assert all(v >= 10 for v in result["id"].to_list())

    def test_vacuum_after_compaction_partitioned(self, make_write_catalog):
        """Partitioned table: vacuum after compaction removes old partition files."""
        cat = make_write_catalog()

        df_init = pl.DataFrame({"region": ["us"], "val": [0]})
        write_ducklake(df_init, cat.metadata_path, "t", data_path=cat.data_path)
        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "t", ["region"], data_path=cat.data_path,
        )

        for i in range(1, 7):
            region = "us" if i % 2 == 0 else "eu"
            df = pl.DataFrame({"region": [region], "val": [i]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                           data_path=cat.data_path)

        files_before = _count_parquet_files(cat.data_path)
        assert files_before >= 7

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        expire_snapshots(cat.metadata_path, keep_last_n=1,
                         data_path=cat.data_path)
        vacuumed = vacuum_ducklake(cat.metadata_path, data_path=cat.data_path)
        assert vacuumed >= 1

        files_final = _count_parquet_files(cat.data_path)
        assert files_final == 2  # one per partition (us, eu)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 7


# ---------------------------------------------------------------------------
# 2. Compaction preserving statistics
# ---------------------------------------------------------------------------


class TestCompactionPreservesStatistics:
    """Verify column stats (min/max, null count) are correct after compaction."""

    def test_integer_stats_correct_after_compaction(self, make_write_catalog):
        """Integer column stats reflect consolidated data."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"v": [10, 20]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({"v": [5, 30]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        df3 = pl.DataFrame({"v": [-100, 1000]})
        write_ducklake(df3, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        row = cat.query_one(
            "SELECT min_value, max_value FROM ducklake_table_column_stats "
            "WHERE table_id = (SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row is not None
        assert str(row[0]) == "-100"
        assert str(row[1]) == "1000"

    def test_stats_after_compaction_with_deletes(self, make_write_catalog):
        """Stats reflect data AFTER deleted rows are removed."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"v": [1, 100]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({"v": [50, 75]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        # Delete the row with max value
        delete_ducklake(cat.metadata_path, "t", pl.col("v") == 100,
                        data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        row = cat.query_one(
            "SELECT min_value, max_value FROM ducklake_table_column_stats "
            "WHERE table_id = (SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row is not None
        assert str(row[0]) == "1"
        assert str(row[1]) == "75"

    def test_null_stats_correct_after_compaction(self, make_write_catalog):
        """contains_null flag is correct after compaction."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"v": [1, 2]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({"v": [None, 4]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        row = cat.query_one(
            "SELECT contains_null FROM ducklake_table_column_stats "
            "WHERE table_id = (SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row is not None
        assert row[0] in (1, True)  # SQLite returns int, PG returns bool

    def test_record_count_correct_after_compaction(self, make_write_catalog):
        """Table-level record_count stat is correct after compaction."""
        cat = make_write_catalog()

        _write_batches(cat, "t", n_batches=5, rows_per_batch=10)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        row = cat.query_one(
            "SELECT record_count FROM ducklake_table_stats "
            "WHERE table_id = (SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row is not None
        assert row[0] == 50

    def test_file_size_updated_after_compaction(self, make_write_catalog):
        """file_size_bytes in ducklake_table_stats is updated after compaction."""
        cat = make_write_catalog()

        _write_batches(cat, "t", n_batches=3, rows_per_batch=10)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        row = cat.query_one(
            "SELECT file_size_bytes FROM ducklake_table_stats "
            "WHERE table_id = (SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row is not None
        assert row[0] > 0


# ---------------------------------------------------------------------------
# 3. Multi-compaction — compact -> insert -> compact again
# ---------------------------------------------------------------------------


class TestMultiCompaction:
    """Multiple rounds of compaction with interleaved operations."""

    def test_compact_insert_compact(self, make_write_catalog):
        """Compact -> insert more -> compact again -> data correct."""
        cat = make_write_catalog()

        # Phase 1: insert + compact
        for i in range(4):
            df = pl.DataFrame({"id": [i * 5 + j for j in range(5)]})
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        snap1 = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap1 > 0

        # Phase 2: insert more
        for i in range(3):
            df = pl.DataFrame({"id": [100 + i * 5 + j for j in range(5)]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                           data_path=cat.data_path)

        # Phase 3: compact again
        snap2 = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap2 > snap1

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 35
        expected = list(range(20)) + list(range(100, 115))
        assert sorted(result["id"].to_list()) == expected

    def test_compact_delete_compact(self, make_write_catalog):
        """Compact -> delete -> compact again -> deleted rows gone."""
        cat = make_write_catalog()

        for i in range(4):
            df = pl.DataFrame({"id": list(range(i * 10, (i + 1) * 10))})
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        # Delete after first compaction
        delete_ducklake(cat.metadata_path, "t", pl.col("id") < 10,
                        data_path=cat.data_path)

        # Insert new data
        df_new = pl.DataFrame({"id": list(range(100, 105))})
        write_ducklake(df_new, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        # Second compaction
        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 35  # 40 - 10 + 5
        ids = sorted(result["id"].to_list())
        assert ids == list(range(10, 40)) + list(range(100, 105))

    def test_compact_update_compact(self, make_write_catalog):
        """Compact -> update -> compact again -> updated values preserved."""
        cat = make_write_catalog()

        for i in range(3):
            df = pl.DataFrame({
                "id": list(range(i * 5, (i + 1) * 5)),
                "val": [f"original_{j}" for j in range(i * 5, (i + 1) * 5)],
            })
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        # Update after first compaction
        update_ducklake(cat.metadata_path, "t",
                        predicate=pl.col("id") < 5,
                        updates={"val": pl.lit("updated")},
                        data_path=cat.data_path)

        # Second compaction
        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 15
        updated = result.filter(pl.col("id") < 5).sort("id")
        assert all(v == "updated" for v in updated["val"].to_list())
        originals = result.filter(pl.col("id") >= 5).sort("id")
        assert all(v.startswith("original_") for v in originals["val"].to_list())

    def test_three_compaction_rounds(self, make_write_catalog):
        """Three successive compaction rounds produce correct data."""
        cat = make_write_catalog()

        # Round 1
        for i in range(3):
            df = pl.DataFrame({"x": [i * 10 + j for j in range(10)]})
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)
        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        # Round 2
        for i in range(2):
            df = pl.DataFrame({"x": [100 + i * 10 + j for j in range(10)]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                           data_path=cat.data_path)
        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        # Round 3
        df = pl.DataFrame({"x": list(range(200, 210))})
        write_ducklake(df, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)
        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 60
        expected = list(range(30)) + list(range(100, 120)) + list(range(200, 210))
        assert sorted(result["x"].to_list()) == expected


# ---------------------------------------------------------------------------
# 4. Author / commit_message on rewrite_data_files
# ---------------------------------------------------------------------------


class TestCompactionAuthorCommitMessage:
    """Rewrite records author and commit_message in snapshot_changes."""

    def test_rewrite_with_author(self, make_write_catalog):
        cat = make_write_catalog()
        _write_batches(cat, "t", n_batches=3, rows_per_batch=5)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
            author="compactor", commit_message="scheduled compaction",
        )

        row = cat.query_one(
            "SELECT author, commit_message FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row[0] == "compactor"
        assert row[1] == "scheduled compaction"

    def test_rewrite_without_author(self, make_write_catalog):
        cat = make_write_catalog()
        _write_batches(cat, "t", n_batches=3, rows_per_batch=5)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        row = cat.query_one(
            "SELECT author, commit_message FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row[0] is None
        assert row[1] is None


# ---------------------------------------------------------------------------
# 5. Compaction after complex schema evolution (multiple ALTERs)
# ---------------------------------------------------------------------------


class TestCompactionAfterComplexSchemaEvolution:
    """Rewrite after multiple schema changes in sequence."""

    def test_add_rename_then_compact(self, make_write_catalog):
        """Add column -> rename column -> insert -> compact."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        alter_ducklake_add_column(cat.metadata_path, "t", "b", pl.Int64)
        alter_ducklake_rename_column(cat.metadata_path, "t", "a", "id")

        df2 = pl.DataFrame({"id": [4, 5], "b": [10, 20]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 5
        assert set(result.columns) == {"id", "b"}
        sorted_result = result.sort("id")
        assert sorted_result["b"].to_list() == [None, None, None, 10, 20]

    def test_add_drop_add_then_compact(self, make_write_catalog):
        """Add column -> drop it -> add different column -> compact."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"id": [1, 2]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        alter_ducklake_add_column(cat.metadata_path, "t", "temp", pl.String)

        df2 = pl.DataFrame({"id": [3], "temp": ["x"]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        alter_ducklake_drop_column(cat.metadata_path, "t", "temp")
        alter_ducklake_add_column(cat.metadata_path, "t", "final_col", pl.Float64)

        df3 = pl.DataFrame({"id": [4], "final_col": [3.14]})
        write_ducklake(df3, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 4
        assert set(result.columns) == {"id", "final_col"}
        assert "temp" not in result.columns

    def test_multiple_renames_then_compact(self, make_write_catalog):
        """Rename column multiple times -> compact reflects final name."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"name_v1": [1, 2, 3]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        alter_ducklake_rename_column(cat.metadata_path, "t", "name_v1", "name_v2")

        df2 = pl.DataFrame({"name_v2": [4, 5]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        alter_ducklake_rename_column(cat.metadata_path, "t", "name_v2", "name_v3")

        df3 = pl.DataFrame({"name_v3": [6]})
        write_ducklake(df3, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.columns == ["name_v3"]
        assert sorted(result["name_v3"].to_list()) == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# 6. Compaction with many partitions
# ---------------------------------------------------------------------------


class TestCompactionManyPartitions:
    """Compaction with many small files across many partitions."""

    def test_many_partitions_compact(self, make_write_catalog):
        """10 partitions x 3 inserts each -> compact -> one file per partition."""
        cat = make_write_catalog()

        regions = [f"region_{i}" for i in range(10)]
        df_init = pl.DataFrame({"region": [regions[0]], "val": [0]})
        write_ducklake(df_init, cat.metadata_path, "t", data_path=cat.data_path)
        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "t", ["region"], data_path=cat.data_path,
        )

        counter = 1
        for batch in range(3):
            for region in regions:
                if batch == 0 and region == regions[0]:
                    continue  # already wrote init
                df = pl.DataFrame({"region": [region], "val": [counter]})
                write_ducklake(df, cat.metadata_path, "t", mode="append",
                               data_path=cat.data_path)
                counter += 1

        data_before = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        data_after = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(data_after) == len(data_before)
        assert_frame_equal(
            data_before.sort("region", "val"),
            data_after.sort("region", "val"),
        )

    def test_partition_with_empty_partition_after_delete(self, make_write_catalog):
        """Delete all rows from one partition -> compact -> partition gone."""
        cat = make_write_catalog()

        df = pl.DataFrame({
            "region": ["us", "us", "eu", "eu", "ap"],
            "val": [1, 2, 3, 4, 5],
        })
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)
        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "t", ["region"], data_path=cat.data_path,
        )

        # Append more to create multiple files
        df2 = pl.DataFrame({"region": ["us", "eu"], "val": [6, 7]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        # Delete all AP rows
        delete_ducklake(cat.metadata_path, "t", pl.col("region") == "ap",
                        data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "ap" not in result["region"].to_list()
        assert set(result["region"].to_list()) == {"us", "eu"}
        assert len(result) == 6


# ---------------------------------------------------------------------------
# 7. File count reduction verification
# ---------------------------------------------------------------------------


class TestFileCountReduction:
    """Verify actual parquet file count is reduced after compaction."""

    def test_many_files_to_one(self, make_write_catalog):
        """10 small appends -> compact + expire + vacuum -> 1 file."""
        cat = make_write_catalog()

        for i in range(10):
            df = pl.DataFrame({"x": [i]})
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        expire_snapshots(cat.metadata_path, keep_last_n=1,
                         data_path=cat.data_path)
        vacuum_ducklake(cat.metadata_path, data_path=cat.data_path)

        files_after = _count_parquet_files(cat.data_path)
        assert files_after == 1

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert sorted(result["x"].to_list()) == list(range(10))

    def test_data_file_count_in_metadata(self, make_write_catalog):
        """After compaction, only 1 active data file entry in metadata."""
        cat = make_write_catalog()
        _write_batches(cat, "t", n_batches=5, rows_per_batch=5)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        # Count active (end_snapshot IS NULL) data files
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file "
            "WHERE end_snapshot IS NULL AND table_id = "
            "(SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row[0] == 1


# ---------------------------------------------------------------------------
# 8. Compaction with NULL-heavy data
# ---------------------------------------------------------------------------


class TestCompactionWithNulls:
    """Compaction preserves NULL values correctly."""

    def test_all_null_column_survives_compaction(self, make_write_catalog):
        """Column with only NULLs survives compaction."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"id": [1, 2], "opt": pl.Series([None, None], dtype=pl.String)})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({"id": [3, 4], "opt": pl.Series([None, None], dtype=pl.String)})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 4
        assert result["opt"].null_count() == 4

    def test_mixed_null_non_null_compact(self, make_write_catalog):
        """Mix of NULL and non-NULL values preserved across compaction."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"id": [1, 2, 3], "val": [None, "b", None]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({"id": [4, 5, 6], "val": ["d", None, "f"]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        df3 = pl.DataFrame({"id": [7, 8], "val": [None, None]})
        write_ducklake(df3, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        result_before = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result_after = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert_frame_equal(result_before.sort("id"), result_after.sort("id"))


# ---------------------------------------------------------------------------
# 9. Compaction after overwrite mode
# ---------------------------------------------------------------------------


class TestCompactionAfterOverwrite:
    """Compaction after overwrite mode creates clean consolidated state."""

    def test_overwrite_then_append_then_compact(self, make_write_catalog):
        """Overwrite -> append -> append -> compact -> single clean file."""
        cat = make_write_catalog()

        # Initial write
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        # Overwrite
        df2 = pl.DataFrame({"a": [10, 20]})
        write_ducklake(df2, cat.metadata_path, "t", mode="overwrite",
                       data_path=cat.data_path)

        # Append more
        df3 = pl.DataFrame({"a": [30]})
        write_ducklake(df3, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        df4 = pl.DataFrame({"a": [40]})
        write_ducklake(df4, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert sorted(result["a"].to_list()) == [10, 20, 30, 40]

    def test_multiple_overwrites_then_compact(self, make_write_catalog):
        """Multiple overwrites -> compact -> only latest data."""
        cat = make_write_catalog()

        for val in [[1, 2], [10, 20], [100, 200]]:
            df = pl.DataFrame({"a": val})
            mode = "error" if val == [1, 2] else "overwrite"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        # Append after final overwrite
        df_extra = pl.DataFrame({"a": [300]})
        write_ducklake(df_extra, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert sorted(result["a"].to_list()) == [100, 200, 300]


# ---------------------------------------------------------------------------
# 10. Compaction with empty result (all rows deleted)
# ---------------------------------------------------------------------------


class TestCompactionAllRowsDeleted:
    """Compaction after all rows are deleted."""

    def test_compact_after_full_delete(self, make_write_catalog):
        """Delete all rows -> compact -> empty table."""
        cat = make_write_catalog()
        _write_batches(cat, "t", n_batches=3, rows_per_batch=5)

        # Delete all rows
        delete_ducklake(cat.metadata_path, "t", pl.col("id") >= 0,
                        data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 0
        assert set(result.columns) == {"id", "val"}

    def test_compact_empty_then_insert(self, make_write_catalog):
        """Delete all -> compact -> insert new data -> data correct."""
        cat = make_write_catalog()
        _write_batches(cat, "t", n_batches=3, rows_per_batch=5)

        delete_ducklake(cat.metadata_path, "t", pl.col("id") >= 0,
                        data_path=cat.data_path)
        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        # Insert fresh data after compaction of empty table
        df_new = pl.DataFrame({"id": [100, 200], "val": ["fresh1", "fresh2"]})
        write_ducklake(df_new, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 2
        assert sorted(result["id"].to_list()) == [100, 200]


# ---------------------------------------------------------------------------
# 11. Compaction with sort keys
# ---------------------------------------------------------------------------


class TestCompactionWithSortKeys:
    """Compaction respects sort key ordering."""

    def test_compact_preserves_sort_order(self, make_write_catalog):
        """Sorted table -> compact -> output sorted."""
        cat = make_write_catalog()

        # Write data in random order across multiple batches
        for vals in [[30, 10, 20], [5, 25, 15], [35, 1, 50]]:
            df = pl.DataFrame({"id": vals, "name": [f"n{v}" for v in vals]})
            mode = "error" if vals[0] == 30 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("id", "ASC")],
            data_path=cat.data_path,
        )

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        ids = result["id"].to_list()
        assert ids == sorted(ids)

    def test_compact_descending_sort(self, make_write_catalog):
        """Descending sort key preserved after compaction."""
        cat = make_write_catalog()

        for i in range(3):
            df = pl.DataFrame({"score": [i * 10 + j for j in range(5)]})
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("score", "DESC")],
            data_path=cat.data_path,
        )

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        scores = result["score"].to_list()
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 12. Compaction with DuckDB-created data
# ---------------------------------------------------------------------------


class TestCompactionWithDuckDBData:
    """Rewrite files originally created by DuckDB."""

    def test_duckdb_inserts_polars_compacts(self, ducklake_catalog):
        """DuckDB creates many small inserts -> Polars compacts."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, name VARCHAR)")
        for i in range(8):
            cat.execute(
                f"INSERT INTO ducklake.t VALUES ({i}, 'row_{i}')"
            )
        cat.close()

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 8
        assert sorted(result["id"].to_list()) == list(range(8))

    def test_duckdb_schema_evolution_polars_compacts(self, ducklake_catalog):
        """DuckDB adds column + inserts -> Polars compacts."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN extra VARCHAR")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'hello'), (4, 'world')")
        cat.close()

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 4
        old = result.filter(pl.col("id") <= 2).sort("id")
        assert old["extra"].null_count() == 2
        new = result.filter(pl.col("id") > 2).sort("id")
        assert new["extra"].to_list() == ["hello", "world"]

    def test_duckdb_deletes_polars_compacts(self, ducklake_catalog):
        """DuckDB creates + deletes -> Polars compacts -> dead rows removed."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.t AS "
            "SELECT i AS id FROM range(100) t(i)"
        )
        cat.execute("DELETE FROM ducklake.t WHERE id % 2 = 0")
        cat.execute("INSERT INTO ducklake.t VALUES (200), (201)")
        cat.close()

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 52  # 50 odd + 2 new
        ids = sorted(result["id"].to_list())
        assert ids == list(range(1, 100, 2)) + [200, 201]
