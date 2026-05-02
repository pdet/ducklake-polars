"""Rewrite data files (compaction) test parity with ducklake-ref.

ducklake-ref has 10 dedicated rewrite_data_files tests covering:
  - Threshold-based rewriting (only rewrite when multiple files or deletes exist)
  - Concurrency during rewrite
  - Partitioned table rewriting
  - Merge-adjacent behavior
  - Row ID preservation

This module closes the gap between ducklake-dataframe and ducklake-ref by
exercising each of those scenarios plus additional edge cases:

  1. Threshold-based: skip rewrite when single file with no deletes
  2. Concurrent rewrite: two threads rewriting simultaneously
  3. Partitioned rewrite: partition structure preserved after rewrite
  4. Row ID preservation: row IDs stable after rewrite
  5. Rewrite after mixed operations: insert + delete + update -> rewrite
  6. Rewrite idempotency: rewrite twice produces same result
  7. Rewrite with delete files: deletes applied during rewrite
  8. File count reduction: actual file count drops
  9. Data integrity: all rows survive rewrite unchanged
 10. Stats preservation: min/max stats correct after rewrite
 11. Merge-adjacent: adjacent small files merged into larger ones
 12. Sort key preservation: sorted tables stay sorted after rewrite
"""

from __future__ import annotations

import os
import threading

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
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
    """Count all .parquet files under a directory tree."""
    count = 0
    for dirpath, _dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith(".parquet"):
                count += 1
    return count


def _write_many_small(cat, table: str, n_batches: int,
                      rows_per_batch: int = 5) -> pl.DataFrame:
    """Write *n_batches* small appends and return the combined expected frame."""
    frames: list[pl.DataFrame] = []
    for i in range(n_batches):
        df = pl.DataFrame({
            "id": list(range(i * rows_per_batch, (i + 1) * rows_per_batch)),
            "val": [f"v{j}" for j in range(i * rows_per_batch,
                                            (i + 1) * rows_per_batch)],
        })
        mode = "error" if i == 0 else "append"
        write_ducklake(df, cat.metadata_path, table, mode=mode,
                       data_path=cat.data_path)
        frames.append(df)
    return pl.concat(frames)


# ===================================================================
# 1. Threshold-based: skip rewrite when nothing to do
# ===================================================================


class TestThresholdBasedRewrite:
    """rewrite_data_files should return -1 when no compaction is needed."""

    def test_single_file_no_deletes_returns_negative_one(self, make_write_catalog):
        """A table with exactly one data file and no deletes -> no rewrite."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        result = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert result == -1

    def test_two_files_triggers_rewrite(self, make_write_catalog):
        """Two data files -> rewrite is performed (returns positive snap)."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)
        df2 = pl.DataFrame({"a": [3, 4]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap > 0

    def test_single_file_with_deletes_triggers_rewrite(self, make_write_catalog):
        """One data file but with a deletion vector -> rewrite is performed."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        delete_ducklake(cat.metadata_path, "t", pl.col("a") <= 2,
                        data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert sorted(result["a"].to_list()) == [3, 4, 5]

    def test_empty_table_returns_negative_one(self, make_write_catalog):
        """Table with zero rows (never inserted) -> no rewrite needed."""
        import duckdb

        cat = make_write_catalog()

        src = cat.attach_source()
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{src}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.t (a INTEGER)")
        con.close()

        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap == -1


# ===================================================================
# 2. Concurrent rewrite: two writers rewriting simultaneously
# ===================================================================


class TestConcurrentRewrite:
    """Concurrent rewrite_data_files on the same table.

    SQLite-only because Postgres deadlocks with concurrent DuckLake writes
    at the Python level (the DuckDB C++ extension handles PG concurrency).
    """

    def test_concurrent_rewrite_both_succeed_or_one_retries(
        self, ducklake_catalog_sqlite
    ):
        """Two threads rewriting the same table -- both succeed via retry."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        for i in range(10):
            cat.execute(f"INSERT INTO ducklake.t VALUES ({i}, 'v{i}')")
        cat.close()

        errors: list[Exception] = []
        results: list[int] = []

        def rewrite():
            try:
                snap = rewrite_data_files_ducklake(
                    cat.metadata_path, "t", data_path=cat.data_path,
                    max_retries=5, retry_wait_ms=50,
                )
                results.append(snap)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=rewrite)
        t2 = threading.Thread(target=rewrite)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        # At least one should succeed; the other either also succeeds or
        # returns -1 (nothing left to rewrite)
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert any(r > 0 or r == -1 for r in results)

        # Data integrity preserved
        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 10
        assert sorted(result["id"].to_list()) == list(range(10))

    def test_concurrent_rewrite_and_insert(self, ducklake_catalog_sqlite):
        """One thread rewrites while another inserts -- both succeed."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        for i in range(8):
            cat.execute(f"INSERT INTO ducklake.t VALUES ({i})")
        cat.close()

        errors: list[Exception] = []

        def rewrite():
            try:
                rewrite_data_files_ducklake(
                    cat.metadata_path, "t", data_path=cat.data_path,
                    max_retries=5, retry_wait_ms=50,
                )
            except Exception as e:
                errors.append(e)

        def insert():
            try:
                df = pl.DataFrame({"id": pl.Series(list(range(100, 110)),
                                                   dtype=pl.Int32)})
                write_ducklake(
                    df, cat.metadata_path, "t", mode="append",
                    data_path=cat.data_path,
                    max_retries=5, retry_wait_ms=50,
                )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=rewrite)
        t2 = threading.Thread(target=insert)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert len(errors) == 0, f"Unexpected errors: {errors}"

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        # Original 8 rows + 10 inserted
        assert len(result) == 18


# ===================================================================
# 3. Partitioned rewrite: partition structure preserved
# ===================================================================


class TestPartitionedRewrite:
    """Rewriting partitioned tables preserves partition structure."""

    def test_partition_files_grouped_after_rewrite(self, make_write_catalog):
        """Multiple inserts across partitions -> rewrite -> one file per partition."""
        cat = make_write_catalog()

        df_init = pl.DataFrame({"region": ["us"], "val": [1]})
        write_ducklake(df_init, cat.metadata_path, "t", data_path=cat.data_path)
        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "t", ["region"], data_path=cat.data_path,
        )

        # Multiple inserts per partition
        for region, val in [("us", 2), ("eu", 10), ("eu", 11), ("us", 3),
                            ("ap", 20), ("ap", 21), ("eu", 12)]:
            df = pl.DataFrame({"region": [region], "val": [val]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                           data_path=cat.data_path)

        data_before = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap > 0

        data_after = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert_frame_equal(
            data_before.sort("region", "val"),
            data_after.sort("region", "val"),
        )

        # After expire+vacuum only partition files remain
        expire_snapshots(cat.metadata_path, keep_last_n=1,
                         data_path=cat.data_path)
        vacuum_ducklake(cat.metadata_path, data_path=cat.data_path)

        # Active data files in metadata = number of distinct partitions
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file "
            "WHERE end_snapshot IS NULL AND table_id = "
            "(SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row[0] == 3  # us, eu, ap

    def test_partitioned_rewrite_preserves_filter_pushdown(self, make_write_catalog):
        """After rewrite, partition filter still works correctly."""
        cat = make_write_catalog()

        df = pl.DataFrame({
            "region": ["us", "eu", "ap", "us", "eu"],
            "val": [1, 2, 3, 4, 5],
        })
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)
        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "t", ["region"], data_path=cat.data_path,
        )

        # Append more to create multiple files per partition
        df2 = pl.DataFrame({"region": ["us", "eu"], "val": [10, 20]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        # Read and filter
        result = read_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        ).filter(pl.col("region") == "us")
        assert all(r == "us" for r in result["region"].to_list())
        assert sorted(result["val"].to_list()) == [1, 4, 10]

    def test_partitioned_rewrite_with_deletes(self, make_write_catalog):
        """Delete rows from one partition -> rewrite -> partition shrinks."""
        cat = make_write_catalog()

        df = pl.DataFrame({
            "region": ["us", "us", "us", "eu", "eu"],
            "val": [1, 2, 3, 10, 20],
        })
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)
        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "t", ["region"], data_path=cat.data_path,
        )

        # Append to create multi-file scenario
        df2 = pl.DataFrame({"region": ["us", "eu"], "val": [4, 30]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        # Delete from US partition
        delete_ducklake(cat.metadata_path, "t",
                        (pl.col("region") == "us") & (pl.col("val") <= 2),
                        data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        us_rows = result.filter(pl.col("region") == "us")
        assert sorted(us_rows["val"].to_list()) == [3, 4]
        eu_rows = result.filter(pl.col("region") == "eu")
        assert sorted(eu_rows["val"].to_list()) == [10, 20, 30]


# ===================================================================
# 4. Row ID preservation after rewrite
# ===================================================================


class TestRowIdPreservation:
    """Row IDs should be renumbered consistently after rewrite.

    After compaction, new row IDs start from 0 and are contiguous
    within each output file. The important property is that
    next_row_id matches the total record count.
    """

    def test_row_id_range_correct_after_rewrite(self, make_write_catalog):
        """next_row_id in table stats matches total row count after rewrite."""
        cat = make_write_catalog()
        _write_many_small(cat, "t", n_batches=5, rows_per_batch=10)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        row = cat.query_one(
            "SELECT record_count, next_row_id FROM ducklake_table_stats "
            "WHERE table_id = (SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row[0] == 50  # record_count
        assert row[1] == 50  # next_row_id

    def test_row_id_correct_after_rewrite_with_deletes(self, make_write_catalog):
        """After delete + rewrite, next_row_id reflects surviving rows."""
        cat = make_write_catalog()
        _write_many_small(cat, "t", n_batches=4, rows_per_batch=10)

        delete_ducklake(cat.metadata_path, "t", pl.col("id") < 15,
                        data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        row = cat.query_one(
            "SELECT record_count, next_row_id FROM ducklake_table_stats "
            "WHERE table_id = (SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row[0] == 25  # 40 - 15 deleted
        assert row[1] == 25

    def test_data_file_row_id_start_zero_after_rewrite(self, make_write_catalog):
        """The consolidated data file has row_id_start = 0."""
        cat = make_write_catalog()
        _write_many_small(cat, "t", n_batches=5, rows_per_batch=5)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        rows = cat.query_all(
            "SELECT row_id_start FROM ducklake_data_file "
            "WHERE end_snapshot IS NULL AND table_id = "
            "(SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        # For non-partitioned: single file with offset 0
        assert len(rows) == 1
        assert rows[0][0] == 0


# ===================================================================
# 5. Rewrite after mixed operations: insert + delete + update
# ===================================================================


class TestRewriteAfterMixedOps:
    """Rewrite after a complex sequence of insert, delete, update."""

    def test_insert_delete_update_rewrite(self, make_write_catalog):
        """insert -> delete -> update -> rewrite -> data correct."""
        cat = make_write_catalog()

        # Insert 3 batches
        for i in range(3):
            df = pl.DataFrame({
                "id": list(range(i * 10, (i + 1) * 10)),
                "status": ["active"] * 10,
            })
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        # Delete some rows
        delete_ducklake(cat.metadata_path, "t", pl.col("id") < 5,
                        data_path=cat.data_path)

        # Update remaining rows in first batch
        update_ducklake(cat.metadata_path, "t",
                        predicate=(pl.col("id") >= 5) & (pl.col("id") < 10),
                        updates={"status": pl.lit("updated")},
                        data_path=cat.data_path)

        # Rewrite
        snap = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 25  # 30 - 5 deleted

        updated_rows = result.filter(
            (pl.col("id") >= 5) & (pl.col("id") < 10)
        ).sort("id")
        assert all(s == "updated" for s in updated_rows["status"].to_list())

        active_rows = result.filter(pl.col("id") >= 10).sort("id")
        assert all(s == "active" for s in active_rows["status"].to_list())

    def test_multiple_updates_then_rewrite(self, make_write_catalog):
        """Multiple updates to same rows -> rewrite keeps final values."""
        cat = make_write_catalog()

        df = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        # Append to create multi-file
        df2 = pl.DataFrame({"id": [4, 5], "val": [40, 50]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        # Update id=1 twice
        update_ducklake(cat.metadata_path, "t",
                        predicate=pl.col("id") == 1,
                        updates={"val": pl.lit(100, dtype=pl.Int64)},
                        data_path=cat.data_path)
        update_ducklake(cat.metadata_path, "t",
                        predicate=pl.col("id") == 1,
                        updates={"val": pl.lit(999, dtype=pl.Int64)},
                        data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t",
                               data_path=cat.data_path).sort("id")
        assert result.filter(pl.col("id") == 1)["val"].to_list() == [999]
        assert sorted(result["id"].to_list()) == [1, 2, 3, 4, 5]


# ===================================================================
# 6. Rewrite idempotency: rewrite twice -> same result
# ===================================================================


class TestRewriteIdempotency:
    """Rewriting an already-compacted table is a no-op."""

    def test_second_rewrite_returns_negative_one(self, make_write_catalog):
        """After compaction, a second rewrite returns -1 (nothing to do)."""
        cat = make_write_catalog()
        _write_many_small(cat, "t", n_batches=5, rows_per_batch=10)

        snap1 = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap1 > 0

        snap2 = rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert snap2 == -1  # nothing to rewrite

    def test_data_unchanged_after_double_rewrite(self, make_write_catalog):
        """Data is identical after first and second rewrite attempt."""
        cat = make_write_catalog()
        expected = _write_many_small(cat, "t", n_batches=4, rows_per_batch=8)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        result1 = read_ducklake(cat.metadata_path, "t",
                                data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        result2 = read_ducklake(cat.metadata_path, "t",
                                data_path=cat.data_path)

        assert_frame_equal(result1.sort("id"), result2.sort("id"))
        assert_frame_equal(result1.sort("id"), expected.sort("id"))


# ===================================================================
# 7. Rewrite with delete files: deletes applied during rewrite
# ===================================================================


class TestRewriteWithDeleteFiles:
    """Deletion vectors are consumed and removed during rewrite."""

    def test_delete_files_removed_after_rewrite(self, make_write_catalog):
        """After rewrite, no active delete files remain in metadata."""
        cat = make_write_catalog()
        _write_many_small(cat, "t", n_batches=3, rows_per_batch=10)

        # Create delete files
        delete_ducklake(cat.metadata_path, "t", pl.col("id") < 5,
                        data_path=cat.data_path)
        delete_ducklake(cat.metadata_path, "t", pl.col("id") >= 25,
                        data_path=cat.data_path)

        # Verify delete files exist
        del_count_before = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_delete_file "
            "WHERE end_snapshot IS NULL AND table_id = "
            "(SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )[0]
        assert del_count_before > 0

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        # No active delete files after rewrite
        del_count_after = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_delete_file "
            "WHERE end_snapshot IS NULL AND table_id = "
            "(SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )[0]
        assert del_count_after == 0

        # Data correct
        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 20  # 30 - 5 - 5
        ids = sorted(result["id"].to_list())
        assert ids == list(range(5, 25))

    def test_partial_delete_applied_in_rewrite(self, make_write_catalog):
        """Delete some rows from each file -> rewrite removes them."""
        cat = make_write_catalog()

        # 3 files, each with 10 rows
        for i in range(3):
            df = pl.DataFrame({"id": list(range(i * 10, (i + 1) * 10))})
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        # Delete even IDs
        delete_ducklake(cat.metadata_path, "t", pl.col("id") % 2 == 0,
                        data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 15  # half of 30
        assert all(v % 2 == 1 for v in result["id"].to_list())


# ===================================================================
# 8. File count reduction: verify file count drops
# ===================================================================


class TestFileCountReduction:
    """Verify actual on-disk file count is reduced after full lifecycle."""

    def test_many_files_consolidated_to_one(self, make_write_catalog):
        """8 small appends -> rewrite + expire + vacuum -> 1 file on disk."""
        cat = make_write_catalog()

        for i in range(8):
            df = pl.DataFrame({"x": [i * 100 + j for j in range(5)]})
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        files_before = _count_parquet_files(cat.data_path)
        assert files_before >= 8

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        expire_snapshots(cat.metadata_path, keep_last_n=1,
                         data_path=cat.data_path)
        vacuum_ducklake(cat.metadata_path, data_path=cat.data_path)

        files_after = _count_parquet_files(cat.data_path)
        assert files_after == 1

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 40

    def test_metadata_shows_single_active_file(self, make_write_catalog):
        """After rewrite, metadata has exactly one active data file."""
        cat = make_write_catalog()
        _write_many_small(cat, "t", n_batches=6, rows_per_batch=5)

        # Before rewrite: 6 active data files
        row_before = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file "
            "WHERE end_snapshot IS NULL AND table_id = "
            "(SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row_before[0] == 6

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        row_after = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file "
            "WHERE end_snapshot IS NULL AND table_id = "
            "(SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row_after[0] == 1


# ===================================================================
# 9. Data integrity: all rows survive rewrite unchanged
# ===================================================================


class TestDataIntegrity:
    """Every row present before rewrite is present and unchanged after."""

    def test_all_rows_preserved(self, make_write_catalog):
        """All rows survive compaction with exact values."""
        cat = make_write_catalog()
        expected = _write_many_small(cat, "t", n_batches=5, rows_per_batch=10)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert_frame_equal(expected.sort("id"), result.sort("id"))

    def test_various_types_preserved(self, make_write_catalog):
        """Integers, floats, strings, booleans all preserved through rewrite."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({
            "int_col": [4, 5],
            "float_col": [4.4, 5.5],
            "str_col": ["d", "e"],
            "bool_col": [False, True],
        })
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        expected = pl.concat([df1, df2])

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert_frame_equal(expected.sort("int_col"), result.sort("int_col"))

    def test_large_data_preserved(self, make_write_catalog):
        """Larger dataset (1000 rows across 10 files) preserved."""
        cat = make_write_catalog()

        all_frames: list[pl.DataFrame] = []
        for i in range(10):
            df = pl.DataFrame({
                "id": list(range(i * 100, (i + 1) * 100)),
                "value": [float(j) * 0.1 for j in range(i * 100, (i + 1) * 100)],
            })
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)
            all_frames.append(df)

        expected = pl.concat(all_frames)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 1000
        assert_frame_equal(expected.sort("id"), result.sort("id"))


# ===================================================================
# 10. Stats preservation: min/max stats correct after rewrite
# ===================================================================


class TestStatsPreservation:
    """Column and table stats are accurate after rewrite."""

    def test_min_max_correct_after_rewrite(self, make_write_catalog):
        """Integer min/max stats reflect consolidated data range."""
        cat = make_write_catalog()

        for vals in [[10, 20], [5, 30], [-50, 1000]]:
            df = pl.DataFrame({"v": vals})
            mode = "error" if vals[0] == 10 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
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
        assert str(row[0]) == "-50"
        assert str(row[1]) == "1000"

    def test_stats_reflect_deletes_after_rewrite(self, make_write_catalog):
        """After delete + rewrite, stats exclude deleted rows."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"v": [1, 100]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)
        df2 = pl.DataFrame({"v": [50, 75]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                       data_path=cat.data_path)

        # Delete the max value
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
        assert str(row[0]) == "1"
        assert str(row[1]) == "75"

    def test_null_flag_correct_after_rewrite(self, make_write_catalog):
        """contains_null reflects actual NULL presence after rewrite."""
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
        assert row[0] in (1, True)

    def test_record_count_stat_correct(self, make_write_catalog):
        """record_count in table stats matches actual row count."""
        cat = make_write_catalog()
        _write_many_small(cat, "t", n_batches=4, rows_per_batch=8)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        row = cat.query_one(
            "SELECT record_count FROM ducklake_table_stats "
            "WHERE table_id = (SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row[0] == 32


# ===================================================================
# 11. Merge-adjacent: adjacent small files merged into larger ones
# ===================================================================


class TestMergeAdjacent:
    """Small files from successive appends are merged into one."""

    def test_small_files_merged(self, make_write_catalog):
        """Many 1-row appends -> rewrite -> single consolidated file."""
        cat = make_write_catalog()

        for i in range(12):
            df = pl.DataFrame({"x": [i]})
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        # 12 active data files before
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file "
            "WHERE end_snapshot IS NULL AND table_id = "
            "(SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row[0] == 12

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        # 1 active data file after
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file "
            "WHERE end_snapshot IS NULL AND table_id = "
            "(SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row[0] == 1

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert sorted(result["x"].to_list()) == list(range(12))

    def test_merge_preserves_column_order(self, make_write_catalog):
        """After merging files, column order is consistent."""
        cat = make_write_catalog()

        for i in range(5):
            df = pl.DataFrame({"a": [i], "b": [f"s{i}"], "c": [float(i)]})
            mode = "error" if i == 0 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.columns == ["a", "b", "c"]
        assert sorted(result["a"].to_list()) == list(range(5))

    def test_merge_adjacent_partitioned(self, make_write_catalog):
        """Merge-adjacent within each partition independently."""
        cat = make_write_catalog()

        df_init = pl.DataFrame({"part": ["a"], "val": [0]})
        write_ducklake(df_init, cat.metadata_path, "t", data_path=cat.data_path)
        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "t", ["part"], data_path=cat.data_path,
        )

        # 4 inserts per partition
        for i in range(1, 9):
            part = "a" if i % 2 == 0 else "b"
            df = pl.DataFrame({"part": [part], "val": [i]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                           data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        # 2 active files (one per partition)
        row = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file "
            "WHERE end_snapshot IS NULL AND table_id = "
            "(SELECT table_id FROM ducklake_table "
            "WHERE table_name = 't' AND end_snapshot IS NULL)"
        )
        assert row[0] == 2

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 9  # init + 8 inserts


# ===================================================================
# 12. Sort key preservation: sorted tables stay sorted after rewrite
# ===================================================================


class TestSortKeyPreservation:
    """Rewrite respects sort key configuration."""

    def test_ascending_sort_preserved(self, make_write_catalog):
        """Data is sorted by sort key after rewrite (ASC)."""
        cat = make_write_catalog()

        # Write out of order across multiple files
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
        assert ids == sorted(ids), f"Expected sorted ASC, got {ids}"

    def test_descending_sort_preserved(self, make_write_catalog):
        """Data is sorted by sort key after rewrite (DESC)."""
        cat = make_write_catalog()

        for i in range(4):
            df = pl.DataFrame({"score": [i * 5 + j for j in range(5)]})
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
        assert scores == sorted(scores, reverse=True), \
            f"Expected sorted DESC, got {scores}"

    def test_multi_column_sort_preserved(self, make_write_catalog):
        """Compound sort key (two columns) preserved after rewrite."""
        cat = make_write_catalog()

        for batch in [
            {"a": [2, 1, 2], "b": [3, 1, 1], "c": ["x", "y", "z"]},
            {"a": [1, 3, 1], "b": [2, 1, 3], "c": ["w", "v", "u"]},
        ]:
            df = pl.DataFrame(batch)
            mode = "error" if batch["a"][0] == 2 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("a", "ASC"), ("b", "ASC")],
            data_path=cat.data_path,
        )

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        pairs = list(zip(result["a"].to_list(), result["b"].to_list()))
        assert pairs == sorted(pairs), f"Expected sorted by (a, b), got {pairs}"

    def test_sort_key_with_deletes_preserved(self, make_write_catalog):
        """Sort order preserved even after deleting rows and rewriting."""
        cat = make_write_catalog()

        for vals in [[50, 10, 30], [20, 40, 60]]:
            df = pl.DataFrame({"id": vals})
            mode = "error" if vals[0] == 50 else "append"
            write_ducklake(df, cat.metadata_path, "t", mode=mode,
                           data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("id", "ASC")],
            data_path=cat.data_path,
        )

        # Delete middle values
        delete_ducklake(cat.metadata_path, "t",
                        (pl.col("id") >= 30) & (pl.col("id") <= 40),
                        data_path=cat.data_path)

        rewrite_data_files_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        ids = result["id"].to_list()
        assert ids == sorted(ids)
        assert sorted(ids) == [10, 20, 50, 60]
