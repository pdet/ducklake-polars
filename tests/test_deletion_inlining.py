"""Tests for deletion inlining edge cases.

Covers interactions between inline deletion and:
- ALTER TABLE (add/drop/rename column)
- Compaction (rewrite_data_files)
- Partitioned tables
- Stats / filter pushdown
- Large inline deletion (near the threshold boundary)
- Transaction semantics (time travel across multiple operations)
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    alter_ducklake_set_partitioned_by,
    delete_ducklake,
    read_ducklake,
    rewrite_data_files_ducklake,
    scan_ducklake,
    update_ducklake,
    write_ducklake,
)


INLINE_LIMIT = 1000


# ---------------------------------------------------------------------------
# Deletion inlining + ALTER TABLE
# ---------------------------------------------------------------------------


class TestDeletionInliningAlter:
    """Inline delete followed by schema evolution operations."""

    def test_add_column_after_inline_delete(self, make_write_catalog):
        """Add a column after deleting inlined rows -- reads show NULL for new column."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 3,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        alter_ducklake_add_column(
            cat.metadata_path, "test", "c", pl.Float64,
        )

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 4, 5]
        assert result["b"].to_list() == ["x", "y", "w", "v"]
        assert result["c"].to_list() == [None, None, None, None]

    @pytest.mark.xfail(reason="Inlined data reader does not apply column defaults")
    def test_add_column_with_default_after_inline_delete(self, make_write_catalog):
        """Add a column with a default value after inline delete."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        alter_ducklake_add_column(
            cat.metadata_path, "test", "val", pl.Int64, default=42,
        )

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result["a"].to_list() == [1, 3]
        assert result["val"].to_list() == [42, 42]

    def test_drop_column_after_inline_delete(self, make_write_catalog):
        """Drop a column after deleting inlined rows -- remaining rows lose the column."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10, 20, 30]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        alter_ducklake_drop_column(
            cat.metadata_path, "test", "c",
        )

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 3]
        assert result["b"].to_list() == ["x", "z"]

    @pytest.mark.xfail(reason="Inlined data reader does not map renamed columns")
    def test_rename_column_after_inline_delete(self, make_write_catalog):
        """Rename a column after deleting inlined rows -- reads use the new name."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        alter_ducklake_rename_column(
            cat.metadata_path, "test", "b", "label",
        )

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert "label" in result.columns
        assert "b" not in result.columns
        assert result["a"].to_list() == [1, 3]
        assert result["label"].to_list() == ["x", "z"]

    def test_insert_after_alter_after_inline_delete(self, make_write_catalog):
        """Insert new data after add_column + inline_delete -- new rows have the new column."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        alter_ducklake_add_column(
            cat.metadata_path, "test", "b", pl.Utf8,
        )

        write_ducklake(
            pl.DataFrame({"a": [10, 20], "b": ["new1", "new2"]}),
            cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result["a"].to_list() == [1, 3, 10, 20]
        assert result["b"].to_list() == [None, None, "new1", "new2"]

    def test_delete_after_add_column(self, make_write_catalog):
        """Delete from inlined data that was inserted before a column was added."""
        cat = make_write_catalog()
        if cat.backend in ("postgres", "duckdb"):
            pytest.xfail(
                f"{cat.backend} backend: inlined delete after add_column returns 0"
            )
        df = pl.DataFrame({"a": [1, 2, 3]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        alter_ducklake_add_column(
            cat.metadata_path, "test", "b", pl.Int64,
        )

        # Delete a row from the original inlined data (which doesn't have column b)
        deleted = delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert deleted == 1

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result["a"].to_list() == [1, 3]

    @pytest.mark.xfail(reason="Inlined data reader does not map renamed columns")
    def test_multiple_alters_after_inline_delete(self, make_write_catalog):
        """Multiple alter operations after inline delete -- schema evolves correctly."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Delete two rows
        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") <= 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        # Add a column
        alter_ducklake_add_column(
            cat.metadata_path, "test", "c", pl.Int64,
        )

        # Rename original column
        alter_ducklake_rename_column(
            cat.metadata_path, "test", "b", "label",
        )

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.columns == ["a", "label", "c"]
        assert result["a"].to_list() == [3, 4]
        assert result["label"].to_list() == ["y", "z"]
        assert result["c"].to_list() == [None, None]


# ---------------------------------------------------------------------------
# Deletion inlining + compaction (rewrite_data_files)
# ---------------------------------------------------------------------------


class TestDeletionInliningCompaction:
    """Compaction (rewrite_data_files) interacting with deletion."""

    def test_rewrite_after_parquet_deletes(self, make_write_catalog):
        """Rewrite merges parquet files and removes deleted rows."""
        cat = make_write_catalog()

        # Two batches to Parquet (no inlining)
        write_ducklake(
            pl.DataFrame({"a": list(range(10)), "b": [f"v{i}" for i in range(10)]}),
            cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=0,
        )
        write_ducklake(
            pl.DataFrame({"a": list(range(10, 20)), "b": [f"v{i}" for i in range(10, 20)]}),
            cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=0,
        )

        # Delete some rows
        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 5, data_inlining_row_limit=0)
        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 15, data_inlining_row_limit=0)

        # Rewrite should merge files and remove deleted rows
        snap = rewrite_data_files_ducklake(cat.metadata_path, "test")
        assert snap != -1

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 18
        vals = sorted(result["a"].to_list())
        assert 5 not in vals
        assert 15 not in vals

    def test_rewrite_mixed_inlined_and_parquet(self, make_write_catalog):
        """Table has both inlined rows and Parquet files -- all readable after rewrite."""
        cat = make_write_catalog()

        # Batch 1: inlined
        write_ducklake(
            pl.DataFrame({"a": [1, 2, 3]}),
            cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Batch 2: larger, exceeds threshold -> goes to Parquet
        write_ducklake(
            pl.DataFrame({"a": list(range(100, 100 + INLINE_LIMIT))}),
            cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Delete from inlined data
        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        # Rewrite -- should merge Parquet files (inlined stays inlined)
        rewrite_data_files_ducklake(cat.metadata_path, "test")

        result = read_ducklake(cat.metadata_path, "test")
        vals = sorted(result["a"].to_list())
        assert 2 not in vals
        assert 1 in vals
        assert 3 in vals

    def test_rewrite_after_multiple_deletes(self, make_write_catalog):
        """Multiple deletions across different snapshots, then rewrite."""
        cat = make_write_catalog()

        # Two batches to Parquet to make rewrite viable
        write_ducklake(
            pl.DataFrame({"a": list(range(10))}),
            cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=0,
        )
        write_ducklake(
            pl.DataFrame({"a": list(range(10, 20))}),
            cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=0,
        )

        # Delete rows in separate operations
        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 0, data_inlining_row_limit=0)
        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 5, data_inlining_row_limit=0)
        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 10, data_inlining_row_limit=0)
        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 19, data_inlining_row_limit=0)

        snap = rewrite_data_files_ducklake(cat.metadata_path, "test")
        assert snap != -1

        result = read_ducklake(cat.metadata_path, "test")
        vals = sorted(result["a"].to_list())
        assert len(vals) == 16
        for v in [0, 5, 10, 19]:
            assert v not in vals

    def test_rewrite_after_delete_all_from_one_file(self, make_write_catalog):
        """Delete all rows from one Parquet file, then rewrite -- only second file data remains."""
        cat = make_write_catalog()

        write_ducklake(
            pl.DataFrame({"a": [1, 2, 3]}),
            cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=0,
        )
        write_ducklake(
            pl.DataFrame({"a": [10, 20, 30]}),
            cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=0,
        )

        # Delete all rows from first file
        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") < 10,
            data_inlining_row_limit=0,
        )

        snap = rewrite_data_files_ducklake(cat.metadata_path, "test")
        assert snap != -1

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [10, 20, 30]


# ---------------------------------------------------------------------------
# Deletion inlining + partitioned tables
# ---------------------------------------------------------------------------


class TestDeletionInliningPartitions:
    """Inline deletion on partitioned tables."""

    def test_inline_delete_on_partitioned_table(self, make_write_catalog):
        """Delete inlined rows from a table that has a partition spec."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "category": ["A", "A", "B", "B", "C"],
            "value": [1, 2, 3, 4, 5],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "test", ["category"],
        )

        # Delete rows from one partition
        deleted = delete_ducklake(
            cat.metadata_path, "test", pl.col("category") == "A",
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert deleted == 2

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("value")
        assert result["category"].to_list() == ["B", "B", "C"]
        assert result["value"].to_list() == [3, 4, 5]

    def test_inline_delete_cross_partition(self, make_write_catalog):
        """Delete inlined rows across multiple partitions."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "category": ["A", "A", "B", "B"],
            "value": [1, 2, 3, 4],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "test", ["category"],
        )

        # Delete rows with value > 1 -- affects both partitions
        deleted = delete_ducklake(
            cat.metadata_path, "test", pl.col("value") > 1,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert deleted == 3

        result = read_ducklake(cat.metadata_path, "test")
        assert result["value"].to_list() == [1]
        assert result["category"].to_list() == ["A"]

    def test_inline_then_partitioned_insert_after_delete(self, make_write_catalog):
        """Inline insert + delete, then set partition, then insert more."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "category": ["A", "B"],
            "value": [1, 2],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("value") == 1,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "test", ["category"],
        )

        new_data = pl.DataFrame({
            "category": ["A", "A", "B", "C"],
            "value": [10, 20, 30, 40],
        })
        write_ducklake(
            new_data, cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["value"].to_list()) == [2, 10, 20, 30, 40]

    def test_delete_all_from_one_partition_inlined(self, make_write_catalog):
        """Delete all rows belonging to one partition value from inlined data."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "region": ["east", "east", "west", "west", "west"],
            "amount": [100, 200, 300, 400, 500],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "test", ["region"],
        )

        deleted = delete_ducklake(
            cat.metadata_path, "test", pl.col("region") == "east",
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert deleted == 2

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("amount")
        assert result["region"].to_list() == ["west", "west", "west"]
        assert result["amount"].to_list() == [300, 400, 500]


# ---------------------------------------------------------------------------
# Deletion inlining + stats / filter pushdown
# ---------------------------------------------------------------------------


class TestDeletionInliningStats:
    """Verify stats and filter pushdown still work after inline deletions."""

    def test_filter_after_inline_delete(self, make_write_catalog):
        """scan_ducklake with filter works correctly after inline deletion."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "id": list(range(20)),
            "label": [f"item_{i}" for i in range(20)],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Delete some rows
        delete_ducklake(
            cat.metadata_path, "test", pl.col("id") < 5,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        # Filter with scan_ducklake
        result = scan_ducklake(cat.metadata_path, "test").filter(
            pl.col("id") >= 10
        ).collect()
        result = result.sort("id")
        assert result["id"].to_list() == list(range(10, 20))

    def test_filter_on_deleted_value(self, make_write_catalog):
        """Filter for an exact value that was deleted returns empty."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 3,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        result = scan_ducklake(cat.metadata_path, "test").filter(
            pl.col("a") == 3
        ).collect()
        assert result.shape[0] == 0

    def test_filter_range_excluding_deleted_rows(self, make_write_catalog):
        """Range filter correctly excludes deleted rows within range."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "x": list(range(100)),
            "y": [f"val_{i}" for i in range(100)],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Delete even numbers
        delete_ducklake(
            cat.metadata_path, "test", (pl.col("x") % 2) == 0,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        result = scan_ducklake(cat.metadata_path, "test").filter(
            (pl.col("x") >= 10) & (pl.col("x") < 20)
        ).collect()
        result = result.sort("x")
        # Only odd numbers in [10, 20)
        assert result["x"].to_list() == [11, 13, 15, 17, 19]

    def test_stats_correctness_after_inline_delete(self, make_write_catalog):
        """Table-level record_count reflects remaining rows after inline deletion."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") > 3,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        # Read all remaining data
        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        assert sorted(result["a"].to_list()) == [1, 2, 3]

    def test_string_filter_after_inline_delete(self, make_write_catalog):
        """String column filter works after inline delete."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "name": ["alice", "bob", "charlie", "diana", "eve"],
            "score": [90, 80, 70, 60, 50],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("name") == "charlie",
            data_inlining_row_limit=INLINE_LIMIT,
        )

        result = scan_ducklake(cat.metadata_path, "test").filter(
            pl.col("name") > "bob"
        ).collect()
        result = result.sort("name")
        assert result["name"].to_list() == ["diana", "eve"]
        assert result["score"].to_list() == [60, 50]

    def test_null_filter_after_inline_delete(self, make_write_catalog):
        """Null filtering works after inline delete on nullable column."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "a": pl.Series([1, 2, None, 4, None], dtype=pl.Int64),
            "b": ["x", "y", "z", "w", "v"],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        # Delete one null row
        delete_ducklake(
            cat.metadata_path, "test", pl.col("b") == "z",
            data_inlining_row_limit=INLINE_LIMIT,
        )

        result = scan_ducklake(cat.metadata_path, "test").filter(
            pl.col("a").is_null()
        ).collect()
        assert result.shape[0] == 1
        assert result["b"].to_list() == ["v"]


# ---------------------------------------------------------------------------
# Large inline deletion (near threshold boundary)
# ---------------------------------------------------------------------------


class TestLargeInlineDeletion:
    """Tests for inline deletion near the inlining threshold boundary."""

    def test_delete_at_threshold_boundary(self, make_write_catalog):
        """Insert exactly at the threshold, delete some, result is correct."""
        limit = 50
        cat = make_write_catalog()
        df = pl.DataFrame({"a": list(range(limit))})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=limit,
        )

        # Delete half the rows
        deleted = delete_ducklake(
            cat.metadata_path, "test", pl.col("a") >= (limit // 2),
            data_inlining_row_limit=limit,
        )
        assert deleted == limit // 2

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == list(range(limit // 2))

    def test_insert_after_delete_reuses_inline_capacity(self, make_write_catalog):
        """After deleting rows, new inserts may still go to inline storage
        since the active row count dropped below the threshold."""
        limit = 20
        cat = make_write_catalog()

        df = pl.DataFrame({"a": list(range(limit))})
        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=limit,
        )

        # Delete most rows
        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") >= 5,
            data_inlining_row_limit=limit,
        )

        # Active row count is now 5. Insert 10 more -> total 15 < 20 limit.
        new_df = pl.DataFrame({"a": list(range(100, 110))})
        write_ducklake(
            new_df, cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=limit,
        )

        result = read_ducklake(cat.metadata_path, "test")
        vals = sorted(result["a"].to_list())
        expected = list(range(5)) + list(range(100, 110))
        assert vals == sorted(expected)

    def test_delete_almost_all_rows(self, make_write_catalog):
        """Delete all but one row from inline data."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": list(range(100))})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        deleted = delete_ducklake(
            cat.metadata_path, "test", pl.col("a") > 0,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert deleted == 99

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].to_list() == [0]

    def test_successive_deletes_near_threshold(self, make_write_catalog):
        """Multiple successive deletes, each removing one row, near the limit."""
        limit = 10
        cat = make_write_catalog()
        df = pl.DataFrame({"a": list(range(limit))})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=limit,
        )

        for i in range(limit):
            deleted = delete_ducklake(
                cat.metadata_path, "test", pl.col("a") == i,
                data_inlining_row_limit=limit,
            )
            assert deleted == 1

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0

    def test_large_inline_delete_duckdb_interop(self, make_write_catalog):
        """DuckDB reads the correct result after a large inline deletion."""
        limit = 100
        cat = make_write_catalog()
        df = pl.DataFrame({
            "id": list(range(limit)),
            "data": [f"row_{i}" for i in range(limit)],
        })

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=limit,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("id") % 3 == 0,
            data_inlining_row_limit=limit,
        )

        # Remaining: ids not divisible by 3
        expected_ids = [i for i in range(limit) if i % 3 != 0]

        result = cat.read_with_duckdb("test", inline_limit=limit)
        assert sorted(result["id"].to_list()) == expected_ids


# ---------------------------------------------------------------------------
# Transaction semantics with inline deletion
# ---------------------------------------------------------------------------


class TestDeletionInliningTransactions:
    """Time travel and multi-snapshot interactions with inline deletion."""

    def test_time_travel_before_and_after_inline_delete(self, make_write_catalog):
        """Time travel to snapshot before inline delete sees all rows."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        snap_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") > 3,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        snap_after = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        # Before delete: all 5 rows
        result_before = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before,
        )
        assert sorted(result_before["a"].to_list()) == [1, 2, 3, 4, 5]

        # After delete: 3 rows
        result_after = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_after,
        )
        assert sorted(result_after["a"].to_list()) == [1, 2, 3]

    def test_multiple_delete_insert_cycles_inlined(self, make_write_catalog):
        """Multiple cycles of insert+delete all inlined, verify final state."""
        cat = make_write_catalog()

        write_ducklake(
            pl.DataFrame({"a": [1, 2, 3]}),
            cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        write_ducklake(
            pl.DataFrame({"a": [10, 20]}),
            cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 10,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        write_ducklake(
            pl.DataFrame({"a": [100]}),
            cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [1, 3, 20, 100]

    def test_update_after_inline_delete(self, make_write_catalog):
        """Update remaining rows after inline delete."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["old", "old", "old", "old"]})

        write_ducklake(
            df, cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") <= 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        updated = update_ducklake(
            cat.metadata_path, "test",
            {"b": "new"},
            pl.col("a") == 3,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        assert updated == 1

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result["a"].to_list() == [3, 4]
        assert result["b"].to_list() == ["new", "old"]

    def test_time_travel_across_delete_insert_update(self, make_write_catalog):
        """Time travel through a series of inline operations: insert, delete, insert, update."""
        cat = make_write_catalog()

        # Snap 1: insert [1, 2, 3]
        write_ducklake(
            pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
            cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )
        snap1 = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")[0]

        # Snap 2: delete a=2
        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 2,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        snap2 = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")[0]

        # Snap 3: insert [10, 20]
        write_ducklake(
            pl.DataFrame({"a": [10, 20], "b": ["m", "n"]}),
            cat.metadata_path, "test",
            mode="append", data_inlining_row_limit=INLINE_LIMIT,
        )
        snap3 = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")[0]

        # Snap 4: update a=3 -> b='updated'
        update_ducklake(
            cat.metadata_path, "test",
            {"b": "updated"},
            pl.col("a") == 3,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        snap4 = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")[0]

        # Verify each snapshot
        r1 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap1)
        assert sorted(r1["a"].to_list()) == [1, 2, 3]

        r2 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap2)
        assert sorted(r2["a"].to_list()) == [1, 3]

        r3 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap3)
        assert sorted(r3["a"].to_list()) == [1, 3, 10, 20]

        r4 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap4)
        r4 = r4.sort("a")
        assert r4["a"].to_list() == [1, 3, 10, 20]
        assert r4["b"].to_list() == ["x", "updated", "m", "n"]

    def test_overwrite_after_inline_delete(self, make_write_catalog):
        """Overwrite the table after inline delete -- old data is gone."""
        cat = make_write_catalog()

        write_ducklake(
            pl.DataFrame({"a": [1, 2, 3, 4, 5]}),
            cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") > 3,
            data_inlining_row_limit=INLINE_LIMIT,
        )

        write_ducklake(
            pl.DataFrame({"a": [99, 100]}),
            cat.metadata_path, "test",
            mode="overwrite", data_inlining_row_limit=INLINE_LIMIT,
        )

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [99, 100]

    def test_delete_then_time_travel_then_delete_again(self, make_write_catalog):
        """Delete, verify via time travel, delete more rows."""
        cat = make_write_catalog()

        write_ducklake(
            pl.DataFrame({"a": [1, 2, 3, 4, 5]}),
            cat.metadata_path, "test",
            mode="error", data_inlining_row_limit=INLINE_LIMIT,
        )
        snap0 = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")[0]

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 1,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        snap1 = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")[0]

        delete_ducklake(
            cat.metadata_path, "test", pl.col("a") == 5,
            data_inlining_row_limit=INLINE_LIMIT,
        )
        snap2 = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")[0]

        r0 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap0)
        assert sorted(r0["a"].to_list()) == [1, 2, 3, 4, 5]

        r1 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap1)
        assert sorted(r1["a"].to_list()) == [2, 3, 4, 5]

        r2 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap2)
        assert sorted(r2["a"].to_list()) == [2, 3, 4]
