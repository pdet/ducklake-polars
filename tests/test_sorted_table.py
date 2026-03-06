"""Sorted table tests — merge-adjacent, flush, and sort interactions.

Extends the sort key tests in test_write_alter.py with:
  - Sorted merge-adjacent (rewrite respects sort order)
  - Sort keys + schema evolution (add/rename column)
  - Sort keys + partitioned tables
  - Sort keys + deletes/updates
  - Sort key validation after multiple operations
  - Sorted flush from inlined data
  - Sort keys with time travel
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    read_ducklake,
    write_ducklake,
    rewrite_data_files_ducklake,
    alter_ducklake_set_sort_keys,
    alter_ducklake_reset_sort_keys,
    alter_ducklake_add_column,
    alter_ducklake_rename_column,
    delete_ducklake,
)


class TestSortedMergeAdjacent:
    """Rewrite (merge-adjacent) respects sort keys."""

    def test_sorted_merge_ascending(self, make_write_catalog):
        """Multiple unsorted files -> rewrite with ASC sort key -> sorted output."""
        cat = make_write_catalog()
        # Write data in reverse batches
        for i in range(5):
            df = pl.DataFrame({
                "id": [50 - i * 10 - j for j in range(10)],
                "val": [f"v{50 - i * 10 - j}" for j in range(10)],
            })
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("id", "ASC")],
            data_path=cat.data_path,
        )

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t",
                                           data_path=cat.data_path)
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        ids = result["id"].to_list()
        assert ids == sorted(ids), "Merged data should be sorted ASC"

    def test_sorted_merge_descending(self, make_write_catalog):
        """Multiple files -> rewrite with DESC sort key -> sorted descending."""
        cat = make_write_catalog()
        for i in range(4):
            df = pl.DataFrame({"id": [i * 5 + j for j in range(5)]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("id", "DESC")],
            data_path=cat.data_path,
        )

        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        ids = result["id"].to_list()
        assert ids == sorted(ids, reverse=True), "Merged data should be sorted DESC"

    def test_sorted_merge_multi_column(self, make_write_catalog):
        """Rewrite with multi-column sort key."""
        cat = make_write_catalog()
        for i in range(3):
            df = pl.DataFrame({
                "dept": ["B", "A", "C"],
                "name": [f"n{i*3+j}" for j in range(3)],
                "id": [i * 3 + j for j in range(3)],
            })
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("dept", "ASC"), ("name", "ASC")],
            data_path=cat.data_path,
        )

        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        depts = result["dept"].to_list()
        assert depts == sorted(depts), "Primary sort column should be sorted"

    def test_sorted_merge_preserves_data(self, make_write_catalog):
        """Sorted rewrite preserves all data values exactly."""
        cat = make_write_catalog()
        all_ids = []
        for i in range(4):
            batch_ids = list(range(i * 10, i * 10 + 10))
            all_ids.extend(batch_ids)
            df = pl.DataFrame({
                "id": batch_ids,
                "val": [f"v{x}" for x in batch_ids],
            })
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        result_before = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("id", "ASC")],
            data_path=cat.data_path,
        )
        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        result_after = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert_frame_equal(result_before.sort("id"), result_after.sort("id"))


class TestSortKeysWithSchemaEvolution:
    """Sort keys interact correctly with schema changes."""

    def test_sort_after_add_column(self, make_write_catalog):
        """Add column then set sort keys on original column."""
        cat = make_write_catalog()
        df = pl.DataFrame({"id": [3, 1, 2], "score": [30, 10, 20]})
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        alter_ducklake_add_column(
            cat.metadata_path, "t", "extra", pl.Int64(),
            data_path=cat.data_path,
        )

        # More data with new column
        df2 = pl.DataFrame({"id": [5, 4], "score": [50, 40], "extra": [10, 20]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("id", "ASC")],
            data_path=cat.data_path,
        )

        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        ids = result["id"].to_list()
        assert ids == sorted(ids)
        assert len(result) == 5

    def test_sort_after_rename_column(self, make_write_catalog):
        """Rename sorted column, verify data preserved after rewrite."""
        cat = make_write_catalog()
        df = pl.DataFrame({"id": [3, 1, 2], "score": [30, 10, 20]})
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        alter_ducklake_rename_column(
            cat.metadata_path, "t", "id", "row_id",
            data_path=cat.data_path,
        )

        df2 = pl.DataFrame({"row_id": [5, 4], "score": [50, 40]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("row_id", "ASC")],
            data_path=cat.data_path,
        )

        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "row_id" in result.columns
        assert len(result) == 5
        ids = result["row_id"].to_list()
        assert ids == sorted(ids)


class TestSortKeysWithDeletes:
    """Sort keys preserved through delete/update operations."""

    def test_sorted_rewrite_after_deletes(self, make_write_catalog):
        """Delete rows then rewrite -> still sorted."""
        cat = make_write_catalog()
        for i in range(3):
            df = pl.DataFrame({"id": [i * 10 + j for j in range(10)]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("id", "ASC")],
            data_path=cat.data_path,
        )

        delete_ducklake(
            cat.metadata_path, "t", pl.col("id") < 5,
            data_path=cat.data_path,
        )

        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        ids = result["id"].to_list()
        assert ids == sorted(ids)
        assert min(ids) >= 5


class TestSortKeysReset:
    """Reset sort keys and verify behavior."""

    def test_reset_then_rewrite_unsorted(self, make_write_catalog):
        """Set sort keys, reset, rewrite -> output may not be sorted."""
        cat = make_write_catalog()
        for i in range(3):
            df = pl.DataFrame({"id": [30 - i * 10 - j for j in range(10)]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("id", "ASC")],
            data_path=cat.data_path,
        )
        alter_ducklake_reset_sort_keys(
            cat.metadata_path, "t",
            data_path=cat.data_path,
        )

        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 30  # All data preserved

    def test_change_sort_key_column(self, make_write_catalog):
        """Change sort key from one column to another."""
        cat = make_write_catalog()
        df = pl.DataFrame({
            "id": [3, 1, 2],
            "name": ["charlie", "alice", "bob"],
        })
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        # Add more data
        df2 = pl.DataFrame({"id": [5, 4], "name": ["eve", "dave"]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        # Sort by name
        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("name", "ASC")],
            data_path=cat.data_path,
        )
        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        names = result["name"].to_list()
        assert names == sorted(names)

        # Add more and change to sort by id
        df3 = pl.DataFrame({"id": [0], "name": ["aaa"]})
        write_ducklake(df3, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        alter_ducklake_set_sort_keys(
            cat.metadata_path, "t", [("id", "ASC")],
            data_path=cat.data_path,
        )
        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        result2 = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        ids = result2["id"].to_list()
        assert ids == sorted(ids)


class TestSortedTimeTravelInteraction:
    """Sort keys and time travel."""

    def test_time_travel_before_sort_key_set(self, ducklake_catalog_sqlite):
        """Time travel to before sort keys were set."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, score INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 30), (1, 10), (2, 20)")
        snap_before = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]
        cat.close()

        alter_ducklake_set_sort_keys(cat.metadata_path, "t", [("id", "ASC")])

        # More data
        write_ducklake(
            pl.DataFrame({"id": pl.Series([5, 4], dtype=pl.Int32),
                          "score": pl.Series([50, 40], dtype=pl.Int32)}),
            cat.metadata_path, "t", mode="append",
        )

        rewrite_data_files_ducklake(cat.metadata_path, "t")

        # Current: sorted
        result_current = read_ducklake(cat.metadata_path, "t")
        assert len(result_current) == 5

        # Time travel: pre-sort snapshot
        result_old = read_ducklake(cat.metadata_path, "t",
                                    snapshot_version=snap_before)
        assert len(result_old) == 3
        assert set(result_old["id"].to_list()) == {1, 2, 3}
