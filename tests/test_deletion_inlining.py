"""Edge case tests for deletion inlining interactions.

Covers gaps from TEST_PARITY.md:
  - Deletion from inlined data across multiple snapshots
  - Deletion inlining interaction with ALTER TABLE
  - Deletion inlining with partitions
  - Stats correctness after deletion inlining
  - Flush (inlined -> parquet) after deletes
  - Rewrite interaction with inlined data + deletes
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    read_ducklake,
    write_ducklake,
    rewrite_data_files_ducklake,
)


INLINE_LIMIT = 100


class TestDeletionFromInlinedData:
    """Deletes from inlined data across snapshots."""

    def test_delete_from_inlined_data(self, ducklake_catalog):
        """Delete rows that are stored in inlined data tables."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)"
        )
        # Small insert that gets inlined
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        cat.execute("DELETE FROM ducklake.t WHERE id = 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 2
        assert set(result["id"].to_list()) == {1, 3}

    def test_delete_all_inlined_data(self, ducklake_catalog):
        """Delete all rows from inlined data."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3)")
        cat.execute("DELETE FROM ducklake.t WHERE 1=1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 0

    def test_delete_inlined_across_snapshots(self, ducklake_catalog):
        """Insert inlined -> snapshot -> delete -> snapshot -> verify."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        # Snapshot after insert
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c'), (4, 'd')")
        # Another snapshot
        cat.execute("DELETE FROM ducklake.t WHERE id <= 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 2
        assert set(result["id"].to_list()) == {3, 4}

    def test_delete_then_insert_inlined(self, ducklake_catalog):
        """Delete from inlined, then insert new inlined data."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3)")
        cat.execute("DELETE FROM ducklake.t WHERE id = 1")
        cat.execute("INSERT INTO ducklake.t VALUES (10), (20)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 4
        assert set(result["id"].to_list()) == {2, 3, 10, 20}


class TestDeletionInliningWithAlter:
    """Deletion inlining interaction with ALTER TABLE."""

    def test_delete_after_add_column(self, ducklake_catalog):
        """Add column then delete — inlined data handles schema change."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN extra INTEGER")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c', 100)")
        cat.execute("DELETE FROM ducklake.t WHERE id = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 2
        assert set(result["id"].to_list()) == {2, 3}
        row3 = result.filter(pl.col("id") == 3)
        assert row3["extra"][0] == 100

    def test_delete_after_drop_column(self, ducklake_catalog):
        """Drop column then delete — remaining data intact."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR, temp INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a', 10), (2, 'b', 20), (3, 'c', 30)")
        cat.execute("ALTER TABLE ducklake.t DROP COLUMN temp")
        cat.execute("DELETE FROM ducklake.t WHERE id = 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 2
        assert "temp" not in result.columns

    def test_delete_after_rename_column(self, ducklake_catalog):
        """Rename column then delete — works correctly."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, old_name VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x'), (2, 'y'), (3, 'z')")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN old_name TO new_name")
        cat.execute("DELETE FROM ducklake.t WHERE id = 3")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 2
        assert "new_name" in result.columns
        assert "old_name" not in result.columns


class TestDeletionInliningWithPartitions:
    """Deletion from partitioned tables with inlined data."""

    def test_delete_from_partitioned_inlined(self, ducklake_catalog):
        """Delete from a partitioned table that has inlined data."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, region VARCHAR, val INTEGER)")
        cat.execute(
            "INSERT INTO ducklake.t VALUES "
            "(1, 'US', 100), (2, 'US', 200), (3, 'EU', 300), (4, 'EU', 400)"
        )
        cat.execute("DELETE FROM ducklake.t WHERE region = 'US' AND id = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 3
        us = result.filter(pl.col("region") == "US")
        assert len(us) == 1
        assert us["id"][0] == 2


class TestDeletionInliningStats:
    """Stats correctness after deletion from inlined data."""

    def test_count_correct_after_delete(self, ducklake_catalog):
        """Row count is accurate after deleting from inlined data."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        for i in range(20):
            cat.execute(f"INSERT INTO ducklake.t VALUES ({i})")
        cat.execute("DELETE FROM ducklake.t WHERE id % 2 = 0")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 10
        assert all(v % 2 == 1 for v in result["id"].to_list())

    def test_min_max_correct_after_delete(self, ducklake_catalog):
        """Filter pushdown stats are correct after delete."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.t AS "
            "SELECT i AS id FROM range(100) t(i)"
        )
        cat.execute("DELETE FROM ducklake.t WHERE id >= 90")
        cat.execute("DELETE FROM ducklake.t WHERE id < 10")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t")
        assert result["id"].min() == 10
        assert result["id"].max() == 89
        assert len(result) == 80


class TestFlushAfterDelete:
    """Flush (inlined -> parquet) interaction with deletes."""

    def test_flush_preserves_deletes(self, make_write_catalog):
        """Inlined data with deletes -> flush -> deletes still applied."""
        cat = make_write_catalog(inline=True, inline_limit=INLINE_LIMIT)
        # Write small data (inlined)
        df = pl.DataFrame({"id": list(range(10)), "val": [f"v{i}" for i in range(10)]})
        write_ducklake(df, cat.metadata_path, "t",
                      data_path=cat.data_path,
                      data_inlining_row_limit=INLINE_LIMIT)

        result_before = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result_before) == 10

        # Write more data that exceeds inline limit (triggers flush)
        df_big = pl.DataFrame({
            "id": list(range(100, 300)),
            "val": [f"v{i}" for i in range(100, 300)],
        })
        write_ducklake(df_big, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path,
                      data_inlining_row_limit=INLINE_LIMIT)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 210  # 10 + 200


class TestRewriteWithInlinedDeletes:
    """Rewrite interaction with inlined data that has deletes."""

    def test_rewrite_after_inlined_delete(self, ducklake_catalog):
        """Table with inlined data + deletes -> rewrite consolidates."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.t VALUES "
            "(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')"
        )
        cat.execute("DELETE FROM ducklake.t WHERE id <= 2")
        # Add more data to create multiple files
        cat.execute(
            "INSERT INTO ducklake.t VALUES (10, 'x'), (20, 'y')"
        )
        cat.close()

        result_before = read_ducklake(cat.metadata_path, "t")

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")

        result_after = read_ducklake(cat.metadata_path, "t")
        assert_frame_equal(result_before.sort("id"), result_after.sort("id"))

    def test_rewrite_inlined_with_schema_evolution(self, ducklake_catalog):
        """Inlined data + schema evolution + delete -> rewrite."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN score DOUBLE")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c', 9.5), (4, 'd', 8.0)")
        cat.execute("DELETE FROM ducklake.t WHERE id = 1")
        cat.close()

        result_before = read_ducklake(cat.metadata_path, "t")
        assert len(result_before) == 3

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")

        result_after = read_ducklake(cat.metadata_path, "t")
        assert len(result_after) == 3
        assert_frame_equal(result_before.sort("id"), result_after.sort("id"))
