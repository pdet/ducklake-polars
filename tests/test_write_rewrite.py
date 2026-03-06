"""Tests for rewrite_data_files — compaction / data file consolidation.

Covers:
  - Basic rewrite: multiple small files -> one consolidated file
  - Rewrite with deletion vectors -> deleted rows removed
  - Rewrite with schema evolution (add/drop/rename columns)
  - Rewrite with partitioned tables
  - Rewrite with sort keys (output respects sort order)
  - No-op rewrite (single file, no deletes -> returns -1)
  - Rewrite after repeated insert+delete cycles
  - Stats correctness after rewrite
  - Time travel still works after rewrite
  - Various data types survive rewrite
"""

from __future__ import annotations

import datetime
import random

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    read_ducklake,
    write_ducklake,
    rewrite_data_files_ducklake,
    alter_ducklake_set_sort_keys,
    alter_ducklake_set_partitioned_by,
)


class TestRewriteBasic:
    """Basic rewrite_data_files behavior."""

    def test_rewrite_merges_multiple_files(self, make_write_catalog):
        """Multiple small appends -> rewrite -> single consolidated file."""
        cat = make_write_catalog()
        for i in range(5):
            df = pl.DataFrame({
                "id": [i * 10 + j for j in range(10)],
                "val": [f"v{i * 10 + j}" for j in range(10)],
            })
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        result_before = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result_before) == 50

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t",
                                           data_path=cat.data_path)
        assert snap > 0

        result_after = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result_after) == 50
        assert_frame_equal(result_before.sort("id"), result_after.sort("id"))

    def test_rewrite_noop_single_file(self, make_write_catalog):
        """Single file with no deletes -> rewrite returns -1 (no-op)."""
        cat = make_write_catalog()
        df = pl.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t",
                                           data_path=cat.data_path)
        assert snap == -1

    def test_rewrite_idempotent(self, make_write_catalog):
        """Rewrite twice -> second should be no-op."""
        cat = make_write_catalog()
        for i in range(3):
            df = pl.DataFrame({"id": [i], "val": [f"v{i}"]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        snap1 = rewrite_data_files_ducklake(cat.metadata_path, "t",
                                            data_path=cat.data_path)
        assert snap1 > 0

        snap2 = rewrite_data_files_ducklake(cat.metadata_path, "t",
                                            data_path=cat.data_path)
        assert snap2 == -1


class TestRewriteWithDeletes:
    """Rewrite removes deleted rows via DuckDB DML, then Polars rewrite."""

    def test_rewrite_removes_deleted_rows(self, ducklake_catalog):
        """Insert rows, delete some via DuckDB, then rewrite via Polars."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.t AS "
            "SELECT i AS id, 'v' || i AS val FROM range(50) t(i)"
        )
        cat.execute("DELETE FROM ducklake.t WHERE id < 10")
        # Append more to create multiple files
        cat.execute(
            "INSERT INTO ducklake.t "
            "SELECT i AS id, 'v' || i AS val FROM range(50, 60) t(i)"
        )
        cat.close()

        result_before = read_ducklake(cat.metadata_path, "t")
        assert len(result_before) == 50  # 50 - 10 + 10

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result_after = read_ducklake(cat.metadata_path, "t")
        assert len(result_after) == 50
        assert_frame_equal(result_before.sort("id"), result_after.sort("id"))

    def test_rewrite_after_multiple_deletes(self, ducklake_catalog):
        """Multiple delete operations -> rewrite cleans all up."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.t AS "
            "SELECT i AS id FROM range(50) t(i)"
        )
        cat.execute("DELETE FROM ducklake.t WHERE id < 5")
        cat.execute("DELETE FROM ducklake.t WHERE id >= 45")
        # Append to create multiple files
        cat.execute("INSERT INTO ducklake.t VALUES (100)")
        cat.close()

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 41  # 50 - 5 - 5 + 1

    def test_rewrite_after_update(self, ducklake_catalog):
        """Update + rewrite preserves updated values."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.t (id INTEGER, name VARCHAR)"
        )
        cat.execute(
            "INSERT INTO ducklake.t VALUES (1, 'alice'), (2, 'bob'), (3, 'carol')"
        )
        cat.execute("UPDATE ducklake.t SET name = 'ALICE' WHERE id = 1")
        cat.execute("INSERT INTO ducklake.t VALUES (4, 'dave')")
        cat.close()

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 4
        alice = result.filter(pl.col("id") == 1)
        assert alice["name"][0] == "ALICE"


class TestRewriteWithSchemaEvolution:
    """Rewrite with schema changes (add/drop/rename columns)."""

    def test_rewrite_after_add_column(self, ducklake_catalog):
        """Add column after initial write -> rewrite fills with NULL."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN extra INTEGER")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c', 10), (4, 'd', 20)")
        cat.close()

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 4

        old_rows = result.filter(pl.col("id") <= 2).sort("id")
        assert old_rows["extra"].null_count() == 2

        new_rows = result.filter(pl.col("id") > 2).sort("id")
        assert new_rows["extra"].to_list() == [10, 20]

    def test_rewrite_after_drop_column(self, ducklake_catalog):
        """Drop column -> rewrite removes it from output."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR, temp INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a', 100), (2, 'b', 200)")
        cat.execute("ALTER TABLE ducklake.t DROP COLUMN temp")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c'), (4, 'd')")
        cat.close()

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t")
        assert "temp" not in result.columns
        assert len(result) == 4

    def test_rewrite_after_rename_column(self, ducklake_catalog):
        """Rename column -> rewrite uses new name."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, old_name VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN old_name TO new_name")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c')")
        cat.close()

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t")
        assert "new_name" in result.columns
        assert "old_name" not in result.columns
        assert len(result) == 3


class TestRewriteWithPartitions:
    """Rewrite with partitioned tables."""

    def test_rewrite_partitioned_table(self, make_write_catalog):
        """Partitioned table -> rewrite produces one file per partition."""
        cat = make_write_catalog()
        # Create table and set partitioning
        df_init = pl.DataFrame({"id": [0], "region": ["US"], "val": ["US_0"]})
        write_ducklake(df_init, cat.metadata_path, "t", data_path=cat.data_path)
        alter_ducklake_set_partitioned_by(
            cat.metadata_path, "t", ["region"], data_path=cat.data_path,
        )

        for region in ["EU", "AP"]:
            for batch in range(3):
                df = pl.DataFrame({
                    "id": [batch],
                    "region": [region],
                    "val": [f"{region}_{batch}"],
                })
                write_ducklake(df, cat.metadata_path, "t", mode="append",
                              data_path=cat.data_path)
        # More US data
        for batch in range(1, 3):
            df = pl.DataFrame({"id": [batch], "region": ["US"], "val": [f"US_{batch}"]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        result_before = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t",
                                           data_path=cat.data_path)
        assert snap > 0

        result_after = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert_frame_equal(
            result_before.sort("id", "region"),
            result_after.sort("id", "region"),
        )

    def test_rewrite_partitioned_with_deletes(self, ducklake_catalog):
        """Partitioned table + deletes -> rewrite cleans up."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.t (id INTEGER, region VARCHAR)"
        )
        cat.execute(
            "INSERT INTO ducklake.t "
            "SELECT i, CASE WHEN i < 10 THEN 'US' ELSE 'EU' END "
            "FROM range(20) t(i)"
        )
        cat.execute("INSERT INTO ducklake.t VALUES (20, 'US'), (21, 'US')")
        cat.execute("DELETE FROM ducklake.t WHERE id < 5")
        cat.close()

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t")
        assert len(result) == 17  # 22 - 5


class TestRewriteWithSortKeys:
    """Rewrite respects sort key ordering."""

    def test_rewrite_preserves_sort_order(self, make_write_catalog):
        """Table with sort keys -> rewritten file is sorted."""
        cat = make_write_catalog()
        # Write data in reverse order across multiple appends
        for i in range(3):
            base = 30 - i * 10
            df = pl.DataFrame({
                "id": [base, base - 1, base - 2],
                "val": [f"v{base}", f"v{base - 1}", f"v{base - 2}"],
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
        assert ids == sorted(ids), "Rewritten data should be sorted by id ASC"


class TestRewriteInsertDeleteCycles:
    """Rewrite after repeated insert+delete cycles."""

    def test_rewrite_after_insert_delete_cycles(self, ducklake_catalog):
        """Repeated insert+delete -> rewrite -> clean state."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.t AS "
            "SELECT i AS id, 'v' || i AS val FROM range(20) t(i)"
        )
        cat.execute("DELETE FROM ducklake.t WHERE id < 5")
        cat.execute(
            "INSERT INTO ducklake.t "
            "SELECT i AS id, 'v' || i AS val FROM range(20, 30) t(i)"
        )
        cat.execute("DELETE FROM ducklake.t WHERE id >= 27")
        cat.close()

        expected_count = 22  # 20 - 5 + 10 - 3
        result_before = read_ducklake(cat.metadata_path, "t")
        assert len(result_before) == expected_count

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap > 0

        result_after = read_ducklake(cat.metadata_path, "t")
        assert len(result_after) == expected_count
        assert_frame_equal(result_before.sort("id"), result_after.sort("id"))


class TestRewriteStatsCorrectness:
    """Rewrite preserves data correctly."""

    def test_rewrite_preserves_all_values(self, make_write_catalog):
        """All values survive rewrite unchanged."""
        cat = make_write_catalog()
        random.seed(42)
        values = [random.randint(-1000, 1000) for _ in range(100)]
        df = pl.DataFrame({"id": list(range(100)), "val": values})
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({
            "id": list(range(100, 150)),
            "val": [random.randint(-1000, 1000) for _ in range(50)],
        })
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        result_before = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        result_after = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert_frame_equal(result_before.sort("id"), result_after.sort("id"))


class TestRewriteTimeTravelInteraction:
    """Time travel works after rewrite (old snapshots preserved)."""

    def test_time_travel_after_rewrite(self, ducklake_catalog_sqlite):
        """Old snapshots still readable after rewrite."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3)")
        snap1 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute("INSERT INTO ducklake.t VALUES (4), (5), (6)")
        cat.close()

        snap_rewrite = rewrite_data_files_ducklake(cat.metadata_path, "t")
        assert snap_rewrite > 0

        result_current = read_ducklake(cat.metadata_path, "t")
        assert len(result_current) == 6

        result_old = read_ducklake(cat.metadata_path, "t",
                                    snapshot_version=snap1)
        assert len(result_old) == 3
        assert set(result_old["id"].to_list()) == {1, 2, 3}

    def test_rewrite_does_not_break_intermediate_snapshots(
        self, ducklake_catalog_sqlite
    ):
        """Each intermediate snapshot is still accessible after rewrite."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        snapshots = []

        for i in range(4):
            vals = ", ".join(f"({i * 5 + j})" for j in range(5))
            cat.execute(f"INSERT INTO ducklake.t VALUES {vals}")
            snap = cat.fetchone(
                "SELECT * FROM ducklake_current_snapshot('ducklake')"
            )[0]
            snapshots.append((snap, (i + 1) * 5))

        cat.close()

        rewrite_data_files_ducklake(cat.metadata_path, "t")

        for snap, expected_count in snapshots:
            result = read_ducklake(cat.metadata_path, "t",
                                    snapshot_version=snap)
            assert len(result) == expected_count, (
                f"Snapshot {snap}: expected {expected_count}, got {len(result)}"
            )


class TestRewriteWithTypes:
    """Rewrite preserves various data types correctly."""

    def test_rewrite_mixed_types(self, make_write_catalog):
        """Various column types survive rewrite unchanged."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({
            "id": [1, 2],
            "name": ["alice", "bob"],
            "score": [3.14, 2.72],
            "active": [True, False],
            "created": [datetime.date(2024, 1, 1), datetime.date(2024, 6, 15)],
        })
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({
            "id": [3, 4],
            "name": ["carol", "dave"],
            "score": [1.41, 1.73],
            "active": [True, True],
            "created": [datetime.date(2024, 12, 25), datetime.date(2025, 1, 1)],
        })
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        result_before = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t",
                                           data_path=cat.data_path)
        assert snap > 0

        result_after = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert_frame_equal(result_before.sort("id"), result_after.sort("id"))

    def test_rewrite_with_nulls(self, make_write_catalog):
        """NULL values survive rewrite."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"id": [1, 2], "val": [None, "b"]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({"id": [3, 4], "val": ["c", None]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t",
                                           data_path=cat.data_path)
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 4
        assert result.sort("id")["val"].to_list() == [None, "b", "c", None]
