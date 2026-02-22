"""Update operation read tests for ducklake-polars.

DuckLake implements UPDATE as DELETE + INSERT. These tests verify that
reading tables after UPDATE operations returns correct results.
"""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake, scan_ducklake


class TestBasicUpdate:
    """Test reading tables after basic UPDATE operations."""

    def test_read_after_update(self, ducklake_catalog):
        """Single-row update: only the targeted row changes."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'old1'), (2, 'old2'), (3, 'old3')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'new2' WHERE a = 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["old1", "new2", "old3"]

    def test_read_after_update_all(self, ducklake_catalog):
        """UPDATE without WHERE clause updates every row."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'old1'), (2, 'old2'), (3, 'old3')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'updated'")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["updated", "updated", "updated"]

    def test_read_after_update_expression(self, ducklake_catalog):
        """UPDATE with an arithmetic expression on the column value."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 10), (2, 20), (3, 30)"
        )
        cat.execute("UPDATE ducklake.test SET b = b + 100 WHERE a >= 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == [10, 120, 130]

    def test_read_after_multiple_updates(self, ducklake_catalog):
        """Multiple sequential UPDATEs targeting different rows."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'orig'), (2, 'orig'), (3, 'orig'), (4, 'orig'), (5, 'orig')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'first' WHERE a = 1")
        cat.execute("UPDATE ducklake.test SET b = 'second' WHERE a = 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 5
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4, 5]
        assert result["b"].to_list() == [
            "first",
            "second",
            "orig",
            "orig",
            "orig",
        ]


class TestUpdateTimeTravelRead:
    """Test time travel reads around UPDATE operations."""

    def test_time_travel_before_update(self, ducklake_catalog):
        """Reading at a snapshot before the UPDATE returns original values."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'old1'), (2, 'old2'), (3, 'old3')"
        )
        snap_before = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute("UPDATE ducklake.test SET b = 'new2' WHERE a = 2")
        cat.close()

        # Read at snapshot before the update: all original values
        result_before = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        )
        result_before = result_before.sort("a")
        assert result_before["a"].to_list() == [1, 2, 3]
        assert result_before["b"].to_list() == ["old1", "old2", "old3"]

        # Read at latest: updated value
        result_latest = read_ducklake(cat.metadata_path, "test")
        result_latest = result_latest.sort("a")
        assert result_latest["a"].to_list() == [1, 2, 3]
        assert result_latest["b"].to_list() == ["old1", "new2", "old3"]

    def test_time_travel_between_updates(self, ducklake_catalog):
        """Reading at snapshots between successive UPDATEs returns correct intermediate states."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'orig'), (2, 'orig'), (3, 'orig')"
        )
        snap1 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute("UPDATE ducklake.test SET b = 'upd1' WHERE a = 1")
        snap2 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute("UPDATE ducklake.test SET b = 'upd2' WHERE a = 2")
        cat.close()

        # snap1: all original
        r1 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap1)
        r1 = r1.sort("a")
        assert r1["b"].to_list() == ["orig", "orig", "orig"]

        # snap2: a=1 updated, rest original
        r2 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap2)
        r2 = r2.sort("a")
        assert r2["b"].to_list() == ["upd1", "orig", "orig"]

        # latest: a=1 and a=2 updated
        r_latest = read_ducklake(cat.metadata_path, "test")
        r_latest = r_latest.sort("a")
        assert r_latest["b"].to_list() == ["upd1", "upd2", "orig"]


class TestUpdateWithFilter:
    """Test scan_ducklake filters after UPDATE operations."""

    def test_filter_after_update(self, ducklake_catalog):
        """Filter excludes rows whose updated value no longer matches."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i AS a, i * 10 AS b FROM range(10) t(i)"
        )
        # Set b = -1 for rows where a < 3
        cat.execute("UPDATE ducklake.test SET b = -1 WHERE a < 3")
        cat.close()

        # Filter for b > 0 should exclude a=0,1,2 (now b=-1)
        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("b") > 0)
            .collect()
        )
        result = result.sort("a")
        assert result["a"].to_list() == [3, 4, 5, 6, 7, 8, 9]
        assert result["b"].to_list() == [30, 40, 50, 60, 70, 80, 90]

    def test_filter_on_updated_column(self, ducklake_catalog):
        """Filter on the column that was updated."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'keep'), (2, 'keep'), (3, 'keep'), (4, 'keep')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'drop' WHERE a <= 2")
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("b") == "keep")
            .collect()
        )
        result = result.sort("a")
        assert result["a"].to_list() == [3, 4]
        assert result["b"].to_list() == ["keep", "keep"]


class TestUpdateMultiFile:
    """Test UPDATE across multiple data files."""

    def test_update_spanning_multiple_files(self, ducklake_catalog):
        """UPDATE affects rows originating from different INSERT batches (different files)."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        # Two separate INSERTs produce two data files
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'old'), (2, 'old')"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (3, 'old'), (4, 'old')"
        )
        # Update rows from both files
        cat.execute("UPDATE ducklake.test SET b = 'new' WHERE a = 2 OR a = 3")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 4
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["old", "new", "new", "old"]

    def test_update_one_file_only(self, ducklake_catalog):
        """UPDATE targeting only one file's range leaves the other file untouched."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        # File 1: a in [1, 2, 3]
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'f1'), (2, 'f1'), (3, 'f1')"
        )
        # File 2: a in [10, 11, 12]
        cat.execute(
            "INSERT INTO ducklake.test VALUES (10, 'f2'), (11, 'f2'), (12, 'f2')"
        )
        # Only update rows from file 1's range
        cat.execute("UPDATE ducklake.test SET b = 'upd' WHERE a <= 3")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 6
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3, 10, 11, 12]
        assert result["b"].to_list() == ["upd", "upd", "upd", "f2", "f2", "f2"]


class TestUpdateWithPartition:
    """Test UPDATE on partitioned tables."""

    def test_update_partitioned_table(self, ducklake_catalog):
        """UPDATE a non-partition column within a single partition."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x')"
        )
        cat.execute("UPDATE ducklake.test SET a = 10 WHERE b = 'x' AND a = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        result = result.sort("a")
        assert result["a"].to_list() == [2, 3, 10]
        assert result["b"].to_list() == ["y", "x", "x"]

    def test_update_partition_column(self, ducklake_catalog):
        """UPDATE the partition column itself, changing a row's partition."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'z' WHERE a = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["z", "y", "x"]
