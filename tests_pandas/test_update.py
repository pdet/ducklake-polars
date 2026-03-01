"""Update operation read tests for ducklake-pandas.

DuckLake implements UPDATE as DELETE + INSERT. These tests verify that
reading tables after UPDATE operations returns correct results.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ducklake_pandas import read_ducklake


class TestBasicUpdate:
    """Test reading tables after basic UPDATE operations."""

    def test_read_after_update(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'old1'), (2, 'old2'), (3, 'old3')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'new2' WHERE a = 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["old1", "new2", "old3"]

    def test_read_after_update_all(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'old1'), (2, 'old2'), (3, 'old3')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'updated'")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["updated", "updated", "updated"]

    def test_read_after_update_expression(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 10), (2, 20), (3, 30)"
        )
        cat.execute("UPDATE ducklake.test SET b = b + 100 WHERE a >= 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == [10, 120, 130]

    def test_read_after_multiple_updates(self, ducklake_catalog):
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
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3, 4, 5]
        assert result["b"].tolist() == ["first", "second", "orig", "orig", "orig"]


class TestUpdateTimeTravelRead:
    """Test time travel reads around UPDATE operations."""

    def test_time_travel_before_update(self, ducklake_catalog):
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

        result_before = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        )
        result_before = result_before.sort_values("a").reset_index(drop=True)
        assert result_before["a"].tolist() == [1, 2, 3]
        assert result_before["b"].tolist() == ["old1", "old2", "old3"]

        result_latest = read_ducklake(cat.metadata_path, "test")
        result_latest = result_latest.sort_values("a").reset_index(drop=True)
        assert result_latest["a"].tolist() == [1, 2, 3]
        assert result_latest["b"].tolist() == ["old1", "new2", "old3"]

    def test_time_travel_between_updates(self, ducklake_catalog):
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

        r1 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap1)
        r1 = r1.sort_values("a").reset_index(drop=True)
        assert r1["b"].tolist() == ["orig", "orig", "orig"]

        r2 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap2)
        r2 = r2.sort_values("a").reset_index(drop=True)
        assert r2["b"].tolist() == ["upd1", "orig", "orig"]

        r_latest = read_ducklake(cat.metadata_path, "test")
        r_latest = r_latest.sort_values("a").reset_index(drop=True)
        assert r_latest["b"].tolist() == ["upd1", "upd2", "orig"]


class TestUpdateMultiFile:
    """Test UPDATE across multiple data files."""

    def test_update_spanning_multiple_files(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'old'), (2, 'old')"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (3, 'old'), (4, 'old')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'new' WHERE a = 2 OR a = 3")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 4
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3, 4]
        assert result["b"].tolist() == ["old", "new", "new", "old"]

    def test_update_one_file_only(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'f1'), (2, 'f1'), (3, 'f1')"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (10, 'f2'), (11, 'f2'), (12, 'f2')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'upd' WHERE a <= 3")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 6
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3, 10, 11, 12]
        assert result["b"].tolist() == ["upd", "upd", "upd", "f2", "f2", "f2"]


class TestUpdateWithPartition:
    """Test UPDATE on partitioned tables."""

    def test_update_partitioned_table(self, ducklake_catalog):
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
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [2, 3, 10]
        assert result["b"].tolist() == ["y", "x", "x"]

    def test_update_partition_column(self, ducklake_catalog):
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
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["z", "y", "x"]
