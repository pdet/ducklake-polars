"""Partition pruning tests for ducklake-polars."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from ducklake_polars import read_ducklake, scan_ducklake


class TestPartition:
    """Test reading partitioned tables."""

    def test_read_partitioned_table(self, ducklake_catalog):
        """Basic read of a partitioned table."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 2)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "x"]

    def test_partitioned_filter(self, ducklake_catalog):
        """Filter on partition column, verify correct data returned."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x')")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("b") == "x").collect()
        assert result.shape[0] == 2
        result = result.sort("a")
        assert result["a"].to_list() == [1, 3]
        assert result["b"].to_list() == ["x", "x"]

    def test_partitioned_multi_column(self, ducklake_catalog):
        """Multiple partition columns."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c INTEGER)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b, c)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'x', 10), (2, 'y', 20), (3, 'x', 10), (4, 'y', 30)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (4, 3)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "y"]
        assert result["c"].to_list() == [10, 20, 10, 30]

    def test_partitioned_with_stats(self, ducklake_catalog):
        """Verify statistics-based filtering works on partitioned data."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        # Insert in separate batches to create multiple files
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'x')")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'y'), (4, 'y')")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("a") > 2).collect()
        result = result.sort("a")
        assert result["a"].to_list() == [3, 4]
        assert result["b"].to_list() == ["y", "y"]

    def test_partitioned_integer_partition_column(self, ducklake_catalog):
        """Partition on an integer column."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 100), (2, 200), (3, 100)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 2)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == [100, 200, 100]

    def test_empty_partitioned_table(self, ducklake_catalog):
        """Read an empty partitioned table."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.close()
        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (0, 2)
        assert result.columns == ["a", "b"]

    def test_partitioned_with_delete(self, ducklake_catalog):
        """Delete rows from a partitioned table."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x')")
        cat.execute("DELETE FROM ducklake.test WHERE a = 1")
        cat.close()
        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result["a"].to_list() == [2, 3]
        assert result["b"].to_list() == ["y", "x"]


class TestPartitionAdvanced:
    """Test advanced partition scenarios."""

    def test_append_to_existing_partition(self, ducklake_catalog):
        """Two inserts into the same partitions; second appends to existing partitions."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y')")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'x'), (4, 'y')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (4, 2)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "y"]

    def test_three_partition_keys(self, ducklake_catalog):
        """Partition on three columns."""
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c INTEGER, d VARCHAR)"
        )
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b, c, d)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'x', 10, 'p'), (2, 'y', 20, 'q'), "
            "(3, 'x', 10, 'p'), (4, 'y', 30, 'r'), "
            "(5, 'z', 10, 'p')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (5, 4)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4, 5]
        assert result["b"].to_list() == ["x", "y", "x", "y", "z"]
        assert result["c"].to_list() == [10, 20, 10, 30, 10]
        assert result["d"].to_list() == ["p", "q", "p", "r", "p"]

    def test_filter_on_each_partition_key(self, ducklake_catalog):
        """Filter on each partition key individually in a three-key partition."""
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c INTEGER, d VARCHAR)"
        )
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b, c, d)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'x', 10, 'p'), (2, 'y', 20, 'q'), "
            "(3, 'x', 10, 'p'), (4, 'y', 30, 'r'), "
            "(5, 'z', 10, 'p')"
        )
        cat.close()

        # Filter on b only
        lf = scan_ducklake(cat.metadata_path, "test")
        result_b = lf.filter(pl.col("b") == "x").collect().sort("a")
        assert result_b["a"].to_list() == [1, 3]
        assert result_b["b"].to_list() == ["x", "x"]

        # Filter on c only
        lf = scan_ducklake(cat.metadata_path, "test")
        result_c = lf.filter(pl.col("c") == 10).collect().sort("a")
        assert result_c["a"].to_list() == [1, 3, 5]
        assert result_c["c"].to_list() == [10, 10, 10]

        # Filter on d only
        lf = scan_ducklake(cat.metadata_path, "test")
        result_d = lf.filter(pl.col("d") == "p").collect().sort("a")
        assert result_d["a"].to_list() == [1, 3, 5]
        assert result_d["d"].to_list() == ["p", "p", "p"]

    def test_large_partitioned_table(self, ducklake_catalog):
        """Insert 1000 rows into a partitioned table and verify counts."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, CASE WHEN i % 3 = 0 THEN 'a' "
            "WHEN i % 3 = 1 THEN 'b' ELSE 'c' END "
            "FROM range(0, 1000) t(i)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 1000

        # Filter on partition column and check counts
        lf = scan_ducklake(cat.metadata_path, "test")
        result_a = lf.filter(pl.col("b") == "a").collect()
        # i % 3 == 0 for i in 0..999: 0, 3, 6, ..., 999 -> 334 values
        assert result_a.shape[0] == 334

        lf = scan_ducklake(cat.metadata_path, "test")
        result_b = lf.filter(pl.col("b") == "b").collect()
        # i % 3 == 1 for i in 0..999: 1, 4, 7, ..., 997 -> 333 values
        assert result_b.shape[0] == 333

        lf = scan_ducklake(cat.metadata_path, "test")
        result_c = lf.filter(pl.col("b") == "c").collect()
        # i % 3 == 2 for i in 0..999: 2, 5, 8, ..., 998 -> 333 values
        assert result_c.shape[0] == 333


class TestPartitionWithSchemaEvolution:
    """Test partitioned tables combined with schema evolution."""

    def test_partition_add_column(self, ducklake_catalog):
        """Add a column to a partitioned table; old rows have NULL for the new column."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y')")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN c INTEGER")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'x', 100), (4, 'y', 200)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (4, 3)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "y"]
        # Old rows have c=NULL, new rows have c values
        assert result["c"].to_list() == [None, None, 100, 200]

    def test_partition_drop_non_partition_column(self, ducklake_catalog):
        """Drop a non-partition column from a partitioned table."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c INTEGER)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'x', 10), (2, 'y', 20)"
        )

        cat.execute("ALTER TABLE ducklake.test DROP COLUMN c")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'x'), (4, 'y')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a", "b"]
        assert result.shape == (4, 2)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "y"]


class TestPartitionWithUpdate:
    """Test UPDATE operations on partitioned tables."""

    def test_update_non_partition_column(self, ducklake_catalog):
        """UPDATE a non-partition column within a partitioned table."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x')"
        )
        cat.execute("UPDATE ducklake.test SET a = 10 WHERE a = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        result = result.sort("a")
        assert result["a"].to_list() == [2, 3, 10]
        assert result["b"].to_list() == ["y", "x", "x"]

    def test_update_partition_column_value(self, ducklake_catalog):
        """UPDATE the partition column itself, moving a row to a different partition."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y')")
        cat.execute("UPDATE ducklake.test SET b = 'z' WHERE a = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 2
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == ["z", "y"]


class TestPartitionTimeTravelRead:
    """Test time travel reads on partitioned tables."""

    def test_partition_time_travel(self, ducklake_catalog):
        """Read a partitioned table at a past snapshot; only batch 1 visible."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        # Batch 1
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y')")
        snap1 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        # Batch 2
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'x'), (4, 'z')")
        cat.close()

        # Read at snap1: only batch 1
        result_v1 = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap1
        )
        assert result_v1.shape[0] == 2
        result_v1 = result_v1.sort("a")
        assert result_v1["a"].to_list() == [1, 2]
        assert result_v1["b"].to_list() == ["x", "y"]

        # Read latest: both batches
        result_latest = read_ducklake(cat.metadata_path, "test")
        assert result_latest.shape[0] == 4
        result_latest = result_latest.sort("a")
        assert result_latest["a"].to_list() == [1, 2, 3, 4]
        assert result_latest["b"].to_list() == ["x", "y", "x", "z"]


class TestNonIdentityPartition:
    """Test non-identity partition transforms (year, month).

    Non-identity partition data is still readable (the partition column is
    stored in the Parquet file), even though file pruning does not use
    the partition transform.
    """

    def test_year_partition_read(self, ducklake_catalog):
        """Read a table partitioned by year(ts)."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (id INTEGER, ts TIMESTAMP)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (year(ts))")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, '2020-03-15 10:00:00'), (2, '2020-07-20 12:00:00'), "
            "(3, '2021-01-10 08:00:00'), (4, '2021-11-25 16:00:00')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (4, 2)
        result = result.sort("id")
        assert result["id"].to_list() == [1, 2, 3, 4]

    def test_month_partition_read(self, ducklake_catalog):
        """Read a table partitioned by month(ts)."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (id INTEGER, ts TIMESTAMP)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (month(ts))")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, '2021-01-15 10:00:00'), (2, '2021-02-20 12:00:00'), "
            "(3, '2021-03-10 08:00:00'), (4, '2021-01-25 16:00:00')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (4, 2)
        result = result.sort("id")
        assert result["id"].to_list() == [1, 2, 3, 4]

    def test_year_partition_filter(self, ducklake_catalog):
        """Filter on ts column in a year(ts)-partitioned table."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (id INTEGER, ts TIMESTAMP)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (year(ts))")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, '2020-03-15 10:00:00'), (2, '2020-07-20 12:00:00'), "
            "(3, '2021-01-10 08:00:00'), (4, '2021-11-25 16:00:00')"
        )
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(
            pl.col("ts") < datetime(2021, 1, 1)
        ).collect().sort("id")
        assert result["id"].to_list() == [1, 2]
