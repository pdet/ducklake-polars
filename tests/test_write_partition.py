"""Tests for partitioned writes in ducklake-dataframe."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    alter_ducklake_set_partitioned_by,
    create_ducklake_table,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# ALTER TABLE SET PARTITIONED BY
# ---------------------------------------------------------------------------


class TestSetPartitionedBy:
    """Test alter_ducklake_set_partitioned_by."""

    def test_set_partitioned_by_single_column(self, make_write_catalog):
        """Set partitioning on a single column."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )

        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        # Check metadata
        pi = cat.query_one("SELECT partition_id, table_id FROM ducklake_partition_info")
        assert pi is not None
        partition_id = pi[0]

        pc = cat.query_all(
            "SELECT partition_key_index, column_id, transform "
            "FROM ducklake_partition_column WHERE partition_id = ?",
            [partition_id],
        )
        assert len(pc) == 1
        assert pc[0][0] == 0  # partition_key_index
        assert pc[0][2] == "identity"

        # Schema version bumped
        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert "altered_table" in change[0]

    def test_set_partitioned_by_multi_column(self, make_write_catalog):
        """Set partitioning on multiple columns."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test",
            {"a": pl.Int64(), "b": pl.String(), "c": pl.Int64()},
        )

        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b", "c"])

        pcs = cat.query_all(
            "SELECT partition_key_index, transform "
            "FROM ducklake_partition_column ORDER BY partition_key_index"
        )
        assert len(pcs) == 2
        assert pcs[0][0] == 0
        assert pcs[1][0] == 1
        assert pcs[0][1] == "identity"
        assert pcs[1][1] == "identity"

    def test_set_partitioned_by_nonexistent_table(self, make_write_catalog):
        cat = make_write_catalog()
        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_set_partitioned_by(cat.metadata_path, "missing", ["a"])

    def test_set_partitioned_by_nonexistent_column(self, make_write_catalog):
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64()},
        )
        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["missing"])

    def test_set_partitioned_by_duckdb_reads(self, make_write_catalog):
        """DuckDB can read the partition spec created by ducklake-dataframe."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        # Insert after partitioning
        df2 = pl.DataFrame({"a": [4, 5], "b": ["z", "x"]})
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 5
        assert sorted(pdf["a"].to_list()) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Partitioned INSERT
# ---------------------------------------------------------------------------


class TestPartitionedInsert:
    """Test writing data to partitioned tables."""

    def test_basic_partitioned_insert(self, make_write_catalog):
        """Insert into a partitioned table creates per-partition files."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "x"]

    def test_partitioned_insert_hive_paths(self, make_write_catalog):
        """Verify Hive-style directory layout."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        # Check data file paths in catalog
        paths = cat.query_all(
            "SELECT path FROM ducklake_data_file WHERE partition_id IS NOT NULL"
        )

        path_strs = [p[0] for p in paths]
        assert any("b=x/" in p for p in path_strs)
        assert any("b=y/" in p for p in path_strs)

    def test_partitioned_insert_multi_column(self, make_write_catalog):
        """Multi-column partition produces nested Hive paths."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test",
            {"a": pl.Int64(), "b": pl.String(), "c": pl.Int64()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b", "c"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"], "c": [10, 20, 10]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "x"]
        assert result["c"].to_list() == [10, 20, 10]

        # Check nested paths
        paths = cat.query_all(
            "SELECT path FROM ducklake_data_file WHERE partition_id IS NOT NULL"
        )

        path_strs = [p[0] for p in paths]
        assert any("b=x/c=10/" in p for p in path_strs)
        assert any("b=y/c=20/" in p for p in path_strs)

    def test_partitioned_insert_partition_values_registered(self, make_write_catalog):
        """Partition values are correctly registered in ducklake_file_partition_value."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        pvs = cat.query_all(
            "SELECT data_file_id, partition_key_index, partition_value "
            "FROM ducklake_file_partition_value ORDER BY data_file_id"
        )

        # Should have one entry per partition file
        assert len(pvs) >= 2
        values = {pv[2] for pv in pvs}
        assert "x" in values
        assert "y" in values

    def test_partitioned_insert_partition_id_on_files(self, make_write_catalog):
        """Data files reference the correct partition_id."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        pi = cat.query_one(
            "SELECT partition_id FROM ducklake_partition_info"
        )
        dfs = cat.query_all(
            "SELECT partition_id FROM ducklake_data_file WHERE partition_id IS NOT NULL"
        )

        # All partitioned files should reference the same partition_id
        for (part_id,) in dfs:
            assert part_id == pi[0]

    def test_partitioned_insert_column_stats(self, make_write_catalog):
        """Column statistics are computed per-partition file."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 5, 3], "b": ["x", "x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        # Stats for partition b=x: min=1, max=5
        # Stats for partition b=y: min=3, max=3
        stats = cat.query_all(
            "SELECT data_file_id, column_id, min_value, max_value "
            "FROM ducklake_file_column_stats "
            "WHERE column_id = (SELECT column_id FROM ducklake_column WHERE column_name='a' LIMIT 1) "
            "ORDER BY data_file_id"
        )

        assert len(stats) >= 2
        mins = {s[2] for s in stats}
        maxs = {s[3] for s in stats}
        assert "1" in mins
        assert "3" in mins
        assert "5" in maxs
        assert "3" in maxs

    def test_partitioned_insert_multiple_batches(self, make_write_catalog):
        """Multiple inserts into a partitioned table append correctly."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        write_ducklake(
            pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
            cat.metadata_path, "test", mode="append",
        )
        write_ducklake(
            pl.DataFrame({"a": [3, 4], "b": ["x", "z"]}),
            cat.metadata_path, "test", mode="append",
        )

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "z"]

    def test_partitioned_insert_integer_partition(self, make_write_catalog):
        """Partition on an integer column."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.Int64()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": [100, 200, 100]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == [100, 200, 100]


# ---------------------------------------------------------------------------
# Partitioned INSERT with filter
# ---------------------------------------------------------------------------


class TestPartitionedFilter:
    """Test filtering on partition columns after partitioned writes."""

    def test_filter_on_partition_column(self, make_write_catalog):
        """Filter on partition column returns correct subset."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("b") == "x").collect().sort("a")
        assert result["a"].to_list() == [1, 3]
        assert result["b"].to_list() == ["x", "x"]

    def test_filter_on_non_partition_column(self, make_write_catalog):
        """Filter on non-partition column in partitioned table."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 5, 3, 2], "b": ["x", "x", "y", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("a") > 2).collect().sort("a")
        assert result["a"].to_list() == [3, 5]


# ---------------------------------------------------------------------------
# Partitioned OVERWRITE
# ---------------------------------------------------------------------------


class TestPartitionedOverwrite:
    """Test overwrite on partitioned tables."""

    def test_overwrite_partitioned(self, make_write_catalog):
        """Overwrite partitioned table replaces all data."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        write_ducklake(
            pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]}),
            cat.metadata_path, "test", mode="append",
        )
        write_ducklake(
            pl.DataFrame({"a": [10, 20], "b": ["z", "w"]}),
            cat.metadata_path, "test", mode="overwrite",
        )

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [10, 20]
        assert result["b"].to_list() == ["z", "w"]


# ---------------------------------------------------------------------------
# DuckDB interop
# ---------------------------------------------------------------------------


class TestPartitionedDuckDBInterop:
    """DuckDB can read partitioned data written by ducklake-dataframe."""

    def test_duckdb_reads_polars_partitioned_insert(self, make_write_catalog):
        """DuckDB reads data written by ducklake-dataframe into partitioned table."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 3
        assert sorted(pdf["a"].to_list()) == [1, 2, 3]
        assert sorted(pdf["b"].to_list()) == ["x", "x", "y"]

    def test_duckdb_reads_polars_multi_partition(self, make_write_catalog):
        """DuckDB reads multi-column partitioned data."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test",
            {"a": pl.Int64(), "b": pl.String(), "c": pl.Int64()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b", "c"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"], "c": [10, 20, 10]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 3
        assert sorted(pdf["a"].to_list()) == [1, 2, 3]

    def test_duckdb_writes_polars_reads_partitioned(self, make_write_catalog):
        """DuckDB creates partitioned table, ducklake-dataframe writes to it."""
        cat = make_write_catalog()

        # Use DuckDB to create partitioned table structure
        if cat.backend == "sqlite":
            source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            source = f"ducklake:postgres:{cat.metadata_path}"

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        con.close()

        # Insert via ducklake-dataframe (cast to Int32 to match DuckDB's INTEGER)
        df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int32), "b": ["x", "y", "x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "x"]

    def test_duckdb_insert_then_polars_insert_partitioned(self, make_write_catalog):
        """DuckDB inserts data, then ducklake-dataframe appends to same partitioned table."""
        cat = make_write_catalog()

        # Use DuckDB to create table, partition, and insert initial data
        if cat.backend == "sqlite":
            source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            source = f"ducklake:postgres:{cat.metadata_path}"

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y')")
        con.close()

        # Append via ducklake-dataframe (cast to Int32 to match DuckDB's INTEGER)
        df = pl.DataFrame({"a": pl.Series([3, 4], dtype=pl.Int32), "b": ["x", "z"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "z"]


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestPartitionedRoundTrip:
    """Write and read partitioned data with ducklake-dataframe."""

    def test_roundtrip_basic(self, make_write_catalog):
        cat = make_write_catalog()
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "category": ["a", "b", "a", "b", "c"],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["category"])

        # Insert more data
        df2 = pl.DataFrame({
            "id": [6, 7],
            "category": ["a", "d"],
            "value": [60.0, 70.0],
        })
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("id")
        assert result["id"].to_list() == [1, 2, 3, 4, 5, 6, 7]
        assert result["category"].to_list() == ["a", "b", "a", "b", "c", "a", "d"]
        assert result["value"].to_list() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    def test_roundtrip_time_travel(self, make_write_catalog):
        """Time travel works with partitioned writes."""
        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        write_ducklake(
            pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
            cat.metadata_path, "test", mode="append",
        )

        snap_after_first = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        write_ducklake(
            pl.DataFrame({"a": [3, 4], "b": ["x", "z"]}),
            cat.metadata_path, "test", mode="append",
        )

        # Latest: 4 rows
        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 4

        # At first insert: 2 rows
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_after_first
        )
        assert result_old.shape[0] == 2
        result_old = result_old.sort("a")
        assert result_old["a"].to_list() == [1, 2]


# ---------------------------------------------------------------------------
# DELETE on partitioned table (written by polars)
# ---------------------------------------------------------------------------


class TestPartitionedDelete:
    """Test DELETE on partitioned tables written by ducklake-dataframe."""

    def test_delete_from_partitioned(self, make_write_catalog):
        from ducklake_polars import delete_ducklake

        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        deleted = delete_ducklake(cat.metadata_path, "test", pl.col("a") == 1)
        assert deleted == 1

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [2, 3]
        assert result["b"].to_list() == ["y", "x"]


# ---------------------------------------------------------------------------
# UPDATE on partitioned table (written by polars)
# ---------------------------------------------------------------------------


class TestPartitionedUpdate:
    """Test UPDATE on partitioned tables written by ducklake-dataframe."""

    def test_update_non_partition_column(self, make_write_catalog):
        from ducklake_polars import update_ducklake

        cat = make_write_catalog()
        create_ducklake_table(
            cat.metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        updated = update_ducklake(
            cat.metadata_path, "test", {"a": 10}, pl.col("a") == 1
        )
        assert updated == 1

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [2, 3, 10]
        assert result["b"].to_list() == ["y", "x", "x"]


# ---------------------------------------------------------------------------
# write_ducklake mode='error' with partition
# ---------------------------------------------------------------------------


class TestWriteModeWithPartition:
    """Test write_ducklake modes combined with partitioning."""

    def test_mode_error_then_partition_then_append(self, make_write_catalog):
        """Create with mode='error', partition, then append."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df1, cat.metadata_path, "test", mode="error")
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        df2 = pl.DataFrame({"a": [3, 4], "b": ["x", "z"]})
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "z"]

    def test_mode_overwrite_on_partitioned(self, make_write_catalog):
        """mode='overwrite' on partitioned table."""
        cat = make_write_catalog()

        df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df1, cat.metadata_path, "test", mode="error")
        alter_ducklake_set_partitioned_by(cat.metadata_path, "test", ["b"])

        # Overwrite with different partition values
        df2 = pl.DataFrame({"a": [10, 20], "b": ["p", "q"]})
        write_ducklake(df2, cat.metadata_path, "test", mode="overwrite")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [10, 20]
        assert result["b"].to_list() == ["p", "q"]
