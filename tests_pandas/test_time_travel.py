"""Time travel tests for ducklake-pandas."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from ducklake_pandas import read_ducklake


class TestTimeTravel:
    """Test reading tables at specific snapshot versions."""

    def test_read_at_version(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (3)")
        cat.close()

        # Read at latest - should have all 3 rows
        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3

        # Read at v1 - should have 1 row
        result_v1 = read_ducklake(cat.metadata_path, "test", snapshot_version=v1)
        assert result_v1.shape[0] == 1
        assert result_v1["a"].tolist() == [1]

        # Read at v2 - should have 2 rows
        result_v2 = read_ducklake(cat.metadata_path, "test", snapshot_version=v2)
        assert result_v2.shape[0] == 2
        assert sorted(result_v2["a"].tolist()) == [1, 2]

    def test_read_at_version_with_filter(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (2), (3)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", snapshot_version=v1)
        result = result[result["a"] == 1]
        assert result.shape == (1, 1)

    def test_read_at_snapshot_time(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        # Get snapshot version after first insert
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        cat.close()

        # Read the snapshot time for v1 from the catalog directly
        ts = cat.query_metadata(
            "SELECT snapshot_time FROM ducklake_snapshot WHERE snapshot_id = ?",
            [v1],
        )[0]

        # Read at the timestamp of v1 - should have 1 row
        result = read_ducklake(cat.metadata_path, "test", snapshot_time=ts)
        assert result.shape[0] == 1
        assert result["a"].tolist() == [1]

    def test_invalid_snapshot_version(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "test", snapshot_version=9999)

    def test_invalid_snapshot_time(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        with pytest.raises(ValueError, match="No snapshot found"):
            read_ducklake(cat.metadata_path, "test", snapshot_time="2000-01-01T00:00:00")


class TestTimeTravelDroppedTable:
    """Test time travel behaviour with dropped tables."""

    def test_read_dropped_table_at_prior_version(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("DROP TABLE ducklake.test")
        cat.close()

        # The table existed at v1, so reading at that version should return data
        result = read_ducklake(cat.metadata_path, "test", snapshot_version=v1)
        assert result.shape[0] == 3
        assert sorted(result["a"].tolist()) == [1, 2, 3]

    def test_read_dropped_table_at_latest_fails(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.execute("DROP TABLE ducklake.test")
        cat.close()

        # The table is dropped at the latest snapshot, so reading should fail
        with pytest.raises(Exception):
            read_ducklake(cat.metadata_path, "test")


class TestTimeTravelEmptyTable:
    """Test time travel with empty tables (created but no data inserted yet)."""

    def test_read_at_creation_snapshot(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        v_create = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        cat.close()

        # At v_create, table exists but has no data yet
        result = read_ducklake(cat.metadata_path, "test", snapshot_version=v_create)
        assert result.shape[0] == 0
        assert "a" in result.columns
        assert "b" in result.columns

    def test_read_before_table_exists(self, ducklake_catalog):
        cat = ducklake_catalog

        # Capture snapshot before the table is created
        # We need at least one snapshot to exist, so create a dummy table first
        cat.execute("CREATE TABLE ducklake.dummy (x INTEGER)")
        v0 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        # Table "test" did not exist at v0, so reading should fail
        with pytest.raises(Exception):
            read_ducklake(cat.metadata_path, "test", snapshot_version=v0)


class TestTimeTravelWithSchema:
    """Test time travel with non-default schemas."""

    def test_time_travel_non_default_schema(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE TABLE ducklake.s1.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.s1.test VALUES (1), (2)")
        snap = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.s1.test VALUES (3), (4)")
        cat.close()

        # Read at old snapshot: should return only first batch
        result_old = read_ducklake(cat.metadata_path, "test", schema="s1", snapshot_version=snap)
        assert result_old.shape[0] == 2
        assert sorted(result_old["a"].tolist()) == [1, 2]

        # Read at latest: should return all rows
        result_latest = read_ducklake(cat.metadata_path, "test", schema="s1")
        assert result_latest.shape[0] == 4
        assert sorted(result_latest["a"].tolist()) == [1, 2, 3, 4]

    def test_read_from_non_default_schema(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE TABLE ducklake.s1.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.s1.test VALUES (10, 'alpha'), (20, 'beta')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", schema="s1")
        assert result.shape[0] == 2
        assert sorted(result["a"].tolist()) == [10, 20]
        assert sorted(result["b"].tolist()) == ["alpha", "beta"]


class TestTimeTravelWithComplexTypes:
    """Test time travel with complex column types (structs, lists)."""

    def test_time_travel_with_struct(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (id INTEGER, s STRUCT(x INTEGER, y VARCHAR))")
        cat.execute("INSERT INTO ducklake.test VALUES (1, {'x': 10, 'y': 'a'})")
        snap = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (2, {'x': 20, 'y': 'b'})")
        cat.close()

        # Read at old snapshot: only first struct row
        result_old = read_ducklake(cat.metadata_path, "test", snapshot_version=snap)
        assert result_old.shape[0] == 1
        assert result_old["id"].tolist() == [1]
        s_val = result_old["s"][0]
        assert s_val["x"] == 10
        assert s_val["y"] == "a"

        # Read at latest: all struct rows
        result_latest = read_ducklake(cat.metadata_path, "test")
        assert result_latest.shape[0] == 2
        assert sorted(result_latest["id"].tolist()) == [1, 2]

    def test_time_travel_with_list(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (id INTEGER, vals INTEGER[])")
        cat.execute("INSERT INTO ducklake.test VALUES (1, [10, 20, 30])")
        snap = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (2, [40, 50])")
        cat.close()

        # Read at old snapshot: only first list row
        result_old = read_ducklake(cat.metadata_path, "test", snapshot_version=snap)
        assert result_old.shape[0] == 1
        assert result_old["id"].tolist() == [1]
        assert list(result_old["vals"].iloc[0]) == [10, 20, 30]

        # Read at latest: all list rows
        result_latest = read_ducklake(cat.metadata_path, "test")
        assert result_latest.shape[0] == 2
        assert sorted(result_latest["id"].tolist()) == [1, 2]
