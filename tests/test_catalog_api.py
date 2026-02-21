"""Tests for the DuckLakeCatalog utility API."""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import DuckLakeCatalog


class TestSnapshots:
    """Test snapshot-related methods."""

    def test_snapshots_after_creates_and_inserts(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (2)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.snapshots()

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["snapshot_id", "snapshot_time", "schema_version"]
        assert result.schema["snapshot_id"] == pl.Int64
        assert result.schema["snapshot_time"] == pl.String
        assert result.schema["schema_version"] == pl.Int64
        # CREATE TABLE + 2 INSERTs = at least 3 snapshots
        assert len(result) >= 3
        # snapshot_ids should be monotonically increasing
        ids = result["snapshot_id"].to_list()
        assert ids == sorted(ids)
        # All snapshot_times should be non-null
        assert result["snapshot_time"].null_count() == 0

    def test_current_snapshot(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        expected = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.current_snapshot()

        assert isinstance(result, int)
        assert result == expected


class TestTableInfo:
    """Test table_info method."""

    def test_table_info_single_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2), (3)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_info()

        assert isinstance(result, pl.DataFrame)
        assert result.columns == [
            "table_name", "table_id", "file_count",
            "file_size_bytes", "delete_file_count", "delete_row_count",
        ]
        assert result.schema["table_name"] == pl.String
        assert result.schema["table_id"] == pl.Int64
        assert result.schema["file_count"] == pl.Int64
        assert result.schema["file_size_bytes"] == pl.Int64
        assert result.schema["delete_file_count"] == pl.Int64
        assert result.schema["delete_row_count"] == pl.Int64

        assert len(result) == 1
        row = result.row(0, named=True)
        assert row["table_name"] == "t1"
        assert row["file_count"] >= 1
        assert row["file_size_bytes"] > 0
        assert row["delete_file_count"] == 0
        assert row["delete_row_count"] == 0

    def test_table_info_multiple_tables(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t2 VALUES ('hello')")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_info()

        assert len(result) == 2
        names = sorted(result["table_name"].to_list())
        assert names == ["t1", "t2"]

    def test_table_info_with_deletes(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2), (3)")
        cat.execute("DELETE FROM ducklake.t1 WHERE a = 2")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_info()

        assert len(result) == 1
        row = result.row(0, named=True)
        assert row["table_name"] == "t1"
        assert row["file_count"] >= 1
        assert row["file_size_bytes"] > 0
        assert row["delete_file_count"] >= 1
        assert row["delete_row_count"] >= 1

    def test_table_info_empty_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_info()

        assert len(result) == 1
        row = result.row(0, named=True)
        assert row["table_name"] == "t1"
        assert row["file_count"] == 0
        assert row["file_size_bytes"] == 0
        assert row["delete_file_count"] == 0
        assert row["delete_row_count"] == 0

    def test_table_info_nonexistent_schema(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_info(schema="nonexistent")

        assert len(result) == 0
        assert "table_name" in result.columns


class TestListFiles:
    """Test list_files method."""

    def test_list_files(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_files("t1")

        assert isinstance(result, pl.DataFrame)
        assert result.columns == [
            "data_file", "data_file_size_bytes",
            "delete_file", "delete_row_count",
        ]
        assert result.schema["data_file"] == pl.String
        assert result.schema["data_file_size_bytes"] == pl.Int64
        assert result.schema["delete_file"] == pl.String
        assert result.schema["delete_row_count"] == pl.Int64

        assert len(result) >= 1
        # Data files should be non-null
        assert result["data_file"].null_count() == 0
        assert result["data_file_size_bytes"].null_count() == 0
        # No deletes, so delete columns should be null
        assert result["delete_file"].null_count() == len(result)

    def test_list_files_with_deletes(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2), (3)")
        cat.execute("DELETE FROM ducklake.t1 WHERE a = 2")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_files("t1")

        assert len(result) >= 1
        # Should have at least one row with a delete file
        has_delete = result.filter(pl.col("delete_file").is_not_null())
        assert len(has_delete) >= 1

    def test_list_files_at_version(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.t1 VALUES (2)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)

        # At v1, should have fewer files than current
        result_v1 = api.list_files("t1", snapshot_version=v1)
        result_current = api.list_files("t1")

        assert len(result_v1) <= len(result_current)

    def test_list_files_empty_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_files("t1")

        assert len(result) == 0
        assert result.columns == [
            "data_file", "data_file_size_bytes",
            "delete_file", "delete_row_count",
        ]


class TestListSchemasAndTables:
    """Test list_schemas and list_tables methods."""

    def test_list_schemas(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_schemas()

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["schema_id", "schema_name"]
        assert "main" in result["schema_name"].to_list()

    def test_list_tables(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_tables()

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["table_id", "table_name"]
        names = sorted(result["table_name"].to_list())
        assert names == ["t1", "t2"]

    def test_list_tables_empty_schema(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_tables(schema="nonexistent")

        assert len(result) == 0
        assert result.columns == ["table_id", "table_name"]

    def test_list_tables_at_snapshot_version(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)

        # At v1, only t1 should exist
        result_v1 = api.list_tables(snapshot_version=v1)
        assert result_v1["table_name"].to_list() == ["t1"]

        # At current, both should exist
        result_current = api.list_tables()
        assert sorted(result_current["table_name"].to_list()) == ["t1", "t2"]

    def test_list_schemas_at_snapshot_version(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_schemas(snapshot_version=v1)

        assert isinstance(result, pl.DataFrame)
        assert "main" in result["schema_name"].to_list()


class TestOptions:
    """Test options and settings methods."""

    def test_options(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.options()

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["key", "value"]
        # Should have at least the data_path key
        keys = result["key"].to_list()
        assert "data_path" in keys

    def test_settings(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.settings()

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["catalog_type", "data_path"]
        assert len(result) == 1

        row = result.row(0, named=True)
        assert row["catalog_type"] in ("sqlite", "postgresql")
        assert row["data_path"] is not None and len(row["data_path"]) > 0


class TestChangeDataFeed:
    """Test change data feed methods: table_insertions, table_deletions, table_changes."""

    def test_table_insertions(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1, 'one')")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.t1 VALUES (2, 'two'), (3, 'three')")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_insertions("t1", v1, v2)

        assert isinstance(result, pl.DataFrame)
        assert "snapshot_id" in result.columns
        assert "a" in result.columns
        assert "b" in result.columns

        # Should only contain the rows inserted after v1
        assert len(result) == 2
        assert sorted(result["a"].to_list()) == [2, 3]
        result_sorted = result.sort("a")
        assert result_sorted["b"].to_list() == ["two", "three"]

    def test_table_insertions_multiple_snapshots(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        v0 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.t1 VALUES (2)")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.t1 VALUES (3)")
        v3 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)

        # v0 to v3: all 3 inserts
        result_all = api.table_insertions("t1", v0, v3)
        assert sorted(result_all["a"].to_list()) == [1, 2, 3]

        # v1 to v3: last 2 inserts
        result_last2 = api.table_insertions("t1", v1, v3)
        assert sorted(result_last2["a"].to_list()) == [2, 3]

        # v2 to v3: last insert only
        result_last = api.table_insertions("t1", v2, v3)
        assert result_last["a"].to_list() == [3]

    def test_table_deletions(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1, 'one'), (2, 'two'), (3, 'three')")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("DELETE FROM ducklake.t1 WHERE a = 2")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_deletions("t1", v1, v2)

        assert isinstance(result, pl.DataFrame)
        assert "snapshot_id" in result.columns
        assert "a" in result.columns
        assert "b" in result.columns

        # Should contain the deleted row
        assert len(result) == 1
        assert result["a"].to_list() == [2]
        assert result["b"].to_list() == ["two"]

    def test_table_deletions_multiple_snapshots(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2), (3), (4), (5)")
        v0 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("DELETE FROM ducklake.t1 WHERE a = 2")
        cat.execute("DELETE FROM ducklake.t1 WHERE a = 4")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_deletions("t1", v0, v2)

        # Both deleted values should appear (DuckLake may generate
        # additional delete files across snapshots)
        deleted_values = set(result["a"].to_list())
        assert 2 in deleted_values
        assert 4 in deleted_values

    def test_table_changes_insert_only(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        v0 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_changes("t1", v0, v1)

        assert "snapshot_id" in result.columns
        assert "change_type" in result.columns
        assert "a" in result.columns

        assert len(result) == 2
        assert set(result["change_type"].to_list()) == {"insert"}

    def test_table_changes_delete_only(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2), (3)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("DELETE FROM ducklake.t1 WHERE a = 2")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_changes("t1", v1, v2)

        assert len(result) == 1
        assert result["change_type"].to_list() == ["delete"]
        assert result["a"].to_list() == [2]

    def test_table_changes_update(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1, 'old'), (2, 'keep')")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("UPDATE ducklake.t1 SET b = 'new' WHERE a = 1")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_changes("t1", v1, v2)

        assert "change_type" in result.columns

        # An UPDATE produces both a deletion (preimage) and insertion (postimage)
        # in the same snapshot, so we expect update_preimage + update_postimage
        change_types = set(result["change_type"].to_list())
        assert "update_preimage" in change_types
        assert "update_postimage" in change_types

        preimage = result.filter(pl.col("change_type") == "update_preimage")
        postimage = result.filter(pl.col("change_type") == "update_postimage")

        assert len(preimage) >= 1
        assert len(postimage) >= 1

        # The preimage should have the old value
        pre_row = preimage.filter(pl.col("a") == 1)
        assert len(pre_row) == 1
        assert pre_row["b"].to_list() == ["old"]

        # The postimage should have the new value
        post_row = postimage.filter(pl.col("a") == 1)
        assert len(post_row) == 1
        assert post_row["b"].to_list() == ["new"]

    def test_table_changes_mixed_insert_and_delete_different_snapshots(self, ducklake_catalog):
        """Inserts and deletes in different snapshots should be labeled correctly."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2), (3)")
        v0 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        # Pure insert in one snapshot
        cat.execute("INSERT INTO ducklake.t1 VALUES (4)")
        # Pure delete in the next snapshot
        cat.execute("DELETE FROM ducklake.t1 WHERE a = 1")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_changes("t1", v0, v2)

        change_types = result["change_type"].to_list()
        # The insert and delete happen in DIFFERENT snapshots,
        # so they should NOT be classified as updates
        assert "insert" in change_types
        assert "delete" in change_types

    def test_table_insertions_empty_range(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_insertions("t1", v1, v1)

        assert len(result) == 0
        assert "snapshot_id" in result.columns
        assert "a" in result.columns
        # Empty frame should have correct types, not String fallback
        assert result.schema["snapshot_id"] == pl.Int64
        assert result.schema["a"] == pl.Int32

    def test_table_deletions_empty_range(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_deletions("t1", v1, v1)

        assert len(result) == 0
        assert "snapshot_id" in result.columns
        assert "a" in result.columns

    def test_table_changes_empty_range(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.table_changes("t1", v1, v1)

        assert len(result) == 0
        assert "snapshot_id" in result.columns
        assert "change_type" in result.columns


class TestErrorCases:
    """Test error handling in catalog API methods."""

    def test_list_files_invalid_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        with pytest.raises(ValueError, match="not found"):
            api.list_files("nonexistent")

    def test_list_files_invalid_snapshot_version(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        with pytest.raises(ValueError, match="not found"):
            api.list_files("t1", snapshot_version=9999)

    def test_table_insertions_invalid_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        with pytest.raises(ValueError, match="not found"):
            api.table_insertions("nonexistent", 0, v1)

    def test_table_changes_invalid_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        with pytest.raises(ValueError, match="not found"):
            api.table_changes("nonexistent", 0, v1)


class TestContextManager:
    """Test that DuckLakeCatalog works as a context manager."""

    def test_context_manager(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        cat.close()

        with DuckLakeCatalog(cat.metadata_path) as api:
            result = api.snapshots()
            assert len(result) >= 1

    def test_data_path_override(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (42)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)
        result = api.list_files("t1")
        assert len(result) >= 1
