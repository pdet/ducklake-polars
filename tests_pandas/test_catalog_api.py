"""Tests for the DuckLakeCatalog utility API (pandas version)."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from ducklake_pandas import DuckLakeCatalog


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

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["snapshot_id", "snapshot_time", "schema_version"]
        # CREATE TABLE + 2 INSERTs = at least 3 snapshots
        assert len(result) >= 3
        # snapshot_ids should be monotonically increasing
        ids = result["snapshot_id"].tolist()
        assert ids == sorted(ids)
        # All snapshot_times should be non-null
        assert result["snapshot_time"].notna().all()

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

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "table_name", "table_id", "file_count",
            "file_size_bytes", "delete_file_count", "delete_row_count",
        ]

        assert len(result) == 1
        row = result.iloc[0]
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
        names = sorted(result["table_name"].tolist())
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
        row = result.iloc[0]
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
        row = result.iloc[0]
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

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "data_file", "data_file_size_bytes",
            "delete_file", "delete_row_count",
        ]

        assert len(result) >= 1
        # Data files should be non-null
        assert result["data_file"].notna().all()
        assert result["data_file_size_bytes"].notna().all()
        # No deletes, so delete columns should be null
        assert result["delete_file"].isna().all()

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
        has_delete = result[result["delete_file"].notna()]
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
        assert list(result.columns) == [
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

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["schema_id", "schema_name"]
        assert "main" in result["schema_name"].tolist()

    def test_list_tables(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_tables()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["table_id", "table_name"]
        names = sorted(result["table_name"].tolist())
        assert names == ["t1", "t2"]

    def test_list_tables_empty_schema(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_tables(schema="nonexistent")

        assert len(result) == 0
        assert list(result.columns) == ["table_id", "table_name"]

    def test_list_tables_at_snapshot_version(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)

        # At v1, only t1 should exist
        result_v1 = api.list_tables(snapshot_version=v1)
        assert result_v1["table_name"].tolist() == ["t1"]

        # At current, both should exist
        result_current = api.list_tables()
        assert sorted(result_current["table_name"].tolist()) == ["t1", "t2"]

    def test_list_schemas_at_snapshot_version(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_schemas(snapshot_version=v1)

        assert isinstance(result, pd.DataFrame)
        assert "main" in result["schema_name"].tolist()


class TestOptions:
    """Test options and settings methods."""

    def test_options(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.options()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["key", "value", "scope", "scope_id"]
        # Should have at least the data_path key
        keys = result["key"].tolist()
        assert "data_path" in keys

    def test_settings(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.settings()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["catalog_type", "data_path"]
        assert len(result) == 1

        row = result.iloc[0]
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

        assert isinstance(result, pd.DataFrame)
        assert "snapshot_id" in result.columns
        assert "a" in result.columns
        assert "b" in result.columns

        # Should only contain the rows inserted after v1
        assert len(result) == 2
        assert sorted(result["a"].tolist()) == [2, 3]
        result_sorted = result.sort_values("a").reset_index(drop=True)
        assert result_sorted["b"].tolist() == ["two", "three"]

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
        assert sorted(result_all["a"].tolist()) == [1, 2, 3]

        # v1 to v3: last 2 inserts
        result_last2 = api.table_insertions("t1", v1, v3)
        assert sorted(result_last2["a"].tolist()) == [2, 3]

        # v2 to v3: last insert only
        result_last = api.table_insertions("t1", v2, v3)
        assert result_last["a"].tolist() == [3]

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

        assert isinstance(result, pd.DataFrame)
        assert "snapshot_id" in result.columns
        assert "a" in result.columns
        assert "b" in result.columns

        # Should contain the deleted row
        assert len(result) == 1
        assert result["a"].tolist() == [2]
        assert result["b"].tolist() == ["two"]

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

        deleted_values = set(result["a"].tolist())
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
        assert set(result["change_type"].tolist()) == {"insert"}

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
        assert result["change_type"].tolist() == ["delete"]
        assert result["a"].tolist() == [2]

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

        change_types = set(result["change_type"].tolist())
        assert "update_preimage" in change_types
        assert "update_postimage" in change_types

        preimage = result[result["change_type"] == "update_preimage"]
        postimage = result[result["change_type"] == "update_postimage"]

        assert len(preimage) >= 1
        assert len(postimage) >= 1

        # The preimage should have the old value
        pre_row = preimage[preimage["a"] == 1]
        assert len(pre_row) == 1
        assert pre_row["b"].tolist() == ["old"]

        # The postimage should have the new value
        post_row = postimage[postimage["a"] == 1]
        assert len(post_row) == 1
        assert post_row["b"].tolist() == ["new"]

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

        change_types = result["change_type"].tolist()
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
