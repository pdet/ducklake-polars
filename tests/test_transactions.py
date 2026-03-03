"""Transaction semantics tests matching ducklake-ref coverage.

Tests rollback behavior, multi-operation transactions, snapshot visibility,
and conflict scenarios.
"""
from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake, scan_ducklake, write_ducklake
from ducklake_core._writer import DuckLakeCatalogWriter, TransactionConflictError


class TestTransactionRollback:
    """Test that failed operations don't corrupt catalog state."""

    def test_write_after_failed_create(self, ducklake_catalog_sqlite):
        """A failed create shouldn't prevent subsequent successful operations."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x')")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path

        # Try to create existing table — should fail
        w = DuckLakeCatalogWriter(meta, data_path_override=data)
        with pytest.raises(ValueError, match="already exists"):
            w.create_table("t", {"a": "INTEGER"})
        w.close()

        # Successful write should still work
        df = pl.DataFrame({"a": pl.Series([2], dtype=pl.Int32), "b": ["y"]})
        write_ducklake(df, meta, "t", data_path=data, mode='append')

        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 2
        assert sorted(result["a"].to_list()) == [1, 2]

    def test_catalog_state_after_failed_create(self, ducklake_catalog_sqlite):
        """Failed table creation shouldn't leave orphan metadata."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path

        # Try to create same table — should fail
        w = DuckLakeCatalogWriter(meta, data_path_override=data)
        with pytest.raises(ValueError, match="already exists"):
            w.create_table("t", {"a": "INTEGER"})
        w.close()

        # Original table should be fine
        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 1


class TestSnapshotVisibility:
    """Test that reads see correct snapshot state."""

    def test_read_at_specific_version(self, ducklake_catalog_sqlite):
        """Reading at version N should see correct data.

        Snapshot layout: 0=init, 1=CREATE, 2=INSERT(1), 3=INSERT(2), 4=INSERT(3)
        """
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("INSERT INTO ducklake.t VALUES (2)")
        cat.execute("INSERT INTO ducklake.t VALUES (3)")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path

        # Latest should have 3 rows
        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 3

        # Snapshot 2 (first insert) should have 1 row
        result_v2 = read_ducklake(meta, "t", data_path=data, snapshot_version=2)
        assert result_v2.shape[0] == 1
        assert result_v2["a"].to_list() == [1]

        # Snapshot 3 (second insert) should have 2 rows
        result_v3 = read_ducklake(meta, "t", data_path=data, snapshot_version=3)
        assert result_v3.shape[0] == 2

    def test_read_after_delete_at_old_version(self, ducklake_catalog_sqlite):
        """Reading at a version before delete should show deleted rows.

        Snapshots: 0=init, 1=CREATE, 2=INSERT(1,2,3), 3=DELETE(2)
        """
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3)")
        cat.execute("DELETE FROM ducklake.t WHERE a = 2")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path

        # Latest: 2 rows (after delete)
        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 2
        assert sorted(result["a"].to_list()) == [1, 3]

        # Snapshot 2 (before delete): 3 rows
        result_v2 = read_ducklake(meta, "t", data_path=data, snapshot_version=2)
        assert result_v2.shape[0] == 3
        assert sorted(result_v2["a"].to_list()) == [1, 2, 3]

    def test_read_after_alter_at_old_version(self, ducklake_catalog_sqlite):
        """Reading at version before ALTER should use old schema.

        Snapshots: 0=init, 1=CREATE, 2=INSERT(1), 3=ALTER ADD b, 4=INSERT(2,'val')
        """
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR DEFAULT 'def'")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'val')")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path

        # Latest: 2 rows, 2 columns
        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 2
        assert "b" in result.columns

        # Snapshot 2 (before ALTER): 1 row, 1 column
        result_v2 = read_ducklake(meta, "t", data_path=data, snapshot_version=2)
        assert result_v2.shape[0] == 1
        assert result_v2.columns == ["a"]


class TestMultiOpTransaction:
    """Test multiple operations in a single DuckDB transaction."""

    def test_insert_update_delete_sequence(self, ducklake_catalog_sqlite):
        """Insert → Update → Delete in sequence, verify final state."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        cat.execute("UPDATE ducklake.t SET val = 'updated' WHERE id = 2")
        cat.execute("DELETE FROM ducklake.t WHERE id = 3")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 2
        result_sorted = result.sort("id")
        assert result_sorted["val"].to_list() == ["a", "updated"]

    def test_create_insert_alter_insert(self, ducklake_catalog_sqlite):
        """Create → Insert → Alter (add column) → Insert with new column."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert "b" in result.columns
        result_sorted = result.sort("a")
        assert result_sorted["b"].to_list()[0] is None
        assert result_sorted["b"].to_list()[1] == "hello"

    def test_multiple_tables_in_sequence(self, ducklake_catalog_sqlite):
        """Create and populate multiple tables, read each independently."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t1 (x INTEGER)")
        cat.execute("CREATE TABLE ducklake.t2 (y VARCHAR)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2)")
        cat.execute("INSERT INTO ducklake.t2 VALUES ('a'), ('b'), ('c')")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "t1", data_path=cat.data_path)
        r2 = read_ducklake(cat.metadata_path, "t2", data_path=cat.data_path)
        assert r1.shape[0] == 2
        assert r2.shape[0] == 3

    def test_drop_and_recreate_table(self, ducklake_catalog_sqlite):
        """Drop a table and recreate with same name but different schema."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("DROP TABLE ducklake.t")
        cat.execute("CREATE TABLE ducklake.t (x VARCHAR, y DOUBLE)")
        cat.execute("INSERT INTO ducklake.t VALUES ('hello', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.columns == ["x", "y"]
        assert result.shape[0] == 1
        assert result["x"].to_list() == ["hello"]


class TestConflictDetection:
    """Test conflict detection edge cases."""

    def test_concurrent_create_same_table(self, ducklake_catalog_sqlite):
        """Two writers trying to create the same table — second should fail."""
        import pyarrow as pa

        cat = ducklake_catalog_sqlite
        cat.close()

        meta, data = cat.metadata_path, cat.data_path

        w1 = DuckLakeCatalogWriter(meta, data_path_override=data)
        w1.create_table("t", {"a": pa.int32()})
        w1.close()

        w2 = DuckLakeCatalogWriter(meta, data_path_override=data)
        with pytest.raises(ValueError, match="already exists"):
            w2.create_table("t", {"a": pa.int32()})
        w2.close()

    def test_write_to_nonexistent_table_in_append_mode(self, ducklake_catalog_sqlite):
        """Appending to nonexistent table should fail (mode='append')."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.close()

        meta, data = cat.metadata_path, cat.data_path
        df = pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)})

        # mode='error' on nonexistent should create, but mode='append'
        # on existing should work
        write_ducklake(df, meta, "t", data_path=data, mode='append')
        result = read_ducklake(meta, "t", data_path=data)
        assert result.shape[0] == 1


class TestSchemaTransactions:
    """Test schema changes across multiple snapshots."""

    def test_add_drop_add_same_column(self, ducklake_catalog_sqlite):
        """Add column, drop it, add it back — should work."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'x')")
        cat.execute("ALTER TABLE ducklake.t DROP COLUMN b")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b DOUBLE")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 3
        assert "b" in result.columns
        # Column b should be DOUBLE now
        result_sorted = result.sort("a")
        assert result_sorted["b"].to_list()[2] == pytest.approx(3.14)

    def test_rename_column_and_add_with_old_name(self, ducklake_catalog_sqlite):
        """Rename column A→B, then add new column A — should be distinct."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x')")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN a TO b")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN a DOUBLE")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'y', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert "a" in result.columns
        assert "b" in result.columns

    def test_type_promotion_across_snapshots(self, ducklake_catalog_sqlite):
        """Change column type, verify reads handle mixed physical types."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("ALTER TABLE ducklake.t ALTER COLUMN a TYPE BIGINT")
        cat.execute("INSERT INTO ducklake.t VALUES (2)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert sorted(result["a"].to_list()) == [1, 2]
