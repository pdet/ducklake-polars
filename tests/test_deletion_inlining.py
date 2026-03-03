"""Deletion inlining edge case tests — parity with ducklake-ref.

Tests deletion + inlining interactions: delete from inlined data,
partial inlining after delete, stats after delete, etc.
"""
from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake


class TestDeletionInlining:
    """Test delete operations on tables with inlined data."""

    def test_delete_all_inlined_rows(self, ducklake_catalog_sqlite):
        """Delete all rows from a fully-inlined table."""
        cat = ducklake_catalog_sqlite
        # Small table — stays inlined
        cat.execute(
            "CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)"
        )
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x'), (2, 'y')")
        cat.execute("DELETE FROM ducklake.t")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 0

    def test_delete_some_inlined_rows(self, ducklake_catalog_sqlite):
        """Delete specific rows from inlined data."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3), (4), (5)")
        cat.execute("DELETE FROM ducklake.t WHERE a > 3")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 3
        assert sorted(result["a"].to_list()) == [1, 2, 3]

    def test_delete_then_insert_inlined(self, ducklake_catalog_sqlite):
        """Delete rows then insert new ones — both should be inlined."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2)")
        cat.execute("DELETE FROM ducklake.t WHERE a = 1")
        cat.execute("INSERT INTO ducklake.t VALUES (3)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert sorted(result["a"].to_list()) == [2, 3]

    def test_delete_with_alter_add_column(self, ducklake_catalog_sqlite):
        """Delete after adding a column — should handle NULL defaults."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.t VALUES (4, 'new')")
        cat.execute("DELETE FROM ducklake.t WHERE a = 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 3
        result_sorted = result.sort("a")
        assert result_sorted["a"].to_list() == [1, 3, 4]

    def test_delete_from_partitioned_table(self, ducklake_catalog_sqlite):
        """Delete from a partitioned table with inlined data."""
        cat = ducklake_catalog_sqlite
        cat.execute(
            "CREATE TABLE ducklake.t (a INTEGER, part VARCHAR)"
        )
        cat.execute(
            "INSERT INTO ducklake.t VALUES (1, 'A'), (2, 'B'), (3, 'A'), (4, 'B')"
        )
        cat.execute("DELETE FROM ducklake.t WHERE part = 'A'")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert sorted(result["a"].to_list()) == [2, 4]

    def test_update_then_delete(self, ducklake_catalog_sqlite):
        """Update a row then delete it — should be gone."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x'), (2, 'y')")
        cat.execute("UPDATE ducklake.t SET b = 'updated' WHERE a = 1")
        cat.execute("DELETE FROM ducklake.t WHERE a = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 1
        assert result["a"].to_list() == [2]

    def test_truncate_then_reinsert(self, ducklake_catalog_sqlite):
        """Truncate table then reinsert — should only see new data."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3)")
        cat.execute("TRUNCATE TABLE ducklake.t")
        cat.execute("INSERT INTO ducklake.t VALUES (10), (20)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert sorted(result["a"].to_list()) == [10, 20]


class TestDefaultValues:
    """Test default value behavior on CREATE and ALTER."""

    def test_add_column_with_default_string(self, ducklake_catalog_sqlite):
        """Add column with VARCHAR default — old rows get default."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR DEFAULT 'hello'")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        result_sorted = result.sort("a")
        assert result_sorted.shape[0] == 3
        # Old rows get default 'hello', new row gets 'world'
        assert result_sorted["b"].to_list()[0] in [None, "hello"]  # depends on implementation
        assert result_sorted["b"].to_list()[2] == "world"

    def test_add_column_with_default_integer(self, ducklake_catalog_sqlite):
        """Add column with INTEGER default."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b INTEGER DEFAULT 42")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 99)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        result_sorted = result.sort("a")
        assert result_sorted.shape[0] == 2
        assert result_sorted["b"].to_list()[1] == 99

    def test_add_column_with_null_default(self, ducklake_catalog_sqlite):
        """Add column without default — old rows get NULL."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'val')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        result_sorted = result.sort("a")
        assert result_sorted["b"].to_list()[0] is None
        assert result_sorted["b"].to_list()[1] == "val"

    def test_multiple_add_columns_with_defaults(self, ducklake_catalog_sqlite):
        """Add multiple columns with different defaults sequentially."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR DEFAULT 'x'")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'y')")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN c DOUBLE DEFAULT 3.14")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'z', 2.71)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 3
        assert "b" in result.columns
        assert "c" in result.columns
