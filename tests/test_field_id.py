"""Tests for field_id-based column mapping after renames."""
from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake, write_ducklake


class TestFieldIdMapping:
    """Test that reads work correctly after column renames."""

    def test_read_after_rename(self, ducklake_catalog_sqlite):
        """Write data, rename column, write more data, read all."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (old_name INTEGER, value VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN old_name TO new_name")
        cat.execute("INSERT INTO ducklake.t VALUES (4, 'd'), (5, 'e'), (6, 'f')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "new_name" in result.columns
        assert "old_name" not in result.columns
        assert result.shape[0] == 6
        assert sorted(result["new_name"].to_list()) == [1, 2, 3, 4, 5, 6]

    def test_read_after_multiple_renames(self, ducklake_catalog_sqlite):
        """Column renamed twice — reads should use final name."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (col_a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2)")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN col_a TO col_b")
        cat.execute("INSERT INTO ducklake.t VALUES (3), (4)")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN col_b TO col_c")
        cat.execute("INSERT INTO ducklake.t VALUES (5), (6)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "col_c" in result.columns
        assert "col_a" not in result.columns
        assert "col_b" not in result.columns
        assert result.shape[0] == 6
        assert sorted(result["col_c"].to_list()) == [1, 2, 3, 4, 5, 6]

    def test_rename_with_multiple_columns(self, ducklake_catalog_sqlite):
        """Rename one column out of several — others stay the same."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (x INTEGER, y VARCHAR, z DOUBLE)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a', 1.0), (2, 'b', 2.0)")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN x TO renamed_x")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c', 3.0)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "renamed_x" in result.columns
        assert "x" not in result.columns
        assert "y" in result.columns
        assert "z" in result.columns
        assert result.shape[0] == 3
        assert sorted(result["renamed_x"].to_list()) == [1, 2, 3]

    def test_no_rename_fast_path(self, ducklake_catalog_sqlite):
        """No renames — field_id mapping shouldn't break fast path."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x'), (2, 'y')")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'z')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 3
        assert sorted(result.columns) == ["a", "b"]

    def test_duckdb_rename_then_read_with_polars(self, ducklake_catalog_sqlite):
        """DuckDB creates, renames, inserts — ducklake-dataframe reads correctly."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (x INTEGER, y VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN x TO z")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c')")

        # Verify DuckDB sees it correctly
        duckdb_result = cat.execute("SELECT * FROM ducklake.t ORDER BY z").fetchall()
        assert duckdb_result == [(1, 'a'), (2, 'b'), (3, 'c')]
        cat.close()

        # Now read with ducklake-dataframe
        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "z" in result.columns
        assert "x" not in result.columns
        assert result.shape[0] == 3
        assert sorted(result["z"].to_list()) == [1, 2, 3]
