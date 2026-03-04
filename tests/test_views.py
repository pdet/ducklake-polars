"""Tests for DuckLake view support."""

from __future__ import annotations

import os
import pytest
import duckdb
import ducklake_polars
import ducklake_pandas


@pytest.fixture
def catalog_with_view(tmp_path):
    """Create a DuckLake catalog with a table and a view."""
    metadata_path = str(tmp_path / "test.ducklake")
    data_path = str(tmp_path / "data")

    conn = duckdb.connect()
    conn.install_extension("ducklake")
    conn.load_extension("ducklake")
    conn.install_extension("sqlite_scanner")
    conn.load_extension("sqlite_scanner")
    conn.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS test (DATA_PATH '{data_path}')"
    )
    conn.execute("CREATE TABLE test.main.users (id INTEGER, name VARCHAR, age INTEGER)")
    conn.execute("INSERT INTO test.main.users VALUES (1, 'Alice', 30), (2, 'Bob', 25)")
    conn.execute("CREATE VIEW test.main.active_users AS SELECT id, name FROM test.main.users WHERE age > 20")
    conn.close()
    return metadata_path


class TestListViews:
    def test_list_views_polars(self, catalog_with_view):
        views = ducklake_polars.list_views(catalog_with_view)
        assert "active_users" in views

    def test_list_views_pandas(self, catalog_with_view):
        views = ducklake_pandas.list_views(catalog_with_view)
        assert "active_users" in views

    def test_list_views_empty(self, tmp_path):
        metadata_path = str(tmp_path / "empty.ducklake")
        data_path = str(tmp_path / "data")
        conn = duckdb.connect()
        conn.install_extension("ducklake")
        conn.load_extension("ducklake")
        conn.install_extension("sqlite_scanner")
        conn.load_extension("sqlite_scanner")
        conn.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS test (DATA_PATH '{data_path}')"
        )
        conn.execute("CREATE TABLE test.main.t (id INTEGER)")
        conn.close()
        views = ducklake_polars.list_views(metadata_path)
        assert views == []


class TestGetView:
    def test_get_view(self, catalog_with_view):
        v = ducklake_polars.get_view(catalog_with_view, "active_users")
        assert v is not None
        assert v["view_name"] == "active_users"
        assert "SELECT" in v["sql"]
        assert v["dialect"] == "duckdb"

    def test_get_view_pandas(self, catalog_with_view):
        v = ducklake_pandas.get_view(catalog_with_view, "active_users")
        assert v is not None
        assert v["view_name"] == "active_users"

    def test_get_view_case_insensitive(self, catalog_with_view):
        v = ducklake_polars.get_view(catalog_with_view, "ACTIVE_USERS")
        assert v is not None

    def test_get_view_not_found(self, catalog_with_view):
        v = ducklake_polars.get_view(catalog_with_view, "nonexistent")
        assert v is None
