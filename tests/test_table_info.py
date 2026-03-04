"""Tests for DuckLake table_info."""

from __future__ import annotations

import pytest
import duckdb
import ducklake_polars
import ducklake_pandas


@pytest.fixture
def catalog(tmp_path):
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
    conn.execute(
        "CREATE TABLE test.main.users (id INTEGER NOT NULL, name VARCHAR, score DOUBLE)"
    )
    conn.execute("INSERT INTO test.main.users VALUES (1, 'Alice', 95.5)")
    conn.close()
    return metadata_path


class TestTableInfo:
    def test_table_info_polars(self, catalog):
        cols = ducklake_polars.table_info(catalog, "users")
        assert len(cols) == 3
        names = [c["column_name"] for c in cols]
        assert names == ["id", "name", "score"]

    def test_table_info_pandas(self, catalog):
        cols = ducklake_pandas.table_info(catalog, "users")
        assert len(cols) == 3

    def test_column_types(self, catalog):
        cols = ducklake_polars.table_info(catalog, "users")
        type_map = {c["column_name"]: c["column_type"].lower() for c in cols}
        assert type_map["id"] in ("int32", "integer")
        assert type_map["name"] == "varchar"
        assert type_map["score"] in ("float64", "double")

    def test_case_insensitive(self, catalog):
        cols = ducklake_polars.table_info(catalog, "USERS")
        assert len(cols) == 3

    def test_nonexistent_table(self, catalog):
        cols = ducklake_polars.table_info(catalog, "nonexistent")
        assert cols == []

    def test_column_order(self, catalog):
        cols = ducklake_polars.table_info(catalog, "users")
        orders = [c["column_order"] for c in cols]
        assert orders == sorted(orders)
