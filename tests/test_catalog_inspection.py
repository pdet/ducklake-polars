"""Tests for DuckLake catalog inspection utilities."""

from __future__ import annotations

import pytest
import ducklake_polars
import ducklake_pandas


@pytest.fixture(params=["sqlite"])
def catalog(request, tmp_path):
    """Create a DuckLake catalog with some tables via DuckDB."""
    import duckdb

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
    conn.execute("CREATE TABLE test.main.users (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO test.main.users VALUES (1, 'Alice'), (2, 'Bob')")
    conn.execute("CREATE TABLE test.main.orders (id INTEGER, user_id INTEGER, amount DOUBLE)")
    conn.execute("INSERT INTO test.main.orders VALUES (1, 1, 99.99)")
    conn.close()
    return metadata_path


class TestListTables:
    def test_list_tables_polars(self, catalog):
        tables = ducklake_polars.list_tables(catalog)
        assert set(tables) == {"users", "orders"}

    def test_list_tables_pandas(self, catalog):
        tables = ducklake_pandas.list_tables(catalog)
        assert set(tables) == {"users", "orders"}

    def test_list_tables_empty_catalog(self, tmp_path):
        import duckdb

        metadata_path = str(tmp_path / "empty.ducklake")
        data_path = str(tmp_path / "empty_data")

        conn = duckdb.connect()
        conn.install_extension("ducklake")
        conn.load_extension("ducklake")
        conn.install_extension("sqlite_scanner")
        conn.load_extension("sqlite_scanner")
        conn.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS test (DATA_PATH '{data_path}')"
        )
        conn.close()
        tables = ducklake_polars.list_tables(metadata_path)
        assert tables == []


class TestListSnapshots:
    def test_list_snapshots(self, catalog):
        snapshots = ducklake_polars.list_snapshots(catalog)
        assert len(snapshots) > 0
        assert "snapshot_id" in snapshots[0]
        assert "snapshot_time" in snapshots[0]

    def test_list_snapshots_limit(self, catalog):
        snapshots = ducklake_polars.list_snapshots(catalog, limit=2)
        assert len(snapshots) <= 2

    def test_list_snapshots_pandas(self, catalog):
        snapshots = ducklake_pandas.list_snapshots(catalog)
        assert len(snapshots) > 0


class TestCatalogInfo:
    def test_catalog_info(self, catalog):
        info = ducklake_polars.catalog_info(catalog)
        assert info["version"] in ("0.3", "0.4", "1.0")
        assert info["table_count"] == 2
        assert info["snapshot_count"] > 0
        assert "data_path" in info

    def test_catalog_info_pandas(self, catalog):
        info = ducklake_pandas.catalog_info(catalog)
        assert info["table_count"] == 2
