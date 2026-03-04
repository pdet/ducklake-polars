"""Tests for snapshot_changes API."""

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
    conn.execute("CREATE TABLE test.main.t (id INTEGER)")
    conn.execute("INSERT INTO test.main.t VALUES (1)")
    conn.execute("INSERT INTO test.main.t VALUES (2)")
    conn.close()
    return metadata_path


class TestSnapshotChanges:
    def test_all_changes(self, catalog):
        changes = ducklake_polars.snapshot_changes(catalog)
        assert len(changes) > 0
        assert all("snapshot_id" in c for c in changes)
        assert all("change" in c for c in changes)

    def test_specific_snapshot(self, catalog):
        changes = ducklake_polars.snapshot_changes(catalog, snapshot_id=1)
        assert len(changes) > 0
        assert all(c["snapshot_id"] == 1 for c in changes)

    def test_has_create_table_change(self, catalog):
        changes = ducklake_polars.snapshot_changes(catalog)
        create_changes = [c for c in changes if "created_table" in c["change"]]
        assert len(create_changes) > 0

    def test_pandas(self, catalog):
        changes = ducklake_pandas.snapshot_changes(catalog)
        assert len(changes) > 0
