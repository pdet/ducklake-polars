"""Encryption is deferred — verify the catalog reader surfaces a clean error."""

from __future__ import annotations

import sqlite3

import pytest

from ducklake_core._catalog import DuckLakeCatalogReader


def test_encrypted_catalog_raises_not_implemented(tmp_path):
    db_path = str(tmp_path / "enc.ducklake")
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE ducklake_metadata (
            key TEXT NOT NULL, value TEXT NOT NULL,
            scope TEXT, scope_id BIGINT
        );
        INSERT INTO ducklake_metadata VALUES ('version', '1.0', NULL, NULL);
        INSERT INTO ducklake_metadata VALUES ('encrypted', 'true', NULL, NULL);
    """)
    con.commit()
    con.close()

    with pytest.raises(NotImplementedError, match="encrypted"):
        with DuckLakeCatalogReader(db_path) as reader:
            reader.get_current_snapshot()


def test_unencrypted_catalog_loads_normally(tmp_path):
    """Sanity check: encrypted='false' (the bootstrap default) is fine."""
    from ducklake_core._bootstrap import bootstrap_catalog

    db_path = str(tmp_path / "ok.ducklake")
    bootstrap_catalog(db_path)
    with DuckLakeCatalogReader(db_path) as reader:
        snap = reader.get_current_snapshot()
        assert snap.snapshot_id == 0
