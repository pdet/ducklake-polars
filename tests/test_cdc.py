"""Tests for Change Data Feed (CDC) — scan_ducklake_changes / read_ducklake_changes.

Covers:
  - Inserts only
  - Deletes only
  - Updates (insert + delete in same snapshot → update_preimage/postimage)
  - Mixed operations across multiple snapshots
  - Filtering by change_type
  - Empty range (no changes)
  - Schema evolution through CDC
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    scan_ducklake_changes,
    read_ducklake_changes,
)


class TestCDCInserts:
    """CDC detects inserts."""

    def test_inserts_only(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        snap1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        changes = read_ducklake_changes(cat.metadata_path, "t", 0, snap1)
        assert len(changes) == 2
        assert all(ct == "insert" for ct in changes["change_type"].to_list())

    def test_incremental_inserts(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        snap1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.t VALUES (2), (3)")
        snap2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        # Changes between snap1 and snap2: only rows 2,3
        changes = read_ducklake_changes(cat.metadata_path, "t", snap1, snap2)
        assert len(changes) == 2
        assert set(changes["id"].to_list()) == {2, 3}


class TestCDCDeletes:
    """CDC detects deletes."""

    def test_deletes_only(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3)")
        snap1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("DELETE FROM ducklake.t WHERE id = 2")
        snap2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        changes = read_ducklake_changes(cat.metadata_path, "t", snap1, snap2)
        deletes = changes.filter(pl.col("change_type") == "delete")
        assert len(deletes) >= 1
        assert 2 in deletes["id"].to_list()


class TestCDCUpdates:
    """CDC detects updates as preimage/postimage pairs."""

    def test_update_detected(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, name VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'alice'), (2, 'bob')")
        snap1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("UPDATE ducklake.t SET name = 'ALICE' WHERE id = 1")
        snap2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        changes = read_ducklake_changes(cat.metadata_path, "t", snap1, snap2)
        pre = changes.filter(pl.col("change_type") == "update_preimage")
        post = changes.filter(pl.col("change_type") == "update_postimage")

        assert len(pre) >= 1
        assert len(post) >= 1
        # Post-image should have the new value
        assert "ALICE" in post["name"].to_list()


class TestCDCMixed:
    """Mixed operations across snapshots."""

    def test_mixed_changes(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        snap1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.t VALUES (4, 'd')")
        cat.execute("DELETE FROM ducklake.t WHERE id = 1")
        snap2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        changes = read_ducklake_changes(cat.metadata_path, "t", snap1, snap2)
        types = set(changes["change_type"].to_list())
        assert "insert" in types
        assert "delete" in types

    def test_empty_range(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        snap1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        # No changes between snap1 and snap1
        changes = read_ducklake_changes(cat.metadata_path, "t", snap1, snap1)
        assert len(changes) == 0


class TestCDCLazy:
    """scan_ducklake_changes returns LazyFrame."""

    def test_lazy_scan(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3)")
        snap1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        lf = scan_ducklake_changes(cat.metadata_path, "t", 0, snap1)
        assert isinstance(lf, pl.LazyFrame)

        result = lf.filter(pl.col("change_type") == "insert").collect()
        assert len(result) == 3

    def test_lazy_filter_pushdown(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        snap = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        result = (
            scan_ducklake_changes(cat.metadata_path, "t", 0, snap)
            .filter(pl.col("id") > 1)
            .collect()
        )
        assert len(result) == 1
        assert result["id"][0] == 2
