"""Tests for cleanup_old_files / delete_orphaned_files."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone

import duckdb
import polars as pl
import pytest

from ducklake_polars import (
    cleanup_old_files_ducklake,
    delete_orphaned_files_ducklake,
    merge_adjacent_files_ducklake,
    read_ducklake,
    write_ducklake,
)


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


# ------------------------------------------------------------------
# cleanup_old_files: removes scheduled-for-deletion entries
# ------------------------------------------------------------------


class TestCleanupOldFiles:
    def test_requires_either_older_than_or_cleanup_all(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, path, "t", mode="error", data_path=data_path)

        with pytest.raises(ValueError, match="older_than or cleanup_all"):
            cleanup_old_files_ducklake(path, data_path=data_path)

    def test_no_op_when_nothing_scheduled(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, path, "t", mode="error", data_path=data_path)

        deleted = cleanup_old_files_ducklake(
            path, cleanup_all=True, data_path=data_path,
        )
        assert deleted == []

    def test_cleans_up_after_merge(self, tmp_path):
        """After merge_adjacent_files, source files are scheduled and
        cleanup_old_files removes them physically."""
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})
        write_ducklake(df1, path, "t", mode="error", data_path=data_path)
        write_ducklake(df2, path, "t", mode="append", data_path=data_path)

        files_before = os.listdir(os.path.join(data_path, "main", "t"))
        assert len(files_before) == 2

        merge_adjacent_files_ducklake(path, "t", data_path=data_path)

        # After merge: 3 parquet files exist on disk (2 source + 1 merged),
        # 2 source files are queued for deletion.
        files_after_merge = os.listdir(os.path.join(data_path, "main", "t"))
        assert len(files_after_merge) == 3

        con = sqlite3.connect(path)
        try:
            scheduled = con.execute(
                "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
            ).fetchone()[0]
        finally:
            con.close()
        assert scheduled == 2

        deleted = cleanup_old_files_ducklake(
            path, cleanup_all=True, data_path=data_path,
        )
        assert len(deleted) == 2

        # Only the merged file remains
        files_after_cleanup = os.listdir(os.path.join(data_path, "main", "t"))
        assert len(files_after_cleanup) == 1

        # Schedule queue is empty
        con = sqlite3.connect(path)
        try:
            scheduled = con.execute(
                "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
            ).fetchone()[0]
        finally:
            con.close()
        assert scheduled == 0

        # Data is still readable
        result = read_ducklake(path, "t")
        assert sorted(result["a"].to_list()) == [1, 2, 3, 4]

    def test_dry_run(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df1 = pl.DataFrame({"a": [1]})
        df2 = pl.DataFrame({"a": [2]})
        write_ducklake(df1, path, "t", mode="error", data_path=data_path)
        write_ducklake(df2, path, "t", mode="append", data_path=data_path)
        merge_adjacent_files_ducklake(path, "t", data_path=data_path)

        deleted = cleanup_old_files_ducklake(
            path, cleanup_all=True, dry_run=True, data_path=data_path,
        )
        assert len(deleted) == 2

        # Nothing actually removed
        files_on_disk = os.listdir(os.path.join(data_path, "main", "t"))
        assert len(files_on_disk) == 3

        con = sqlite3.connect(path)
        try:
            scheduled = con.execute(
                "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
            ).fetchone()[0]
        finally:
            con.close()
        assert scheduled == 2

    def test_older_than_filters_recent(self, tmp_path):
        """Files queued seconds ago are not removed by an older threshold."""
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df1 = pl.DataFrame({"a": [1]})
        df2 = pl.DataFrame({"a": [2]})
        write_ducklake(df1, path, "t", mode="error", data_path=data_path)
        write_ducklake(df2, path, "t", mode="append", data_path=data_path)
        merge_adjacent_files_ducklake(path, "t", data_path=data_path)

        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        deleted = cleanup_old_files_ducklake(
            path, older_than=cutoff, data_path=data_path,
        )
        assert deleted == []

        # The schedule queue still has 2 entries
        con = sqlite3.connect(path)
        try:
            scheduled = con.execute(
                "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
            ).fetchone()[0]
        finally:
            con.close()
        assert scheduled == 2


# ------------------------------------------------------------------
# delete_orphaned_files
# ------------------------------------------------------------------


class TestDeleteOrphanedFiles:
    def test_removes_unreferenced(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, path, "t", mode="error", data_path=data_path)

        # Drop an orphan into the table directory
        table_dir = os.path.join(data_path, "main", "t")
        orphan = os.path.join(table_dir, "stray.parquet")
        # Reuse an existing parquet file as content
        existing = os.listdir(table_dir)[0]
        import shutil
        shutil.copy(os.path.join(table_dir, existing), orphan)

        deleted = delete_orphaned_files_ducklake(path, data_path=data_path)
        assert orphan in deleted
        assert not os.path.exists(orphan)

    def test_dry_run(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, path, "t", mode="error", data_path=data_path)

        table_dir = os.path.join(data_path, "main", "t")
        orphan = os.path.join(table_dir, "stray.parquet")
        existing = os.listdir(table_dir)[0]
        import shutil
        shutil.copy(os.path.join(table_dir, existing), orphan)

        deleted = delete_orphaned_files_ducklake(
            path, dry_run=True, data_path=data_path,
        )
        assert orphan in deleted
        assert os.path.exists(orphan)

    def test_does_not_remove_scheduled(self, tmp_path):
        """Files queued for deletion are not orphans — cleanup_old_files owns them."""
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        df1 = pl.DataFrame({"a": [1]})
        df2 = pl.DataFrame({"a": [2]})
        write_ducklake(df1, path, "t", mode="error", data_path=data_path)
        write_ducklake(df2, path, "t", mode="append", data_path=data_path)
        merge_adjacent_files_ducklake(path, "t", data_path=data_path)

        deleted = delete_orphaned_files_ducklake(path, data_path=data_path)
        # The two scheduled files are NOT orphans
        assert deleted == []

        # All 3 parquet files still on disk
        files_on_disk = os.listdir(os.path.join(data_path, "main", "t"))
        assert len(files_on_disk) == 3


# ------------------------------------------------------------------
# DuckDB roundtrip
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_sees_scheduled_deletions(tmp_path):
    """A DuckLake-extension reader sees the same scheduled-for-deletion rows."""
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")

    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [3, 4]})
    write_ducklake(df1, path, "t", mode="error", data_path=data_path)
    write_ducklake(df2, path, "t", mode="append", data_path=data_path)
    merge_adjacent_files_ducklake(path, "t", data_path=data_path)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{path}' AS d (DATA_PATH '{data_path}')"
    )
    rows = con.execute("SELECT a FROM d.t ORDER BY a").fetchall()
    con.close()
    assert [r[0] for r in rows] == [1, 2, 3, 4]
