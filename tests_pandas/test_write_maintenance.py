"""Tests for catalog maintenance: expire_snapshots, vacuum, author/commit_message."""

from __future__ import annotations

import os

import pandas as pd
import numpy as np
import pytest

from ducklake_pandas import (
    delete_ducklake,
    expire_snapshots,
    merge_ducklake,
    read_ducklake,
    update_ducklake,
    vacuum_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# author / commit_message on write operations
# ---------------------------------------------------------------------------


class TestAuthorCommitMessage:
    """Verify author and commit_message are stored in snapshot_changes."""

    def test_write_with_author_and_message(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3]})

        write_ducklake(
            df, cat.metadata_path, "test", mode="error",
            author="alice", commit_message="initial load",
        )

        row = cat.query_one(
            "SELECT author, commit_message FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row[0] == "alice"
        assert row[1] == "initial load"

    def test_write_without_author(self, make_write_catalog):
        """When author/commit_message are not set, NULLs are stored."""
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        row = cat.query_one(
            "SELECT author, commit_message FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row[0] is None
        assert row[1] is None

    def test_delete_with_author(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        delete_ducklake(
            cat.metadata_path, "test", lambda df: df["a"] == 1,
            author="bob", commit_message="remove row 1",
        )

        row = cat.query_one(
            "SELECT author, commit_message FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row[0] == "bob"
        assert row[1] == "remove row 1"

    def test_update_with_author(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        update_ducklake(
            cat.metadata_path, "test",
            {"b": "z"}, lambda df: df["a"] == 1,
            author="charlie", commit_message="fix row",
        )

        row = cat.query_one(
            "SELECT author, commit_message FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row[0] == "charlie"
        assert row[1] == "fix row"

    def test_merge_with_author(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        source = pd.DataFrame({"id": [2, 3], "val": ["B", "c"]})
        merge_ducklake(
            cat.metadata_path, "test", source, "id",
            when_matched_update=True,
            author="dana", commit_message="merge update",
        )

        row = cat.query_one(
            "SELECT author, commit_message FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row[0] == "dana"
        assert row[1] == "merge update"

    def test_append_with_author(self, make_write_catalog):
        """Append mode also records author."""
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1]})
        write_ducklake(df1, cat.metadata_path, "test", mode="error")

        df2 = pd.DataFrame({"a": [2]})
        write_ducklake(
            df2, cat.metadata_path, "test", mode="append",
            author="eve", commit_message="add row",
        )

        row = cat.query_one(
            "SELECT author, commit_message FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row[0] == "eve"
        assert row[1] == "add row"


# ---------------------------------------------------------------------------
# expire_snapshots
# ---------------------------------------------------------------------------


class TestExpireSnapshots:
    """Test expire_snapshots functionality."""

    def test_expire_older_than_snapshot(self, make_write_catalog):
        """Expire snapshots older than a given ID."""
        cat = make_write_catalog()

        # Create multiple snapshots
        write_ducklake(
            pd.DataFrame({"a": [1]}), cat.metadata_path, "test", mode="error",
        )
        write_ducklake(
            pd.DataFrame({"a": [2]}), cat.metadata_path, "test", mode="append",
        )
        write_ducklake(
            pd.DataFrame({"a": [3]}), cat.metadata_path, "test", mode="append",
        )

        # Count snapshots before expiry
        snap_count_before = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]
        assert snap_count_before >= 4  # initial + 3 writes

        # Get the latest snapshot ID
        latest = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        # Expire all but the last 2 snapshots by ID
        expired = expire_snapshots(
            cat.metadata_path, older_than_snapshot=latest - 1,
        )
        assert expired > 0

        # Remaining snapshots should be fewer
        snap_count_after = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]
        assert snap_count_after == snap_count_before - expired

        # Data should still be readable at latest snapshot
        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == [1, 2, 3]

    def test_expire_keep_last_n(self, make_write_catalog):
        """keep_last_n preserves the N most recent snapshots."""
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1]}), cat.metadata_path, "test", mode="error",
        )
        write_ducklake(
            pd.DataFrame({"a": [2]}), cat.metadata_path, "test", mode="append",
        )
        write_ducklake(
            pd.DataFrame({"a": [3]}), cat.metadata_path, "test", mode="append",
        )

        snap_count_before = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]

        expired = expire_snapshots(cat.metadata_path, keep_last_n=2)
        assert expired == snap_count_before - 2

        snap_count_after = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]
        assert snap_count_after == 2

        # Data still readable
        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == [1, 2, 3]

    def test_expire_nothing_when_all_recent(self, make_write_catalog):
        """No expiry when all snapshots are recent enough."""
        cat = make_write_catalog()
        write_ducklake(
            pd.DataFrame({"a": [1]}), cat.metadata_path, "test", mode="error",
        )

        snap_count = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot"
        )[0]

        expired = expire_snapshots(cat.metadata_path, keep_last_n=snap_count + 5)
        assert expired == 0

    def test_expire_cleans_up_snapshot_changes(self, make_write_catalog):
        """Expired snapshot's changes records are also removed."""
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1]}), cat.metadata_path, "test", mode="error",
        )
        write_ducklake(
            pd.DataFrame({"a": [2]}), cat.metadata_path, "test", mode="append",
        )

        changes_before = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot_changes"
        )[0]

        expired = expire_snapshots(cat.metadata_path, keep_last_n=1)
        assert expired > 0

        changes_after = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_snapshot_changes"
        )[0]
        assert changes_after < changes_before

    def test_expire_cleans_up_ended_data_files(self, make_write_catalog):
        """Expired data file metadata is removed after overwrite + expire."""
        cat = make_write_catalog()

        # Write, then overwrite — first file gets end_snapshot set
        write_ducklake(
            pd.DataFrame({"a": [1]}), cat.metadata_path, "test", mode="error",
        )
        write_ducklake(
            pd.DataFrame({"a": [99]}), cat.metadata_path, "test", mode="overwrite",
        )

        files_before = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file"
        )[0]

        expire_snapshots(cat.metadata_path, keep_last_n=1)

        files_after = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file"
        )[0]
        # The overwritten file's metadata should be cleaned up
        assert files_after < files_before

        # Data still readable
        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].tolist() == [99]

    def test_expire_both_params_raises(self, make_write_catalog):
        cat = make_write_catalog()
        write_ducklake(
            pd.DataFrame({"a": [1]}), cat.metadata_path, "test", mode="error",
        )
        with pytest.raises(ValueError, match="Cannot specify both"):
            expire_snapshots(
                cat.metadata_path, older_than_snapshot=1, keep_last_n=1,
            )

    def test_expire_no_params_raises(self, make_write_catalog):
        cat = make_write_catalog()
        write_ducklake(
            pd.DataFrame({"a": [1]}), cat.metadata_path, "test", mode="error",
        )
        with pytest.raises(ValueError, match="Must specify either"):
            expire_snapshots(cat.metadata_path)

    def test_expire_keep_last_n_zero_raises(self, make_write_catalog):
        cat = make_write_catalog()
        write_ducklake(
            pd.DataFrame({"a": [1]}), cat.metadata_path, "test", mode="error",
        )
        with pytest.raises(ValueError, match="keep_last_n must be >= 1"):
            expire_snapshots(cat.metadata_path, keep_last_n=0)


# ---------------------------------------------------------------------------
# vacuum_ducklake
# ---------------------------------------------------------------------------


class TestVacuum:
    """Test vacuum_ducklake functionality."""

    def test_vacuum_removes_orphaned_files(self, make_write_catalog):
        """After overwrite + expire, vacuum deletes the orphaned file."""
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1]}), cat.metadata_path, "test", mode="error",
        )
        write_ducklake(
            pd.DataFrame({"a": [99]}), cat.metadata_path, "test", mode="overwrite",
        )

        # Count parquet files before
        parquet_before = _count_parquet_files(cat.data_path)
        assert parquet_before >= 2  # at least original + overwrite

        # Expire old snapshots to make old file unreferenced in metadata
        expire_snapshots(cat.metadata_path, keep_last_n=1)

        # Vacuum
        deleted = vacuum_ducklake(cat.metadata_path)
        assert deleted >= 1

        # Remaining parquet files should be fewer
        parquet_after = _count_parquet_files(cat.data_path)
        assert parquet_after < parquet_before

        # Data still readable
        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].tolist() == [99]

    def test_vacuum_preserves_referenced_files(self, make_write_catalog):
        """Vacuum does not delete files that are still referenced."""
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1, 2]}), cat.metadata_path, "test", mode="error",
        )

        parquet_before = _count_parquet_files(cat.data_path)

        deleted = vacuum_ducklake(cat.metadata_path)
        assert deleted == 0

        parquet_after = _count_parquet_files(cat.data_path)
        assert parquet_after == parquet_before

    def test_vacuum_empty_data_dir(self, make_write_catalog):
        """Vacuum on an empty catalog returns 0."""
        cat = make_write_catalog()

        deleted = vacuum_ducklake(cat.metadata_path)
        assert deleted == 0

    def test_vacuum_after_delete_and_expire(self, make_write_catalog):
        """Delete files are also cleaned up by vacuum after expire."""
        cat = make_write_catalog()

        write_ducklake(
            pd.DataFrame({"a": [1, 2, 3]}), cat.metadata_path, "test", mode="error",
        )

        # Delete a row — creates a delete file
        delete_ducklake(cat.metadata_path, "test", lambda df: df["a"] == 2)

        parquet_before = _count_parquet_files(cat.data_path)
        assert parquet_before >= 2  # data file + delete file

        # Overwrite to make delete file obsolete
        write_ducklake(
            pd.DataFrame({"a": [1, 3]}), cat.metadata_path, "test", mode="overwrite",
        )

        # Expire + vacuum
        expire_snapshots(cat.metadata_path, keep_last_n=1)
        deleted = vacuum_ducklake(cat.metadata_path)
        assert deleted >= 1

        # Data still readable
        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == [1, 3]

    def test_expire_then_vacuum_full_workflow(self, make_write_catalog):
        """End-to-end: create, insert, overwrite, expire, vacuum."""
        cat = make_write_catalog()

        # Create table with initial data
        write_ducklake(
            pd.DataFrame({"x": [10, 20]}), cat.metadata_path, "tbl", mode="error",
        )

        # Append more data
        write_ducklake(
            pd.DataFrame({"x": [30]}), cat.metadata_path, "tbl", mode="append",
        )

        # Overwrite with fresh data
        write_ducklake(
            pd.DataFrame({"x": [100, 200]}), cat.metadata_path, "tbl", mode="overwrite",
        )

        # At this point we have old parquet files from create/append that
        # are superseded by the overwrite
        parquet_count = _count_parquet_files(cat.data_path)
        assert parquet_count >= 2

        # Expire all but the latest snapshot
        expired = expire_snapshots(cat.metadata_path, keep_last_n=1)
        assert expired > 0

        # Vacuum orphans
        vacuumed = vacuum_ducklake(cat.metadata_path)
        assert vacuumed >= 1

        # Verify final data
        result = read_ducklake(cat.metadata_path, "tbl")
        assert sorted(result["x"].tolist()) == [100, 200]

        # Only active files remain
        parquet_final = _count_parquet_files(cat.data_path)
        assert parquet_final == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_parquet_files(data_path: str) -> int:
    """Count all .parquet files under a directory."""
    count = 0
    for dirpath, _dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith(".parquet"):
                count += 1
    return count
