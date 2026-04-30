"""Pandas parity tests for cleanup / orphan / merge-adjacent maintenance."""

from __future__ import annotations

import datetime
import os
import sqlite3

import pandas as pd

from ducklake_pandas import (
    cleanup_old_files_ducklake,
    delete_orphaned_files_ducklake,
    merge_adjacent_files_ducklake,
    write_ducklake,
)


def _make_two_file_catalog(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pd.DataFrame({"a": [1]}), path, "events",
        mode="error", data_path=data_path,
    )
    write_ducklake(
        pd.DataFrame({"a": [2]}), path, "events",
        mode="append", data_path=data_path,
    )
    return path, data_path


def _data_files(path):
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT data_file_id, path FROM ducklake_data_file "
            "WHERE end_snapshot IS NULL"
        ).fetchall()
    finally:
        con.close()


def test_pandas_merge_adjacent_files(tmp_path):
    path, data_path = _make_two_file_catalog(tmp_path)
    before = _data_files(path)
    assert len(before) == 2

    new_snap = merge_adjacent_files_ducklake(
        path, "events", data_path=data_path,
    )
    assert new_snap >= 0

    after = _data_files(path)
    # The two source files were replaced with a single merged file.
    assert len(after) == 1


def test_pandas_cleanup_after_merge(tmp_path):
    path, data_path = _make_two_file_catalog(tmp_path)
    merge_adjacent_files_ducklake(path, "events", data_path=data_path)

    deleted = cleanup_old_files_ducklake(
        path, cleanup_all=True, data_path=data_path,
    )
    # Two source files were scheduled for deletion by the merge.
    assert len(deleted) == 2

    # ducklake_files_scheduled_for_deletion is now empty.
    con = sqlite3.connect(path)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        ).fetchone()[0]
    finally:
        con.close()
    assert n == 0


def test_pandas_delete_orphaned_files(tmp_path):
    path, data_path = _make_two_file_catalog(tmp_path)
    # Drop a stray Parquet file the catalog never knew about.
    stray = os.path.join(data_path, "main", "events", "stray.parquet")
    os.makedirs(os.path.dirname(stray), exist_ok=True)
    with open(stray, "wb") as f:
        f.write(b"not a real parquet file")

    deleted = delete_orphaned_files_ducklake(
        path, data_path=data_path,
    )
    assert any("stray.parquet" in p for p in deleted)
    assert not os.path.exists(stray)


def test_pandas_cleanup_dry_run(tmp_path):
    path, data_path = _make_two_file_catalog(tmp_path)
    merge_adjacent_files_ducklake(path, "events", data_path=data_path)

    listed = cleanup_old_files_ducklake(
        path, cleanup_all=True, dry_run=True, data_path=data_path,
    )
    # Dry run reports the same files but leaves them in the table.
    con = sqlite3.connect(path)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        ).fetchone()[0]
    finally:
        con.close()
    assert n == len(listed) > 0
