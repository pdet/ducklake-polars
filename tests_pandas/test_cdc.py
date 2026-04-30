"""Pandas parity tests for change data feed (read_ducklake_changes)."""

from __future__ import annotations

import sqlite3

import pandas as pd

from ducklake_pandas import (
    delete_ducklake,
    read_ducklake_changes,
    update_ducklake,
    write_ducklake,
)


def _setup(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pd.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
        path, "t", mode="error", data_path=data_path,
    )
    return path, data_path


def _latest_snapshot(path):
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
    finally:
        con.close()


def test_pandas_changes_basic_insert(tmp_path):
    path, data_path = _setup(tmp_path)
    snap = _latest_snapshot(path)
    df = read_ducklake_changes(path, "t", 0, snap, data_path=data_path)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"snapshot_id", "change_type", "id", "v"}
    inserts = df[df["change_type"] == "insert"]
    assert len(inserts) == 3


def test_pandas_changes_includes_delete(tmp_path):
    path, data_path = _setup(tmp_path)
    delete_ducklake(
        path, "t", lambda df: df["id"] == 2, data_path=data_path,
    )

    snap = _latest_snapshot(path)
    df = read_ducklake_changes(path, "t", 0, snap, data_path=data_path)
    assert "delete" in df["change_type"].unique().tolist()
    deleted = df[df["change_type"] == "delete"]
    assert deleted["id"].tolist() == [2]


def test_pandas_changes_detects_update(tmp_path):
    path, data_path = _setup(tmp_path)
    update_ducklake(
        path, "t", {"v": "X"}, lambda df: df["id"] == 1,
        data_path=data_path,
    )
    snap = _latest_snapshot(path)
    df = read_ducklake_changes(path, "t", 0, snap, data_path=data_path)
    types = set(df["change_type"].unique().tolist())
    assert {"update_preimage", "update_postimage"} <= types
