"""Pandas parity tests for non-identity partition transforms."""

from __future__ import annotations

import datetime
import sqlite3

import pandas as pd

from ducklake_pandas import (
    alter_ducklake_set_partitioned_by,
    read_ducklake,
    write_ducklake,
)


def _make_table(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "ts": [
            datetime.datetime(2024, 1, 15, 10, 0),
            datetime.datetime(2024, 1, 15, 14, 0),
            datetime.datetime(2025, 6, 20, 9, 0),
        ],
    })
    write_ducklake(df, path, "events", mode="error", data_path=data_path)
    return path, data_path


def test_pandas_set_partitioned_by_year_transform(tmp_path):
    path, data_path = _make_table(tmp_path)
    alter_ducklake_set_partitioned_by(
        path, "events", [("ts", "year")], data_path=data_path,
    )

    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT pc.transform "
            "FROM ducklake_partition_column pc "
            "JOIN ducklake_partition_info pi ON pc.partition_id = pi.partition_id "
            "WHERE pi.end_snapshot IS NULL"
        ).fetchall()
    finally:
        con.close()
    assert rows == [("year",)]


def test_pandas_set_partitioned_by_identity_via_string(tmp_path):
    """Bare-string entries default to identity transform."""
    path, data_path = _make_table(tmp_path)
    alter_ducklake_set_partitioned_by(
        path, "events", ["id"], data_path=data_path,
    )

    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT pc.transform "
            "FROM ducklake_partition_column pc "
            "JOIN ducklake_partition_info pi ON pc.partition_id = pi.partition_id "
            "WHERE pi.end_snapshot IS NULL"
        ).fetchall()
    finally:
        con.close()
    assert rows == [("identity",)]


def test_pandas_partitioned_read_round_trip(tmp_path):
    path, data_path = _make_table(tmp_path)
    alter_ducklake_set_partitioned_by(
        path, "events", [("ts", "month")], data_path=data_path,
    )
    # Append more data so the partition transform actually runs.
    write_ducklake(
        pd.DataFrame({
            "id": [4, 5],
            "ts": [
                datetime.datetime(2024, 2, 1, 10, 0),
                datetime.datetime(2024, 3, 1, 10, 0),
            ],
        }),
        path, "events", mode="append", data_path=data_path,
    )

    df = read_ducklake(path, "events", data_path=data_path).sort_values("id")
    assert df["id"].tolist() == [1, 2, 3, 4, 5]
