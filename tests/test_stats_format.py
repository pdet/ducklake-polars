"""Stat-string formatting tests (audit_compat.md P1 #5/#6/#10).

Three behaviors:

* Timestamp stats use ``+00`` (DuckDB convention), not ``+00:00``.
* TIME columns receive ISO-formatted min/max stats.
* Boolean columns emit no min/max (DuckDB-compatible).
"""

from __future__ import annotations

import datetime
import sqlite3

import polars as pl

from ducklake_polars import write_ducklake


def _stats_for(path, column_id):
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT min_value, max_value FROM ducklake_file_column_stats "
            "WHERE column_id = ?",
            [column_id],
        ).fetchone()
    finally:
        con.close()


def test_timestamp_stats_use_plus_zero_zero(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pl.DataFrame(
        {
            "ts": [
                datetime.datetime(2024, 1, 15, 10, 0, tzinfo=datetime.timezone.utc),
                datetime.datetime(2024, 6, 20, 9, 0, tzinfo=datetime.timezone.utc),
            ],
        }
    )
    write_ducklake(df, path, "events", mode="error", data_path=data_path)

    stats = _stats_for(path, 1)
    assert stats is not None
    min_val, max_val = stats
    # Either no offset (UTC stripped) or ``+00`` form — never ``+00:00``.
    assert "+00:00" not in (min_val or "")
    assert "+00:00" not in (max_val or "")


def test_time_column_emits_min_max(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pl.DataFrame(
        {
            "t": [
                datetime.time(10, 0, 0),
                datetime.time(14, 30, 5),
            ],
        }
    )
    write_ducklake(df, path, "events", mode="error", data_path=data_path)
    stats = _stats_for(path, 1)
    assert stats is not None
    min_val, max_val = stats
    assert min_val is not None and max_val is not None
    # ISO-formatted (HH:MM:SS[.ffffff])
    assert min_val.startswith("10:00:00")
    assert max_val.startswith("14:30:05")


def test_boolean_column_skips_min_max(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pl.DataFrame({"flag": [True, False, True]})
    write_ducklake(df, path, "events", mode="error", data_path=data_path)
    stats = _stats_for(path, 1)
    assert stats is not None
    assert stats[0] is None
    assert stats[1] is None
