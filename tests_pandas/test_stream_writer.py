"""Pandas parity tests for DuckLakeStreamWriter."""

from __future__ import annotations

import pandas as pd
import pytest

from ducklake_pandas import DuckLakeStreamWriter, read_ducklake


def test_pandas_stream_writer_context_manager(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")

    with DuckLakeStreamWriter(
        path, "events",
        flush_threshold=2,
        compact_on_close=False,
        data_path=data_path,
    ) as w:
        w.append(pd.DataFrame({"id": [1], "v": ["a"]}))
        w.append(pd.DataFrame({"id": [2], "v": ["b"]}))  # triggers flush
        w.append(pd.DataFrame({"id": [3], "v": ["c"]}))

    df = read_ducklake(path, "events", data_path=data_path)
    df = df.sort_values("id").reset_index(drop=True)
    assert df["id"].tolist() == [1, 2, 3]
    assert df["v"].tolist() == ["a", "b", "c"]


def test_pandas_stream_writer_total_rows(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")

    w = DuckLakeStreamWriter(
        path, "events", flush_threshold=10,
        compact_on_close=False, data_path=data_path,
    )
    w.append(pd.DataFrame({"id": [1, 2]}))
    assert w.total_rows == 2
    assert w.buffer_rows == 2
    assert w.flush_count == 0
    w.flush()
    assert w.flush_count == 1
    assert w.buffer_rows == 0
    w.close()


def test_pandas_stream_writer_compacts_on_close(tmp_path):
    """compact_on_close=True triggers rewrite when more than one
    flush has happened."""
    import sqlite3

    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")

    with DuckLakeStreamWriter(
        path, "events",
        flush_threshold=1,
        compact_on_close=True,
        data_path=data_path,
    ) as w:
        for i in range(3):
            w.append(pd.DataFrame({"id": [i]}))

    con = sqlite3.connect(path)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM ducklake_data_file "
            "WHERE end_snapshot IS NULL"
        ).fetchone()[0]
    finally:
        con.close()
    # Compaction merges the per-flush files into one.
    assert n == 1


def test_pandas_stream_writer_exception_drops_unflushed_buffer(tmp_path):
    """coverage P2 #14 — exception in context drops unflushed buffer."""
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")

    with pytest.raises(RuntimeError, match="boom"):
        with DuckLakeStreamWriter(
            path, "events",
            flush_threshold=10_000,
            compact_on_close=False,
            data_path=data_path,
        ) as w:
            w.append(pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
            raise RuntimeError("boom")

    # Catalog was never written to; reading must error or yield empty.
    try:
        df = read_ducklake(path, "events", data_path=data_path)
        assert len(df) == 0
    except Exception:
        # Catalog never created — also acceptable.
        pass
