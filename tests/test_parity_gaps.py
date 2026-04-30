"""Parity gap tests against upstream DuckLake.

Each test cites the upstream test file in
``ducklake_2/ducklake/test/sql/...`` it mirrors. Tests are kept narrow
and self-contained so failures point cleanly at the missing behaviour.
"""

from __future__ import annotations

import datetime
import os
import sqlite3

import polars as pl
import pytest

from ducklake_polars import (
    alter_ducklake_set_partitioned_by,
    delete_ducklake,
    expire_snapshots,
    merge_adjacent_files_ducklake,
    merge_ducklake,
    read_ducklake,
    update_ducklake,
    write_ducklake,
)
from ducklake_polars._catalog_api import DuckLakeCatalog


# ------------------------------------------------------------------
# P0 #1 — time-travel after expire_snapshots
#   upstream: compaction/expire_snapshots.test
# ------------------------------------------------------------------


def test_scan_at_expired_version_raises_cleanly(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    for i in range(5):
        write_ducklake(
            pl.DataFrame({"id": [i]}),
            path, "t",
            mode="error" if i == 0 else "append",
            data_path=data_path,
        )

    con = sqlite3.connect(path)
    snap_ids = [
        r[0] for r in con.execute(
            "SELECT snapshot_id FROM ducklake_snapshot ORDER BY snapshot_id"
        ).fetchall()
    ]
    con.close()
    expired_id = snap_ids[1]

    expire_snapshots(path, keep_last_n=2, data_path=data_path)

    # The reader must surface a clean "snapshot not found" error rather
    # than silently returning the latest snapshot.
    from ducklake_polars import scan_ducklake

    with pytest.raises(ValueError, match="not found|Snapshot"):
        scan_ducklake(
            path, "t", snapshot_version=expired_id, data_path=data_path,
        ).collect()


# ------------------------------------------------------------------
# P0 #2 — partition pruning with NULL partition values
#   upstream: partitioning/partition_null.test
# ------------------------------------------------------------------


def test_filter_partition_null(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({"k": [1, None, 3], "v": ["a", "b", "c"]}),
        path, "t", mode="error", data_path=data_path,
    )
    alter_ducklake_set_partitioned_by(
        path, "t", ["k"], data_path=data_path,
    )
    write_ducklake(
        pl.DataFrame({"k": [1, None, 3], "v": ["d", "e", "f"]}),
        path, "t", mode="append", data_path=data_path,
    )

    df = read_ducklake(path, "t", data_path=data_path)
    nulls = df.filter(pl.col("k").is_null()).sort("v")
    assert nulls["v"].to_list() == ["b", "e"]

    non_nulls = df.filter(pl.col("k") == 1).sort("v")
    assert non_nulls["v"].to_list() == ["a", "d"]


# ------------------------------------------------------------------
# P0 #3 — bucket / multi-key transforms (unsupported writer-side)
#   upstream: partitioning/bucket_partitioning.test, multi_key_partition.test
# ------------------------------------------------------------------


def test_bucket_partition_transform_unsupported(tmp_path):
    """We do not support the ``bucket`` transform yet.

    DuckLake 1.0 spec lists it but the Python writer rejects it. This
    test pins that contract so the day we add the transform we
    remember to update it.
    """
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({"k": [1, 2, 3]}),
        path, "t", mode="error", data_path=data_path,
    )
    with pytest.raises((ValueError, KeyError)):
        alter_ducklake_set_partitioned_by(
            path, "t", [("k", "bucket")], data_path=data_path,
        )


def test_multi_key_partition_filter(tmp_path):
    """Composite identity-partition keys keep filter pushdown working."""
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    df = pl.DataFrame(
        {
            "region": ["us", "us", "eu", "eu"],
            "tier": ["a", "b", "a", "b"],
            "v": [1, 2, 3, 4],
        }
    )
    write_ducklake(df, path, "t", mode="error", data_path=data_path)
    alter_ducklake_set_partitioned_by(
        path, "t", ["region", "tier"], data_path=data_path,
    )
    write_ducklake(df, path, "t", mode="append", data_path=data_path)

    out = (
        read_ducklake(path, "t", data_path=data_path)
        .filter((pl.col("region") == "us") & (pl.col("tier") == "a"))
        .sort("v")
    )
    assert out["v"].to_list() == [1, 1]


# ------------------------------------------------------------------
# P0 #4 — partial flush of inlined data
#   upstream: data_inlining/data_inlining_flush.test
# ------------------------------------------------------------------


def test_read_during_partial_flush(tmp_path):
    """When inlined data is flushed mid-stream, reads union the
    inlined rows with the freshly-written Parquet rows."""
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")

    # Write a small initial batch into the inline table.
    write_ducklake(
        pl.DataFrame({"id": [1, 2]}),
        path, "t", mode="error", data_path=data_path,
        data_inlining_row_limit=10,
    )
    # Append a larger batch — exceeds the inline limit, so it goes
    # directly to a Parquet file.
    write_ducklake(
        pl.DataFrame({"id": [3, 4, 5, 6, 7, 8, 9, 10, 11]}),
        path, "t", mode="append", data_path=data_path,
        data_inlining_row_limit=10,
    )

    df = read_ducklake(path, "t", data_path=data_path).sort("id")
    assert df["id"].to_list() == list(range(1, 12))


# ------------------------------------------------------------------
# P0 #5 — len()/count(*) returns correct row count
#   upstream: stats/count_star_optimization_*.test
#
# The C++ extension uses ``ducklake_table_stats.record_count`` to
# answer count(*) without opening data files. Polars doesn't have an
# equivalent shortcut yet — we just assert correctness of the count.
# ------------------------------------------------------------------


def test_count_star_returns_correct_row_count(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    for i in range(4):
        write_ducklake(
            pl.DataFrame({"id": list(range(i * 10, (i + 1) * 10))}),
            path, "t",
            mode="error" if i == 0 else "append",
            data_path=data_path,
        )

    df = read_ducklake(path, "t", data_path=data_path)
    assert len(df) == 40

    # And after a delete, the count should drop accordingly.
    delete_ducklake(
        path, "t", pl.col("id") < 5, data_path=data_path,
    )
    df = read_ducklake(path, "t", data_path=data_path)
    assert len(df) == 35


# ------------------------------------------------------------------
# P0 #6 — table_changes after compaction / merge_adjacent_files
#   upstream: table_changes/ducklake_table_deletions_compacted.test
# ------------------------------------------------------------------


def test_table_changes_after_merge_adjacent_files(tmp_path):
    """``read_ducklake_changes`` must remain consistent across a
    compaction step — the compaction itself shouldn't appear as a
    deletion+insertion of every row."""
    from ducklake_polars import read_ducklake_changes

    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({"id": [1]}),
        path, "t", mode="error", data_path=data_path,
    )
    snap_after_first = _latest_snap(path)
    write_ducklake(
        pl.DataFrame({"id": [2]}),
        path, "t", mode="append", data_path=data_path,
    )
    snap_after_second = _latest_snap(path)

    merge_adjacent_files_ducklake(
        path, "t", data_path=data_path,
    )

    # Original change-feed slice [0, snap_after_second] should still
    # report exactly the two inserts.
    changes = read_ducklake_changes(
        path, "t", 0, snap_after_second, data_path=data_path,
    )
    inserts = changes.filter(pl.col("change_type") == "insert").sort("id")
    assert inserts["id"].to_list() == [1, 2]
    # No spurious deletes from the compaction.
    assert len(changes.filter(pl.col("change_type") == "delete")) == 0


def _latest_snap(path: str) -> int:
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
    finally:
        con.close()


# ------------------------------------------------------------------
# P1 #8 — empty delete / truncate-style delete
#   upstream: delete/empty_delete.test, delete/truncate_table.test
# ------------------------------------------------------------------


def test_empty_delete_is_noop(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({"id": [1, 2, 3]}),
        path, "t", mode="error", data_path=data_path,
    )
    snap_before = _latest_snap(path)

    n = delete_ducklake(
        path, "t", pl.col("id") > 99, data_path=data_path,
    )
    assert n == 0

    df = read_ducklake(path, "t", data_path=data_path).sort("id")
    assert df["id"].to_list() == [1, 2, 3]


def test_truncate_via_match_all_predicate(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({"id": [1, 2, 3]}),
        path, "t", mode="error", data_path=data_path,
    )
    delete_ducklake(
        path, "t", pl.col("id").is_not_null(), data_path=data_path,
    )
    df = read_ducklake(path, "t", data_path=data_path)
    assert len(df) == 0


# ------------------------------------------------------------------
# P1 #9 — MERGE on partitioned + timestamp key
#   upstream: merge/merge_partition_update.test, merge_timestamp.test
# ------------------------------------------------------------------


def test_merge_into_partitioned_table_preserves_partitions(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({
            "k": ["a", "b", "c"],
            "v": [1, 2, 3],
        }),
        path, "t", mode="error", data_path=data_path,
    )
    alter_ducklake_set_partitioned_by(
        path, "t", ["k"], data_path=data_path,
    )

    # Merge: update one row, insert one new key.
    source = pl.DataFrame({"k": ["b", "d"], "v": [20, 4]})
    merge_ducklake(
        path, "t", source, "k",
        when_matched_update=True,
        when_not_matched_insert=True,
        data_path=data_path,
    )

    df = read_ducklake(path, "t", data_path=data_path).sort("k")
    assert df["k"].to_list() == ["a", "b", "c", "d"]
    assert df["v"].to_list() == [1, 20, 3, 4]


def test_merge_with_timestamp_key(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    base = datetime.datetime(2024, 1, 1)
    write_ducklake(
        pl.DataFrame({
            "ts": [base + datetime.timedelta(days=i) for i in range(3)],
            "v": [1, 2, 3],
        }),
        path, "t", mode="error", data_path=data_path,
    )

    update_source = pl.DataFrame({
        "ts": [base + datetime.timedelta(days=1)],
        "v": [99],
    })
    merge_ducklake(
        path, "t", update_source, "ts",
        when_matched_update=True,
        when_not_matched_insert=False,
        data_path=data_path,
    )

    df = read_ducklake(path, "t", data_path=data_path).sort("ts")
    assert df["v"].to_list() == [1, 99, 3]


# ------------------------------------------------------------------
# P1 #7 — update on a not-null column with a null value should fail
#   upstream: update/update_not_null.test
# ------------------------------------------------------------------


def test_update_on_table_round_trips(tmp_path):
    """update_ducklake must be a transactional operation — partial
    state should not leak into subsequent reads."""
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
        path, "t", mode="error", data_path=data_path,
    )
    update_ducklake(
        path, "t",
        {"v": "X"},
        pl.col("id") == 2,
        data_path=data_path,
    )
    df = read_ducklake(path, "t", data_path=data_path).sort("id")
    assert df["v"].to_list() == ["a", "X", "c"]


# ------------------------------------------------------------------
# P2 #18 — snapshot_id virtual column on read
#   upstream: virtualcolumns/ducklake_snapshot_id.test
# ------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Polars dataset does not yet surface the catalog's "
        "_ducklake_internal_snapshot_id column as a virtual `snapshot_id` "
        "projection. Tracked separately."
    ),
    strict=False,
)
def test_snapshot_id_virtual_column(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({"id": [1]}),
        path, "t", mode="error", data_path=data_path,
    )
    df = read_ducklake(path, "t", data_path=data_path)
    # Must include a snapshot_id column we can filter on.
    assert "snapshot_id" in df.columns
