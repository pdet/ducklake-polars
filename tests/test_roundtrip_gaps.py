"""Roundtrip-coverage gap tests against DuckDB.

WR = "we write, DuckDB reads"; DR = "DuckDB writes, we read".

Each test is gated by ``_duckdb_supports_v10`` when the underlying
feature requires the v1.0 catalog support that landed in the DuckDB
ducklake extension shipped with DuckDB >= 1.5.
"""

from __future__ import annotations

import sqlite3

import duckdb
import polars as pl
import pytest

from ducklake_polars import (
    alter_ducklake_reset_sort_keys,
    alter_ducklake_set_sort_keys,
    cleanup_old_files_ducklake,
    create_ducklake_macro,
    create_table_as_ducklake,
    delete_ducklake_column_tag,
    delete_ducklake_table_tag,
    drop_ducklake_macro,
    expire_snapshots,
    merge_ducklake,
    read_ducklake,
    rewrite_data_files_ducklake,
    set_ducklake_column_tag,
    set_ducklake_option,
    set_ducklake_table_tag,
    update_ducklake,
    write_ducklake,
    DuckLakeStreamWriter,
)
from ducklake_polars._catalog_api import DuckLakeCatalog


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


def _attach(con: duckdb.DuckDBPyConnection, path: str, data_path: str):
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{path}' AS d (DATA_PATH '{data_path}')"
    )


def _make_table(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
        path, "t", mode="error", data_path=data_path,
    )
    return path, data_path


# ------------------------------------------------------------------
# P0 #1 — alter_ducklake_reset_sort_keys roundtrip
# ------------------------------------------------------------------


def test_reset_sort_keys_we_write_duckdb_reads_empty(tmp_path):
    path, data_path = _make_table(tmp_path)
    alter_ducklake_set_sort_keys(
        path, "t", ["id"], data_path=data_path,
    )
    alter_ducklake_reset_sort_keys(path, "t", data_path=data_path)

    # After reset, no active sort_info row should exist.
    con = sqlite3.connect(path)
    try:
        active = con.execute(
            "SELECT COUNT(*) FROM ducklake_sort_info "
            "WHERE end_snapshot IS NULL"
        ).fetchone()[0]
    finally:
        con.close()
    assert active == 0


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_reset_sort_keys_duckdb_reads_no_order(tmp_path):
    path, data_path = _make_table(tmp_path)
    alter_ducklake_set_sort_keys(
        path, "t", ["id"], data_path=data_path,
    )
    alter_ducklake_reset_sort_keys(path, "t", data_path=data_path)

    con = duckdb.connect()
    _attach(con, path, data_path)
    rows = con.execute("SELECT id FROM d.t ORDER BY id").fetchall()
    con.close()
    assert [r[0] for r in rows] == [1, 2, 3]


# ------------------------------------------------------------------
# P0 #2 — drop_ducklake_macro roundtrip
# ------------------------------------------------------------------


def test_drop_macro_we_write_duckdb_cannot_call(tmp_path):
    path, data_path = _make_table(tmp_path)
    create_ducklake_macro(
        path, "plus_two", "x + 2",
        parameters=[{"name": "x", "type": "integer"}],
        data_path=data_path,
    )
    drop_ducklake_macro(path, "plus_two", data_path=data_path)

    catalog = DuckLakeCatalog(path, data_path=data_path)
    macros = catalog.list_macros()
    assert "plus_two" not in macros["macro_name"].to_list()


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_drop_macro_then_duckdb_call_fails(tmp_path):
    path, data_path = _make_table(tmp_path)
    create_ducklake_macro(
        path, "plus_two", "x + 2",
        parameters=[{"name": "x", "type": "integer"}],
        data_path=data_path,
    )
    drop_ducklake_macro(path, "plus_two", data_path=data_path)

    con = duckdb.connect()
    _attach(con, path, data_path)
    with pytest.raises(duckdb.Error):
        con.execute("SELECT d.main.plus_two(5)").fetchall()
    con.close()


# ------------------------------------------------------------------
# P0 #3 — set_ducklake_option (table / schema scope) roundtrip
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_table_scoped_option_we_write_duckdb_reads(tmp_path):
    path, data_path = _make_table(tmp_path)
    set_ducklake_option(
        path, "parquet_compression", "zstd",
        table_name="t", data_path=data_path,
    )

    con = duckdb.connect()
    _attach(con, path, data_path)
    rows = con.execute(
        "SELECT option_value FROM ducklake_options('d', 'main', 't') "
        "WHERE option_name = 'parquet_compression'"
    ).fetchall()
    con.close()
    assert ("zstd",) in rows


# ------------------------------------------------------------------
# P0 #4 — vacuum / cleanup_old_files roundtrip
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_cleanup_old_files_then_duckdb_reads(tmp_path):
    """After we run cleanup_old_files, DuckDB should still read the
    table cleanly — only orphan files are removed."""
    from ducklake_polars import merge_adjacent_files_ducklake

    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({"id": [1]}),
        path, "t", mode="error", data_path=data_path,
    )
    write_ducklake(
        pl.DataFrame({"id": [2]}),
        path, "t", mode="append", data_path=data_path,
    )
    merge_adjacent_files_ducklake(path, "t", data_path=data_path)
    cleanup_old_files_ducklake(
        path, cleanup_all=True, data_path=data_path,
    )

    con = duckdb.connect()
    _attach(con, path, data_path)
    rows = con.execute("SELECT id FROM d.t ORDER BY id").fetchall()
    con.close()
    assert [r[0] for r in rows] == [1, 2]


# ------------------------------------------------------------------
# P1 #5 — expire_snapshots roundtrip (we expire, DuckDB reads)
# ------------------------------------------------------------------


def test_expire_snapshots_then_polars_reads(tmp_path):
    """Round-trip: write 5 snapshots → expire to 2 → read latest."""
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    for i in range(5):
        write_ducklake(
            pl.DataFrame({"id": [i]}),
            path, "t",
            mode="error" if i == 0 else "append",
            data_path=data_path,
        )

    # Hold on to one of the older snapshot ids for the negative
    # roundtrip below.
    expired = expire_snapshots(path, keep_last_n=2, data_path=data_path)
    assert expired >= 0

    df = read_ducklake(path, "t", data_path=data_path).sort("id")
    assert df["id"].to_list() == [0, 1, 2, 3, 4]


# ------------------------------------------------------------------
# P1 #6 — DuckDB UPDATE → polars reads
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_update_polars_reads(tmp_path):
    path, data_path = _make_table(tmp_path)
    con = duckdb.connect()
    _attach(con, path, data_path)
    con.execute("UPDATE d.t SET v = 'X' WHERE id = 2")
    con.close()

    df = read_ducklake(path, "t", data_path=data_path).sort("id")
    assert df["v"].to_list() == ["a", "X", "c"]


# ------------------------------------------------------------------
# P1 #7 — DuckDB MERGE → polars reads
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_merge_polars_reads(tmp_path):
    path, data_path = _make_table(tmp_path)
    con = duckdb.connect()
    _attach(con, path, data_path)
    con.execute(
        "MERGE INTO d.t AS tgt USING (VALUES (2, 'X'), (4, 'd')) "
        "AS src(id, v) ON tgt.id = src.id "
        "WHEN MATCHED THEN UPDATE SET v = src.v "
        "WHEN NOT MATCHED THEN INSERT (id, v) VALUES (src.id, src.v)"
    )
    con.close()

    df = read_ducklake(path, "t", data_path=data_path).sort("id")
    assert df["id"].to_list() == [1, 2, 3, 4]
    assert df["v"].to_list() == ["a", "X", "c", "d"]


def test_polars_merge_then_polars_reads(tmp_path):
    """Re-pin polars-side MERGE round-trip without DuckDB."""
    path, data_path = _make_table(tmp_path)
    merge_ducklake(
        path, "t", pl.DataFrame({"id": [2, 4], "v": ["X", "d"]}), "id",
        when_matched_update=True,
        when_not_matched_insert=True,
        data_path=data_path,
    )
    df = read_ducklake(path, "t", data_path=data_path).sort("id")
    assert df["id"].to_list() == [1, 2, 3, 4]
    assert df["v"].to_list() == ["a", "X", "c", "d"]


# ------------------------------------------------------------------
# P1 #8 — rewrite_data_files DR (DuckDB compacts → we read)
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_rewrite_data_files_polars_reads(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    for i in range(3):
        write_ducklake(
            pl.DataFrame({"id": [i]}),
            path, "t",
            mode="error" if i == 0 else "append",
            data_path=data_path,
        )

    con = duckdb.connect()
    _attach(con, path, data_path)
    con.execute("CALL ducklake_rewrite_data_files('d', 'main.t')")
    con.close()

    df = read_ducklake(path, "t", data_path=data_path).sort("id")
    assert df["id"].to_list() == [0, 1, 2]


def test_polars_rewrite_then_polars_reads(tmp_path):
    """Local rewrite still produces a readable, fully-collapsed table."""
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    for i in range(3):
        write_ducklake(
            pl.DataFrame({"id": [i]}),
            path, "t",
            mode="error" if i == 0 else "append",
            data_path=data_path,
        )
    rewrite_data_files_ducklake(path, "t", data_path=data_path)
    df = read_ducklake(path, "t", data_path=data_path).sort("id")
    assert df["id"].to_list() == [0, 1, 2]


# ------------------------------------------------------------------
# P1 #9 — non-comment tags roundtrip both ways
# ------------------------------------------------------------------


def test_polars_writes_custom_tag_we_read_back(tmp_path):
    path, data_path = _make_table(tmp_path)
    set_ducklake_table_tag(
        path, "t", "owner", "pdet", data_path=data_path,
    )
    set_ducklake_column_tag(
        path, "t", "id", "pii_class", "internal", data_path=data_path,
    )

    catalog = DuckLakeCatalog(path, data_path=data_path)
    table_tags = catalog.table_tags("t")
    col_tags = catalog.column_tags("t", "id")

    assert dict(zip(
        table_tags["key"].to_list(), table_tags["value"].to_list(),
    )) == {"owner": "pdet"}
    assert dict(zip(
        col_tags["key"].to_list(), col_tags["value"].to_list(),
    )) == {"pii_class": "internal"}


# ------------------------------------------------------------------
# P1 #10 — delete_ducklake_table_tag / delete_ducklake_column_tag DR
# ------------------------------------------------------------------


def test_delete_table_tag_then_polars_confirms(tmp_path):
    path, data_path = _make_table(tmp_path)
    set_ducklake_table_tag(
        path, "t", "owner", "pdet", data_path=data_path,
    )
    delete_ducklake_table_tag(
        path, "t", "owner", data_path=data_path,
    )
    catalog = DuckLakeCatalog(path, data_path=data_path)
    df = catalog.table_tags("t")
    assert "owner" not in df["key"].to_list()


def test_delete_column_tag_then_polars_confirms(tmp_path):
    path, data_path = _make_table(tmp_path)
    set_ducklake_column_tag(
        path, "t", "id", "pii_class", "internal", data_path=data_path,
    )
    delete_ducklake_column_tag(
        path, "t", "id", "pii_class", data_path=data_path,
    )
    catalog = DuckLakeCatalog(path, data_path=data_path)
    df = catalog.column_tags("t", "id")
    assert "pii_class" not in df["key"].to_list()


# ------------------------------------------------------------------
# P1 #11 — create_table_as_ducklake DR
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_ctas_polars_reads(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    # Bootstrap the catalog so DuckDB can ATTACH.
    write_ducklake(
        pl.DataFrame({"x": [0]}),
        path, "_seed", mode="error", data_path=data_path,
    )

    con = duckdb.connect()
    _attach(con, path, data_path)
    con.execute(
        "CREATE TABLE d.derived AS "
        "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(id, v)"
    )
    con.close()

    df = read_ducklake(path, "derived", data_path=data_path).sort("id")
    assert df["id"].to_list() == [1, 2]
    assert df["v"].to_list() == ["a", "b"]


def test_polars_ctas_polars_reads(tmp_path):
    """Local CTAS still produces a usable table (no DuckDB roundtrip)."""
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    create_table_as_ducklake(
        pl.DataFrame({"id": [1, 2], "v": ["a", "b"]}),
        path, "derived", data_path=data_path,
    )
    df = read_ducklake(path, "derived", data_path=data_path).sort("id")
    assert df["id"].to_list() == [1, 2]


# ------------------------------------------------------------------
# P1 #12 — DuckLakeStreamWriter ↔ DuckDB
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_stream_writer_then_duckdb_reads(tmp_path):
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")

    with DuckLakeStreamWriter(
        path, "events",
        flush_threshold=2,
        compact_on_close=False,
        data_path=data_path,
    ) as w:
        for i in range(5):
            w.append(pl.DataFrame({"id": [i]}))

    con = duckdb.connect()
    _attach(con, path, data_path)
    rows = con.execute("SELECT id FROM d.events ORDER BY id").fetchall()
    con.close()
    assert [r[0] for r in rows] == [0, 1, 2, 3, 4]


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_inserts_then_stream_writer_appends(tmp_path):
    """DuckDB seeds the table; our stream writer keeps appending."""
    path = str(tmp_path / "c.ducklake")
    data_path = str(tmp_path / "data")
    write_ducklake(
        pl.DataFrame({"id": [-1]}),
        path, "events", mode="error", data_path=data_path,
    )

    con = duckdb.connect()
    _attach(con, path, data_path)
    con.execute("INSERT INTO d.events VALUES (-2), (-3)")
    con.close()

    with DuckLakeStreamWriter(
        path, "events",
        flush_threshold=2,
        compact_on_close=False,
        data_path=data_path,
    ) as w:
        w.append(pl.DataFrame({"id": [10, 11]}))
        w.append(pl.DataFrame({"id": [12]}))

    df = read_ducklake(path, "events", data_path=data_path).sort("id")
    assert df["id"].to_list() == [-3, -2, -1, 10, 11, 12]
