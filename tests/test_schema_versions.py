"""Per-table ``ducklake_schema_versions`` tracking (DuckLake 1.0).

DuckLake 1.0 added a ``table_id`` column to ``ducklake_schema_versions``.
Catalog-wide DDL (CREATE SCHEMA, DROP SCHEMA, macros) yields a NULL
``table_id``, while table-scoped DDL (ALTER TABLE …) records the
table_id of the affected table. The DuckDB extension uses this to look
up the begin_snapshot of a particular table-schema version.
"""

from __future__ import annotations

import sqlite3

import duckdb
import polars as pl
import pytest

from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    create_ducklake_macro,
    create_ducklake_schema,
    write_ducklake,
)


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


def _make_table(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pl.DataFrame({"id": [1, 2, 3]})
    write_ducklake(df, path, "events", mode="error", data_path=data_path)
    return path, data_path


def _schema_versions(path):
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT begin_snapshot, schema_version, table_id "
            "FROM ducklake_schema_versions "
            "ORDER BY begin_snapshot"
        ).fetchall()
    finally:
        con.close()


def _table_id(path, table_name):
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT table_id FROM ducklake_table "
            "WHERE table_name = ? AND end_snapshot IS NULL",
            [table_name],
        ).fetchone()[0]
    finally:
        con.close()


# ------------------------------------------------------------------
# Table-scoped writes carry table_id
# ------------------------------------------------------------------


class TestTableScoped:
    def test_add_column_records_table_id(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        tid = _table_id(path, "events")

        before = _schema_versions(path)
        alter_ducklake_add_column(
            path, "events", "value", pl.Int64(), data_path=data_path,
        )
        after = _schema_versions(path)

        # A new schema_versions row was added — and it carries our table_id.
        new_rows = after[len(before):]
        assert len(new_rows) == 1
        assert new_rows[0][2] == tid

    def test_rename_column_records_table_id(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        tid = _table_id(path, "events")

        alter_ducklake_rename_column(
            path, "events", "id", "renamed_id", data_path=data_path,
        )
        rows = _schema_versions(path)
        # last row is the rename
        assert rows[-1][2] == tid

    def test_drop_column_records_table_id(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        alter_ducklake_add_column(
            path, "events", "extra", pl.Int64(), data_path=data_path,
        )
        tid = _table_id(path, "events")
        alter_ducklake_drop_column(
            path, "events", "extra", data_path=data_path,
        )
        rows = _schema_versions(path)
        assert rows[-1][2] == tid


# ------------------------------------------------------------------
# Catalog-wide DDL leaves table_id NULL
# ------------------------------------------------------------------


class TestCatalogScoped:
    def test_create_schema_null_table_id(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        create_ducklake_schema(path, "analytics", data_path=data_path)
        rows = _schema_versions(path)
        # The latest write is CREATE SCHEMA → table_id IS NULL.
        assert rows[-1][2] is None

    def test_create_macro_null_table_id(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        create_ducklake_macro(
            path, "add_one", "x + 1",
            parameters=[{"name": "x", "type": "integer"}],
            data_path=data_path,
        )
        rows = _schema_versions(path)
        assert rows[-1][2] is None


# ------------------------------------------------------------------
# Bootstrap leaves the table empty (1.0 forbids NULL table_id seeds).
# ------------------------------------------------------------------


def test_bootstrap_no_seed_row(tmp_path):
    path, data_path = _make_table(tmp_path)
    rows = _schema_versions(path)
    # All rows should have a non-NULL table_id (the seed row was removed).
    assert all(r[2] is not None for r in rows), rows


# ------------------------------------------------------------------
# DuckDB roundtrip — the per-table rows should let DuckDB resolve a
# table's begin_snapshot for a given schema_version.
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_resolves_per_table_schema_version(tmp_path):
    path, data_path = _make_table(tmp_path)
    alter_ducklake_add_column(
        path, "events", "extra", pl.Int64(), data_path=data_path,
    )

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{path}' AS d (DATA_PATH '{data_path}')"
    )
    cols = [r[0] for r in con.execute("DESCRIBE d.events").fetchall()]
    rows = con.execute(
        "SELECT id, extra FROM d.events ORDER BY id"
    ).fetchall()
    con.close()
    assert "extra" in cols
    assert [r[0] for r in rows] == [1, 2, 3]
    # Newly added column is NULL for pre-existing rows.
    assert all(r[1] is None for r in rows)
