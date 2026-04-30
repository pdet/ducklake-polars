"""Tests for default_value_type='expression' (DuckLake v0.4+)."""

from __future__ import annotations

import os
import sqlite3

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from ducklake_core._catalog import DuckLakeCatalogReader
from ducklake_polars import write_ducklake


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


# ------------------------------------------------------------------
# Writer: add_column with default_expression
# ------------------------------------------------------------------


class TestAddColumnExpression:
    def _make_table(self, tmp_path):
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df = pl.DataFrame({"id": [1, 2, 3]})
        write_ducklake(df, path, "t", mode="error", data_path=data_path)
        return path, data_path

    def test_add_column_literal_default(self, tmp_path):
        from ducklake_core._writer import DuckLakeCatalogWriter

        path, data_path = self._make_table(tmp_path)
        with DuckLakeCatalogWriter(path, data_path_override=data_path) as w:
            w.add_column("t", "x", pa.int64(), default=42)

        with DuckLakeCatalogReader(path, data_path_override=data_path) as r:
            snap = r.get_current_snapshot()
            tbl = r.get_table("t", "main", snap.snapshot_id)
            cols = {c.column_name: c for c in r.get_columns(tbl.table_id, snap.snapshot_id)}
            x = cols["x"]
            assert x.default_value == "42"
            assert x.default_value_type == "literal"
            assert x.default_value_dialect is None

    def test_add_column_expression_default(self, tmp_path):
        from ducklake_core._writer import DuckLakeCatalogWriter

        path, data_path = self._make_table(tmp_path)
        with DuckLakeCatalogWriter(path, data_path_override=data_path) as w:
            w.add_column(
                "t", "created_at", pa.timestamp("us"),
                default_expression="now()",
            )

        with DuckLakeCatalogReader(path, data_path_override=data_path) as r:
            snap = r.get_current_snapshot()
            tbl = r.get_table("t", "main", snap.snapshot_id)
            cols = {c.column_name: c for c in r.get_columns(tbl.table_id, snap.snapshot_id)}
            ca = cols["created_at"]
            assert ca.default_value == "now()"
            assert ca.default_value_type == "expression"
            assert ca.default_value_dialect == "duckdb"

    def test_add_column_expression_with_custom_dialect(self, tmp_path):
        from ducklake_core._writer import DuckLakeCatalogWriter

        path, data_path = self._make_table(tmp_path)
        with DuckLakeCatalogWriter(path, data_path_override=data_path) as w:
            w.add_column(
                "t", "rand_col", pa.float64(),
                default_expression="random()",
                default_dialect="duckdb",
            )

        # Verify the raw catalog row
        con = sqlite3.connect(path)
        try:
            row = con.execute(
                "SELECT default_value, default_value_type, default_value_dialect "
                "FROM ducklake_column WHERE column_name = 'rand_col'"
            ).fetchone()
        finally:
            con.close()
        assert row == ("random()", "expression", "duckdb")

    def test_default_and_expression_are_mutually_exclusive(self, tmp_path):
        from ducklake_core._writer import DuckLakeCatalogWriter

        path, data_path = self._make_table(tmp_path)
        with DuckLakeCatalogWriter(path, data_path_override=data_path) as w:
            with pytest.raises(ValueError, match="mutually exclusive"):
                w.add_column(
                    "t", "x", pa.int64(),
                    default=1, default_expression="now()",
                )

    def test_no_default_means_no_default_value_type(self, tmp_path):
        from ducklake_core._writer import DuckLakeCatalogWriter

        path, data_path = self._make_table(tmp_path)
        with DuckLakeCatalogWriter(path, data_path_override=data_path) as w:
            w.add_column("t", "y", pa.int64())

        # Bootstrap-created tables get default_value_type='literal' and value=NULL.
        # When the column has no default, we still emit literal/NULL — matches
        # DuckLake-extension behavior since the raw default_value is NULL.
        con = sqlite3.connect(path)
        try:
            row = con.execute(
                "SELECT default_value, default_value_type, default_value_dialect "
                "FROM ducklake_column WHERE column_name = 'y'"
            ).fetchone()
        finally:
            con.close()
        assert row[0] is None
        assert row[1] == "literal"
        assert row[2] is None


# ------------------------------------------------------------------
# Reader: surface default fields from existing 0.4/1.0 catalogs
# ------------------------------------------------------------------


class TestColumnInfoDefaults:
    def test_v04_catalog_with_expression_default(self, tmp_path):
        """Synthesize a v0.4 catalog row with an expression default and read it."""
        # Use the bootstrap-produced catalog (1.0) since the schema is identical.
        path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        df = pl.DataFrame({"id": [1]})
        write_ducklake(df, path, "t", mode="error", data_path=data_path)

        # Manually tag a column with an expression default
        con = sqlite3.connect(path)
        try:
            con.execute(
                "UPDATE ducklake_column SET default_value = 'now()', "
                "default_value_type = 'expression', default_value_dialect = 'duckdb' "
                "WHERE column_name = 'id'"
            )
            con.commit()
        finally:
            con.close()

        with DuckLakeCatalogReader(path, data_path_override=data_path) as r:
            snap = r.get_current_snapshot()
            tbl = r.get_table("t", "main", snap.snapshot_id)
            cols = r.get_columns(tbl.table_id, snap.snapshot_id)
            assert len(cols) == 1
            assert cols[0].default_value == "now()"
            assert cols[0].default_value_type == "expression"
            assert cols[0].default_value_dialect == "duckdb"


# ------------------------------------------------------------------
# DuckDB roundtrip — only meaningful with DuckDB ≥ 1.5
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_reads_expression_default(tmp_path):
    """DuckDB sees the expression-default column we added."""
    from ducklake_core._writer import DuckLakeCatalogWriter

    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pl.DataFrame({"id": [1, 2]})
    write_ducklake(df, path, "t", mode="error", data_path=data_path)

    with DuckLakeCatalogWriter(path, data_path_override=data_path) as w:
        w.add_column(
            "t", "ts", pa.timestamp("us"),
            default_expression="now()",
        )

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{path}' AS d (DATA_PATH '{data_path}')"
    )
    cols = con.execute(
        "SELECT column_name, column_default "
        "FROM information_schema.columns "
        "WHERE table_name = 't' ORDER BY ordinal_position"
    ).fetchall()
    con.close()
    # The 'ts' column should report a non-NULL default (the expression).
    by_name = dict(cols)
    assert by_name["ts"] is not None
    assert "now" in by_name["ts"].lower()
