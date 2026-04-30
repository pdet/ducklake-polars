"""Tests for macro writes (CREATE/DROP MACRO)."""

from __future__ import annotations

import sqlite3

import duckdb
import polars as pl
import pytest

from ducklake_polars import (
    create_ducklake_macro,
    drop_ducklake_macro,
    write_ducklake,
)
from ducklake_polars._catalog import DuckLakeCatalogReader
from ducklake_polars._catalog_api import DuckLakeCatalog


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


def _make_catalog(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pl.DataFrame({"a": [1]})
    write_ducklake(df, path, "t", mode="error", data_path=data_path)
    return path, data_path


# ------------------------------------------------------------------
# create_ducklake_macro
# ------------------------------------------------------------------


class TestCreateMacro:
    def test_simple_scalar_macro(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)

        macro_id = create_ducklake_macro(
            path, "add_one", "a + 1",
            parameters=[{"name": "a", "type": "integer"}],
            data_path=data_path,
        )
        assert macro_id >= 0

        con = sqlite3.connect(path)
        try:
            row = con.execute(
                "SELECT macro_name FROM ducklake_macro "
                "WHERE end_snapshot IS NULL"
            ).fetchone()
            impl = con.execute(
                "SELECT dialect, sql, type FROM ducklake_macro_impl "
                "WHERE macro_id = ?",
                [macro_id],
            ).fetchone()
            params = con.execute(
                "SELECT parameter_name, parameter_type, default_value, "
                "default_value_type FROM ducklake_macro_parameters "
                "WHERE macro_id = ? ORDER BY column_id",
                [macro_id],
            ).fetchall()
        finally:
            con.close()

        assert row == ("add_one",)
        assert impl == ("duckdb", "a + 1", "scalar")
        # parameter_type canonicalized to DuckLake form ("integer" -> "int32"),
        # default_value persisted as empty string (DuckDB's macro reader
        # crashes on NULL there, matching the C++ writer).
        assert params == [("a", "int32", "", "unknown")]

    def test_macro_with_default_parameter(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)

        create_ducklake_macro(
            path, "greet", "concat('hi ', name)",
            parameters=[
                {"name": "name", "type": "varchar", "default": "world"},
            ],
            data_path=data_path,
        )
        con = sqlite3.connect(path)
        try:
            row = con.execute(
                "SELECT default_value, default_value_type "
                "FROM ducklake_macro_parameters"
            ).fetchone()
        finally:
            con.close()
        assert row == ("world", "literal")

    def test_table_macro(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        create_ducklake_macro(
            path, "ones", "SELECT 1 AS x",
            macro_type="table", data_path=data_path,
        )
        con = sqlite3.connect(path)
        try:
            t = con.execute(
                "SELECT type FROM ducklake_macro_impl"
            ).fetchone()
        finally:
            con.close()
        assert t == ("table",)

    def test_duplicate_macro_raises(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        create_ducklake_macro(path, "m", "1", data_path=data_path)
        with pytest.raises(ValueError, match="already exists"):
            create_ducklake_macro(path, "m", "2", data_path=data_path)

    def test_or_replace_overwrites(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        create_ducklake_macro(path, "m", "1", data_path=data_path)
        create_ducklake_macro(path, "m", "2", or_replace=True, data_path=data_path)

        con = sqlite3.connect(path)
        try:
            active = con.execute(
                "SELECT COUNT(*) FROM ducklake_macro WHERE end_snapshot IS NULL"
            ).fetchone()[0]
            ended = con.execute(
                "SELECT COUNT(*) FROM ducklake_macro WHERE end_snapshot IS NOT NULL"
            ).fetchone()[0]
        finally:
            con.close()
        assert active == 1
        assert ended == 1


# ------------------------------------------------------------------
# drop_ducklake_macro
# ------------------------------------------------------------------


class TestDropMacro:
    def test_drop_existing(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        create_ducklake_macro(path, "m", "42", data_path=data_path)
        drop_ducklake_macro(path, "m", data_path=data_path)

        con = sqlite3.connect(path)
        try:
            active = con.execute(
                "SELECT COUNT(*) FROM ducklake_macro WHERE end_snapshot IS NULL"
            ).fetchone()[0]
        finally:
            con.close()
        assert active == 0

    def test_drop_missing_raises(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            drop_ducklake_macro(path, "missing", data_path=data_path)


# ------------------------------------------------------------------
# Reader sees the new macro
# ------------------------------------------------------------------


def test_catalog_api_lists_macro(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    create_ducklake_macro(
        path, "double_it", "x * 2",
        parameters=[{"name": "x", "type": "integer"}],
        data_path=data_path,
    )
    catalog = DuckLakeCatalog(path, data_path=data_path)
    macros = catalog.list_macros()
    rows = macros.to_dicts() if hasattr(macros, "to_dicts") else macros.to_pydict()
    names = (
        [r["macro_name"] for r in rows] if isinstance(rows, list)
        else rows["macro_name"]
    )
    assert "double_it" in names


# ------------------------------------------------------------------
# get_macro: full round-trip
# ------------------------------------------------------------------


def test_catalog_get_macro_round_trip(tmp_path):
    """``DuckLakeCatalog.get_macro`` should surface the macro body,
    type, dialect, and parameter list (with default values) for a
    macro round-tripped through the writer."""
    path, data_path = _make_catalog(tmp_path)

    # Scalar macro with default parameter.
    create_ducklake_macro(
        path, "greet", "concat('hi ', who)",
        parameters=[
            {"name": "who", "type": "varchar", "default": "world"},
        ],
        data_path=data_path,
    )
    # Table macro to make sure type round-trips too.
    create_ducklake_macro(
        path, "ones", "SELECT 1 AS x",
        macro_type="table", data_path=data_path,
    )

    catalog = DuckLakeCatalog(path, data_path=data_path)
    df = catalog.get_macro("greet")

    rows = df.to_dicts() if hasattr(df, "to_dicts") else df.to_dict("records")
    assert len(rows) == 1
    row = rows[0]
    assert row["macro_name"] == "greet"
    assert row["macro_type"] == "scalar"
    assert row["dialect"] == "duckdb"
    assert row["sql"] == "concat('hi ', who)"
    # Parameter list is comma-separated; default value should appear.
    assert "who" in row["parameters"]
    assert "DEFAULT" in row["parameters"].upper()
    assert "world" in row["parameters"]

    df_table = catalog.get_macro("ones")
    rows_t = (
        df_table.to_dicts()
        if hasattr(df_table, "to_dicts")
        else df_table.to_dict("records")
    )
    assert rows_t[0]["macro_type"] == "table"
    assert rows_t[0]["sql"].lstrip().upper().startswith("SELECT")


def test_catalog_get_macro_missing_raises(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    catalog = DuckLakeCatalog(path, data_path=data_path)
    with pytest.raises(ValueError, match="not found"):
        catalog.get_macro("ghost")


def test_catalog_get_macro_dialect_filter(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    create_ducklake_macro(
        path, "id_macro", "x", dialect="duckdb",
        parameters=[{"name": "x", "type": "integer"}],
        data_path=data_path,
    )
    catalog = DuckLakeCatalog(path, data_path=data_path)
    df = catalog.get_macro("id_macro", dialect="duckdb")
    assert (df.shape[0] if hasattr(df, "shape") else len(df)) == 1
    df_empty = catalog.get_macro("id_macro", dialect="spark")
    assert (df_empty.shape[0] if hasattr(df_empty, "shape") else len(df_empty)) == 0


def test_list_macros_columns_include_type(tmp_path):
    """Beyond row count: ``list_macros`` should expose macro_id,
    macro_name, and macro_type so callers can branch on macro kind
    without re-reading the catalog."""
    path, data_path = _make_catalog(tmp_path)
    create_ducklake_macro(
        path, "scalar1", "1", data_path=data_path,
    )
    create_ducklake_macro(
        path, "table1", "SELECT 1", macro_type="table", data_path=data_path,
    )
    catalog = DuckLakeCatalog(path, data_path=data_path)
    df = catalog.list_macros()
    rows = df.to_dicts() if hasattr(df, "to_dicts") else df.to_dict("records")
    by_name = {r["macro_name"]: r for r in rows}
    assert by_name["scalar1"]["macro_type"] == "scalar"
    assert by_name["table1"]["macro_type"] == "table"
    assert isinstance(by_name["scalar1"]["macro_id"], int)


# ------------------------------------------------------------------
# DuckDB roundtrip
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_executes_our_macro(tmp_path):
    """A macro we create should be callable from DuckDB."""
    path, data_path = _make_catalog(tmp_path)
    create_ducklake_macro(
        path, "plus_two", "x + 2",
        parameters=[{"name": "x", "type": "integer"}],
        data_path=data_path,
    )

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{path}' AS d (DATA_PATH '{data_path}')"
    )
    result = con.execute("SELECT d.main.plus_two(5)").fetchone()
    con.close()
    assert result == (7,)
