"""Pandas parity tests for set_ducklake_option."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from ducklake_pandas import (
    create_ducklake_schema,
    set_ducklake_option,
    write_ducklake,
)


def _make_catalog(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pd.DataFrame({"a": [1, 2, 3]})
    write_ducklake(df, path, "events", mode="error", data_path=data_path)
    return path, data_path


def _select(path, sql, params=()):
    con = sqlite3.connect(path)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def test_pandas_set_catalog_option(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    set_ducklake_option(
        path, "parquet_compression", "zstd", data_path=data_path,
    )
    rows = _select(
        path,
        "SELECT value, scope, scope_id FROM ducklake_metadata "
        "WHERE key = 'parquet_compression'",
    )
    assert rows == [("zstd", None, None)]


def test_pandas_set_table_scoped_option(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    set_ducklake_option(
        path, "auto_compact", True,
        table_name="events", data_path=data_path,
    )
    rows = _select(
        path,
        "SELECT scope, scope_id IS NOT NULL FROM ducklake_metadata "
        "WHERE key = 'auto_compact'",
    )
    assert rows == [("table", 1)]


def test_pandas_set_schema_scoped_option(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    create_ducklake_schema(path, "analytics", data_path=data_path)
    set_ducklake_option(
        path, "auto_compact", True,
        schema="analytics", data_path=data_path,
    )
    rows = _select(
        path,
        "SELECT scope FROM ducklake_metadata WHERE key = 'auto_compact'",
    )
    assert rows == [("schema",)]


def test_pandas_unknown_option_rejected(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    with pytest.raises(ValueError, match="Unsupported option"):
        set_ducklake_option(
            path, "not_a_real_option", "x", data_path=data_path,
        )
