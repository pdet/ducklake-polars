"""Pandas parity tests for create/drop ducklake macro."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from ducklake_pandas import (
    create_ducklake_macro,
    drop_ducklake_macro,
    write_ducklake,
)
from ducklake_pandas._catalog_api import DuckLakeCatalog


def _make_catalog(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pd.DataFrame({"a": [1]})
    write_ducklake(df, path, "t", mode="error", data_path=data_path)
    return path, data_path


def test_pandas_create_and_drop_macro(tmp_path):
    path, data_path = _make_catalog(tmp_path)

    macro_id = create_ducklake_macro(
        path, "add_one", "a + 1",
        parameters=[{"name": "a", "type": "integer"}],
        data_path=data_path,
    )
    assert macro_id >= 0

    con = sqlite3.connect(path)
    try:
        active = con.execute(
            "SELECT COUNT(*) FROM ducklake_macro WHERE end_snapshot IS NULL"
        ).fetchone()[0]
    finally:
        con.close()
    assert active == 1

    drop_ducklake_macro(path, "add_one", data_path=data_path)
    con = sqlite3.connect(path)
    try:
        active = con.execute(
            "SELECT COUNT(*) FROM ducklake_macro WHERE end_snapshot IS NULL"
        ).fetchone()[0]
    finally:
        con.close()
    assert active == 0


def test_pandas_get_macro_via_catalog(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    create_ducklake_macro(
        path, "double_it", "x * 2",
        parameters=[{"name": "x", "type": "integer"}],
        data_path=data_path,
    )
    catalog = DuckLakeCatalog(path, data_path=data_path)
    df = catalog.get_macro("double_it")
    assert len(df) == 1
    assert df["macro_name"].iloc[0] == "double_it"
    assert df["sql"].iloc[0] == "x * 2"


def test_pandas_drop_missing_raises(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        drop_ducklake_macro(path, "missing", data_path=data_path)
