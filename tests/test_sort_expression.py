"""Tests for expression-based sort keys (DuckLake v1.0)."""

from __future__ import annotations

import datetime
import sqlite3

import duckdb
import polars as pl
import pytest

from ducklake_polars import (
    alter_ducklake_reset_sort_keys,
    alter_ducklake_set_sort_keys,
    write_ducklake,
)
from ducklake_polars._catalog_api import DuckLakeCatalog


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


def _make_table(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "ts": [
            datetime.datetime(2024, 1, 15, 10, 0),
            datetime.datetime(2024, 1, 15, 14, 0),
            datetime.datetime(2024, 6, 20, 9, 0),
        ],
    })
    write_ducklake(df, path, "events", mode="error", data_path=data_path)
    return path, data_path


def _read_sort_expressions(path):
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT se.sort_key_index, se.expression, se.dialect, "
            "se.sort_direction, se.null_order "
            "FROM ducklake_sort_expression se "
            "JOIN ducklake_sort_info si ON se.sort_id = si.sort_id "
            "WHERE si.end_snapshot IS NULL "
            "ORDER BY se.sort_key_index"
        ).fetchall()
    finally:
        con.close()


class TestColumnSortKeys:
    """Existing column-name behaviour stays intact."""

    def test_simple_column_default_asc_nulls_last(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        alter_ducklake_set_sort_keys(path, "events", ["ts"], data_path=data_path)
        rows = _read_sort_expressions(path)
        assert rows == [(0, "ts", "duckdb", "ASC", "NULLS_LAST")]

    def test_column_with_direction_and_null_order(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        alter_ducklake_set_sort_keys(
            path, "events", [("ts", "DESC", "NULLS_FIRST")],
            data_path=data_path,
        )
        rows = _read_sort_expressions(path)
        assert rows == [(0, "ts", "duckdb", "DESC", "NULLS_FIRST")]


class TestExpressionSortKeys:
    """New expression-based form (dict)."""

    def test_arbitrary_expression(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        alter_ducklake_set_sort_keys(
            path, "events",
            [{"expression": "date_trunc('hour', ts)", "direction": "ASC"}],
            data_path=data_path,
        )
        rows = _read_sort_expressions(path)
        assert rows == [
            (0, "date_trunc('hour', ts)", "duckdb", "ASC", "NULLS_LAST"),
        ]

    def test_custom_dialect(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        alter_ducklake_set_sort_keys(
            path, "events",
            [{"expression": "to_unixtime(ts)", "dialect": "spark"}],
            data_path=data_path,
        )
        rows = _read_sort_expressions(path)
        assert rows[0][2] == "spark"
        assert rows[0][1] == "to_unixtime(ts)"

    def test_mixed_column_and_expression(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        alter_ducklake_set_sort_keys(
            path, "events",
            [
                "id",
                {"expression": "lower(cast(ts AS VARCHAR))", "direction": "DESC"},
            ],
            data_path=data_path,
        )
        rows = _read_sort_expressions(path)
        assert rows == [
            (0, "id", "duckdb", "ASC", "NULLS_LAST"),
            (1, "lower(cast(ts AS VARCHAR))", "duckdb", "DESC", "NULLS_LAST"),
        ]

    def test_expression_skips_column_check(self, tmp_path):
        """Bare-column form errors on missing columns; expression form does not."""
        path, data_path = _make_table(tmp_path)
        # Bare column form: the column 'missing' is rejected.
        with pytest.raises(ValueError, match="Column 'missing' not found"):
            alter_ducklake_set_sort_keys(
                path, "events", ["missing"], data_path=data_path,
            )
        # Expression form: ``missing()`` is opaque to us — accepted.
        alter_ducklake_set_sort_keys(
            path, "events",
            [{"expression": "missing(ts)"}],
            data_path=data_path,
        )

    def test_invalid_direction_rejected(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        with pytest.raises(ValueError, match="Invalid sort direction"):
            alter_ducklake_set_sort_keys(
                path, "events",
                [{"expression": "ts", "direction": "BACKWARD"}],
                data_path=data_path,
            )

    def test_reset_clears_expressions(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        alter_ducklake_set_sort_keys(
            path, "events",
            [{"expression": "year(ts)"}],
            data_path=data_path,
        )
        alter_ducklake_reset_sort_keys(path, "events", data_path=data_path)
        rows = _read_sort_expressions(path)
        assert rows == []


# ------------------------------------------------------------------
# DuckLakeCatalog.sort_keys — surface the active sort order
# ------------------------------------------------------------------


class TestCatalogSortKeysAPI:
    def test_no_sort_keys_returns_empty(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        catalog = DuckLakeCatalog(path, data_path=data_path)
        df = catalog.sort_keys("events")
        assert df.shape[0] == 0
        # Schema should still be present.
        assert "expression" in df.columns

    def test_simple_columns_round_trip(self, tmp_path):
        path, data_path = _make_table(tmp_path)
        alter_ducklake_set_sort_keys(
            path, "events",
            ["id", ("ts", "DESC", "NULLS_FIRST")],
            data_path=data_path,
        )
        catalog = DuckLakeCatalog(path, data_path=data_path)
        df = catalog.sort_keys("events")
        rows = df.to_dicts()
        assert [r["expression"] for r in rows] == ["id", "ts"]
        assert [r["sort_direction"] for r in rows] == ["ASC", "DESC"]
        assert [r["null_order"] for r in rows] == [
            "NULLS_LAST", "NULLS_FIRST",
        ]


# ------------------------------------------------------------------
# DuckDB roundtrip
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_reads_expression_sort_key(tmp_path):
    path, data_path = _make_table(tmp_path)
    alter_ducklake_set_sort_keys(
        path, "events",
        [{"expression": "date_trunc('hour', ts)", "direction": "ASC"}],
        data_path=data_path,
    )

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{path}' AS d (DATA_PATH '{data_path}')"
    )
    rows = con.execute("SELECT id FROM d.events ORDER BY id").fetchall()
    con.close()
    assert [r[0] for r in rows] == [1, 2, 3]
