"""Tests for ``set_ducklake_option`` (scoped catalog settings)."""

from __future__ import annotations

import sqlite3

import duckdb
import polars as pl
import pytest

from ducklake_polars import (
    create_ducklake_schema,
    set_ducklake_option,
    write_ducklake,
)
from ducklake_polars._catalog_api import DuckLakeCatalog


def _duckdb_supports_v10() -> bool:
    parts = duckdb.__version__.split(".")
    return len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (1, 5)


def _make_catalog(tmp_path):
    path = str(tmp_path / "catalog.ducklake")
    data_path = str(tmp_path / "data")
    df = pl.DataFrame({"a": [1, 2, 3]})
    write_ducklake(df, path, "events", mode="error", data_path=data_path)
    return path, data_path


def _select(path, sql, params=()):
    con = sqlite3.connect(path)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


# ------------------------------------------------------------------
# Catalog-wide (scope IS NULL)
# ------------------------------------------------------------------


class TestCatalogScope:
    def test_set_simple_string(self, tmp_path):
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

    def test_update_overwrites_value(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        set_ducklake_option(path, "parquet_compression", "zstd", data_path=data_path)
        set_ducklake_option(path, "parquet_compression", "snappy", data_path=data_path)
        rows = _select(
            path,
            "SELECT value FROM ducklake_metadata WHERE key = 'parquet_compression'",
        )
        assert rows == [("snappy",)]

    def test_bool_normalised(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        set_ducklake_option(path, "auto_compact", True, data_path=data_path)
        set_ducklake_option(path, "per_thread_output", "false", data_path=data_path)
        rows = dict(_select(
            path,
            "SELECT key, value FROM ducklake_metadata "
            "WHERE key IN ('auto_compact', 'per_thread_output')",
        ))
        assert rows == {"auto_compact": "true", "per_thread_output": "false"}

    def test_memory_value_parsed(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        set_ducklake_option(
            path, "target_file_size", "128MB", data_path=data_path,
        )
        rows = _select(
            path,
            "SELECT value FROM ducklake_metadata WHERE key = 'target_file_size'",
        )
        assert rows == [(str(128 * 1024 * 1024),)]

    def test_parquet_version(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        set_ducklake_option(path, "parquet_version", 2, data_path=data_path)
        rows = _select(
            path,
            "SELECT value FROM ducklake_metadata WHERE key = 'parquet_version'",
        )
        assert rows == [("V2",)]


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


class TestValidation:
    def test_unknown_option_rejected(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        with pytest.raises(ValueError, match="Unsupported option"):
            set_ducklake_option(path, "not_a_real_option", "x", data_path=data_path)

    def test_bad_codec_rejected(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        with pytest.raises(ValueError, match="Unsupported codec"):
            set_ducklake_option(
                path, "parquet_compression", "rotzip", data_path=data_path,
            )

    def test_bad_parquet_version(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        with pytest.raises(ValueError, match="Parquet version"):
            set_ducklake_option(path, "parquet_version", 7, data_path=data_path)

    def test_zero_row_group_size(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        with pytest.raises(ValueError, match="cannot be 0"):
            set_ducklake_option(
                path, "parquet_row_group_size", 0, data_path=data_path,
            )

    def test_threshold_out_of_range(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        with pytest.raises(ValueError, match="between 0 and 1"):
            set_ducklake_option(
                path, "rewrite_delete_threshold", 2.5, data_path=data_path,
            )


# ------------------------------------------------------------------
# Scoped (schema / table)
# ------------------------------------------------------------------


class TestScopedOptions:
    def test_table_scope(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        set_ducklake_option(
            path, "target_file_size", "32MB",
            table_name="events", data_path=data_path,
        )
        rows = _select(
            path,
            "SELECT scope, scope_id IS NOT NULL FROM ducklake_metadata "
            "WHERE key = 'target_file_size'",
        )
        assert rows == [("table", 1)]

    def test_schema_scope(self, tmp_path):
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

    def test_table_and_catalog_independent(self, tmp_path):
        """Catalog-wide and table-scoped rows for the same key coexist."""
        path, data_path = _make_catalog(tmp_path)
        set_ducklake_option(
            path, "parquet_compression", "zstd", data_path=data_path,
        )
        set_ducklake_option(
            path, "parquet_compression", "snappy",
            table_name="events", data_path=data_path,
        )
        rows = sorted(_select(
            path,
            "SELECT value, COALESCE(scope, '_catalog') "
            "FROM ducklake_metadata WHERE key = 'parquet_compression'",
        ))
        assert rows == [("snappy", "table"), ("zstd", "_catalog")]

    def test_table_update_in_place(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        set_ducklake_option(
            path, "parquet_compression", "zstd",
            table_name="events", data_path=data_path,
        )
        set_ducklake_option(
            path, "parquet_compression", "snappy",
            table_name="events", data_path=data_path,
        )
        rows = _select(
            path,
            "SELECT value FROM ducklake_metadata WHERE key = 'parquet_compression'",
        )
        assert rows == [("snappy",)]

    def test_missing_table_raises(self, tmp_path):
        path, data_path = _make_catalog(tmp_path)
        with pytest.raises(Exception, match="not found"):
            set_ducklake_option(
                path, "parquet_compression", "zstd",
                table_name="ghost", data_path=data_path,
            )


# ------------------------------------------------------------------
# Reader exposes scope columns
# ------------------------------------------------------------------


def test_options_dataframe_has_scope_columns(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    set_ducklake_option(
        path, "auto_compact", True,
        table_name="events", data_path=data_path,
    )
    catalog = DuckLakeCatalog(path, data_path=data_path)
    df = catalog.options()
    assert {"key", "value", "scope", "scope_id"}.issubset(df.columns)
    rows = df.filter(pl.col("key") == "auto_compact").to_dicts()
    assert rows
    assert rows[0]["scope"] == "table"
    assert rows[0]["scope_id"] is not None


# ------------------------------------------------------------------
# DuckDB roundtrip
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not _duckdb_supports_v10(),
    reason="DuckDB < 1.5 cannot read v1.0 catalogs",
)
def test_duckdb_reads_catalog_option(tmp_path):
    path, data_path = _make_catalog(tmp_path)
    set_ducklake_option(
        path, "parquet_compression", "zstd", data_path=data_path,
    )

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{path}' AS d (DATA_PATH '{data_path}')"
    )
    rows = con.execute(
        "SELECT value FROM ducklake_options('d') "
        "WHERE option_name = 'parquet_compression'"
    ).fetchall()
    con.close()
    assert rows == [("zstd",)]
