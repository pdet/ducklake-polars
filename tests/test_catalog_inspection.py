"""Tests for DuckLake macro support (list_macros, get_macro)."""

from __future__ import annotations

import polars as pl
import pandas as pd
import pytest

from ducklake_polars import DuckLakeCatalog as PolarsCatalog
from ducklake_polars import list_macros as polars_list_macros
from ducklake_polars import get_macro as polars_get_macro

from ducklake_pandas import DuckLakeCatalog as PandasCatalog
from ducklake_pandas import list_macros as pandas_list_macros
from ducklake_pandas import get_macro as pandas_get_macro


class TestListMacrosEmpty:
    """Test list_macros on a catalog with no macros."""

    def test_empty_catalog_polars(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        result = polars_list_macros(cat.metadata_path)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["macro_id", "macro_name", "macro_type"]
        assert len(result) == 0

    def test_empty_catalog_pandas(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        result = pandas_list_macros(cat.metadata_path)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["macro_id", "macro_name", "macro_type"]
        assert len(result) == 0


class TestListMacrosWithMacros:
    """Test list_macros when macros exist."""

    def test_scalar_macro_polars(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute(
            "CREATE MACRO ducklake.double_it(x) AS x * 2"
        )
        cat.close()

        result = polars_list_macros(cat.metadata_path)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert result["macro_name"][0] == "double_it"
        assert result["macro_type"][0] == "scalar"

    def test_scalar_macro_pandas(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute(
            "CREATE MACRO ducklake.double_it(x) AS x * 2"
        )
        cat.close()

        result = pandas_list_macros(cat.metadata_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["macro_name"].iloc[0] == "double_it"
        assert result["macro_type"].iloc[0] == "scalar"

    def test_multiple_macros(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE MACRO ducklake.add_one(x) AS x + 1")
        cat.execute("CREATE MACRO ducklake.add_two(x) AS x + 2")
        cat.close()

        result = polars_list_macros(cat.metadata_path)
        assert len(result) == 2
        names = sorted(result["macro_name"].to_list())
        assert names == ["add_one", "add_two"]

    def test_catalog_api_class(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE MACRO ducklake.triple(x) AS x * 3")
        cat.close()

        api = PolarsCatalog(cat.metadata_path)
        result = api.list_macros()
        assert len(result) == 1
        assert result["macro_name"][0] == "triple"

    def test_table_macro(self, ducklake_catalog):
        """Test listing table macros."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute(
            "CREATE MACRO ducklake.my_range(n) AS TABLE "
            "SELECT * FROM range(n)"
        )
        cat.close()

        result = polars_list_macros(cat.metadata_path)
        assert len(result) == 1
        assert result["macro_name"][0] == "my_range"
        assert result["macro_type"][0] == "table"


class TestGetMacro:
    """Test get_macro retrieves macro definition details."""

    def test_get_scalar_macro(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE MACRO ducklake.double_it(x) AS x * 2")
        cat.close()

        result = polars_get_macro(cat.metadata_path, "double_it")
        assert isinstance(result, pl.DataFrame)
        assert len(result) >= 1
        assert result["macro_name"][0] == "double_it"
        assert result["macro_type"][0] == "scalar"
        # SQL should contain the macro body
        assert "2" in result["sql"][0]

    def test_get_macro_pandas(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE MACRO ducklake.inc(x) AS x + 1")
        cat.close()

        result = pandas_get_macro(cat.metadata_path, "inc")
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1
        assert result["macro_name"].iloc[0] == "inc"

    def test_get_macro_not_found(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.close()

        with pytest.raises(ValueError, match="not found"):
            polars_get_macro(cat.metadata_path, "nonexistent")

    def test_get_macro_with_parameters(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute(
            "CREATE MACRO ducklake.add_vals(x, y) AS x + y"
        )
        cat.close()

        result = polars_get_macro(cat.metadata_path, "add_vals")
        assert len(result) >= 1
        params = result["parameters"][0]
        # Should contain both parameter names
        assert "x" in params
        assert "y" in params

    def test_get_macro_catalog_api(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE MACRO ducklake.negate(x) AS -x")
        cat.close()

        api = PolarsCatalog(cat.metadata_path)
        result = api.get_macro("negate")
        assert len(result) >= 1
        assert result["macro_name"][0] == "negate"

    def test_get_macro_case_insensitive(self, ducklake_catalog):
        """Macro lookup should be case-insensitive."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE MACRO ducklake.MyMacro(x) AS x + 10")
        cat.close()

        # DuckDB lowercases the macro name, so look up with lowercase
        result = polars_get_macro(cat.metadata_path, "mymacro")
        assert len(result) >= 1


class TestMacroDropped:
    """Test that dropped macros don't appear in list_macros."""

    def test_dropped_macro_not_listed(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE MACRO ducklake.temp_macro(x) AS x")
        cat.execute("DROP MACRO ducklake.temp_macro")
        cat.close()

        result = polars_list_macros(cat.metadata_path)
        assert len(result) == 0
