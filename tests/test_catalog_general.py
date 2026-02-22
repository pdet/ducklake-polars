"""Catalog and general tests for ducklake-polars."""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import DuckLakeCatalog, read_ducklake, scan_ducklake


class TestNonDefaultSchema:
    """Test reading from non-default (custom) schemas."""

    def test_read_from_custom_schema(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE TABLE ducklake.s1.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.s1.test VALUES (1, 'hello'), (2, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", schema="s1")
        assert result.shape == (2, 2)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == ["hello", "world"]

    def test_scan_from_custom_schema(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE TABLE ducklake.s1.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.s1.test VALUES (1, 'alpha'), (2, 'beta'), (3, 'gamma')"
        )
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test", schema="s1")
            .filter(pl.col("a") > 1)
            .collect()
        )
        result = result.sort("a")
        assert result.shape[0] == 2
        assert result["a"].to_list() == [2, 3]
        assert result["b"].to_list() == ["beta", "gamma"]

    def test_multiple_schemas(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE SCHEMA ducklake.s2")
        cat.execute("CREATE TABLE ducklake.s1.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.s1.test VALUES (1, 'from_s1')")
        cat.execute("CREATE TABLE ducklake.s2.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.s2.test VALUES (2, 'from_s2')")
        cat.close()

        result_s1 = read_ducklake(cat.metadata_path, "test", schema="s1")
        result_s2 = read_ducklake(cat.metadata_path, "test", schema="s2")

        assert result_s1.shape == (1, 2)
        assert result_s1["a"].to_list() == [1]
        assert result_s1["b"].to_list() == ["from_s1"]

        assert result_s2.shape == (1, 2)
        assert result_s2["a"].to_list() == [2]
        assert result_s2["b"].to_list() == ["from_s2"]

    def test_default_schema_not_found_in_custom(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE TABLE ducklake.s1.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.s1.test VALUES (1)")
        cat.close()

        with pytest.raises(Exception):
            read_ducklake(cat.metadata_path, "test")


class TestQuotedIdentifiers:
    """Test tables and columns with special naming."""

    def test_table_with_spaces(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 2)
        assert result["a"].to_list() == [1]
        assert result["b"].to_list() == ["hello"]

    def test_column_with_special_chars(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            'CREATE TABLE ducklake.test ("my_col" INTEGER, "MixedCase" VARCHAR, "another_one" DOUBLE)'
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (42, 'test_val', 3.14)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 3)
        assert "my_col" in result.columns
        assert "MixedCase" in result.columns
        assert "another_one" in result.columns
        assert result["my_col"].to_list() == [42]
        assert result["MixedCase"].to_list() == ["test_val"]
        assert result["another_one"].to_list() == [3.14]


class TestDropAndRecreate:
    """Test behavior when tables are dropped and recreated."""

    def test_drop_and_recreate_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'old')")
        cat.execute("DROP TABLE ducklake.test")
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (99, 2.718)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 2)
        assert sorted(result.columns) == ["a", "c"]
        assert result["a"].to_list() == [99]
        assert result["c"].to_list() == [2.718]

    def test_read_after_drop_fails(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.execute("DROP TABLE ducklake.test")
        cat.close()

        with pytest.raises(Exception):
            read_ducklake(cat.metadata_path, "test")


class TestCatalogListWithSchemas:
    """Test DuckLakeCatalog list methods with custom schemas."""

    def test_list_schemas_includes_custom(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_schemas()

        schema_names = result["schema_name"].to_list()
        assert "main" in schema_names
        assert "s1" in schema_names

    def test_list_tables_in_custom_schema(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE TABLE ducklake.s1.my_table (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        result = api.list_tables(schema="s1")

        assert isinstance(result, pl.DataFrame)
        assert len(result) >= 1
        table_names = result["table_name"].to_list()
        assert "my_table" in table_names


class TestMultipleTablesRead:
    """Test reading multiple tables across different schemas."""

    def test_read_two_tables_different_schemas(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE SCHEMA ducklake.s2")
        cat.execute("CREATE TABLE ducklake.s1.orders (order_id INTEGER, amount DOUBLE)")
        cat.execute(
            "INSERT INTO ducklake.s1.orders VALUES (1, 99.99), (2, 149.50)"
        )
        cat.execute(
            "CREATE TABLE ducklake.s2.customers (customer_id INTEGER, name VARCHAR)"
        )
        cat.execute(
            "INSERT INTO ducklake.s2.customers VALUES (10, 'Alice'), (20, 'Bob')"
        )
        cat.close()

        orders = read_ducklake(cat.metadata_path, "orders", schema="s1")
        customers = read_ducklake(cat.metadata_path, "customers", schema="s2")

        orders = orders.sort("order_id")
        assert orders.shape == (2, 2)
        assert orders["order_id"].to_list() == [1, 2]
        assert orders["amount"].to_list() == [99.99, 149.50]

        customers = customers.sort("customer_id")
        assert customers.shape == (2, 2)
        assert customers["customer_id"].to_list() == [10, 20]
        assert customers["name"].to_list() == ["Alice", "Bob"]


class TestEdgeCaseReads:
    """Test edge cases for reading data."""

    def test_wide_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cols = ", ".join(f"c{i} INTEGER" for i in range(50))
        cat.execute(f"CREATE TABLE ducklake.test ({cols})")
        vals = ", ".join(str(i) for i in range(50))
        cat.execute(f"INSERT INTO ducklake.test VALUES ({vals})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert len(result.columns) == 50
        assert result.shape == (1, 50)
        # Verify a few columns have the expected values
        assert result["c0"].to_list() == [0]
        assert result["c25"].to_list() == [25]
        assert result["c49"].to_list() == [49]

    def test_many_small_inserts(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        for i in range(20):
            cat.execute(
                f"INSERT INTO ducklake.test VALUES ({i}, 'row_{i}')"
            )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 20
        result = result.sort("a")
        assert result["a"].to_list() == list(range(20))
        assert result["b"].to_list() == [f"row_{i}" for i in range(20)]

    def test_empty_string_values(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, '')")
        cat.execute("INSERT INTO ducklake.test VALUES (2, NULL)")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'notempty')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (3, 2)

        b_values = result["b"].to_list()
        # Row with a=1 has empty string (not NULL)
        assert b_values[0] == ""
        assert b_values[0] is not None
        # Row with a=2 has NULL
        assert b_values[1] is None
        # Row with a=3 has a real string
        assert b_values[2] == "notempty"
