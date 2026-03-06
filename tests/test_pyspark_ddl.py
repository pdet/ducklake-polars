"""Tests for PySpark DDL and catalog metadata operations."""

from __future__ import annotations

import os
import tempfile
import shutil

import duckdb
import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, LongType


@pytest.fixture(scope="module")
def spark():
    """Create a local Spark session for testing."""
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("ducklake-pyspark-ddl-test")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Xss4m")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def ducklake_catalog():
    """Create a DuckLake catalog with a test table."""
    tmpdir = tempfile.mkdtemp(prefix="ducklake_pyspark_ddl_")
    meta = os.path.join(tmpdir, "test.ducklake")
    data = os.path.join(tmpdir, "data")
    os.makedirs(data)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{meta}' AS lake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.execute(
        "CREATE TABLE lake.users AS "
        "SELECT i AS id, 'user_' || i AS name, "
        "CAST(i * 10.5 AS DOUBLE) AS score "
        "FROM range(10) t(i)"
    )
    con.close()

    yield meta, data, tmpdir

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def empty_catalog():
    """Create an empty DuckLake catalog (just the 'main' schema)."""
    tmpdir = tempfile.mkdtemp(prefix="ducklake_pyspark_ddl_empty_")
    meta = os.path.join(tmpdir, "test.ducklake")
    data = os.path.join(tmpdir, "data")
    os.makedirs(data)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{meta}' AS lake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    # Create a table so the catalog is initialized
    con.execute(
        "CREATE TABLE lake.items AS "
        "SELECT i AS id, 'item_' || i AS label "
        "FROM range(5) t(i)"
    )
    con.close()

    yield meta, data, tmpdir

    shutil.rmtree(tmpdir, ignore_errors=True)


# ==================================================================
# ALTER TABLE tests
# ==================================================================


class TestAlterAddColumn:
    def test_add_column_with_pyspark_type(self, ducklake_catalog):
        from ducklake_pyspark import alter_ducklake_add_column, table_info

        meta, data, _ = ducklake_catalog
        alter_ducklake_add_column(meta, "users", "age", IntegerType())

        info = table_info(meta, "users")
        col_names = [c["column_name"] for c in info]
        assert "age" in col_names

    def test_add_column_with_string_type(self, ducklake_catalog):
        from ducklake_pyspark import alter_ducklake_add_column, table_info

        meta, data, _ = ducklake_catalog
        alter_ducklake_add_column(meta, "users", "email", "VARCHAR")

        info = table_info(meta, "users")
        col_names = [c["column_name"] for c in info]
        assert "email" in col_names

    def test_add_column_with_default(self, ducklake_catalog):
        from ducklake_pyspark import alter_ducklake_add_column, table_info

        meta, data, _ = ducklake_catalog
        alter_ducklake_add_column(meta, "users", "active", "BOOLEAN", default=True)

        info = table_info(meta, "users")
        col_names = [c["column_name"] for c in info]
        assert "active" in col_names


class TestAlterDropColumn:
    def test_drop_column(self, ducklake_catalog):
        from ducklake_pyspark import alter_ducklake_drop_column, table_info

        meta, data, _ = ducklake_catalog
        alter_ducklake_drop_column(meta, "users", "score")

        info = table_info(meta, "users")
        col_names = [c["column_name"] for c in info]
        assert "score" not in col_names


class TestAlterRenameColumn:
    def test_rename_column(self, ducklake_catalog):
        from ducklake_pyspark import alter_ducklake_rename_column, table_info

        meta, data, _ = ducklake_catalog
        alter_ducklake_rename_column(meta, "users", "name", "username")

        info = table_info(meta, "users")
        col_names = [c["column_name"] for c in info]
        assert "username" in col_names
        assert "name" not in col_names


class TestAlterSetType:
    def test_set_type(self, ducklake_catalog):
        from ducklake_pyspark import alter_ducklake_set_type, table_info

        meta, data, _ = ducklake_catalog
        alter_ducklake_set_type(meta, "users", "id", "BIGINT")

        info = table_info(meta, "users")
        id_col = [c for c in info if c["column_name"] == "id"][0]
        assert id_col["column_type"].upper() in ("BIGINT", "INT64")


class TestAlterPartitioning:
    def test_set_partitioned_by(self, ducklake_catalog):
        from ducklake_pyspark import alter_ducklake_set_partitioned_by

        meta, data, _ = ducklake_catalog
        # Should not raise
        alter_ducklake_set_partitioned_by(meta, "users", ["name"])


class TestAlterSortKeys:
    def test_set_and_reset_sort_keys(self, ducklake_catalog):
        from ducklake_pyspark import (
            alter_ducklake_set_sort_keys,
            alter_ducklake_reset_sort_keys,
        )

        meta, data, _ = ducklake_catalog
        alter_ducklake_set_sort_keys(meta, "users", ["id", ("score", "DESC")])
        # Reset should not raise
        alter_ducklake_reset_sort_keys(meta, "users")


# ==================================================================
# TABLE operations
# ==================================================================


class TestDropTable:
    def test_drop_table(self, ducklake_catalog):
        from ducklake_pyspark import drop_ducklake_table, list_tables

        meta, data, _ = ducklake_catalog
        drop_ducklake_table(meta, "users")

        tables = list_tables(meta)
        assert "users" not in tables


class TestRenameTable:
    def test_rename_table(self, ducklake_catalog):
        from ducklake_pyspark import rename_ducklake_table, list_tables

        meta, data, _ = ducklake_catalog
        rename_ducklake_table(meta, "users", "people")

        tables = list_tables(meta)
        assert "people" in tables
        assert "users" not in tables


# ==================================================================
# SCHEMA operations
# ==================================================================


class TestSchemaOperations:
    def test_create_and_drop_schema(self, ducklake_catalog):
        from ducklake_pyspark import (
            create_ducklake_schema,
            drop_ducklake_schema,
            list_schemas,
        )

        meta, data, _ = ducklake_catalog
        create_ducklake_schema(meta, "analytics")

        schemas = list_schemas(meta)
        assert "analytics" in schemas

        drop_ducklake_schema(meta, "analytics")
        schemas = list_schemas(meta)
        assert "analytics" not in schemas

    def test_drop_schema_cascade(self, empty_catalog):
        from ducklake_pyspark import (
            create_ducklake_schema,
            drop_ducklake_schema,
            list_schemas,
        )

        meta, data, _ = empty_catalog
        create_ducklake_schema(meta, "temp_schema")
        schemas = list_schemas(meta)
        assert "temp_schema" in schemas

        # Drop with cascade (even though empty)
        drop_ducklake_schema(meta, "temp_schema", cascade=True)
        schemas = list_schemas(meta)
        assert "temp_schema" not in schemas


# ==================================================================
# VIEW operations
# ==================================================================


class TestViewOperations:
    def test_create_and_drop_view(self, ducklake_catalog):
        from ducklake_pyspark import (
            create_ducklake_view,
            drop_ducklake_view,
            list_views,
            get_view,
        )

        meta, data, _ = ducklake_catalog
        create_ducklake_view(meta, "high_scores", "SELECT * FROM users WHERE score > 50")

        views = list_views(meta)
        assert "high_scores" in views

        view_def = get_view(meta, "high_scores")
        assert view_def is not None
        assert "score > 50" in view_def["sql"]

        drop_ducklake_view(meta, "high_scores")
        views = list_views(meta)
        assert "high_scores" not in views

    def test_create_view_or_replace(self, ducklake_catalog):
        from ducklake_pyspark import create_ducklake_view, get_view

        meta, data, _ = ducklake_catalog
        create_ducklake_view(meta, "my_view", "SELECT 1 AS x")
        create_ducklake_view(meta, "my_view", "SELECT 2 AS y", or_replace=True)

        view_def = get_view(meta, "my_view")
        assert "2" in view_def["sql"]


# ==================================================================
# TAG operations
# ==================================================================


class TestTagOperations:
    def test_table_tags(self, ducklake_catalog):
        from ducklake_pyspark import (
            set_ducklake_table_tag,
            delete_ducklake_table_tag,
        )

        meta, data, _ = ducklake_catalog
        set_ducklake_table_tag(meta, "users", "comment", "User table")
        # Should not raise
        delete_ducklake_table_tag(meta, "users", "comment")

    def test_column_tags(self, ducklake_catalog):
        from ducklake_pyspark import (
            set_ducklake_column_tag,
            delete_ducklake_column_tag,
        )

        meta, data, _ = ducklake_catalog
        set_ducklake_column_tag(meta, "users", "id", "comment", "Primary key")
        # Should not raise
        delete_ducklake_column_tag(meta, "users", "id", "comment")


# ==================================================================
# MAINTENANCE operations
# ==================================================================


class TestMaintenanceOperations:
    def test_expire_snapshots(self, ducklake_catalog):
        from ducklake_pyspark import expire_snapshots

        meta, data, _ = ducklake_catalog
        count = expire_snapshots(meta, keep_last_n=1)
        assert isinstance(count, int)

    def test_vacuum(self, ducklake_catalog):
        from ducklake_pyspark import vacuum_ducklake

        meta, data, _ = ducklake_catalog
        count = vacuum_ducklake(meta)
        assert isinstance(count, int)

    def test_rewrite_data_files(self, empty_catalog):
        from ducklake_pyspark import rewrite_data_files_ducklake

        meta, data, _ = empty_catalog
        result = rewrite_data_files_ducklake(meta, "items")
        assert isinstance(result, int)


# ==================================================================
# CATALOG READ operations
# ==================================================================


class TestCatalogReadOperations:
    def test_list_schemas(self, ducklake_catalog):
        from ducklake_pyspark import list_schemas

        meta, data, _ = ducklake_catalog
        schemas = list_schemas(meta)
        assert isinstance(schemas, list)
        assert "main" in schemas

    def test_list_tables(self, ducklake_catalog):
        from ducklake_pyspark import list_tables

        meta, data, _ = ducklake_catalog
        tables = list_tables(meta)
        assert isinstance(tables, list)
        assert "users" in tables

    def test_list_views_empty(self, ducklake_catalog):
        from ducklake_pyspark import list_views

        meta, data, _ = ducklake_catalog
        views = list_views(meta)
        assert isinstance(views, list)

    def test_list_snapshots(self, ducklake_catalog):
        from ducklake_pyspark import list_snapshots

        meta, data, _ = ducklake_catalog
        snapshots = list_snapshots(meta)
        assert isinstance(snapshots, list)
        assert len(snapshots) >= 1

    def test_snapshot_changes(self, ducklake_catalog):
        from ducklake_pyspark import snapshot_changes

        meta, data, _ = ducklake_catalog
        changes = snapshot_changes(meta)
        assert isinstance(changes, list)

    def test_catalog_info(self, ducklake_catalog):
        from ducklake_pyspark import catalog_info

        meta, data, _ = ducklake_catalog
        info = catalog_info(meta)
        assert isinstance(info, dict)

    def test_table_info(self, ducklake_catalog):
        from ducklake_pyspark import table_info

        meta, data, _ = ducklake_catalog
        info = table_info(meta, "users")
        assert isinstance(info, list)
        col_names = [c["column_name"] for c in info]
        assert "id" in col_names
        assert "name" in col_names
        assert "score" in col_names

    def test_get_view_nonexistent(self, ducklake_catalog):
        from ducklake_pyspark import get_view

        meta, data, _ = ducklake_catalog
        result = get_view(meta, "nonexistent")
        assert result is None


# ==================================================================
# Integration: DDL then read
# ==================================================================


class TestDDLThenRead:
    """Ensure DDL operations work end-to-end with PySpark reads."""

    def test_add_column_then_read(self, ducklake_catalog, spark):
        from ducklake_pyspark import alter_ducklake_add_column, read_ducklake

        meta, data, _ = ducklake_catalog
        alter_ducklake_add_column(meta, "users", "status", StringType())

        df = read_ducklake(spark, meta, "users")
        assert "status" in df.columns
        assert df.count() == 10

    def test_rename_column_then_read(self, ducklake_catalog, spark):
        from ducklake_pyspark import alter_ducklake_rename_column, read_ducklake

        meta, data, _ = ducklake_catalog
        alter_ducklake_rename_column(meta, "users", "name", "full_name")

        df = read_ducklake(spark, meta, "users")
        assert "full_name" in df.columns
        assert "name" not in df.columns

    def test_create_schema_and_list(self, ducklake_catalog):
        from ducklake_pyspark import create_ducklake_schema, list_schemas

        meta, data, _ = ducklake_catalog
        create_ducklake_schema(meta, "staging")
        schemas = list_schemas(meta)
        assert "staging" in schemas
        assert "main" in schemas
