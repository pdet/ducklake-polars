"""Tests for PySpark integration."""

from __future__ import annotations

import os
import tempfile
import shutil

import duckdb
import pytest

# Skip all tests if PySpark is not available
pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark():
    """Create a local Spark session for testing."""
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("ducklake-pyspark-test")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Xss4m")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def ducklake_catalog():
    """Create a DuckLake catalog with test data."""
    tmpdir = tempfile.mkdtemp(prefix="ducklake_pyspark_test_")
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
        "CASE WHEN i % 2 = 0 THEN 'US' ELSE 'EU' END AS region, "
        "CAST(i * 10.5 AS DOUBLE) AS score "
        "FROM range(100) t(i)"
    )
    con.close()

    yield meta, data, tmpdir

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def ducklake_multi_snapshot():
    """Create a DuckLake catalog with multiple snapshots for time travel."""
    tmpdir = tempfile.mkdtemp(prefix="ducklake_pyspark_tt_")
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
    # Snapshot 1: 10 rows
    con.execute(
        "CREATE TABLE lake.events AS "
        "SELECT i AS id, CAST(i AS DOUBLE) AS value "
        "FROM range(10) t(i)"
    )
    # Snapshot 2: 10 more rows
    con.execute(
        "INSERT INTO lake.events "
        "SELECT i AS id, CAST(i AS DOUBLE) AS value "
        "FROM range(10, 20) t(i)"
    )
    # Snapshot 3: add a column
    con.execute("ALTER TABLE lake.events ADD COLUMN status VARCHAR DEFAULT 'active'")
    con.close()

    yield meta, data, tmpdir

    shutil.rmtree(tmpdir, ignore_errors=True)


class TestReadDuckLake:
    """Basic read tests."""

    def test_read_basic(self, spark, ducklake_catalog):
        from ducklake_pyspark import read_ducklake

        meta, _, _ = ducklake_catalog
        df = read_ducklake(spark, meta, "users")

        assert df.count() == 100
        assert set(df.columns) == {"id", "name", "region", "score"}

    def test_read_column_selection(self, spark, ducklake_catalog):
        from ducklake_pyspark import read_ducklake

        meta, _, _ = ducklake_catalog
        df = read_ducklake(spark, meta, "users", columns=["id", "name"])

        assert df.count() == 100
        assert set(df.columns) == {"id", "name"}

    def test_read_filter(self, spark, ducklake_catalog):
        from ducklake_pyspark import read_ducklake

        meta, _, _ = ducklake_catalog
        df = read_ducklake(spark, meta, "users")
        filtered = df.filter(df.region == "US")

        assert filtered.count() == 50

    def test_read_values(self, spark, ducklake_catalog):
        from ducklake_pyspark import read_ducklake

        meta, _, _ = ducklake_catalog
        df = read_ducklake(spark, meta, "users")

        row = df.filter(df.id == 0).collect()[0]
        assert row["id"] == 0
        assert row["name"] == "user_0"
        assert row["region"] == "US"
        assert abs(row["score"] - 0.0) < 0.01

    def test_read_types(self, spark, ducklake_catalog):
        from ducklake_pyspark import read_ducklake
        from pyspark.sql.types import LongType, StringType, DoubleType

        meta, _, _ = ducklake_catalog
        df = read_ducklake(spark, meta, "users")
        schema = df.schema

        assert isinstance(schema["id"].dataType, LongType)
        assert isinstance(schema["name"].dataType, StringType)
        assert isinstance(schema["region"].dataType, StringType)
        assert isinstance(schema["score"].dataType, DoubleType)

    def test_table_not_found(self, spark, ducklake_catalog):
        from ducklake_pyspark import read_ducklake

        meta, _, _ = ducklake_catalog
        with pytest.raises(ValueError, match="not found"):
            read_ducklake(spark, meta, "nonexistent_table")

    def test_spark_operations(self, spark, ducklake_catalog):
        """Verify standard Spark operations work on the DataFrame."""
        from ducklake_pyspark import read_ducklake

        meta, _, _ = ducklake_catalog
        df = read_ducklake(spark, meta, "users")

        # Aggregation
        result = df.groupBy("region").count().collect()
        counts = {row["region"]: row["count"] for row in result}
        assert counts["US"] == 50
        assert counts["EU"] == 50

        # SQL
        df.createOrReplaceTempView("users_view")
        sql_result = spark.sql("SELECT COUNT(*) as cnt FROM users_view WHERE id < 10").collect()
        assert sql_result[0]["cnt"] == 10


class TestTimeTravelPySpark:
    """Time travel tests."""

    def test_read_at_snapshot(self, spark, ducklake_multi_snapshot):
        from ducklake_pyspark import read_ducklake

        meta, _, _ = ducklake_multi_snapshot

        # Snapshot 1: 10 rows, 2 columns
        df1 = read_ducklake(spark, meta, "events", snapshot_version=1)
        assert df1.count() == 10

        # Snapshot 2: 20 rows, 2 columns
        df2 = read_ducklake(spark, meta, "events", snapshot_version=2)
        assert df2.count() == 20

    def test_read_latest(self, spark, ducklake_multi_snapshot):
        from ducklake_pyspark import read_ducklake

        meta, _, _ = ducklake_multi_snapshot

        # Latest should have 20 rows + status column
        df = read_ducklake(spark, meta, "events")
        assert df.count() == 20
        assert "status" in df.columns

    def test_snapshot_version_and_time_conflict(self, spark, ducklake_multi_snapshot):
        from ducklake_pyspark import read_ducklake
        from datetime import datetime

        meta, _, _ = ducklake_multi_snapshot
        with pytest.raises(ValueError, match="Cannot specify both"):
            read_ducklake(
                spark, meta, "events",
                snapshot_version=1,
                snapshot_time=datetime.now(),
            )
