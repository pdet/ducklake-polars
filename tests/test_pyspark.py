"""Tests for PySpark integration."""

from __future__ import annotations

import os
import tempfile
import shutil

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# If REQUIRE_PYSPARK=1, a missing PySpark is a hard failure (used in CI on
# Python >= 3.10). Otherwise, silently skip — useful for environments where
# PySpark / a JVM is not available.
if os.environ.get("REQUIRE_PYSPARK") == "1":
    import pyspark  # noqa: F401  (fail loudly if missing)
else:
    pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


def _get_backends():
    """Return pytest parameters for available backends."""
    backends = [
        pytest.param("sqlite", id="sqlite"),
        pytest.param("duckdb", id="duckdb"),
    ]
    if os.environ.get("DUCKLAKE_PG_DSN"):
        backends.append(
            pytest.param("postgres", id="postgres", marks=pytest.mark.postgres)
        )
    return backends


def _meta_path(backend: str, tmpdir: str) -> str:
    if backend == "sqlite":
        return os.path.join(tmpdir, "test.ducklake")
    if backend == "duckdb":
        return os.path.join(tmpdir, "test.duckdb")
    return os.environ["DUCKLAKE_PG_DSN"]


def _attach_source(backend: str, meta: str) -> str:
    if backend == "sqlite":
        return f"ducklake:sqlite:{meta}"
    if backend == "duckdb":
        return f"ducklake:duckdb:{meta}"
    return f"ducklake:postgres:{meta}"


def _cleanup_postgres(meta: str) -> None:
    """Drop all tables in the PostgreSQL public schema for test isolation."""
    import psycopg2

    con = psycopg2.connect(meta)
    try:
        con.autocommit = True
        cur = con.cursor()
        try:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            tables = [row[0] for row in cur.fetchall()]
            for table in tables:
                safe = table.replace('"', '""')
                cur.execute(f'DROP TABLE IF EXISTS "{safe}" CASCADE')
        finally:
            cur.close()
    finally:
        con.close()


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


@pytest.fixture(params=_get_backends())
def ducklake_catalog(request):
    """Create a DuckLake catalog with test data, parametrized over backends."""
    backend = request.param
    tmpdir = tempfile.mkdtemp(prefix="ducklake_pyspark_test_")
    meta = _meta_path(backend, tmpdir)
    data = os.path.join(tmpdir, "data")
    os.makedirs(data)

    if backend == "postgres":
        _cleanup_postgres(meta)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH '{_attach_source(backend, meta)}' AS lake "
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
    if backend == "postgres":
        _cleanup_postgres(meta)


@pytest.fixture(params=_get_backends())
def ducklake_multi_snapshot(request):
    """Create a DuckLake catalog with multiple snapshots for time travel."""
    backend = request.param
    tmpdir = tempfile.mkdtemp(prefix="ducklake_pyspark_tt_")
    meta = _meta_path(backend, tmpdir)
    data = os.path.join(tmpdir, "data")
    os.makedirs(data)

    if backend == "postgres":
        _cleanup_postgres(meta)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH '{_attach_source(backend, meta)}' AS lake "
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
    if backend == "postgres":
        _cleanup_postgres(meta)


@pytest.fixture(params=_get_backends())
def empty_ducklake(request):
    """Create an initialized but empty DuckLake catalog."""
    backend = request.param
    tmpdir = tempfile.mkdtemp(prefix="ducklake_pyspark_empty_")
    meta = _meta_path(backend, tmpdir)
    data = os.path.join(tmpdir, "data")
    os.makedirs(data)

    if backend == "postgres":
        _cleanup_postgres(meta)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH '{_attach_source(backend, meta)}' AS lake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.execute("DETACH lake")
    con.close()

    yield meta, data, tmpdir

    shutil.rmtree(tmpdir, ignore_errors=True)
    if backend == "postgres":
        _cleanup_postgres(meta)


# ------------------------------------------------------------------
# Read tests (existing)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Write tests
# ------------------------------------------------------------------


class TestWriteDuckLake:
    """Write operation tests."""

    def test_write_basic(self, spark, empty_ducklake):
        """Write a PySpark DataFrame, then read it back."""
        from ducklake_pyspark import write_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        df = spark.createDataFrame(
            [(1, "alice", 10.0), (2, "bob", 20.0), (3, "charlie", 30.0)],
            ["id", "name", "score"],
        )

        write_ducklake(df, meta, "test_table")

        result = read_ducklake(spark, meta, "test_table")
        assert result.count() == 3
        assert set(result.columns) == {"id", "name", "score"}

        rows = {r["id"]: r for r in result.collect()}
        assert rows[1]["name"] == "alice"
        assert rows[2]["name"] == "bob"
        assert abs(rows[3]["score"] - 30.0) < 0.01

    def test_write_error_mode_existing_table(self, spark, ducklake_catalog):
        """mode='error' should fail when the table already exists."""
        from ducklake_pyspark import write_ducklake

        meta, _, _ = ducklake_catalog
        df = spark.createDataFrame([(1, "x")], ["id", "name"])

        with pytest.raises(ValueError, match="already exists"):
            write_ducklake(df, meta, "users", mode="error")

    def test_write_append(self, spark, empty_ducklake):
        """Write then append more data."""
        from ducklake_pyspark import write_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        df1 = spark.createDataFrame([(1, "alice"), (2, "bob")], ["id", "name"])
        df2 = spark.createDataFrame([(3, "charlie"), (4, "dave")], ["id", "name"])

        write_ducklake(df1, meta, "append_test", mode="append")
        write_ducklake(df2, meta, "append_test", mode="append")

        result = read_ducklake(spark, meta, "append_test")
        assert result.count() == 4

    def test_write_overwrite(self, spark, empty_ducklake):
        """Overwrite replaces all data."""
        from ducklake_pyspark import write_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        df1 = spark.createDataFrame([(1, "alice"), (2, "bob")], ["id", "name"])
        df2 = spark.createDataFrame([(3, "charlie")], ["id", "name"])

        write_ducklake(df1, meta, "overwrite_test", mode="append")
        assert read_ducklake(spark, meta, "overwrite_test").count() == 2

        write_ducklake(df2, meta, "overwrite_test", mode="overwrite")
        result = read_ducklake(spark, meta, "overwrite_test")
        assert result.count() == 1
        assert result.collect()[0]["name"] == "charlie"


class TestCreateTable:
    """create_ducklake_table and create_table_as_ducklake tests."""

    def test_create_empty_table(self, spark, empty_ducklake):
        """Create an empty table with a PySpark schema."""
        from ducklake_pyspark import create_ducklake_table, read_ducklake

        meta, _, _ = empty_ducklake
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("score", DoubleType(), True),
        ])

        create_ducklake_table(meta, "empty_table", schema)

        result = read_ducklake(spark, meta, "empty_table")
        assert result.count() == 0
        assert set(result.columns) == {"id", "name", "score"}

    def test_create_table_as(self, spark, empty_ducklake):
        """CTAS: create table with data in one snapshot."""
        from ducklake_pyspark import create_table_as_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        df = spark.createDataFrame(
            [(1, "alice", 10.0), (2, "bob", 20.0)],
            ["id", "name", "score"],
        )

        create_table_as_ducklake(df, meta, "ctas_table")

        result = read_ducklake(spark, meta, "ctas_table")
        assert result.count() == 2
        assert set(result.columns) == {"id", "name", "score"}


class TestDeleteDuckLake:
    """delete_ducklake tests."""

    def test_delete_basic(self, spark, empty_ducklake):
        """Delete rows matching a SQL predicate."""
        from ducklake_pyspark import write_ducklake, delete_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        df = spark.createDataFrame(
            [(i, f"user_{i}", float(i * 10)) for i in range(10)],
            ["id", "name", "score"],
        )
        write_ducklake(df, meta, "del_test")

        deleted = delete_ducklake(meta, "del_test", "id >= 5")
        assert deleted == 5

        result = read_ducklake(spark, meta, "del_test")
        assert result.count() == 5
        ids = sorted([r["id"] for r in result.collect()])
        assert ids == [0, 1, 2, 3, 4]

    def test_delete_no_match(self, spark, empty_ducklake):
        """Delete with no matching rows returns 0."""
        from ducklake_pyspark import write_ducklake, delete_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])
        write_ducklake(df, meta, "del_nomatch")

        deleted = delete_ducklake(meta, "del_nomatch", "id > 100")
        assert deleted == 0
        assert read_ducklake(spark, meta, "del_nomatch").count() == 2


class TestUpdateDuckLake:
    """update_ducklake tests."""

    def test_update_basic(self, spark, empty_ducklake):
        """Update rows matching a SQL predicate."""
        from ducklake_pyspark import write_ducklake, update_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        df = spark.createDataFrame(
            [(1, "alice", 10.0), (2, "bob", 20.0), (3, "charlie", 30.0)],
            ["id", "name", "score"],
        )
        write_ducklake(df, meta, "upd_test")

        updated = update_ducklake(
            meta, "upd_test",
            {"score": 99.0},
            "id <= 2",
        )
        assert updated == 2

        result = read_ducklake(spark, meta, "upd_test")
        assert result.count() == 3

        rows = {r["id"]: r for r in result.collect()}
        assert abs(rows[1]["score"] - 99.0) < 0.01
        assert abs(rows[2]["score"] - 99.0) < 0.01
        assert abs(rows[3]["score"] - 30.0) < 0.01

    def test_update_no_match(self, spark, empty_ducklake):
        """Update with no matching rows returns 0."""
        from ducklake_pyspark import write_ducklake, update_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        df = spark.createDataFrame([(1, "alice")], ["id", "name"])
        write_ducklake(df, meta, "upd_nomatch")

        updated = update_ducklake(
            meta, "upd_nomatch", {"name": "unknown"}, "id > 100"
        )
        assert updated == 0


class TestMergeDuckLake:
    """merge_ducklake tests."""

    def test_merge_insert_only(self, spark, empty_ducklake):
        """Merge with only inserts (no matched rows)."""
        from ducklake_pyspark import write_ducklake, merge_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        target = spark.createDataFrame(
            [(1, "alice", 10.0), (2, "bob", 20.0)],
            ["id", "name", "score"],
        )
        write_ducklake(target, meta, "merge_test")

        source = spark.createDataFrame(
            [(3, "charlie", 30.0), (4, "dave", 40.0)],
            ["id", "name", "score"],
        )

        updated, inserted = merge_ducklake(
            meta, "merge_test", source, "id",
            when_matched_update=True,
            when_not_matched_insert=True,
        )
        assert updated == 0
        assert inserted == 2

        result = read_ducklake(spark, meta, "merge_test")
        assert result.count() == 4

    def test_merge_update_and_insert(self, spark, empty_ducklake):
        """Merge that both updates existing and inserts new rows."""
        from ducklake_pyspark import write_ducklake, merge_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        target = spark.createDataFrame(
            [(1, "alice", 10.0), (2, "bob", 20.0)],
            ["id", "name", "score"],
        )
        write_ducklake(target, meta, "merge_upsert")

        source = spark.createDataFrame(
            [(2, "bob_updated", 25.0), (3, "charlie", 30.0)],
            ["id", "name", "score"],
        )

        updated, inserted = merge_ducklake(
            meta, "merge_upsert", source, "id",
            when_matched_update=True,
            when_not_matched_insert=True,
        )
        assert updated == 1
        assert inserted == 1

        result = read_ducklake(spark, meta, "merge_upsert")
        assert result.count() == 3

        rows = {r["id"]: r for r in result.collect()}
        assert rows[1]["name"] == "alice"
        assert rows[2]["name"] == "bob_updated"
        assert abs(rows[2]["score"] - 25.0) < 0.01
        assert rows[3]["name"] == "charlie"

    def test_merge_update_dict(self, spark, empty_ducklake):
        """Merge with dict-based update for matched rows."""
        from ducklake_pyspark import write_ducklake, merge_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        target = spark.createDataFrame(
            [(1, "alice", 10.0), (2, "bob", 20.0)],
            ["id", "name", "score"],
        )
        write_ducklake(target, meta, "merge_dict")

        source = spark.createDataFrame(
            [(1, "alice_new", 999.0), (2, "bob_new", 999.0)],
            ["id", "name", "score"],
        )

        updated, inserted = merge_ducklake(
            meta, "merge_dict", source, "id",
            when_matched_update={"score": 77.0},
            when_not_matched_insert=False,
        )
        assert updated == 2
        assert inserted == 0

        result = read_ducklake(spark, meta, "merge_dict")
        assert result.count() == 2

        rows = {r["id"]: r for r in result.collect()}
        # Names should stay, score should be updated
        assert rows[1]["name"] == "alice"
        assert abs(rows[1]["score"] - 77.0) < 0.01
        assert rows[2]["name"] == "bob"
        assert abs(rows[2]["score"] - 77.0) < 0.01


class TestAddFilesDuckLake:
    """add_files_ducklake tests."""

    def test_add_parquet_files(self, spark, empty_ducklake):
        """Register external Parquet files into a DuckLake table."""
        from ducklake_pyspark import (
            create_ducklake_table,
            add_files_ducklake,
            read_ducklake,
        )

        meta, data_dir, tmpdir = empty_ducklake

        # Create a Parquet file externally
        table = pa.table({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
        parquet_dir = os.path.join(data_dir, "main", "ext_table")
        os.makedirs(parquet_dir, exist_ok=True)
        parquet_path = os.path.join(parquet_dir, "data.parquet")
        pq.write_table(table, parquet_path)

        # Create the table schema
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("value", DoubleType(), True),
        ])
        create_ducklake_table(meta, "ext_table", schema)

        # Register the Parquet file
        add_files_ducklake(meta, "ext_table", [parquet_path])

        result = read_ducklake(spark, meta, "ext_table")
        assert result.count() == 3


class TestWriteReadRoundtrip:
    """End-to-end roundtrip tests: write with PySpark, read with PySpark."""

    def test_write_read_roundtrip(self, spark, empty_ducklake):
        """Full roundtrip: write → read → verify values."""
        from ducklake_pyspark import write_ducklake, read_ducklake

        meta, _, _ = empty_ducklake

        data = [(i, f"user_{i}", "US" if i % 2 == 0 else "EU", float(i * 10))
                for i in range(50)]
        df = spark.createDataFrame(data, ["id", "name", "region", "score"])

        write_ducklake(df, meta, "roundtrip")

        result = read_ducklake(spark, meta, "roundtrip")
        assert result.count() == 50
        assert set(result.columns) == {"id", "name", "region", "score"}

        # Verify aggregation
        us_count = result.filter(result.region == "US").count()
        assert us_count == 25

    def test_write_delete_read(self, spark, empty_ducklake):
        """Write → delete → read roundtrip."""
        from ducklake_pyspark import write_ducklake, delete_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        df = spark.createDataFrame(
            [(i, f"u{i}") for i in range(20)], ["id", "name"]
        )
        write_ducklake(df, meta, "wd_test")

        delete_ducklake(meta, "wd_test", "id >= 10")

        result = read_ducklake(spark, meta, "wd_test")
        assert result.count() == 10

    def test_write_update_read(self, spark, empty_ducklake):
        """Write → update → read roundtrip."""
        from ducklake_pyspark import write_ducklake, update_ducklake, read_ducklake

        meta, _, _ = empty_ducklake
        df = spark.createDataFrame(
            [(1, "old_name"), (2, "keep_me")], ["id", "name"]
        )
        write_ducklake(df, meta, "wu_test")

        update_ducklake(meta, "wu_test", {"name": "new_name"}, "id = 1")

        result = read_ducklake(spark, meta, "wu_test")
        rows = {r["id"]: r for r in result.collect()}
        assert rows[1]["name"] == "new_name"
        assert rows[2]["name"] == "keep_me"
