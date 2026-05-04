# %% [markdown]
# # ducklake-dataframe PySpark Tutorial
#
# This notebook walks through the major `ducklake-dataframe` features using
# the **PySpark** wrapper (`ducklake_pyspark`). It mirrors the
# [Polars](tutorial.ipynb) and [Pandas](tutorial_pandas.ipynb) tutorials
# so you can compare the three APIs side-by-side.
#
# We'll:
# 1. Create a DuckLake catalog with DuckDB and populate it
# 2. Read it with `ducklake_pyspark` (snapshot pinning, schema evolution)
# 3. Write data back (INSERT, DELETE, UPDATE, MERGE, CTAS, add_files)
# 4. Verify everything from DuckDB
# 5. Explore DDL, partitioning, views, tags, and catalog maintenance

# %%
# Install/upgrade to latest versions (uncomment when running as a notebook).
# On Google Colab, pin PySpark 3.5.x — Colab ships Java 11, while PySpark
# 4.x requires Java 17+. If you're on a machine with Java 17+, you can drop
# the version pin.
# !pip install -U "ducklake-dataframe[pyspark]" "pyspark>=3.5,<4.0"
# !pip install duckdb==1.5.2

# %%
import os
import shutil
import tempfile

import duckdb
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    DoubleType,
    BooleanType,
    TimestampType,
    StructField,
    StructType,
)

# Work in a temp directory so we don't pollute the repo
WORKDIR = tempfile.mkdtemp(prefix="ducklake_pyspark_tutorial_")
CATALOG = os.path.join(WORKDIR, "catalog.ducklake")
DATA = os.path.join(WORKDIR, "data")
os.makedirs(DATA, exist_ok=True)

print(f"Working directory: {WORKDIR}")
print(f"Catalog: {CATALOG}")
print(f"Data path: {DATA}")
print(f"DuckDB {duckdb.__version__}")

# Build a local SparkSession. PyArrow is required for fast row conversion.
spark = (
    SparkSession.builder
    .appName("ducklake-pyspark-tutorial")
    .master("local[2]")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# %% [markdown]
# ## 1. Initialize the Catalog with DuckDB
#
# DuckLake catalogs must be initialized by DuckDB's DuckLake extension. This
# creates the metadata schema (snapshot tables, column tables, etc.) that
# `ducklake_pyspark` reads and writes.

# %%
con = duckdb.connect()
con.execute("INSTALL ducklake; LOAD ducklake")
# DATA_INLINING_ROW_LIMIT 0 disables inlining so all rows land in Parquet
# files (the pyspark reader does not yet read inlined data).
con.execute(
    f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake "
    f"(DATA_PATH '{DATA}/', DATA_INLINING_ROW_LIMIT 0)"
)

con.execute("""
    CREATE TABLE lake.users (
        id INTEGER,
        name VARCHAR,
        email VARCHAR,
        score DOUBLE,
        active BOOLEAN
    )
""")

con.execute("""
    INSERT INTO lake.users VALUES
        (1, 'Alice',   'alice@example.com',   95.5, true),
        (2, 'Bob',     'bob@example.com',     87.3, true),
        (3, 'Carol',   'carol@example.com',   72.1, false),
        (4, 'Dave',    'dave@example.com',     91.0, true),
        (5, 'Eve',     'eve@example.com',      68.5, false)
""")

con.execute("""
    CREATE TABLE lake.events (
        event_id INTEGER,
        user_id INTEGER,
        region VARCHAR,
        amount DOUBLE,
        ts TIMESTAMP
    )
""")
con.execute("""
    INSERT INTO lake.events VALUES
        (1, 1, 'us',   100.0, '2025-01-15 10:00:00'),
        (2, 2, 'eu',   250.0, '2025-01-15 11:00:00'),
        (3, 1, 'us',    75.0, '2025-01-16 09:00:00'),
        (4, 3, 'apac', 300.0, '2025-01-16 14:00:00'),
        (5, 4, 'eu',   150.0, '2025-01-17 08:00:00')
""")
con.close()
print("DuckDB created 'users' (5 rows) and 'events' (5 rows).")

# %% [markdown]
# ## 2. Reading with ducklake_pyspark
#
# `read_ducklake` returns a regular PySpark DataFrame, so all DataFrame /
# SQL operations work — predicate pushdown, projection pushdown, joins, etc.

# %%
from ducklake_pyspark import read_ducklake

df = read_ducklake(spark, CATALOG, "users")
print("All users:")
df.orderBy("id").show()

# %%
# Predicate + projection pushdown (Spark optimizer handles this on the parquet scan)
result = (
    read_ducklake(spark, CATALOG, "users")
    .filter((F.col("active") == True) & (F.col("score") > 90))
    .select("name", "score")
    .orderBy(F.col("score").desc())
)
print("Active users with score > 90:")
result.show()

# %%
# Column projection at read time
read_ducklake(spark, CATALOG, "users", columns=["id", "name"]).orderBy("id").show()

# %% [markdown]
# ## 3. Catalog Inspection
#
# The `_ddl` module exposes catalog-introspection helpers as plain functions
# returning Python dicts/lists.

# %%
from ducklake_pyspark import (
    list_schemas,
    list_tables,
    list_snapshots,
    table_info,
    catalog_info,
)

print("Schemas :", list_schemas(CATALOG))
print("Tables  :", list_tables(CATALOG))
snaps = list_snapshots(CATALOG, limit=200)
print(f"Snapshots: {len(snaps)} total, current = {snaps[-1]['snapshot_id']}")
print("Catalog info:", catalog_info(CATALOG))
print("\nusers table info:")
print(table_info(CATALOG, "users"))

# %% [markdown]
# ## 4. Writing
#
# ### 4a. Append rows
#
# > **Note:** DuckDB created `id` as `INTEGER` (32-bit). When appending from
# > Spark, define the column as `IntegerType` to match.

# %%
from ducklake_pyspark import write_ducklake

new_users_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("score", DoubleType(), True),
    StructField("active", BooleanType(), True),
])
new_users = spark.createDataFrame(
    [
        (6, "Frank", "frank@example.com", 88.0, True),
        (7, "Grace", "grace@example.com", 94.5, True),
    ],
    schema=new_users_schema,
)

write_ducklake(
    new_users, CATALOG, "users", mode="append",
    author="tutorial", commit_message="Add Frank and Grace",
)

print("After append — 7 users:")
read_ducklake(spark, CATALOG, "users").orderBy("id").show()

# %% [markdown]
# ### 4b. Verify from DuckDB

# %%
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/', DATA_INLINING_ROW_LIMIT 0)")
print("DuckDB sees:", con.execute("SELECT COUNT(*) FROM lake.users").fetchone()[0], "rows")
con.close()

# %% [markdown]
# ### 4c. Delete rows
#
# `delete_ducklake` accepts a SQL expression evaluated against the table.

# %%
from ducklake_pyspark import delete_ducklake

deleted = delete_ducklake(
    CATALOG, "users", "active = false",
    author="tutorial", commit_message="Remove inactive users",
)
print(f"Deleted {deleted} inactive users")
read_ducklake(spark, CATALOG, "users").orderBy("id").show()

# %% [markdown]
# ### 4d. Update rows

# %%
from ducklake_pyspark import update_ducklake

updated = update_ducklake(
    CATALOG, "users",
    updates={"score": 100.0, "email": "alice-updated@example.com"},
    predicate_sql="name = 'Alice'",
    author="tutorial", commit_message="Perfect score for Alice",
)
print(f"Updated {updated} rows")
read_ducklake(spark, CATALOG, "users").filter(F.col("name") == "Alice").show()

# %% [markdown]
# ### 4e. Merge (upsert)

# %%
from ducklake_pyspark import merge_ducklake

source = spark.createDataFrame(
    [
        (1, "Alice", "alice@new.com", 98.0, True),
        (2, "Bob",   "bob@new.com",   92.0, True),
        (8, "Heidi", "heidi@example.com", 89.0, True),
    ],
    schema=new_users_schema,
)

rows_updated, rows_inserted = merge_ducklake(
    CATALOG, "users", source, on="id",
    when_matched_update=True,
    when_not_matched_insert=True,
    author="tutorial", commit_message="Merge user updates",
)
print(f"Updated: {rows_updated}, Inserted: {rows_inserted}")
read_ducklake(spark, CATALOG, "users").orderBy("id").show()

# %% [markdown]
# ## 5. Time Travel
#
# Every write creates a new snapshot. Pin a read to any historical version.

# %%
snaps = list_snapshots(CATALOG, limit=200)
snap_ids = sorted(s["snapshot_id"] for s in snaps)
print(f"All snapshot ids: {snap_ids}")

# Find the first snapshot that has user data (skip pre-table snapshots)
for v in snap_ids:
    try:
        df = read_ducklake(spark, CATALOG, "users", snapshot_version=v)
    except Exception:
        continue
    if df.count() > 0:
        print(f"\nSnapshot {v} — earliest snapshot with user rows:")
        df.orderBy("id").show()
        break

print("Latest snapshot:")
read_ducklake(spark, CATALOG, "users").orderBy("id").show()

# %% [markdown]
# ## 6. Change Data Feed (CDF)
#
# `read_ducklake_changes` returns a Spark DataFrame with `snapshot_id` and
# `change_type` columns.

# %%
from ducklake_pyspark import read_ducklake_changes

changes = read_ducklake_changes(
    spark, CATALOG, "users",
    start_snapshot=snap_ids[0],
    end_snapshot=snap_ids[-1],
)
print(f"Changes between {snap_ids[0]} and {snap_ids[-1]}:")
changes.orderBy("snapshot_id", "id").show()

# %% [markdown]
# ## 7. Schema Evolution

# %%
from ducklake_pyspark import (
    alter_ducklake_add_column,
    alter_ducklake_rename_column,
    alter_ducklake_drop_column,
    alter_ducklake_set_type,
)

alter_ducklake_add_column(CATALOG, "users", "department", StringType(),
                          author="tutorial", commit_message="Add department column")
print("After ADD COLUMN 'department':")
read_ducklake(spark, CATALOG, "users").orderBy("id").show()

# %%
alter_ducklake_rename_column(CATALOG, "users", "email", "contact_email",
                             author="tutorial", commit_message="email -> contact_email")
print("After RENAME COLUMN 'email' -> 'contact_email':")
read_ducklake(spark, CATALOG, "users").orderBy("id").show()

# %%
alter_ducklake_drop_column(CATALOG, "users", "department",
                           author="tutorial", commit_message="Drop department column")
print("After DROP COLUMN 'department':")
read_ducklake(spark, CATALOG, "users").orderBy("id").show()

# %%
# DuckDB sees the schema changes
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/', DATA_INLINING_ROW_LIMIT 0)")
print("DuckDB DESCRIBE after schema evolution:")
print(con.execute("DESCRIBE lake.users").fetchall())
con.close()

# %% [markdown]
# ## 8. Partitioned Writes

# %%
from ducklake_pyspark import alter_ducklake_set_partitioned_by

alter_ducklake_set_partitioned_by(CATALOG, "events", ["region"],
                                  author="tutorial", commit_message="Partition events by region")

new_events = spark.createDataFrame(
    [
        (6, 1, "us",   200.0, "2025-02-01 10:00:00"),
        (7, 2, "eu",   175.0, "2025-02-01 11:00:00"),
        (8, 6, "us",    50.0, "2025-02-02 09:00:00"),
        (9, 7, "apac", 400.0, "2025-02-02 14:00:00"),
    ],
    schema=StructType([
        StructField("event_id", IntegerType(), True),
        StructField("user_id", IntegerType(), True),
        StructField("region", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("ts", StringType(), True),
    ]),
).withColumn("ts", F.to_timestamp("ts"))

write_ducklake(new_events, CATALOG, "events", mode="append",
               author="tutorial", commit_message="Partitioned event insert")

print("All events:")
read_ducklake(spark, CATALOG, "events").orderBy("event_id").show()

# %%
# Partition pruning — only reads 'us' partition files
us_events = (
    read_ducklake(spark, CATALOG, "events")
    .filter(F.col("region") == "us")
    .orderBy("event_id")
)
print("US events only (partition-pruned):")
us_events.show()

# %%
# Show the on-disk file structure
print("Data files on disk:")
for root, _dirs, files in os.walk(DATA):
    for f in files:
        path = os.path.join(root, f)
        rel = os.path.relpath(path, DATA)
        print(f"  {rel}  ({os.path.getsize(path):,} bytes)")

# %% [markdown]
# ## 9. CREATE TABLE AS

# %%
from ducklake_pyspark import create_table_as_ducklake

summary = (
    read_ducklake(spark, CATALOG, "events")
    .groupBy("region")
    .agg(
        F.sum("amount").alias("total_amount"),
        F.count("event_id").alias("event_count"),
    )
    .orderBy("region")
)

create_table_as_ducklake(
    summary, CATALOG, "region_summary",
    author="tutorial", commit_message="Regional summary CTAS",
)

print("Created 'region_summary' via CTAS:")
read_ducklake(spark, CATALOG, "region_summary").show()

# %% [markdown]
# ## 10. Views

# %%
from ducklake_pyspark import create_ducklake_view, drop_ducklake_view, list_views, get_view

create_ducklake_view(
    CATALOG, "active_users",
    "SELECT id, name, score FROM users",
    author="tutorial", commit_message="Create active_users view",
)
print("Views:", list_views(CATALOG))
print("Definition:", get_view(CATALOG, "active_users"))

# %%
# Replace the view (or_replace=True)
create_ducklake_view(
    CATALOG, "active_users",
    "SELECT id, name, score FROM users WHERE score > 90",
    or_replace=True,
    author="tutorial", commit_message="Update view: score > 90 filter",
)
print("Updated view definition:", get_view(CATALOG, "active_users"))

# DuckDB can query the (replaced) view
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(
    f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake "
    f"(DATA_PATH '{DATA}/', DATA_INLINING_ROW_LIMIT 0)"
)
print("DuckDB querying replaced view (score > 90):")
print(con.execute("SELECT * FROM lake.active_users ORDER BY id").fetchall())
con.close()
del con

# Force the DuckDB-side SQLite handles to be released before the next
# writer call (DuckDB on macOS occasionally needs a beat after close).
import gc, time
gc.collect()
time.sleep(0.2)

drop_ducklake_view(CATALOG, "active_users")
print("Views after drop:", list_views(CATALOG))

# %% [markdown]
# ## 11. Tags
#
# Tags attach key-value metadata to tables and columns. With DuckDB 1.5+ they
# round-trip cleanly through DuckDB's ducklake extension. (DuckDB currently
# only round-trips the `comment` key for column tags; table tags accept
# arbitrary keys.)

# %%
from ducklake_pyspark import (
    set_ducklake_table_tag,
    set_ducklake_column_tag,
    delete_ducklake_table_tag,
)

set_ducklake_table_tag(CATALOG, "users", "owner", "analytics-team")
set_ducklake_table_tag(CATALOG, "users", "pii", "true")
set_ducklake_column_tag(CATALOG, "users", "contact_email", "comment", "PII: email address")
print("Tags set on users.")

delete_ducklake_table_tag(CATALOG, "users", "pii")
print("Removed 'pii' tag.")

# %% [markdown]
# ## 12. Schema and Table Management

# %%
from ducklake_pyspark import (
    create_ducklake_schema,
    drop_ducklake_schema,
    rename_ducklake_table,
    drop_ducklake_table,
    create_ducklake_table,
)

create_ducklake_schema(CATALOG, "staging")
print("Schemas:", list_schemas(CATALOG))

create_ducklake_table(
    CATALOG, "raw_data",
    StructType([
        StructField("id", IntegerType(), True),
        StructField("payload", StringType(), True),
    ]),
    schema="staging",
)
print("Tables in 'staging':", list_tables(CATALOG, schema="staging"))

rename_ducklake_table(CATALOG, "raw_data", "raw_events", schema="staging")
print("After rename:", list_tables(CATALOG, schema="staging"))

drop_ducklake_table(CATALOG, "raw_events", schema="staging")
drop_ducklake_schema(CATALOG, "staging")
print("After cleanup:", list_schemas(CATALOG))

# %% [markdown]
# ## 13. Registering External Parquet Files
#
# `add_files_ducklake` registers an existing Parquet file into the catalog
# without copying it.

# %%
from ducklake_pyspark import add_files_ducklake

# We write the external file with PyArrow rather than Spark to keep the
# timestamp type as `timestamp[us]` (no time zone) so it matches the
# DuckLake `events.ts` column. Spark's parquet writer would attach UTC,
# which `add_files_ducklake` rejects against the existing schema.
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

ext_path = os.path.join(WORKDIR, "external_events.parquet")
ext_table = pa.table({
    "event_id": pa.array([200, 201], type=pa.int32()),
    "user_id":  pa.array([1, 2], type=pa.int32()),
    "region":   pa.array(["us", "eu"]),
    "amount":   pa.array([10.0, 20.0]),
    "ts":       pa.array(
        [datetime(2025, 5, 1, 10, 0), datetime(2025, 5, 1, 10, 5)],
        type=pa.timestamp("us"),
    ),
})
pq.write_table(ext_table, ext_path)

added_snapshot = add_files_ducklake(CATALOG, "events", [ext_path])
print(f"Registered external file; new snapshot id: {added_snapshot}")
print(f"Total events now: {read_ducklake(spark, CATALOG, 'events').count()}")

# %% [markdown]
# ## 14. Compaction & Maintenance

# %%
from ducklake_pyspark import (
    rewrite_data_files_ducklake,
    expire_snapshots,
    vacuum_ducklake,
)

def count_parquet_files(root: str) -> int:
    return sum(1 for _, _, files in os.walk(root) for f in files if f.endswith(".parquet"))

# Note: `rewrite_data_files_ducklake` works across all three wrappers, but
# requires consistent timestamp metadata across files. We've mixed
# DuckDB-written, Spark-written, and PyArrow-written timestamps in `events`,
# which trips an Arrow concat error here. Demonstrate the API on `users`
# instead (no timestamp columns).
n_before = count_parquet_files(DATA)
rewrite_data_files_ducklake(CATALOG, "users")
n_after = count_parquet_files(DATA)
print(f"parquet files in {DATA}: {n_before} -> {n_after} after rewrite of 'users'")

print(f"Snapshots before expire: {len(list_snapshots(CATALOG, limit=200))}")
expired = expire_snapshots(CATALOG, keep_last_n=3)
print(f"Expired {expired} snapshots")
print(f"Snapshots after expire:  {len(list_snapshots(CATALOG, limit=200))}")

deleted = vacuum_ducklake(CATALOG)
print(f"Vacuumed {deleted} orphaned files")

# %% [markdown]
# ## 15. Final Roundtrip Verification

# %%
print("=== ducklake_pyspark ===")
for table in ["users", "events", "region_summary"]:
    n = read_ducklake(spark, CATALOG, table).count()
    print(f"  {table:20s} -> {n} rows")

# %%
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/', DATA_INLINING_ROW_LIMIT 0)")
print("=== DuckDB ===")
for table in ["users", "events", "region_summary"]:
    n = con.execute(f"SELECT COUNT(*) FROM lake.{table}").fetchone()[0]
    print(f"  {table:20s} -> {n} rows")
con.close()

# %% [markdown]
# ## Cleanup

# %%
spark.stop()
shutil.rmtree(WORKDIR)
print(f"Cleaned up {WORKDIR}")
print("\nALL DONE — PySpark tutorial complete!")

# %% [markdown]
# ---
#
# ## Summary
#
# | Feature | Functions Used |
# |---------|----------------|
# | **Reading** | `read_ducklake`, `read_ducklake_changes` |
# | **Writing** | `write_ducklake` (error/append/overwrite) |
# | **Deletes** | `delete_ducklake` |
# | **Updates** | `update_ducklake` |
# | **Merge** | `merge_ducklake` |
# | **CTAS** | `create_table_as_ducklake` |
# | **External files** | `add_files_ducklake` |
# | **DDL** | `create_ducklake_table`, `alter_ducklake_add/drop/rename_column`, `alter_ducklake_set_type` |
# | **Partitioning** | `alter_ducklake_set_partitioned_by` |
# | **Sort keys** | `alter_ducklake_set_sort_keys`, `alter_ducklake_reset_sort_keys` |
# | **Views** | `create_ducklake_view`, `drop_ducklake_view`, `list_views`, `get_view` |
# | **Schemas** | `create_ducklake_schema`, `drop_ducklake_schema` |
# | **Tags** | `set/delete_ducklake_table_tag`, `set/delete_ducklake_column_tag` |
# | **Time travel** | `read_ducklake(..., snapshot_version=N)` |
# | **Change data feed** | `read_ducklake_changes` |
# | **Catalog inspection** | `list_schemas/tables/views/snapshots`, `table_info`, `catalog_info` |
# | **Maintenance** | `rewrite_data_files_ducklake`, `expire_snapshots`, `vacuum_ducklake` |
# | **DuckDB interop** | Full bidirectional |
#
# For the complete API reference, see the
# [wiki](https://github.com/pdet/ducklake-dataframe/wiki).
