#!/usr/bin/env python3
"""
ducklake-dataframe Pandas Tutorial
===================================

This script walks through every major ducklake-dataframe feature using
the **Pandas** wrapper (``ducklake_pandas``).  It mirrors the Polars
tutorial notebook so you can compare the two APIs side-by-side.

Requirements
------------
    pip install ducklake-dataframe[pandas]
    pip install duckdb==1.5.2   # macros, custom column tags, merge_adjacent_files
"""

import os, shutil, tempfile, datetime
import duckdb
import pandas as pd
import numpy as np

# ── work directory ──────────────────────────────────────────────────
WORKDIR  = tempfile.mkdtemp(prefix="ducklake_pandas_tutorial_")
CATALOG  = os.path.join(WORKDIR, "catalog.ducklake")
DATAPATH = os.path.join(WORKDIR, "data/")
print(f"Working directory: {WORKDIR}")
print(f"DuckDB {duckdb.__version__}")

# ====================================================================
# 1.  Initialize the catalog with DuckDB
# ====================================================================
print("\n" + "=" * 60)
print("1.  Initialize the catalog")
print("=" * 60)

con = duckdb.connect()
con.execute("INSTALL ducklake; LOAD ducklake")
con.execute(
    f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake "
    f"(DATA_PATH '{DATAPATH}')"
)
con.execute("""
    CREATE TABLE lake.users (
        id     INTEGER,
        name   VARCHAR,
        region VARCHAR
    )
""")
con.execute("""
    INSERT INTO lake.users VALUES
        (1, 'Alice',   'us'),
        (2, 'Bob',     'eu'),
        (3, 'Charlie', 'us'),
        (4, 'Diana',   'eu'),
        (5, 'Eve',     'us')
""")
con.execute("""
    CREATE TABLE lake.events (
        ts     TIMESTAMP,
        user_id INTEGER,
        action  VARCHAR,
        region  VARCHAR
    )
""")
con.execute("""
    INSERT INTO lake.events VALUES
        ('2025-03-01 10:00:00', 1, 'login',  'us'),
        ('2025-03-01 10:05:00', 2, 'login',  'eu'),
        ('2025-03-01 10:10:00', 1, 'click',  'us'),
        ('2025-03-01 10:15:00', 3, 'login',  'us'),
        ('2025-03-01 10:20:00', 2, 'click',  'eu')
""")
con.close()
print("Catalog created with 'users' (5 rows) and 'events' (5 rows)")


# ====================================================================
# 2.  Reading with ducklake_pandas
# ====================================================================
print("\n" + "=" * 60)
print("2.  Reading")
print("=" * 60)

from ducklake_pandas import read_ducklake

df = read_ducklake(CATALOG, "users")
print(f"\nFull table ({len(df)} rows):")
print(df.to_string(index=False))

# Predicate pushdown – lambda receives a DataFrame, returns a boolean Series
us_users = read_ducklake(
    CATALOG, "users",
    predicate=lambda d: d["region"] == "us",
)
print(f"\nUS users only ({len(us_users)} rows):")
print(us_users.to_string(index=False))

# Column projection
names = read_ducklake(CATALOG, "users", columns=["id", "name"])
print(f"\nProjected columns: {list(names.columns)}")


# ====================================================================
# 3.  Catalog inspection
# ====================================================================
print("\n" + "=" * 60)
print("3.  Catalog inspection")
print("=" * 60)

from ducklake_pandas import DuckLakeCatalog

catalog = DuckLakeCatalog(CATALOG)
print(f"Schemas : {catalog.list_schemas()}")
print(f"Tables  : {list(catalog.list_tables()['table_name'])}")
print(f"Snapshots: {len(catalog.snapshots())} total, current = {catalog.current_snapshot()}")
print(f"\nTable info:\n{catalog.table_info()}")


# ====================================================================
# 4.  Writing
# ====================================================================
print("\n" + "=" * 60)
print("4.  Writing")
print("=" * 60)

# ── 4a. Append ──────────────────────────────────────────────────────
from ducklake_pandas import write_ducklake

new_users = pd.DataFrame({
    "id":     [6, 7],
    "name":   ["Frank", "Grace"],
    "region": ["eu", "us"],
})
write_ducklake(new_users, CATALOG, "users", mode="append")
print(f"After append: {len(read_ducklake(CATALOG, 'users'))} rows")

# ── 4b. DuckDB interop ─────────────────────────────────────────────
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATAPATH}')")
print(f"DuckDB sees: {con.execute('SELECT COUNT(*) FROM lake.users').fetchone()[0]} rows")
con.close()

# ── 4c. Delete ──────────────────────────────────────────────────────
from ducklake_pandas import delete_ducklake

deleted = delete_ducklake(
    CATALOG, "users",
    predicate=lambda d: d["id"] == 2,
)
print(f"\nDeleted {deleted} row (Bob)")

# ── 4d. Update ──────────────────────────────────────────────────────
from ducklake_pandas import update_ducklake

updated = update_ducklake(
    CATALOG, "users",
    updates={"region": "apac"},
    predicate=lambda d: d["name"] == "Eve",
)
print(f"Updated {updated} row (Eve → apac)")

# ── 4e. Merge ───────────────────────────────────────────────────────
from ducklake_pandas import merge_ducklake

source = pd.DataFrame({
    "id":     [1, 8],
    "name":   ["Alice2", "Hank"],
    "region": ["us", "eu"],
})
rows_upd, rows_ins = merge_ducklake(
    CATALOG, "users", source, on="id",
    when_matched_update=True,
    when_not_matched_insert=True,
)
print(f"Merge: {rows_upd} updated, {rows_ins} inserted")

# ── 4f. Verify ──────────────────────────────────────────────────────
print(f"\nFinal users ({len(read_ducklake(CATALOG, 'users'))} rows):")
print(read_ducklake(CATALOG, "users").sort_values("id").to_string(index=False))


# ====================================================================
# 5.  Time travel
# ====================================================================
print("\n" + "=" * 60)
print("5.  Time travel")
print("=" * 60)

catalog = DuckLakeCatalog(CATALOG)
snapshots = catalog.snapshots()
print(f"Snapshots:\n{snapshots[['snapshot_id', 'snapshot_time']].to_string(index=False)}")

# Pick the snapshot after the initial INSERT (not the empty CREATE TABLE snapshot)
snap_v = int(snapshots["snapshot_id"].iloc[-2])  # second-to-last
old_df = read_ducklake(CATALOG, "users", snapshot_version=snap_v)
print(f"\nAt snapshot {snap_v}: {len(old_df)} rows")


# ====================================================================
# 5b. Change Data Feed
# ====================================================================
print("\n" + "=" * 60)
print("5b. Change Data Feed")
print("=" * 60)

from ducklake_pandas import read_ducklake_changes

snap_ids = sorted(catalog.snapshots()["snapshot_id"].tolist())
changes = read_ducklake_changes(
    CATALOG, "users", start_version=snap_ids[0], end_version=snap_ids[-1],
)
print(f"{len(changes)} changes between snapshots {snap_ids[0]} and {snap_ids[-1]}")
print(changes.head().to_string(index=False))
print(old_df.to_string(index=False))


# ====================================================================
# 6.  Schema evolution
# ====================================================================
print("\n" + "=" * 60)
print("6.  Schema evolution")
print("=" * 60)

from ducklake_pandas import (
    alter_ducklake_add_column,
    alter_ducklake_rename_column,
    alter_ducklake_drop_column,
    alter_ducklake_set_type,
)

alter_ducklake_add_column(CATALOG, "users", "email", "VARCHAR")
print("Added 'email' column")

alter_ducklake_add_column(CATALOG, "users", "department", "VARCHAR")
print("Added 'department' column")

alter_ducklake_rename_column(CATALOG, "users", "email", "contact")
print("Renamed email → contact")

alter_ducklake_drop_column(CATALOG, "users", "department")
print("Dropped 'department' column")

alter_ducklake_set_type(CATALOG, "users", "id", "BIGINT")
print("Changed id type to BIGINT")

df = read_ducklake(CATALOG, "users")
print(f"\nSchema after evolution: {list(df.dtypes.items())}")


# ====================================================================
# 7.  Partitioned writes
# ====================================================================
print("\n" + "=" * 60)
print("7.  Partitioned writes")
print("=" * 60)

from ducklake_pandas import alter_ducklake_set_partitioned_by

alter_ducklake_set_partitioned_by(CATALOG, "events", ["region"])
print("Set events partitioned by region")

new_events = pd.DataFrame({
    "ts":      pd.to_datetime(["2025-03-02 09:00:00", "2025-03-02 09:05:00"]),
    "user_id": [6, 7],
    "action":  ["signup", "signup"],
    "region":  ["us", "eu"],
})
write_ducklake(new_events, CATALOG, "events", mode="append")

us_events = read_ducklake(
    CATALOG, "events",
    predicate=lambda d: d["region"] == "us",
)
print(f"US events: {len(us_events)} rows (partition pruning)")


# ====================================================================
# 8.  Data inlining
# ====================================================================
print("\n" + "=" * 60)
print("8.  Data inlining")
print("=" * 60)

from ducklake_pandas import create_ducklake_table

create_ducklake_table(CATALOG, "tiny", {"ts": "TIMESTAMP", "value": "DOUBLE"})

# Small insert → inlined in the catalog DB
small = pd.DataFrame({
    "ts":    pd.to_datetime(["2025-01-01 00:00:00"]),
    "value": [3.14],
})
write_ducklake(small, CATALOG, "tiny", mode="append", data_inlining_row_limit=100)
print(f"Inlined 1 row, table has {len(read_ducklake(CATALOG, 'tiny'))} rows")

# Larger insert → Parquet
big = pd.DataFrame({
    "ts":    pd.to_datetime([f"2025-01-01 00:00:{i:02d}" for i in range(50)]),
    "value": [float(i) for i in range(50)],
})
write_ducklake(big, CATALOG, "tiny", mode="append")
print(f"After large append: {len(read_ducklake(CATALOG, 'tiny'))} rows")


# ====================================================================
# 9.  CREATE TABLE AS
# ====================================================================
print("\n" + "=" * 60)
print("9.  CREATE TABLE AS")
print("=" * 60)

from ducklake_pandas import create_table_as_ducklake

summary = read_ducklake(CATALOG, "events").groupby("region").size().reset_index(name="cnt")
create_table_as_ducklake(summary, CATALOG, "region_summary")
print(f"Created region_summary:\n{read_ducklake(CATALOG, 'region_summary').to_string(index=False)}")


# ====================================================================
# 10. Views
# ====================================================================
print("\n" + "=" * 60)
print("10. Views")
print("=" * 60)

from ducklake_pandas import create_ducklake_view

create_ducklake_view(
    CATALOG, "active_users",
    "SELECT id, name FROM users WHERE region = 'us'",
)
print("Created view 'active_users'")

# Replace view
create_ducklake_view(
    CATALOG, "active_users",
    "SELECT id, name, contact FROM users WHERE region = 'us'",
    or_replace=True,
)
print("Replaced view 'active_users'")

# Read view definition via DuckDB
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATAPATH}')")
from ducklake_pandas import list_views
print(f"Views in catalog: {list_views(CATALOG)}")
con.close()


# ====================================================================
# 11. Tags
# ====================================================================
print("\n" + "=" * 60)
print("11. Tags")
print("=" * 60)

from ducklake_pandas import (
    set_ducklake_table_tag,
    set_ducklake_column_tag,
    delete_ducklake_table_tag,
)

set_ducklake_table_tag(CATALOG, "users", "owner", "analytics-team")
set_ducklake_table_tag(CATALOG, "users", "pii", "true")
# DuckDB's ducklake extension currently only round-trips the `comment` key
# for columns; table tags do accept arbitrary keys on DuckDB 1.5+.
set_ducklake_column_tag(CATALOG, "users", "contact", "comment", "PII: email")
print("Set tags on users table and contact column")

delete_ducklake_table_tag(CATALOG, "users", "pii")
print("Deleted 'pii' tag")


# ====================================================================
# 11.5. Macros
# ====================================================================
print("\n" + "=" * 60)
print("11.5. Macros (requires DuckDB 1.5+)")
print("=" * 60)

from ducklake_pandas import create_ducklake_macro, drop_ducklake_macro

create_ducklake_macro(
    CATALOG, "add_one", "a + 1",
    parameters=[{"name": "a", "type": "integer"}],
)
catalog = DuckLakeCatalog(CATALOG)
print("Macros:", catalog.list_macros()["macro_name"].tolist())

con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATAPATH}')")
print("add_one(41) =", con.execute("SELECT lake.add_one(41)").fetchone()[0])
con.close()

drop_ducklake_macro(CATALOG, "add_one")


# ====================================================================
# 12. Schema and table management
# ====================================================================
print("\n" + "=" * 60)
print("12. Schema and table management")
print("=" * 60)

from ducklake_pandas import (
    create_ducklake_schema,
    drop_ducklake_schema,
    rename_ducklake_table,
    drop_ducklake_table,
)

create_ducklake_schema(CATALOG, "staging")
print("Created 'staging' schema")
create_ducklake_table(CATALOG, "raw_data", {"x": "INTEGER"}, schema="staging")
print("Created staging.raw_data")
rename_ducklake_table(CATALOG, "raw_data", "raw_events", schema="staging")
print("Renamed staging.raw_data -> staging.raw_events")
drop_ducklake_table(CATALOG, "raw_events", schema="staging")
print("Dropped staging.raw_events")
drop_ducklake_schema(CATALOG, "staging")
print("Dropped 'staging' schema")


# ====================================================================
# 12.5. Streaming ingestion
# ====================================================================
print("\n" + "=" * 60)
print("12.5.  Streaming ingestion")
print("=" * 60)

from ducklake_pandas import DuckLakeStreamWriter

with DuckLakeStreamWriter(
    CATALOG, "events", flush_threshold=2, compact_on_close=False,
) as writer:
    writer.append(pd.DataFrame({
        "ts":      pd.to_datetime(["2025-04-01 09:00:00", "2025-04-01 09:05:00"]),
        "user_id": [1, 2],
        "action":  ["login", "login"],
        "region":  ["us", "eu"],
    }))  # auto-flush
    writer.append(pd.DataFrame({
        "ts":      pd.to_datetime(["2025-04-01 09:10:00"]),
        "user_id": [1],
        "action":  ["click"],
        "region":  ["us"],
    }))  # flushes on close
print(f"Streamed {writer.total_rows} rows in {writer.flush_count} flushes")


# ====================================================================
# 12.6. Registering external Parquet files
# ====================================================================
print("\n" + "=" * 60)
print("12.6.  add_files_ducklake")
print("=" * 60)

from ducklake_pandas import add_files_ducklake

ext_path = os.path.join(WORKDIR, "external.parquet")
pd.DataFrame({
    "ts":      pd.to_datetime(["2025-05-01 10:00:00", "2025-05-01 10:05:00"]),
    "user_id": [1, 2],
    "action":  ["signup", "signup"],
    "region":  ["us", "eu"],
}).to_parquet(ext_path)
added = add_files_ducklake(CATALOG, "events", [ext_path])
print(f"Registered {added} external file(s); events row count: {len(read_ducklake(CATALOG, 'events'))}")


# ====================================================================
# 12.7. Compaction
# ====================================================================
print("\n" + "=" * 60)
print("12.7.  rewrite_data_files")
print("=" * 60)

from datetime import datetime, timedelta, timezone
from ducklake_pandas import (
    merge_adjacent_files_ducklake,
    cleanup_old_files_ducklake,
)

n_before = len(DuckLakeCatalog(CATALOG).list_files("events"))
merge_adjacent_files_ducklake(CATALOG, "events", min_file_size=1, max_file_size=10_000_000)
n_after = len(DuckLakeCatalog(CATALOG).list_files("events"))
print(f"events: {n_before} files -> {n_after} files")

future = datetime.now(timezone.utc) + timedelta(days=1)
removed = cleanup_old_files_ducklake(CATALOG, older_than=future)
print(f"Cleaned up {len(removed)} retired data files")


# ====================================================================
# 13. Maintenance
# ====================================================================
print("\n" + "=" * 60)
print("13. Maintenance")
print("=" * 60)

from ducklake_pandas import expire_snapshots, vacuum_ducklake

catalog = DuckLakeCatalog(CATALOG)
print(f"Before: {len(catalog.snapshots())} snapshots")

expired = expire_snapshots(CATALOG, keep_last_n=3)
print(f"Expired {expired} snapshots")

vacuumed = vacuum_ducklake(CATALOG)
print(f"Vacuumed {vacuumed} orphan files")

catalog = DuckLakeCatalog(CATALOG)
print(f"After:  {len(catalog.snapshots())} snapshots")


# ====================================================================
# 14. Final roundtrip verification
# ====================================================================
print("\n" + "=" * 60)
print("14. Final roundtrip verification")
print("=" * 60)

print("=== ducklake_pandas ===")
for tbl in ["users", "events", "tiny", "region_summary"]:
    n = len(read_ducklake(CATALOG, tbl))
    print(f"  {tbl:20s} → {n} rows")



# ====================================================================
# Cleanup
# ====================================================================
shutil.rmtree(WORKDIR)
print(f"\nCleaned up {WORKDIR}")
print("\n✅  ALL DONE — Pandas tutorial complete!")
