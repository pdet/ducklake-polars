# DuckLake Compatibility Gap Analysis

**ducklake-dataframe / ducklake-pandas vs. DuckDB's native ducklake extension**

_Generated: 2026-03-01 · DuckLake spec version: 0.3_

---

## Feature Matrix

### DDL — Data Definition Language

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| CREATE TABLE | ✅ Full SQL syntax | ✅ `create_ducklake_table()` | ✅ `create_ducklake_table()` | ✅ Full |
| CREATE TABLE AS SELECT | ✅ `CREATE TABLE ... AS` | ✅ `create_table_as_ducklake()` | ✅ `create_table_as_ducklake()` | ✅ Full |
| DROP TABLE | ✅ `DROP TABLE` | ✅ `drop_ducklake_table()` | ✅ `drop_ducklake_table()` | ✅ Full |
| RENAME TABLE | ✅ `ALTER TABLE ... RENAME TO` | ✅ `rename_ducklake_table()` | ✅ `rename_ducklake_table()` | ✅ Full |
| ALTER TABLE ADD COLUMN | ✅ Full (with defaults) | ✅ `alter_ducklake_add_column()` | ✅ `alter_ducklake_add_column()` | ✅ Full |
| ALTER TABLE DROP COLUMN | ✅ Full | ✅ `alter_ducklake_drop_column()` | ✅ `alter_ducklake_drop_column()` | ✅ Full |
| ALTER TABLE RENAME COLUMN | ✅ Full | ✅ `alter_ducklake_rename_column()` | ✅ `alter_ducklake_rename_column()` | ✅ Full |
| ALTER TABLE SET PARTITIONED BY | ✅ Identity transforms | ✅ `alter_ducklake_set_partitioned_by()` | ✅ `alter_ducklake_set_partitioned_by()` | ✅ Full |
| ALTER TABLE SET TYPE (column type change) | ✅ Full | ✅ `alter_ducklake_set_type()` | ✅ `alter_ducklake_set_type()` | ✅ Full |
| CREATE SCHEMA | ✅ Full | ✅ `create_ducklake_schema()` | ✅ `create_ducklake_schema()` | ✅ Full |
| DROP SCHEMA | ✅ Full (with CASCADE) | ✅ `drop_ducklake_schema()` | ✅ `drop_ducklake_schema()` | ✅ Full |
| CREATE VIEW | ✅ Full SQL | ✅ `create_ducklake_view()` | ✅ `create_ducklake_view()` | ✅ Full |
| DROP VIEW | ✅ Full | ✅ `drop_ducklake_view()` | ✅ `drop_ducklake_view()` | ✅ Full |
| CREATE VIEW OR REPLACE | ✅ Full | ✅ `or_replace=True` | ✅ `or_replace=True` | ✅ Full |
| Tags (table/column metadata) | ✅ `ducklake_tag`, `ducklake_column_tag` | ✅ `set_ducklake_table_tag()` / `set_ducklake_column_tag()` | ✅ `set_ducklake_table_tag()` / `set_ducklake_column_tag()` | ✅ Full |

### DML — Data Manipulation Language

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| INSERT | ✅ Full SQL | ✅ `write_ducklake()` | ✅ `write_ducklake()` | ✅ Full |
| DELETE (predicate) | ✅ `DELETE FROM ... WHERE` | ✅ `delete_ducklake()` (pl.Expr) | ✅ `delete_ducklake()` (callable) | ✅ Full |
| UPDATE (predicate) | ✅ `UPDATE ... SET ... WHERE` | ✅ `update_ducklake()` | ✅ `update_ducklake()` | ✅ Full |
| MERGE / UPSERT | ✅ Full SQL MERGE | ✅ `merge_ducklake()` | ✅ `merge_ducklake()` | ✅ Full |
| SELECT (query) | ✅ Full SQL | ✅ `scan_ducklake()` / `read_ducklake()` | ✅ `read_ducklake()` | ✅ Full |
| Lazy evaluation | ✅ DuckDB lazy execution | ✅ Polars LazyFrame | ❌ Eager only | ⚠️ Partial |

### Write Modes

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Error (fail if exists) | ✅ `CREATE TABLE` | ✅ `mode="error"` | ✅ `mode="error"` | ✅ Full |
| Append | ✅ `INSERT INTO` | ✅ `mode="append"` | ✅ `mode="append"` | ✅ Full |
| Overwrite | ✅ Delete all + insert | ✅ `mode="overwrite"` | ✅ `mode="overwrite"` | ✅ Full |

### Partitioning

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Identity transforms | ✅ Full | ✅ Hive-style directories | ✅ Hive-style directories | ✅ Full |
| Partition pruning (read) | ✅ Full | ✅ Via partition values in stats | ✅ Via predicate-based partition pruning | ✅ Full |
| Partitioned writes | ✅ Full | ✅ One Parquet file per partition | ✅ One Parquet file per partition | ✅ Full |
| Non-identity transforms (year, month, bucket, truncate) | ❌ Not in spec | ❌ Not implemented | ❌ Not implemented | N/A |

### Schema Evolution

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Add column | ✅ Full | ✅ `missing_columns="insert"` fills NULLs | ✅ NULL fill for missing columns | ✅ Full |
| Drop column | ✅ Full | ✅ `extra_columns="ignore"` | ✅ Skip columns not in schema | ✅ Full |
| Rename column | ✅ Full (field_id mapping) | ✅ Column history-based rename | ✅ Column history-based rename | ✅ Full |
| Type change | ✅ Full | ✅ `alter_ducklake_set_type()` | ✅ `alter_ducklake_set_type()` | ✅ Full |
| Reading across schema versions | ✅ Transparent | ✅ Groups files by rename map | ✅ Per-file column matching | ✅ Full |
| Struct field add/drop/rename | ✅ Full | ✅ Struct field history tracking | ✅ Struct field history tracking | ✅ Full |

### Time Travel

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Snapshot version-based | ✅ `AT (VERSION => n)` | ✅ `snapshot_version=n` | ✅ `snapshot_version=n` | ✅ Full |
| Timestamp-based | ✅ `AT (TIMESTAMP => ts)` | ✅ `snapshot_time=ts` | ✅ `snapshot_time=ts` | ✅ Full |
| Snapshot listing | ✅ `FROM snapshots()` | ✅ `DuckLakeCatalog.snapshots()` | ✅ `DuckLakeCatalog.snapshots()` | ✅ Full |

### Data Inlining

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Inline small inserts | ✅ Automatic (configurable threshold) | ✅ `data_inlining_row_limit` | ✅ `data_inlining_row_limit` | ✅ Full |
| Read inlined data | ✅ Transparent | ✅ Transparent | ✅ Transparent | ✅ Full |
| Delete inlined rows | ✅ Full | ✅ Sets `end_snapshot` on rows | ✅ Sets `end_snapshot` on rows | ✅ Full |
| Promotion (inline → Parquet) | ✅ Automatic on threshold | ⚠️ Manual (overwrite flushes) | ⚠️ Manual (overwrite flushes) | ⚠️ Partial |

### Statistics

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| File-level column stats (min/max/null_count) | ✅ Full | ✅ Written on insert | ✅ Written on insert | ✅ Full |
| Table-level column stats | ✅ Full | ✅ Written on insert | ✅ Written on insert | ✅ Full |
| Table-level row stats | ✅ Full | ✅ `record_count`, `file_size_bytes` | ✅ `record_count`, `file_size_bytes` | ✅ Full |
| contains_nan tracking | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| File pruning via stats | ✅ Full | ✅ Via Polars `_table_statistics` | ✅ Via predicate-based stats pruning | ✅ Full |

### Predicate Pushdown

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Filter pushdown to Parquet scan | ✅ Full (DuckDB optimizer) | ✅ Full (Polars optimizer) | ❌ No lazy evaluation | ⚠️ Partial |
| Projection pushdown | ✅ Full | ✅ Full (Polars optimizer) | ✅ Via `columns=` parameter | ✅ Full |

### Delete Files

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Iceberg position-delete format | ✅ Full | ✅ `_deletion_files` in scan_parquet | ✅ Manual position filtering | ✅ Full |
| Cumulative delete files | ✅ Full | ✅ On overwrite (cumulative rewrite) | ✅ On overwrite (cumulative rewrite) | ✅ Full |

### Sort Keys

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Sort keys (SET/RESET SORTED BY) | ✅ ASC/DESC, NULLS_FIRST/LAST | ✅ `alter_ducklake_set_sort_keys()` / `reset` | ✅ `alter_ducklake_set_sort_keys()` / `reset` | ✅ Full |

### Encryption

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Parquet file encryption | ✅ Via `encryption_key` in metadata | ❌ Not implemented | ❌ Not implemented | ❌ Missing |
| Encryption key management | ✅ Per-file encryption_key column | ❌ Not implemented | ❌ Not implemented | ❌ Missing |

### Column Mapping

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| field_id-based mapping | ✅ Full (Parquet field_ids) | ⚠️ Read: uses history; Write: map_by_name | ⚠️ Read: field_id + name fallback | ⚠️ Partial |
| map_by_name mapping | ✅ Full | ✅ Registered on every write | ✅ Registered on every write | ✅ Full |
| ducklake_column_mapping table | ✅ Full | ✅ Written on insert | ✅ Written on insert | ✅ Full |
| ducklake_name_mapping table | ✅ Full | ✅ Written on insert | ✅ Written on insert | ✅ Full |

### Data Types

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Integer types (INT8–INT64, UINT8–UINT64) | ✅ | ✅ | ✅ | ✅ Full |
| Float types (FLOAT, DOUBLE) | ✅ | ✅ | ✅ | ✅ Full |
| BOOLEAN | ✅ | ✅ | ✅ | ✅ Full |
| VARCHAR / STRING | ✅ | ✅ | ✅ | ✅ Full |
| BLOB / BINARY | ✅ | ✅ | ✅ | ✅ Full |
| DATE | ✅ | ✅ | ✅ | ✅ Full |
| TIME / TIMETZ / TIME_NS | ✅ | ✅ | ✅ | ✅ Full |
| TIMESTAMP (all precisions) | ✅ | ✅ | ✅ | ✅ Full |
| TIMESTAMPTZ | ✅ | ✅ | ✅ | ✅ Full |
| DECIMAL(p,s) | ✅ | ✅ | ✅ | ✅ Full |
| UUID | ✅ | ✅ (Binary) | ✅ (Binary) | ✅ Full |
| JSON | ✅ | ✅ (String) | ✅ (String) | ✅ Full |
| HUGEINT / UHUGEINT | ✅ | ⚠️ Int128 (DuckDB writes as Float64 in Parquet) | ⚠️ large_binary | ⚠️ Partial |
| INTERVAL | ✅ | ⚠️ Duration("us") (Polars limitation) | ⚠️ Duration("us") | ⚠️ Partial |
| LIST(T) | ✅ | ✅ Recursive | ✅ Recursive | ✅ Full |
| STRUCT(...) | ✅ | ✅ Recursive | ✅ Recursive | ✅ Full |
| MAP(K,V) | ✅ | ⚠️ List(Struct(key,value)) — Polars reader issue | ✅ via Arrow map_ | ⚠️ Partial |
| UNION | ✅ | ❌ Not implemented | ❌ Not implemented | ❌ Missing |
| ENUM | ✅ | ✅ (→ VARCHAR) | ✅ (→ VARCHAR) | ✅ Full |
| GEOMETRY | ✅ | ✅ (Binary) | ✅ (Binary) | ✅ Full |
| VARIANT | ✅ | ✅ (String) | ✅ (String) | ✅ Full |
| BIT | ✅ | ✅ (String) | ✅ (String) | ✅ Full |

### Metadata Backends

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| DuckDB (native .ducklake) | ✅ Primary | ✅ Read via sqlite3 (same format) | ✅ Read via sqlite3 | ✅ Full |
| SQLite | ✅ Via sqlite extension | ✅ Via Python sqlite3 | ✅ Via Python sqlite3 | ✅ Full |
| PostgreSQL | ✅ Via postgres extension | ✅ Via psycopg2 | ✅ Via psycopg2 | ✅ Full |
| MySQL | ✅ Via mysql extension | ❌ Not implemented | ❌ Not implemented | ❌ Missing |

### Catalog API

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| `ducklake_snapshots()` | ✅ | ✅ `DuckLakeCatalog.snapshots()` | ✅ `DuckLakeCatalog.snapshots()` | ✅ Full |
| `ducklake_current_snapshot()` | ✅ | ✅ `.current_snapshot()` | ✅ `.current_snapshot()` | ✅ Full |
| `ducklake_table_info()` | ✅ | ✅ `.table_info()` | ✅ `.table_info()` | ✅ Full |
| `ducklake_list_files()` | ✅ | ✅ `.list_files()` | ✅ `.list_files()` | ✅ Full |
| `ducklake_table_insertions()` | ✅ | ✅ `.table_insertions()` | ✅ `.table_insertions()` | ✅ Full |
| `ducklake_table_deletions()` | ✅ | ✅ `.table_deletions()` | ✅ `.table_deletions()` | ✅ Full |
| `ducklake_table_changes()` | ✅ | ✅ `.table_changes()` | ✅ `.table_changes()` | ✅ Full |
| `ducklake_options()` | ✅ | ✅ `.options()` | ✅ `.options()` | ✅ Full |
| `ducklake_settings()` | ✅ | ✅ `.settings()` | ✅ `.settings()` | ✅ Full |
| Schema listing | ✅ `information_schema` | ✅ `.list_schemas()` | ✅ `.list_schemas()` | ✅ Full |
| Table listing | ✅ `information_schema` | ✅ `.list_tables()` | ✅ `.list_tables()` | ✅ Full |

### Maintenance

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| EXPIRE SNAPSHOTS | ✅ `CALL ducklake_expire_snapshots()` | ✅ `expire_snapshots()` | ✅ `expire_snapshots()` | ✅ Full |
| VACUUM | ✅ `CALL ducklake_vacuum()` | ✅ `vacuum_ducklake()` | ✅ `vacuum_ducklake()` | ✅ Full |
| Scheduled file deletion tracking | ✅ `ducklake_files_scheduled_for_deletion` | ✅ Reads/writes the table | ✅ Reads/writes the table | ✅ Full |

### Interoperability

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Write with DuckDB → read with us | ✅ N/A | ✅ Full (tested extensively) | ✅ Full (tested extensively) | ✅ Full |
| Write with us → read with DuckDB | ✅ N/A | ✅ Full (map_by_name + field_id) | ✅ Full (map_by_name + field_id) | ✅ Full |
| Mixed operations (both tools) | ✅ N/A | ✅ Snapshot-consistent | ✅ Snapshot-consistent | ✅ Full |

### Other

| Feature | DuckDB ducklake | ducklake-dataframe | ducklake-pandas | Status |
|---|---|---|---|---|
| Author / commit message metadata | ✅ `ducklake_snapshot_changes` | ✅ `author=`, `commit_message=` | ✅ `author=`, `commit_message=` | ✅ Full |
| Concurrent writes (transactions) | ✅ DuckDB transaction isolation | ❌ No conflict detection | ❌ No conflict detection | ❌ Missing |
| Object storage (S3, GCS, Azure) | ✅ Via DuckDB httpfs/s3 | ❌ Local filesystem only | ❌ Local filesystem only | ❌ Missing |
| Schema version tracking | ✅ `ducklake_schema_versions` | ✅ Read and written | ✅ Read and written | ✅ Full |

---

## Recently Completed

The following gaps have been resolved since the initial analysis:

- ✅ **ENUM type support** — Mapped to VARCHAR on both Polars and Pandas. Roundtrip-tested with DuckDB interop.
- ✅ **Tags (table/column metadata)** — Full read/write support via `set_ducklake_table_tag()`, `set_ducklake_column_tag()`, `delete_ducklake_table_tag()`, `delete_ducklake_column_tag()` on both Polars and Pandas. Exposed via `DuckLakeCatalog.table_tags()` / `.column_tags()`.
- ✅ **Sort Keys** — `alter_ducklake_set_sort_keys()` implemented. Data is sorted before writing Parquet. Metadata stored and read correctly.
- ✅ **Pandas partition pruning** — Predicate-based partition pruning via `_can_skip_file_by_partition()`. Files whose partition values don't match are skipped.
- ✅ **Pandas stats pruning** — Predicate-based file pruning via `_can_skip_file_by_stats()`. Files whose min/max stats exclude the predicate range are skipped.
- ✅ **ALTER TABLE SET TYPE** — `alter_ducklake_set_type()` implemented on both Polars and Pandas. Column type changes tracked in schema history.

---

## Priority Gaps

### 1. 🔴 Object Storage Support (S3, GCS, Azure) — High Impact

DuckDB's ducklake extension can read/write Parquet files from S3, GCS, and Azure Blob Storage via its httpfs extension. Our implementation is strictly local filesystem. This is the single biggest barrier to production adoption since most real-world data lakes live on object storage.

**Impact:** Blocks use for any cloud-based data lake. This is table stakes for production usage.

### 2. 🟠 Concurrent Write Safety — Medium-High Impact

DuckDB handles concurrent writes through its own transaction isolation. Our writer has no conflict detection — two concurrent `write_ducklake()` calls to the same table can corrupt metadata. At minimum, we need optimistic concurrency control (check snapshot version before commit).

**Impact:** Multi-process or multi-user setups are unsafe. Single-writer scenarios are fine.

### 3. 🟡 UNION Type — Low Impact

DuckDB supports the UNION type, which is not mapped in our type system. This is rare in practice.

**Impact:** Niche. UNION types are uncommon in real schemas.

---

## Recommended Roadmap

### Phase 1 — Production Readiness (Immediate)

1. **Object storage support** — Add fsspec or pyarrow.fs-based file I/O abstraction. Support S3 (via s3fs), GCS (via gcsfs), and Azure (via adlfs). This unblocks all cloud deployments.

2. **Optimistic concurrency control** — Before committing a new snapshot, verify the latest snapshot_id hasn't changed since we started. Retry or raise on conflict. Simple but critical.

### Phase 2 — Completeness (Next)

3. **MySQL backend** — Add a MySQL backend adapter alongside SQLite and PostgreSQL. The DuckLake schema is standard SQL; should be straightforward.

4. **UNION type support** — Map UNION to a struct-of-optionals or string fallback.

5. **Encryption support** — Read the `encryption_key` from metadata and use it to decrypt Parquet files. Write encrypted files and store the key. Depends on pyarrow encryption support.

6. **Automatic inline promotion** — When inlined row count exceeds threshold on insert, automatically promote to Parquet (currently only happens on overwrite).

7. **Concurrent write transactions** — Full MVCC-style conflict detection with retry logic, beyond simple optimistic checks.
