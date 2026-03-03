# DuckLake DataFrame ↔ DuckDB Reference Test Parity Report

Generated: 2026-03-03

## Summary

| Metric | ducklake-ref (DuckDB) | ducklake-dataframe (Polars `tests/`) |
|---|---|---|
| Test files / modules | 345 `.test` files across 48 categories | 31 `.py` modules, ~895 test functions |
| Test categories | 48 directories | ~20 logical areas |

The ducklake-dataframe test suite is **extensive for the features it supports** (basic CRUD, schema evolution, partitioning, inlining, views, maintenance, merge, interop, stats/filters, types, time travel, concurrency, tags). However, there are significant **categories in ducklake-ref that have zero or minimal coverage** in ducklake-dataframe.

---

## Coverage Matrix

### ✅ Well-Covered (ducklake-dataframe has solid tests)

| ducklake-ref category | ref tests | DF coverage | Notes |
|---|---|---|---|
| **delete** | 11 | ✅ `test_delete.py` + `test_write_delete.py` (40+ tests) | Basic, multi-file, time travel, truncate, sequential, rollback |
| **update** | 7 | ✅ `test_update.py` + `test_write_update.py` (30+ tests) | Single/multi column, expressions, spanning files, time travel |
| **alter** | 25 | ✅ `test_write_alter.py` + `test_schema_evolution.py` (60+ tests) | Add/drop/rename column, type promotion, struct evolution, sort keys |
| **types** | 11 | ✅ `test_types.py` (50+ tests) | All integer/float/temporal/compound types, NaN, infinity, null bytes, variant |
| **data_inlining** | 31 | ✅ `test_inlined.py` + `test_write_inline.py` (40+ tests) | Basic, types, filter, delete, update, flush, threshold, time travel |
| **partitioning** | 10 | ✅ `test_partition.py` + `test_write_partition.py` (40+ tests) | Single/multi key, filter, year/month, delete, update, interop |
| **time_travel** | 2 | ✅ `test_time_travel.py` (15+ tests) | Version, timestamp, schema, struct/list, dropped tables |
| **view** | 8 | ✅ `test_write_views.py` (20+ tests) | Create, drop, replace, schema, interop, metadata |
| **merge** | 5 | ✅ `test_write_merge.py` (20+ tests) | Upsert, insert-only, update-only, composite key, interop |
| **table_changes** | 8 | ✅ `test_catalog_api.py` (insertions/deletions/changes tests) | Insertions, deletions, mixed changes, multi-snapshot ranges |
| **stats** | 11 | ✅ `test_stats_filter.py` (20+ tests) | Integer/date/varchar/decimal filter pushdown, sort+limit, null filters |
| **concurrent** | 4 | ✅ `test_concurrent.py` (5 tests) | Concurrent inserts, retry, conflict detection |
| **catalog** | 4 | ✅ `test_catalog_general.py` + `test_catalog_api.py` | Schemas, tables, drop/recreate, quoted identifiers |
| **functions** | 2 | ✅ `test_catalog_api.py` | `snapshots()`, `table_info()` covered via catalog API |
| **schema_evolution** | 1 | ✅ `test_schema_evolution.py` + `test_field_id.py` | Field IDs, renames, struct evolution |
| **insert** | 3 | ✅ `test_write.py` + `test_basic.py` | Insert modes, interop, multiple inserts, CTAS |

### ⚠️ Partially Covered

| ducklake-ref category | ref tests | DF coverage | Gaps |
|---|---|---|---|
| **compaction** | 27 | ⚠️ `test_write_maintenance.py` has expire_snapshots + vacuum | **No merge-adjacent / merge-rewrite tests**. Missing: compaction after alter table, compaction with partitioned tables, compaction size limits, multi-compaction, cleanup-after-compaction. ducklake-ref has 27 tests; DF has ~7 maintenance tests covering only expire + vacuum. |
| **deletion_inlining** | 15 | ⚠️ Partial via `test_write_inline.py` delete tests | DF tests inline deletion from polars side, but ducklake-ref has dedicated tests for: deletion inlining with alter, compaction, concurrency, encryption, large data, partitions, stats, transaction semantics. These edge cases are missing. |
| **sorted_table** | 26 | ⚠️ `test_write_alter.py` has ~10 sort-key tests | DF covers `set_sort_keys` / `reset_sort_keys` + basic validation. Missing: **sorted merge-adjacent** (the main use case — 13 tests in ref), flush-sorted inlining, sorted expressions, macro-based sort expressions, spatial hilbert sort. |
| **comments** | 5 | ⚠️ `test_tags.py` covers tags (comments) | DF maps DuckDB COMMENT → tags. But missing: comment on column directly, comment mixed with other operations in same transaction, comment schema versioning. |
| **general** | 13 | ⚠️ Partial | Missing: database_size, attach_at_snapshot, read_only mode, generated_columns (error), metadata_cache, prepared_statement, missing_parquet error handling, recursive_metadata_catalog. Only `data_path` and `paths` partially covered. |
| **default** | 4 | ⚠️ `test_schema_evolution.py` has add_column_with_default | Missing: default_values on CREATE TABLE, default_expressions (complex defaults), struct_field_default. |
| **transaction** | 12 | ⚠️ `test_concurrent.py` covers conflicts | Missing: transaction rollback semantics, transaction-local visibility, create conflicts, schema transactions, insert+update+delete in single transaction, conflict cleanup, inlining within transactions. |
| **constraints** | 3 | ⚠️ Minimal | NOT NULL is tested implicitly but no dedicated constraint tests. Missing: NOT NULL on create, NOT NULL with drop column, unsupported constraint error handling. |

### ❌ Not Covered (zero tests in ducklake-dataframe)

| ducklake-ref category | ref tests | What it tests | Priority |
|---|---|---|---|
| **add_files** | 31 | Adding external Parquet files to DuckLake (hive, nested, type checking, rollback, rename, compaction) | 🟡 Medium — Important for interop with external file ingestion workflows |
| **rewrite_data_files** | 10 | Rewriting data files after deletes to reclaim space (threshold, concurrency, partitioning, merge-adjacent, row ID preservation) | 🟡 Medium — Important for storage optimization |
| **encryption** | 2 | Encrypted parquet files (ENCRYPTED option), partitioning + encryption | 🟡 Medium — Security feature |
| **macros** | 10 | DuckLake-persisted scalar/table macros, macro transactions, time travel, multiple implementations, defined types | 🔴 High — Macros are a catalog feature; if DF exposes catalog, these should be tested |
| **rowid** | 2 | Virtual row ID tracking, row ID stability after updates | 🟡 Medium — Internal but useful for MERGE correctness |
| **virtualcolumns** | 2 | `file_row_number`, `snapshot_id` virtual columns | 🟡 Medium — Useful for auditing |
| **snapshot_info** | 2 | `ducklake_current_commit()`, `ducklake_last_commit()` functions | 🟢 Low — DuckDB-specific functions |
| **audit** | 1 | Author/commit message tracking on snapshots | ✅ Covered by `test_write_maintenance.py` author tests — **miscategorized here, actually covered** |
| **attach** | 2 | ATTACH REPLACE, different data paths | 🟢 Low — DuckDB SQL-specific |
| **autoloading** | 1 | Auto-loading data path from extension | 🟢 Low — DuckDB-specific |
| **checkpoint** | 4 | Checkpointing behavior, interleaved updates, views | 🟢 Low — DuckDB-internal |
| **cleanup** | 2 | Cleanup old files, create-drop cleanup | 🟡 Medium — Overlaps with vacuum but distinct |
| **cloud** | 1 | Cloud storage test cases (S3/MinIO) | 🟡 Medium — Cloud support |
| **geo** | 5 | GEOMETRY type support, spatial add_files, inlining, merge, nested | 🟡 Medium — Spatial extension feature |
| **initialize** | 2 | Creating new DuckLake, read_only mode initialization | 🟢 Low — DuckDB-specific |
| **issues** | 1 | Late materialization bug regression | 🟢 Low — Regression test |
| **list_files** | 1 | `ducklake_list_files()` function | ✅ Covered via `test_catalog_api.py` list_files tests |
| **metadata** | 5 | DuckDB tables metadata, settings for different backends (Postgres, SQLite, MinIO) | 🟢 Low — DuckDB metadata queries |
| **migration** | 4 | Schema migration from older DuckLake versions | 🟡 Medium — Version compatibility |
| **remove_orphans** | 2 | Remove orphaned files (mixed paths) | ✅ Covered by `test_write_maintenance.py` vacuum tests |
| **secrets** | 1 | DuckLake secrets/connection management | 🟢 Low — DuckDB-specific |
| **settings** | 5 | Parquet compression, row group size, per-table settings, per-thread output, max retry | 🟡 Medium — Configuration |

---

## Top Priority Gaps

### 1. **Compaction / Merge-Adjacent** (27 ref tests → ~0 DF tests)
The biggest gap. ducklake-ref has extensive tests for:
- Merging adjacent small files into larger ones
- Compaction with partitioned tables
- Compaction after ALTER TABLE
- Compaction size limits and max file counts
- Multi-table compaction
- Cleanup after compaction

**Recommendation:** Add `test_write_compaction.py` with merge-adjacent tests.

### 2. **Sorted Table / Sorted Merge** (26 ref tests → ~10 DF tests)
DF covers setting sort keys but not the **actual sorted merge behavior**:
- Merge-adjacent respecting sort order
- Sorted flush from inlined data
- Sort expressions (not just column names)
- Macro-based sort expressions
- Reset/rollback of sort configuration

**Recommendation:** Expand `test_write_alter.py` sort tests or add `test_sorted_table.py`.

### 3. **Deletion Inlining Edge Cases** (15 ref tests → ~3 DF tests)
DF tests basic inline deletion but misses:
- Deletion from inlined data across multiple snapshots
- Deletion inlining with concurrent operations
- Deletion inlining interaction with ALTER TABLE
- Deletion inlining with partitions
- Stats correctness after deletion inlining

**Recommendation:** Add dedicated deletion-inlining section to `test_write_inline.py`.

### 4. **Rewrite Data Files** (10 ref tests → 0 DF tests)
No DF tests for rewriting data files to remove deleted rows:
- Rewrite after repeated insert+delete cycles
- Rewrite threshold parameters
- Rewrite with concurrency
- Rewrite with partitioning
- Row ID preservation after rewrite

**Recommendation:** Add `test_write_rewrite.py` if the feature is supported.

### 5. **Transaction Semantics** (12 ref tests → ~5 DF tests)
DF tests concurrent inserts but misses:
- Transaction-local visibility (read your own writes before commit)
- Rollback semantics
- Multi-operation transactions (insert + update + delete)
- Create table conflict detection
- Schema-level transactions

**Recommendation:** Add `test_transactions.py` with proper transaction isolation tests.

### 6. **Macros** (10 ref tests → 0 DF tests)
If DuckLake catalog supports macros, this is a significant gap:
- Scalar and table macros persisted in catalog
- Macro transactions and time travel
- Multiple macro implementations
- Defined types

**Recommendation:** Add `test_macros.py` if macros are supported via the DataFrame API.

### 7. **Add Files (External File Ingestion)** (31 ref tests → 0 DF tests)
Adding external Parquet files is a core DuckLake feature:
- Adding files with schema validation
- Type checking (integer, float, decimal, timestamp, UUID)
- Hive-partitioned file addition
- Handling extra/missing columns

**Recommendation:** Add `test_add_files.py` if the feature is exposed.

---

## Features Intentionally Out of Scope

These ducklake-ref categories are **DuckDB SQL-specific** and don't need DataFrame API coverage:

- `attach` / `autoloading` / `initialize` — DuckDB ATTACH semantics
- `checkpoint` — DuckDB internal checkpointing
- `secrets` — DuckDB secret management
- `metadata` — DuckDB system table queries
- `migration` — Schema version migration (DuckDB handles this)
- `issues` / `clickbench` / `tpch` — Regression/benchmark tests

---

## Existing DF-Only Tests (not in ducklake-ref)

ducklake-dataframe has tests with **no direct ducklake-ref equivalent**:

| DF test module | What it covers |
|---|---|
| `test_backend.py` | Backend detection (SQLite vs PostgreSQL), URI parsing |
| `test_schema_mapping.py` | DuckDB type string → Polars type mapping |
| `test_field_id.py` | Parquet field ID tracking after renames (50+ tests) |
| `test_property.py` | Property-based / fuzz-style tests (Hypothesis-like) |
| `test_edge_cases.py` | Exhaustive edge cases (100+ tests): wide tables, special chars, Unicode, boundaries |
| `test_interop.py` | Bidirectional DuckDB ↔ Polars interop (150+ tests) |
| `test_tags.py` | Tag/property system (maps to DuckDB COMMENT) |
| `test_duckdb_backend.py` | DuckDB-specific backend API tests |

These are **strengths** of the DF test suite — the interop and edge case coverage goes well beyond what ducklake-ref tests.
