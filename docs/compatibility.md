# Compatibility

This page tracks what is **missing** from each ducklake-dataframe wrapper (Polars / Pandas / PySpark) relative to DuckDB's native DuckLake extension — the reference implementation of the [DuckLake spec](https://ducklake.select/).

The wrappers all delegate to a shared `ducklake_core` engine, so the gaps below are mostly **engine-wide** (apply to all three). A few are **wrapper-specific** — those are flagged.

> **Legend:**
> ✅ Supported · ⚠️ Partial · ❌ Not supported (in this wrapper) · 🔵 Polars-only · 🐼 Pandas-only · ⚡ PySpark-only

## Read path

| Capability | DuckDB-DuckLake | Polars | Pandas | PySpark |
|---|---|---|---|---|
| Basic SELECT | ✅ | ✅ | ✅ | ✅ |
| Lazy / pushdown reads | ✅ | ✅ (`scan_ducklake`) | ❌ — eager only | ✅ (Spark DataFrames are lazy) |
| Predicate pushdown | ✅ | ✅ (Polars optimizer) | ⚠️ via `predicate=` callable on file pruning only | ✅ (Spark's native Parquet pushdown) |
| Projection pushdown | ✅ | ✅ | ⚠️ via `columns=` only | ✅ |
| Time travel by snapshot id | ✅ | ✅ | ✅ | ✅ |
| Time travel by timestamp | ✅ | ✅ | ✅ | ✅ |
| Schema-evolved reads (add/drop/rename/set type) | ✅ | ✅ | ✅ | ✅ |
| Field-id / name-mapping reads | ✅ | ✅ | ✅ | ✅ |
| Inlined data | ✅ | ✅ | ✅ | ✅ |
| Partition pruning | ✅ | ✅ | ✅ | ✅ |
| Min/max stats pruning | ✅ | ✅ | ✅ | ✅ |
| Positional delete files (Iceberg-compatible) | ✅ | ✅ | ✅ | ✅ |
| **Encrypted Parquet** | ✅ | ❌ | ❌ | ❌ |

## Write & DML

| Capability | DuckDB-DuckLake | Polars | Pandas | PySpark |
|---|---|---|---|---|
| INSERT (append / overwrite / error) | ✅ | ✅ | ✅ | ✅ |
| DELETE (predicate) | ✅ | ✅ (`pl.Expr`) | ✅ (callable) | ✅ (SQL string) |
| UPDATE | ✅ | ✅ | ✅ | ✅ |
| MERGE / upsert | ✅ | ✅ | ✅ | ✅ |
| `WHEN NOT MATCHED BY SOURCE` clauses | ✅ | ✅ | ✅ | ✅ |
| CREATE TABLE AS | ✅ | ✅ | ✅ | ✅ |
| ADD FILES (register existing Parquet) | ✅ | ✅ | ✅ | ✅ |
| Partitioned writes (identity) | ✅ | ✅ | ✅ | ✅ |
| Partition transforms `year`/`month`/`day`/`hour` | ✅ | ✅ | ✅ | ✅ |
| Partition transforms `bucket(N, col)` | ✅ | ❌ | ❌ | ❌ |
| Partition transforms `truncate(N, col)` | ✅ | ❌ | ❌ | ❌ |
| Sort keys (column lists, ASC/DESC, NULLS FIRST/LAST) | ✅ | ✅ | ✅ | ✅ |
| **Expression sort keys** (arbitrary SQL exprs) | ✅ | ❌ | ❌ | ❌ |
| Schema evolution on write (`schema_evolution="merge"`) | ✅ | ✅ | ✅ | ✅ |
| Data inlining of small writes | ✅ | ✅ | ✅ | ✅ |
| **Auto-flush of inlined → Parquet at threshold** | ✅ (`ducklake_flush_inlined_data`, runs in `CHECKPOINT`) | ❌ — only flushed on overwrite | ❌ | ❌ |
| Streaming / micro-batch writer | n/a | 🔵 `DuckLakeStreamWriter` | 🐼 `DuckLakeStreamWriter` | ❌ |

## DDL

| Capability | DuckDB-DuckLake | Polars | Pandas | PySpark |
|---|---|---|---|---|
| CREATE / DROP SCHEMA (with `cascade`) | ✅ | ✅ | ✅ | ✅ |
| CREATE / DROP / RENAME TABLE | ✅ | ✅ | ✅ | ✅ |
| CREATE / DROP VIEW (`OR REPLACE`) | ✅ | ✅ | ✅ | ✅ |
| ALTER ADD / DROP / RENAME COLUMN | ✅ | ✅ | ✅ | ✅ |
| ALTER SET TYPE (column promotion) | ✅ | ✅ | ✅ | ✅ |
| **Column DEFAULT** (literal) | ✅ | ✅ (`alter_*_add_column(default=)`) | ✅ | ✅ |
| **Expression column defaults** | ✅ | ❌ | ❌ | ❌ |
| **Nested struct field add/drop/rename** | ✅ | ⚠️ — top-level columns only at the API; nested reads work | ⚠️ | ⚠️ |
| SET / RESET PARTITIONED BY | ✅ | ✅ | ✅ | ✅ |
| SET / RESET SORTED BY | ✅ | ✅ | ✅ | ✅ |
| **NOT NULL constraint** | ✅ | ❌ | ❌ | ❌ |
| **PRIMARY KEY / FOREIGN KEY / UNIQUE / CHECK** | ❌ (DuckLake itself doesn't support these) | ❌ | ❌ | ❌ |
| Tags / `COMMENT ON` (table + column) | ✅ | ✅ | ✅ | ✅ |
| **Custom column tag keys** (v1.0+) | ✅ | ✅ | ✅ | ✅ |

## Maintenance

| Capability | DuckDB-DuckLake | Polars | Pandas | PySpark |
|---|---|---|---|---|
| `expire_snapshots` (by id or `keep_last_n`) | ✅ | ✅ | ✅ | ✅ |
| `vacuum` / orphan cleanup | ✅ | ✅ (`vacuum_ducklake`) | ✅ | ✅ |
| `rewrite_data_files` (full compaction + apply deletes) | ✅ | ✅ | ✅ | ✅ |
| `merge_adjacent_files` (lightweight, v1.0+) | ✅ | ✅ | ✅ | ❌ |
| `cleanup_old_files` (drain delete queue) | ✅ | ✅ | ✅ | ❌ |
| `delete_orphaned_files` (filesystem scan) | ✅ | ✅ | ✅ | ❌ |
| **`flush_inlined_data`** | ✅ | ❌ | ❌ | ❌ |
| **Bundled `CHECKPOINT` maintenance** (flush → expire → merge → rewrite → cleanup → orphans) | ✅ | ❌ — call individually | ❌ | ❌ |
| `set_option` (per-table / column / global) | ✅ | ✅ | ✅ | ❌ |
| Auto-collected file / row-group statistics | ✅ | ✅ | ✅ | ✅ |
| **GEOMETRY bounding-box stats** | ✅ | ❌ — GEOMETRY columns map to `Binary` | ❌ | ❌ |

## Catalog inspection

| Capability | DuckDB-DuckLake | Polars | Pandas | PySpark |
|---|---|---|---|---|
| `snapshots()` / `current_snapshot()` | ✅ | ✅ | ✅ | ✅ (top-level helpers only) |
| `table_info` (file count / size / delete files) | ✅ | ✅ | ✅ | ✅ |
| `list_files` | ✅ | ✅ | ✅ | ❌ — no top-level helper |
| `list_schemas` / `list_tables` / `list_views` | ✅ | ✅ | ✅ | ✅ |
| Tags / sort-key inspection | ✅ | ✅ | ✅ | ⚠️ — no class API |
| Change data feed (`table_insertions`/`table_deletions`/`table_changes`) | ✅ | ✅ | ✅ | ⚠️ — only `read_ducklake_changes()`, no class methods |
| `DuckLakeCatalog` class | n/a (DuckDB uses SQL) | ✅ | ✅ | ❌ |

## Advanced features

| Capability | DuckDB-DuckLake | Polars | Pandas | PySpark |
|---|---|---|---|---|
| Macros (scalar + table, time-travel-aware) | ✅ | ⚠️ create/drop only — cannot evaluate macros without DuckDB at read time | ⚠️ | ❌ |
| Multi-statement transactions / `BEGIN; ... COMMIT;` (incl. DDL) | ✅ | ❌ — every function is its own snapshot | ❌ | ❌ |
| Multi-table commits in one snapshot | ✅ | ❌ | ❌ | ❌ |
| **Iceberg-compatible deletion vectors (Puffin)** | ✅ | ❌ — only Iceberg positional delete files are read/written | ❌ | ❌ |
| **`iceberg_to_ducklake()`** (Iceberg → DuckLake metadata copy) | ✅ (via Iceberg extension) | ❌ | ❌ | ❌ |
| **GEOMETRY type** | ✅ (with bbox stats and nested geometry) | ⚠️ schema-mapped to `Binary`, no spatial ops | ⚠️ | ⚠️ |
| **VARIANT type** | ✅ (binary-encoded with shredding) | ⚠️ schema-mapped to `String`; binary interop with DuckDB **not** supported | ⚠️ | ⚠️ |
| **UNION type** | ✅ | ❌ raises `UnsupportedUnionTypeError` (Polars; pandas/pyspark error similarly) | ❌ | ❌ |
| **HUGEINT / UHUGEINT precision** | ✅ (128-bit) | ⚠️ DuckDB writes as Float64 in Parquet — precision loss for large values | ⚠️ | ⚠️ |
| **INTERVAL reads** | ✅ | ❌ — Polars Parquet reader cannot decode `month_day_millisecond_interval` | ✅ via Pandas/PyArrow | ⚠️ — Spark Parquet support varies |
| **MAP type** | ✅ native `MAP(K, V)` | ⚠️ Polars reads as `List(Struct(key, value))` | ✅ | ⚠️ |
| **Encrypted catalogs / Parquet** | ✅ | ❌ explicit error on encrypted catalogs | ❌ | ❌ |

## Concurrency

| Capability | DuckDB-DuckLake | Polars | Pandas | PySpark |
|---|---|---|---|---|
| Snapshot isolation / OCC at commit | ✅ | ✅ | ✅ | ✅ |
| Configurable retry on conflict | ✅ | ✅ (`max_retries=`, `retry_wait_ms=`, `retry_backoff=`) | ✅ | ✅ |
| Cross-table transactions in one commit | ✅ | ❌ | ❌ | ❌ |
| Distributed lock manager | ❌ (none) | ❌ | ❌ | ❌ |
| Single-writer SQLite (filesystem locking + WAL) | ✅ | ✅ | ✅ | ✅ |
| Multi-writer PostgreSQL (MVCC) | ✅ | ✅ | ✅ | ✅ |

## Catalog backends

| Backend | DuckDB-DuckLake | Polars | Pandas | PySpark |
|---|---|---|---|---|
| SQLite | ✅ | ✅ | ✅ | ✅ |
| PostgreSQL | ✅ | ✅ | ✅ | ✅ |
| DuckDB-format (read-only via `sqlite3`) | ✅ | ✅ | ✅ | ✅ |
| **MySQL** | ✅ | ❌ | ❌ | ❌ |

## Object storage

Identical across all three wrappers — backed by [fsspec](https://filesystem-spec.readthedocs.io/) via the `[s3]` / `[gcs]` / `[azure]` extras. Local filesystem is the most heavily tested path.

| Backend | DuckDB-DuckLake | ducklake-dataframe (all wrappers) |
|---|---|---|
| Local filesystem | ✅ | ✅ |
| S3 / S3-compatible | ✅ | ✅ |
| Google Cloud Storage | ✅ | ✅ |
| Azure Blob Storage | ✅ | ✅ |

## Wrapper-specific gaps at a glance

### Polars (`ducklake_polars`)

Closest to feature parity. Notable gaps versus DuckDB-DuckLake:

- No encrypted catalogs / encrypted Parquet
- No `bucket()` / `truncate()` partition transforms
- No expression sort keys / expression column defaults
- No `flush_inlined_data` and no bundled `CHECKPOINT` maintenance
- No multi-statement / multi-table transactions (one function call = one snapshot)
- MAP and INTERVAL Parquet reads blocked by upstream Polars limitations
- No GEOMETRY/VARIANT spatial/binary semantics — types are schema-mapped to `Binary` / `String`
- Macros: create/drop only; cannot **evaluate** a macro without a DuckDB runtime

### Pandas (`ducklake_pandas`)

Same engine-wide gaps as Polars, plus:

- **No lazy reads** — there is no `scan_ducklake`; reads are always eager
- Predicate pushdown is limited to file pruning via the `predicate=` callable; no row-level pushdown into Parquet
- DML predicates are callables, not declarative expressions

### PySpark (`ducklake_pyspark`)

Same engine-wide gaps as Polars, plus the largest **wrapper-level** gap surface:

- No `DuckLakeCatalog` class — only the top-level helpers (`list_schemas`, `list_tables`, `list_views`, `list_snapshots`, `snapshot_changes`, `catalog_info`, `get_view`, `table_info`)
- No `list_files` helper
- No `DuckLakeStreamWriter`
- No `merge_adjacent_files_ducklake`
- No `cleanup_old_files_ducklake` / `delete_orphaned_files_ducklake`
- No `create_ducklake_macro` / `drop_ducklake_macro`
- No `set_ducklake_option`
- DML predicates are SQL strings rather than expressions
- Change data feed uses `start_snapshot=` / `end_snapshot=` instead of `start_version=` / `end_version=`

## Where the gaps come from

- **Engine-wide gaps** (encryption, MySQL, expression sort/defaults, bucket/truncate transforms, multi-statement transactions, `flush_inlined_data`, `CHECKPOINT`) live in `ducklake_core` and need work in the writer / catalog layer to land. Adding them generally lights the feature up for all three wrappers at once.
- **Type-mapping gaps** (UNION, GEOMETRY, VARIANT semantics, HUGEINT precision, MAP/INTERVAL on Polars) are constrained either by the spec/DuckDB encoding (HUGEINT-as-Float64, VARIANT binary format) or by the engine's Parquet reader (Polars MAP/INTERVAL).
- **Wrapper gaps** (Pandas lazy reads, PySpark catalog class & maintenance helpers) are wrapper-only and could be filled without touching `ducklake_core`.

## Tracking

This page is hand-maintained against `src/ducklake_*/__init__.py` and the DuckLake spec. If you add a feature to `ducklake_core`, update the relevant rows here and remove the corresponding row from "Known limitations" in `README.md`.
