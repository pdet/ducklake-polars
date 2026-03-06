# Benchmarks

Performance benchmarks for ducklake-dataframe. Self-tracking benchmarks detect regressions; comparison benchmarks measure DuckLake vs PyIceberg on the same workloads.

## Benchmark Suite

### `bench_read_write.py` — Polars vs Arrow vs Pandas

Measures ducklake-dataframe performance across output formats (Polars, Arrow, Pandas) for self-tracking regression detection.

```bash
python benchmarks/bench_read_write.py                          # Default: 100K rows, 10 cols
python benchmarks/bench_read_write.py --rows 1000000 --cols 20 # Custom
```

| Benchmark | What it tells you |
|-----------|-------------------|
| Write | Time to write N rows from each format into DuckLake |
| Read | Time to read N rows back into each format |
| Scan + Filter | Benefit of predicate pushdown (Polars lazy) vs post-filter (Arrow/Pandas) |
| Aggregation | Group-by performance across formats |
| Multi-file Read | Read latency scaling with file count (10, 50 files) |

### `bench_streaming.py` — DuckLake vs Iceberg: Streaming Ingestion

Simulates a streaming workload with many small appends, then measures read, filter, aggregation, compaction, and time travel.

```bash
python benchmarks/bench_streaming.py                             # Default: 100 batches × 1K rows
python benchmarks/bench_streaming.py --batches 200 --batch-size 5000
python benchmarks/bench_streaming.py --output results.json       # Save JSON
```

| Scenario | Description |
|----------|-------------|
| Streaming Append | N batches of M rows each |
| Read-After-Write | Append then immediately read back (per batch) |
| Scan + Filter | Predicate pushdown after streaming ingestion |
| Aggregation | Group-by aggregation over all ingested data |
| Compaction | Merge small files into larger ones |
| Time Travel | Read historical snapshot |

### `bench_schema_evolution.py` — DuckLake vs Iceberg: Schema Evolution

Measures DDL cost and read performance after many schema changes. Tests how well each system handles schema-heavy workloads.

```bash
python benchmarks/bench_schema_evolution.py                       # Default: 50 evolutions, 100K rows
python benchmarks/bench_schema_evolution.py --evolutions 100 --rows 500000
```

| Scenario | Description |
|----------|-------------|
| Add Column | N sequential ADD COLUMN operations |
| Read After Evolution | Read with N extra (NULL) columns |
| Rename Column | N sequential RENAME COLUMN operations |
| Read After Renames | Read with old Parquet files needing name reconciliation |
| Schema Churn | N cycles of ADD + DROP column |
| Wide Table Projection | Read 5 columns from a 200-column table |

### `bench_dml.py` — DuckLake vs Iceberg: DML Operations

Compares delete, update, and merge (upsert) performance. Shows how position-delete files affect read latency and how compaction recovers it.

```bash
python benchmarks/bench_dml.py                                    # Default: 100K rows, 20 delete rounds
python benchmarks/bench_dml.py --rows 500000 --delete-rounds 50
```

| Scenario | Description |
|----------|-------------|
| Selective Delete | Delete 10% of rows by predicate |
| Read After Deletes | Read performance with accumulated delete files |
| Bulk Update | Update 20% of rows (DuckLake: atomic; Iceberg: read-modify-write) |
| Upsert Merge | 50% existing + 50% new rows via merge/upsert |
| Delete Cascade | Many small deletes accumulating delete files |
| Read Degradation | Read before vs after compaction (DuckLake only) |

### `bench_backends.py` — Catalog Backend Comparison

Compares DuckLake performance across catalog backends: SQLite, DuckDB, and PostgreSQL (when available). Determines which backend to use for best performance.

```bash
python benchmarks/bench_backends.py                                    # Default: 100K rows, 50 appends (SQLite + DuckDB)
python benchmarks/bench_backends.py --rows 500000 --appends 100
python benchmarks/bench_backends.py --pg-dsn "postgresql://user:pass@localhost/testdb"  # Include PostgreSQL
```

| Scenario | Description |
|----------|-------------|
| Cold Start | Create catalog + table + insert + read (end-to-end) |
| Sequential Writes | N appends measuring catalog commit overhead |
| Read After Writes | Read performance with N accumulated snapshots |
| Schema Evolution | ADD + RENAME column DDL cost |
| Scan + Filter | Predicate pushdown with 5 data files |
| Snapshot History | List snapshots from catalog |
| Mixed Workload | Interleaved write + read cycles |

> **Result:** SQLite is 4–18× faster than DuckDB for catalog operations. All DuckLake vs Iceberg benchmarks use SQLite (the fastest backend).

### `bench_catalog.py` — DuckLake vs Iceberg: Catalog & Metadata

Pure metadata operations: catalog startup, table listing, snapshot history, time travel, and partition pruning.

```bash
python benchmarks/bench_catalog.py                                # Default: 100 snapshots, 50K rows
python benchmarks/bench_catalog.py --snapshots 200 --rows 100000
```

| Scenario | Description |
|----------|-------------|
| Cold Start | Create catalog + table + insert + read (end-to-end) |
| Snapshot History | List N snapshots after N writes |
| Multi-Table Catalog | Create 100 tables, list them |
| Partition Pruning | Scan 1 of 20 partitions with selective filter |
| Time Travel | Read at snapshot k out of N |

## Interpreting Results

- **Rows/s** (or **Ops/s** for metadata) is the key metric — higher is better
- Results include slowdown ratios (`Nx slower`) relative to the fastest system
- Run on the same hardware for meaningful comparisons across commits
- All benchmarks use temp directories and clean up after themselves
- Use `--output results.json` to save structured results for tracking over time

## Dependencies

All benchmarks require `polars`, `pyarrow`, and `duckdb`. The comparison benchmarks also require `pyiceberg`:

```bash
pip install ducklake-dataframe[polars] pyiceberg[sql-sqlite]
```
