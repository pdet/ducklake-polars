# DuckLake Benchmarks

Performance comparison of ducklake-polars, ducklake-pandas, and raw DuckDB for read/write operations on DuckLake catalogs.

## Running

```bash
# Default: 1K, 10K, 100K rows, 3 runs each
python benchmarks/bench_read_write.py

# Custom sizes and runs
python benchmarks/bench_read_write.py --sizes 1000,10000,100000,1000000 --runs 5
```

## What's Measured

| Benchmark | Description |
|-----------|-------------|
| **Write** | Insert N rows into a new DuckLake table |
| **Read** | Read an entire table back into a DataFrame |
| **Arrow conversion** | `pl.DataFrame ↔ pa.Table` round-trip overhead |
| **Filter pushdown** | Full scan vs filtered scan (`id > N/2`) |

**Schema:** `(id INTEGER, name VARCHAR, value DOUBLE, category VARCHAR, ts TIMESTAMP WITH TIME ZONE)`

### Notes

- **raw-duckdb** includes DuckLake extension load time per run (realistic cold-start scenario)
- ducklake-polars/pandas read Parquet directly via PyArrow/Polars, skipping the DuckDB engine
- Median of N runs is reported to reduce variance
- All benchmarks use SQLite metadata backend with data inlining disabled

## Sample Results

Machine: ARM64 (aarch64), 8 GB RAM, Ubuntu 22.04

### Write

| Rows | Method | Time | Rows/s |
|------|--------|------|--------|
| 1K | ducklake-pandas | 18.7ms | 53K |
| 1K | ducklake-polars | 9.7ms | 103K |
| 1K | raw-duckdb | 293.2ms | 3K |
| 10K | ducklake-pandas | 18.8ms | 531K |
| 10K | ducklake-polars | 16.1ms | 620K |
| 10K | raw-duckdb | 308.5ms | 32K |
| 100K | ducklake-pandas | 68.8ms | 1.5M |
| 100K | ducklake-polars | 59.9ms | 1.7M |
| 100K | raw-duckdb | 332.7ms | 301K |

### Read

| Rows | Method | Time | Rows/s |
|------|--------|------|--------|
| 1K | ducklake-pandas | 4.2ms | 236K |
| 1K | ducklake-polars | 3.5ms | 284K |
| 1K | raw-duckdb | 249.3ms | 4K |
| 10K | ducklake-pandas | 4.0ms | 2.5M |
| 10K | ducklake-polars | 2.7ms | 3.7M |
| 10K | raw-duckdb | 299.5ms | 33K |
| 100K | ducklake-pandas | 42.0ms | 2.4M |
| 100K | ducklake-polars | 10.8ms | 9.2M |
| 100K | raw-duckdb | 893.9ms | 112K |

### Arrow Conversion Overhead

| Rows | Direction | Time | Rows/s |
|------|-----------|------|--------|
| 1K | pl → arrow | 205µs | 4.9M |
| 1K | arrow → pl | 101µs | 9.9M |
| 10K | pl → arrow | 292µs | 34.3M |
| 10K | arrow → pl | 231µs | 43.4M |
| 100K | pl → arrow | 2.3ms | 42.6M |
| 100K | arrow → pl | 2.4ms | 41.1M |

### Filter Pushdown

| Rows | Operation | Time | Rows/s |
|------|-----------|------|--------|
| 1K | full scan | 2.7ms | 372K |
| 1K | filtered scan | 3.0ms | 328K |
| 10K | full scan | 4.5ms | 2.2M |
| 10K | filtered scan | 3.7ms | 2.7M |
| 100K | full scan | 7.9ms | 12.7M |
| 100K | filtered scan | 8.1ms | 12.3M |

## Interpreting Results

- **ducklake-polars is consistently faster than ducklake-pandas** — Polars' native Parquet reader and columnar engine give it an edge, especially at larger data sizes.
- **Both Python libraries outperform raw DuckDB** for these benchmarks because raw DuckDB includes extension load time. In a long-running session with a persistent connection, DuckDB would be faster.
- **Arrow conversion overhead is negligible** — sub-millisecond for 10K rows, ~2ms for 100K rows. The `pl.DataFrame ↔ pa.Table` bridge adds minimal cost.
- **Filter pushdown** shows comparable times to full scans at these sizes — the Parquet files are small enough that I/O overhead dominates. At larger scales (1M+ rows, multiple files), pushdown benefits become more pronounced through min/max statistics-based file pruning.
