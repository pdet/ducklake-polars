# Benchmarks

Self-tracking performance benchmarks for ducklake-dataframe and ducklake-pandas.

Use these to detect regressions and measure the impact of optimizations. Not for comparing against other systems.

## Running

```bash
# Default: 1K, 10K, 100K rows, 3 runs each
python benchmarks/bench_read_write.py

# Custom sizes and runs
python benchmarks/bench_read_write.py --sizes 1000,10000,100000,1000000 --runs 5
```

## What's Measured

| Benchmark | What it tells you |
|-----------|-------------------|
| Write | Time to write N rows to a DuckLake table (polars vs pandas wrapper) |
| Read | Time to read N rows back (polars vs pandas wrapper) |
| Arrow conversion | Overhead of DataFrame ↔ pa.Table at the wrapper boundary |
| Filter pushdown | Benefit of predicate pushdown in scan_ducklake (polars only) |

## Interpreting Results

- **Rows/s** is the key metric — higher is better
- **Arrow conversion** shows the wrapper tax — should be negligible vs I/O
- **Filtered scan** should be faster than full scan for large tables (statistics-based pruning)
- Run on the same hardware for meaningful comparisons across commits
