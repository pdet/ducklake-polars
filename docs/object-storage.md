# Object Storage

ducklake-dataframe supports reading and writing Parquet data files on cloud object storage via [fsspec](https://filesystem-spec.readthedocs.io/)-compatible backends. The catalog metadata itself remains in SQLite or PostgreSQL — only the data files live on object storage.

## Supported providers

| Provider | Extra | Package |
|---|---|---|
| Amazon S3 | `[s3]` | `s3fs` |
| Google Cloud Storage | `[gcs]` | `gcsfs` |
| Azure Blob Storage | `[azure]` | `adlfs` |

## Installation

```bash
# S3
pip install ducklake-dataframe[polars,s3]

# Google Cloud Storage
pip install ducklake-dataframe[polars,gcs]

# Azure Blob Storage
pip install ducklake-dataframe[polars,azure]

# All storage backends
pip install ducklake-dataframe[all]
```

## S3

### Setup

```python
from ducklake_polars import write_ducklake, scan_ducklake

# Write with S3 data path
write_ducklake(
    df,
    "catalog.ducklake",
    "users",
    mode="error",
    data_path="s3://my-bucket/warehouse/",
)

# Read back
lf = scan_ducklake(
    "catalog.ducklake",
    "users",
    data_path="s3://my-bucket/warehouse/",
)
```

### Authentication

`s3fs` uses standard AWS credential resolution:

- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- AWS credentials file (`~/.aws/credentials`)
- IAM roles (when running on EC2/ECS/Lambda)
- SSO profiles

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_DEFAULT_REGION=us-east-1
```

### S3-compatible stores

`s3fs` works with any S3-compatible API (MinIO, Ceph, R2, etc.) via the `endpoint_url` environment variable:

```bash
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```

## Google Cloud Storage

### Setup

```python
write_ducklake(
    df,
    "catalog.ducklake",
    "users",
    mode="error",
    data_path="gs://my-bucket/warehouse/",
)

lf = scan_ducklake(
    "catalog.ducklake",
    "users",
    data_path="gs://my-bucket/warehouse/",
)
```

### Authentication

`gcsfs` uses standard Google Cloud credential resolution:

- Application Default Credentials (`gcloud auth application-default login`)
- Service account key file (`GOOGLE_APPLICATION_CREDENTIALS`)
- Compute Engine metadata (when running on GCP)

## Azure Blob Storage

### Setup

```python
write_ducklake(
    df,
    "catalog.ducklake",
    "users",
    mode="error",
    data_path="az://my-container/warehouse/",
)

lf = scan_ducklake(
    "catalog.ducklake",
    "users",
    data_path="az://my-container/warehouse/",
)
```

### Authentication

`adlfs` uses standard Azure credential resolution:

- Environment variables (`AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_STORAGE_ACCOUNT_KEY`)
- Azure CLI login (`az login`)
- Managed Identity (when running on Azure)
- Connection strings

## Architecture

```
┌──────────────────────────┐
│  SQLite / PostgreSQL     │  ← Catalog metadata (local or networked)
│  (catalog.ducklake)      │
└──────────┬───────────────┘
           │ references
           ▼
┌──────────────────────────┐
│  S3 / GCS / Azure        │  ← Parquet data files
│  s3://bucket/data/       │
└──────────────────────────┘
```

The catalog stores references to Parquet files with their paths. When the `data_path` is an object storage URI, ducklake-dataframe uses the corresponding fsspec filesystem to read and write the Parquet files.

## Mixing local and remote

You can keep the catalog metadata locally (fast, no network latency for metadata operations) while storing the actual data on object storage:

```python
# Catalog on local disk, data on S3
write_ducklake(
    df,
    "/local/catalog.ducklake",      # local SQLite
    "users",
    mode="append",
    data_path="s3://data-lake/warehouse/",  # remote data
)
```

Or use PostgreSQL for the catalog and object storage for data — fully separating metadata from data:

```python
PG_DSN = "postgresql://user:pass@host/db"
write_ducklake(df, PG_DSN, "users", mode="append", data_path="s3://data-lake/warehouse/")
```

## Known limitations

- Object storage support is a current area of development. The local filesystem is the most tested path.
- Large-scale object storage operations (thousands of files) may benefit from connection pooling and retry configuration in the underlying fsspec library.
- Ensure the `data_path` parameter is consistent across all reads and writes to the same catalog — mismatched paths will cause file-not-found errors.
