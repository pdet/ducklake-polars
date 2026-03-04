"""DuckLake catalog writer — Polars wrapper around Arrow-based core."""

from __future__ import annotations

from typing import Any, Callable

import polars as pl
import pyarrow as pa

from ducklake_core._writer import DuckLakeCatalogWriter as _CoreWriter


def _polars_schema_to_arrow_dict(
    polars_schema: pl.Schema | dict[str, pl.DataType],
) -> dict[str, pa.DataType]:
    """Convert a Polars schema to a dict of Arrow types."""
    if isinstance(polars_schema, pl.Schema):
        schema_dict = dict(polars_schema)
    else:
        schema_dict = polars_schema
    # Create an empty DataFrame with the given schema and convert to Arrow
    # to get reliable type mapping.
    empty = pl.DataFrame(schema=schema_dict)
    arrow_schema = empty.to_arrow().schema
    return {field.name: field.type for field in arrow_schema}


def _polars_predicate_to_arrow(
    pred: pl.Expr,
) -> Callable[[pa.Table], pa.ChunkedArray]:
    """Wrap a Polars expression predicate as an Arrow-compatible callable."""

    def apply(table: pa.Table) -> pa.ChunkedArray:
        df = pl.from_arrow(table)
        mask = df.with_columns(pred.alias("__pred__"))["__pred__"]
        return mask.to_arrow()

    return apply


def _polars_expr_to_arrow_callable(
    expr: pl.Expr, col_name: str,
) -> Callable[[pa.Table], pa.ChunkedArray]:
    """Wrap a Polars expression (for update values) as an Arrow callable."""

    def apply(table: pa.Table) -> pa.ChunkedArray:
        df = pl.from_arrow(table)
        result = df.with_columns(expr.alias(col_name))[col_name]
        return result.to_arrow()

    return apply


def _convert_updates(updates: dict[str, Any]) -> dict[str, Any]:
    """Convert a Polars-style updates dict to an Arrow-compatible one.

    ``pl.Expr`` values are wrapped in callables; literal values are
    passed through unchanged.
    """
    result: dict[str, Any] = {}
    for col_name, value in updates.items():
        if isinstance(value, pl.Expr):
            result[col_name] = _polars_expr_to_arrow_callable(value, col_name)
        else:
            result[col_name] = value
    return result


def _convert_matched_update(
    when_matched_update: dict[str, Any] | bool | None,
) -> dict[str, Any] | bool | None:
    """Convert ``when_matched_update`` for the core merge API."""
    if when_matched_update is None or when_matched_update is True:
        return when_matched_update
    if isinstance(when_matched_update, dict):
        return _convert_updates(when_matched_update)
    return when_matched_update


class DuckLakeCatalogWriter:
    """
    Writes metadata to a DuckLake catalog (SQLite or PostgreSQL).

    Thin Polars wrapper around the Arrow-based core writer. Converts
    ``pl.DataFrame`` → ``pa.Table`` and ``pl.Expr`` → Arrow callables
    at method boundaries before delegating.
    """

    def __init__(
        self,
        metadata_path: str,
        *,
        data_path_override: str | None = None,
        data_inlining_row_limit: int = 0,
        author: str | None = None,
        commit_message: str | None = None,
        max_retries: int = 3,
        retry_wait_ms: float = 100,
        retry_backoff: float = 2.0,
    ) -> None:
        self._core = _CoreWriter(
            metadata_path,
            data_path_override=data_path_override,
            data_inlining_row_limit=data_inlining_row_limit,
            author=author,
            commit_message=commit_message,
            max_retries=max_retries,
            retry_wait_ms=retry_wait_ms,
            retry_backoff=retry_backoff,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate internal/private attribute access to the core writer."""
        return getattr(self._core, name)

    # ------------------------------------------------------------------
    # Context manager / lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._core.close()

    def __enter__(self) -> DuckLakeCatalogWriter:
        self._core.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._core.__exit__(*args)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data_path(self) -> str:
        return self._core.data_path

    # ------------------------------------------------------------------
    # CREATE TABLE
    # ------------------------------------------------------------------

    def create_table(
        self,
        table_name: str,
        polars_schema: pl.Schema | dict[str, pl.DataType],
        *,
        schema_name: str = "main",
    ) -> int:
        arrow_schema = _polars_schema_to_arrow_dict(polars_schema)
        return self._core.create_table(
            table_name, arrow_schema, schema_name=schema_name,
        )

    def create_table_with_data(
        self,
        table_name: str,
        df: pl.DataFrame,
        *,
        schema_name: str = "main",
    ) -> int:
        return self._core.create_table_with_data(
            table_name, df.to_arrow(), schema_name=schema_name,
        )

    # ------------------------------------------------------------------
    # INSERT / OVERWRITE
    # ------------------------------------------------------------------

    def insert_data(
        self,
        df: pl.DataFrame,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        return self._core.insert_data(
            df.to_arrow(), table_name, schema_name=schema_name,
        )

    def overwrite_data(
        self,
        df: pl.DataFrame,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        return self._core.overwrite_data(
            df.to_arrow(), table_name, schema_name=schema_name,
        )

    # ------------------------------------------------------------------
    # DELETE / UPDATE / MERGE
    # ------------------------------------------------------------------

    def delete_data(
        self,
        predicate: pl.Expr,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        return self._core.delete_data(
            _polars_predicate_to_arrow(predicate),
            table_name,
            schema_name=schema_name,
        )

    def update_data(
        self,
        updates: dict[str, Any],
        predicate: pl.Expr,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        return self._core.update_data(
            _convert_updates(updates),
            _polars_predicate_to_arrow(predicate),
            table_name,
            schema_name=schema_name,
        )

    def merge_data(
        self,
        source_df: pl.DataFrame,
        table_name: str,
        on: str | list[str],
        *,
        when_matched_update: dict[str, Any] | bool | None = None,
        when_not_matched_insert: bool = True,
        schema_name: str = "main",
    ) -> tuple[int, int]:
        return self._core.merge_data(
            source_df.to_arrow(),
            table_name,
            on,
            when_matched_update=_convert_matched_update(when_matched_update),
            when_not_matched_insert=when_not_matched_insert,
            schema_name=schema_name,
        )

    # ------------------------------------------------------------------
    # ADD FILES
    # ------------------------------------------------------------------

    def add_files(
        self,
        table_name: str,
        file_paths: list[str],
        *,
        schema_name: str = "main",
    ) -> int:
        return self._core.add_files(
            table_name, file_paths, schema_name=schema_name,
        )

    # ------------------------------------------------------------------
    # ALTER TABLE
    # ------------------------------------------------------------------

    def add_column(
        self,
        table_name: str,
        column_name: str,
        polars_dtype: pl.DataType,
        *,
        default: Any = None,
        schema_name: str = "main",
    ) -> None:
        arrow_dtype = pl.Series("x", [], dtype=polars_dtype).to_arrow().type
        self._core.add_column(
            table_name, column_name, arrow_dtype,
            default=default, schema_name=schema_name,
        )

    def rename_column(
        self,
        table_name: str,
        old_column_name: str,
        new_column_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        self._core.rename_column(
            table_name, old_column_name, new_column_name,
            schema_name=schema_name,
        )

    def drop_column(
        self,
        table_name: str,
        column_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        self._core.drop_column(
            table_name, column_name, schema_name=schema_name,
        )

    def set_column_type(
        self,
        table_name: str,
        column_name: str,
        new_type: str,
        *,
        schema_name: str = "main",
    ) -> None:
        self._core.set_column_type(
            table_name, column_name, new_type, schema_name=schema_name,
        )

    # ------------------------------------------------------------------
    # TABLE operations
    # ------------------------------------------------------------------

    def drop_table(
        self,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        self._core.drop_table(table_name, schema_name=schema_name)

    def rename_table(
        self,
        old_table_name: str,
        new_table_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        self._core.rename_table(
            old_table_name, new_table_name, schema_name=schema_name,
        )

    def set_partitioned_by(
        self,
        table_name: str,
        column_names: list[str],
        *,
        schema_name: str = "main",
    ) -> None:
        self._core.set_partitioned_by(
            table_name, column_names, schema_name=schema_name,
        )

    # ------------------------------------------------------------------
    # SCHEMA operations
    # ------------------------------------------------------------------

    def create_schema(self, schema_name: str) -> int:
        return self._core.create_schema(schema_name)

    def drop_schema(self, schema_name: str, *, cascade: bool = False) -> None:
        self._core.drop_schema(schema_name, cascade=cascade)

    # ------------------------------------------------------------------
    # VIEW operations
    # ------------------------------------------------------------------

    def create_view(
        self,
        view_name: str,
        sql: str,
        *,
        schema_name: str = "main",
        or_replace: bool = False,
        column_aliases: str = "",
    ) -> int:
        return self._core.create_view(
            view_name, sql,
            schema_name=schema_name,
            or_replace=or_replace,
            column_aliases=column_aliases,
        )

    def drop_view(
        self,
        view_name: str,
        *,
        schema_name: str = "main",
    ) -> None:
        self._core.drop_view(view_name, schema_name=schema_name)

    # ------------------------------------------------------------------
    # MAINTENANCE
    # ------------------------------------------------------------------

    def rewrite_data_files(
        self,
        table_name: str,
        *,
        schema_name: str = "main",
    ) -> int:
        return self._core.rewrite_data_files(
            table_name, schema_name=schema_name,
        )

    def expire_snapshots(
        self,
        *,
        older_than_snapshot: int | None = None,
        keep_last_n: int | None = None,
    ) -> int:
        return self._core.expire_snapshots(
            older_than_snapshot=older_than_snapshot,
            keep_last_n=keep_last_n,
        )

    def vacuum(self) -> int:
        return self._core.vacuum()
