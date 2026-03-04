"""Custom exception hierarchy for DuckLake.

All specific exceptions inherit from both :class:`DuckLakeError` and
``ValueError`` so that existing ``except ValueError`` handlers continue
to work while new code can catch the precise error type.
"""


class DuckLakeError(Exception):
    """Base exception for all DuckLake errors."""


class TableNotFoundError(DuckLakeError, ValueError):
    """Raised when a requested table does not exist in the catalog."""


class SchemaNotFoundError(DuckLakeError, ValueError):
    """Raised when a requested schema does not exist in the catalog."""


class CatalogVersionError(DuckLakeError, ValueError):
    """Raised when the catalog version is unsupported or missing."""
