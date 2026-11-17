#!/usr/bin/env python3
"""Read-only MCP tools for the local SQL Server instance."""

import asyncio
import builtins
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator


def print(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("file", sys.stderr)
    builtins.print(*args, **kwargs)


try:
    import pyodbc
except ImportError:
    print("pyodbc is required. Install project requirements with: pip install -r requirements.txt")
    raise

from mcp.server.fastmcp import Context, FastMCP

DEFAULT_INSTANCE = r".\SQLEXPRESS"
DEFAULT_DATABASE = "MyDatabase"
MAX_QUERY_ROWS = 200
QUERY_TIMEOUT_SECONDS = 15
FORBIDDEN_QUERY_KEYWORDS = re.compile(
    r"\b(?:"
    r"alter|backup|bulk|create|dbcc|delete|drop|execute|exec|insert|into|"
    r"merge|openquery|openrowset|opendatasource|restore|shutdown|truncate|"
    r"update|waitfor"
    r")\b",
    re.IGNORECASE,
)


def _connection_string() -> str:
    drivers = pyodbc.drivers()
    driver = next(
        (name for name in ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server")
         if name in drivers),
        None,
    )
    if driver is None:
        raise RuntimeError(
            "Microsoft ODBC Driver 17 or 18 for SQL Server is required."
        )

    instance = os.getenv("ARGUS_SQLSERVER_INSTANCE", DEFAULT_INSTANCE)
    database = os.getenv("ARGUS_SQLSERVER_DATABASE", DEFAULT_DATABASE)
    if not instance.strip() or not database.strip():
        raise ValueError("SQL Server instance and database settings cannot be empty.")

    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={instance};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=5;"
    )


@asynccontextmanager
async def sqlserver_lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    connection = pyodbc.connect(_connection_string(), timeout=5)
    connection.timeout = QUERY_TIMEOUT_SECONDS
    try:
        yield {"connection": connection}
    finally:
        connection.close()


mcp = FastMCP("Argus SQL Server", lifespan=sqlserver_lifespan)


def _connection(ctx: Context) -> Any:
    return ctx.request_context.lifespan_context["connection"]


def _fetch_dicts(cursor: Any) -> list[dict[str, Any]]:
    columns = [column[0] for column in cursor.description or ()]
    return [
        {
            column: value if value is None or isinstance(value, (str, int, float, bool))
            else value.isoformat() if hasattr(value, "isoformat")
            else str(value)
            for column, value in zip(columns, row)
        }
        for row in cursor.fetchall()
    ]


@mcp.tool()
def get_connection_info(ctx: Context) -> dict[str, Any]:
    """Return the connected SQL Server instance, database, and Windows login."""
    cursor = _connection(ctx).cursor()
    try:
        cursor.execute(
            "SELECT CAST(SERVERPROPERTY('ServerName') AS nvarchar(128)) AS server_name, "
            "DB_NAME() AS database_name, SYSTEM_USER AS login_name"
        )
        return _fetch_dicts(cursor)[0]
    finally:
        cursor.close()


@mcp.tool()
def list_databases(ctx: Context) -> list[dict[str, Any]]:
    """List online SQL Server databases visible to the current Windows login."""
    cursor = _connection(ctx).cursor()
    try:
        cursor.execute(
            "SELECT name, state_desc FROM sys.databases "
            "WHERE state_desc = 'ONLINE' ORDER BY name"
        )
        return _fetch_dicts(cursor)
    finally:
        cursor.close()


@mcp.tool()
def list_tables(ctx: Context, schema_name: str | None = None) -> list[dict[str, Any]]:
    """List tables in the configured database, optionally filtered by schema."""
    cursor = _connection(ctx).cursor()
    try:
        cursor.execute(
            "SELECT s.name AS schema_name, t.name AS table_name "
            "FROM sys.tables AS t "
            "JOIN sys.schemas AS s ON s.schema_id = t.schema_id "
            "WHERE (? IS NULL OR s.name = ?) "
            "ORDER BY s.name, t.name",
            schema_name,
            schema_name,
        )
        return _fetch_dicts(cursor)
    finally:
        cursor.close()


@mcp.tool()
def describe_table(
    ctx: Context, table_name: str, schema_name: str = "dbo"
) -> list[dict[str, Any]]:
    """Describe columns for a table in the configured database."""
    if not table_name.strip() or not schema_name.strip():
        raise ValueError("Table and schema names cannot be empty.")

    cursor = _connection(ctx).cursor()
    try:
        cursor.execute(
            "SELECT c.name AS column_name, ty.name AS data_type, "
            "c.max_length, c.precision, c.scale, c.is_nullable, "
            "c.is_identity, c.column_id "
            "FROM sys.columns AS c "
            "JOIN sys.types AS ty ON ty.user_type_id = c.user_type_id "
            "WHERE c.object_id = OBJECT_ID(QUOTENAME(?) + N'.' + QUOTENAME(?)) "
            "ORDER BY c.column_id",
            schema_name,
            table_name,
        )
        return _fetch_dicts(cursor)
    finally:
        cursor.close()


@mcp.tool()
def execute_read_query(ctx: Context, query: str) -> dict[str, Any]:
    """Run one read-only SELECT query and return at most 200 rows."""
    statement = query.strip()
    if not re.match(r"^SELECT\b", statement, re.IGNORECASE):
        raise ValueError("Only a single SELECT query is allowed.")
    if ";" in statement or "--" in statement or "/*" in statement or "*/" in statement:
        raise ValueError("Semicolons and SQL comments are not allowed.")
    if FORBIDDEN_QUERY_KEYWORDS.search(statement):
        raise ValueError("The query contains a keyword that is not allowed.")

    cursor = _connection(ctx).cursor()
    try:
        cursor.execute(statement)
        if cursor.description is None:
            raise ValueError("The query did not return a result set.")
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(MAX_QUERY_ROWS + 1)
        truncated = len(rows) > MAX_QUERY_ROWS
        rows = rows[:MAX_QUERY_ROWS]
        return {
            "columns": columns,
            "rows": [
                {
                    column: value if value is None or isinstance(value, (str, int, float, bool))
                    else value.isoformat() if hasattr(value, "isoformat")
                    else str(value)
                    for column, value in zip(columns, row)
                }
                for row in rows
            ],
            "row_count": len(rows),
            "truncated": truncated,
        }
    finally:
        cursor.close()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
