import asyncio

import pytest

import argus_sqlserver


def test_sqlserver_mcp_registers_read_only_tools():
    tools = asyncio.run(argus_sqlserver.mcp.list_tools())
    names = {tool.name for tool in tools}

    assert names == {
        "get_connection_info",
        "list_databases",
        "list_tables",
        "describe_table",
        "execute_read_query",
    }


@pytest.mark.parametrize(
    "query",
    [
        "UPDATE dbo.Users SET IsAdmin = 1",
        "SELECT * FROM dbo.Users; DROP TABLE dbo.Users",
        "SELECT * FROM dbo.Users -- ignore the rest",
        "SELECT * FROM dbo.Users /* comment */",
        "SELECT * INTO dbo.CopyOfUsers FROM dbo.Users",
        "EXEC xp_cmdshell 'whoami'",
    ],
)
def test_read_query_rejects_non_read_only_or_multi_statement_sql(query):
    with pytest.raises(ValueError):
        argus_sqlserver.execute_read_query(None, query)


def test_read_query_requires_a_select_statement():
    with pytest.raises(ValueError, match="SELECT"):
        argus_sqlserver.execute_read_query(None, "SHOW TABLES")
