import asyncio
from types import SimpleNamespace

import argus_server
import pytest


def test_mysql_tool_reads_database_from_mcp_lifespan_context():
    class FakeDatabase:
        def get_tables(self):
            return [{"table_name": "example"}]

    ctx = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context={"db": FakeDatabase()})
    )

    result = argus_server.list_mysql_tables(ctx)

    assert result == {"success": True, "tables": ["example"], "count": 1}


def test_mcp_servers_expose_disjoint_tool_groups():
    async def list_tools():
        return {
            "desktop": await argus_server.mcp.list_tools(),
            "mysql": await argus_server.mysql_mcp.list_tools(),
            "ml": await argus_server.ml_mcp.list_tools(),
        }

    tools = asyncio.run(list_tools())
    names = {
        server: {tool.name for tool in server_tools}
        for server, server_tools in tools.items()
    }

    assert "open_youtube_with_search" in names["desktop"]
    assert "get_system_info" in names["desktop"]
    assert not any(name.startswith("mysql_") for name in names["desktop"])

    assert len(names["mysql"]) > 200
    assert "describe_table" in names["mysql"]
    assert "add_partition" in names["mysql"]
    assert names["desktop"].isdisjoint(names["mysql"])

    assert len(names["ml"]) == 7
    assert names["desktop"].isdisjoint(names["ml"])
    assert names["mysql"].isdisjoint(names["ml"])


def test_open_app_with_url_resolves_brave_from_path(monkeypatch):
    calls = []

    monkeypatch.setattr(
        argus_server.shutil, "which", lambda _name: r"C:\Brave\brave.exe"
    )
    monkeypatch.setattr(
        argus_server.subprocess,
        "Popen",
        lambda command: calls.append(command) or SimpleNamespace(pid=123),
    )

    result = asyncio.run(argus_server.open_app_with_url("brave", "https://youtube.com"))

    assert calls == [[r"C:\Brave\brave.exe", "https://youtube.com"]]
    assert "Opened brave (PID: 123)" in result


def test_bluetooth_tools_are_registered_on_desktop_server():
    async def list_names():
        return {tool.name for tool in await argus_server.mcp.list_tools()}

    names = asyncio.run(list_names())

    assert "bluetooth_status" in names
    assert "bluetooth_set_enabled" in names


def test_bluetooth_status_runs_local_device_query(monkeypatch):
    calls = []

    monkeypatch.setattr(argus_server.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        argus_server.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(
            returncode=0,
            stdout='[{"FriendlyName":"RZ616 Bluetooth Adapter","InstanceId":"USB\\\\BT","Status":"OK"}]',
            stderr="",
        ),
    )

    result = asyncio.run(argus_server.bluetooth_status())

    assert "Name: RZ616 Bluetooth Adapter" in result
    assert "InstanceId: USB\\BT" in result
    assert "Status: OK" in result
    assert calls[0][0][:4] == [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
    ]
    assert "Get-PnpDevice -Class Bluetooth" in calls[0][0][4]


def test_bluetooth_set_enabled_uses_instance_id_as_environment_value(monkeypatch):
    calls = []
    instance_id = "BTH\\DEV_123'; Write-Output unsafe"

    monkeypatch.setattr(argus_server.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        argus_server.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(returncode=0, stdout='{"Status":"Disabled"}', stderr=""),
    )

    result = asyncio.run(argus_server.bluetooth_set_enabled(instance_id, False))

    command, kwargs = calls[0]
    assert result == '{"Status":"Disabled"}'
    assert kwargs["env"]["ARGUS_BLUETOOTH_INSTANCE_ID"] == instance_id
    assert instance_id not in command[4]
    assert "Disable-PnpDevice" in command[4]
    assert "$device.Class -ne 'Bluetooth'" in command[4]
    assert "$device.FriendlyName -notmatch" in command[4]


def test_bluetooth_set_enabled_rejects_empty_instance_id():
    result = asyncio.run(argus_server.bluetooth_set_enabled(" ", False))

    assert "InstanceId must not be empty" in result


def test_bluetooth_status_reports_non_windows_platform(monkeypatch):
    monkeypatch.setattr(argus_server.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        argus_server.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("must not invoke Windows PowerShell"),
    )

    result = asyncio.run(argus_server.bluetooth_status())

    assert "supported only on Windows" in result
