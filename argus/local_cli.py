"""Small offline command-line helpers for common local computer tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import psutil

APP_COMMANDS = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
}

NON_DOCUMENT_EXTENSIONS = {
    ".bat",
    ".cmd",
    ".com",
    ".exe",
    ".lnk",
    ".msi",
    ".ps1",
    ".scr",
}

FOLDER_NAMES = {
    "downloads": "Downloads",
    "desktop": "Desktop",
    "documents": "Documents",
    "home": "",
}
MAX_LIST_ENTRIES = 100
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _location(value: str) -> Path:
    shortcut = FOLDER_NAMES.get(value.casefold())
    if shortcut is not None:
        return Path.home() / shortcut
    return Path(value).expanduser()


def _require_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError(
            "These local desktop commands are currently supported on Windows."
        )


def show_info() -> None:
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Computer: {platform.node()}")
    print(f"User: {os.getenv('USERNAME', os.getenv('USER', 'unknown'))}")
    print(f"Python: {platform.python_version()}")
    print(f"Working directory: {Path.cwd()}")


def show_health() -> None:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.home().anchor))
    cpu_percent = psutil.cpu_percent(interval=0.2)
    print(f"CPU usage: {cpu_percent:.1f}%")
    print(
        f"Memory: {memory.percent:.1f}% "
        f"({memory.available // (1024 ** 3)} GB available / "
        f"{memory.total // (1024 ** 3)} GB total)"
    )
    print(
        f"System drive: {disk.percent:.1f}% used "
        f"({disk.free // (1024 ** 3)} GB free)"
    )


def open_target(target: str) -> None:
    _require_windows()
    key = target.casefold()
    command = APP_COMMANDS.get(key)
    if command is not None:
        process = subprocess.Popen(command)
        print(f"Opened {key} (PID: {process.pid})")
        return

    if key in FOLDER_NAMES:
        folder = _location(key)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        process = subprocess.Popen(["explorer.exe", str(folder)])
        print(f"Opened folder: {folder} (PID: {process.pid})")
        return

    path = Path(target).expanduser()
    if not path.exists():
        available = ", ".join(sorted((*APP_COMMANDS, *FOLDER_NAMES)))
        raise FileNotFoundError(
            f"Target not found: {target}. Supported shortcuts: {available}"
        )
    resolved = path.resolve(strict=True)
    if resolved.is_dir():
        process = subprocess.Popen(["explorer.exe", str(resolved)])
        print(f"Opened folder: {resolved} (PID: {process.pid})")
        return
    if resolved.suffix.casefold() in NON_DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Refusing to launch executable or script files through open: {resolved}"
        )
    os.startfile(str(resolved))
    print(f"Opened file: {resolved}")


def list_location(value: str) -> None:
    folder = _location(value)
    if not folder.exists():
        raise FileNotFoundError(f"Path not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    entries = sorted(
        folder.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold())
    )
    for entry in entries[:MAX_LIST_ENTRIES]:
        marker = "[DIR] " if entry.is_dir() else "      "
        print(f"{marker}{entry.name}")
    if len(entries) > MAX_LIST_ENTRIES:
        print(f"...and {len(entries) - MAX_LIST_ENTRIES} more entries")
    if not entries:
        print("Folder is empty.")


def delete_file(value: str) -> bool:
    target = Path(value).expanduser()
    if target.is_symlink():
        raise ValueError("Refusing to delete a symbolic link.")
    resolved = target.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Only individual files can be deleted: {resolved}")

    print(f"File to permanently delete: {resolved}")
    if input("Type DELETE to confirm: ") != "DELETE":
        print("Cancelled; the file was not deleted.")
        return False

    resolved.unlink()
    print(f"Deleted: {resolved}")
    return True


async def _run_mcp(
    mode: str,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "argus_server.py"), "--server", "desktop"],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
    )
    async with stdio_client(server, errlog=sys.stderr) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            if mode == "list":
                result = await session.list_tools()
                tools = result.tools
                if tool_name:
                    query = tool_name.casefold()
                    tools = [
                        tool
                        for tool in tools
                        if query in tool.name.casefold()
                        or query in (tool.description or "").casefold()
                    ]
                for tool in tools:
                    print(f"{tool.name}: {tool.description or '(no description)'}")
                print(f"{len(tools)} tool(s)")
                return 0

            result = await session.call_tool(tool_name, arguments or {})
            for content in result.content:
                text = getattr(content, "text", None)
                if text is not None:
                    print(text)
                else:
                    print(json.dumps(content.model_dump(mode="json"), indent=2))
            if result.structuredContent is not None:
                print(json.dumps(result.structuredContent, indent=2, default=str))
            return 1 if result.isError else 0


def run_mcp(
    mode: str,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> int:
    return asyncio.run(_run_mcp(mode, tool_name, arguments))


def confirm_mcp_call(tool_name: str, arguments: dict[str, Any]) -> bool:
    print(f"MCP tool: {tool_name}")
    print(f"Arguments: {json.dumps(arguments, indent=2)}")
    try:
        response = input("Type CALL to execute this local tool: ")
    except EOFError:
        print("No interactive input is available; the MCP tool was not called.")
        return False
    if response != "CALL":
        print("Cancelled; the MCP tool was not called.")
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus-local",
        description="Run a few explicit local Argus tasks without an AI model or internet.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="show basic local system information")
    commands.add_parser("health", help="show local CPU, memory, and system-drive usage")

    open_parser = commands.add_parser("open", help="open a supported app or exact path")
    open_parser.add_argument(
        "target",
        help="notepad, calculator, downloads, desktop, documents, home, or a file/folder path",
    )

    list_parser = commands.add_parser(
        "list", help="list a folder (defaults to Downloads)"
    )
    list_parser.add_argument("location", nargs="?", default="downloads")

    delete_parser = commands.add_parser(
        "delete", help="delete one exact file after confirmation"
    )
    delete_parser.add_argument(
        "file", help="exact file path; directories and wildcards are not accepted"
    )

    mcp_parser = commands.add_parser(
        "mcp", help="connect directly to the local Argus MCP server (no AI model)"
    )
    mcp_commands = mcp_parser.add_subparsers(dest="mcp_command", required=True)
    mcp_list_parser = mcp_commands.add_parser("list", help="list local Argus MCP tools")
    mcp_list_parser.add_argument(
        "--filter", help="show only tool names or descriptions containing this text"
    )
    mcp_call_parser = mcp_commands.add_parser(
        "call", help="call one named local Argus MCP tool with JSON arguments"
    )
    mcp_call_parser.add_argument("tool", help="exact MCP tool name")
    mcp_call_parser.add_argument(
        "--args",
        default="{}",
        metavar="JSON",
        help='JSON object of tool arguments, for example \'{"app_name":"notepad"}\'',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            show_info()
        elif args.command == "health":
            show_health()
        elif args.command == "open":
            open_target(args.target)
        elif args.command == "list":
            list_location(args.location)
        elif args.command == "delete":
            delete_file(args.file)
        elif args.command == "mcp":
            if args.mcp_command == "list":
                return run_mcp("list", args.filter)
            try:
                arguments = json.loads(args.args)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON for --args: {error}") from error
            if not isinstance(arguments, dict):
                raise ValueError("--args must be a JSON object")
            if not confirm_mcp_call(args.tool, arguments):
                return 1
            return run_mcp("call", args.tool, arguments)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
