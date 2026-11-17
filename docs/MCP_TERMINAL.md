# Use Argus from GitHub Copilot CLI

The Argus server uses MCP over stdio. Starting `argus_server.py` directly only
starts the MCP server; Copilot CLI supplies the chat interface and turns natural
language requests into tool calls.

## Offline work from the terminal

Use the local launcher from PowerShell or Command Prompt. It uses this
checkout's `.venv` and makes no model or network requests:

```powershell
D:\argus2.o\start_argus_local.bat --help
D:\argus2.o\start_argus_local.bat health
D:\argus2.o\start_argus_local.bat open notepad
D:\argus2.o\start_argus_local.bat open downloads
D:\argus2.o\start_argus_local.bat list downloads
```

The local CLI deliberately accepts explicit commands, not plain English. It
supports system information and health, Notepad and Calculator, common folders
or an exact path, listing directories, and deleting one exact file. It will not
launch executable or script files through `open`. Deletion rejects directories
and symbolic links and requires typing `DELETE` after showing the resolved
path:

```powershell
D:\argus2.o\start_argus_local.bat open "$HOME\Documents\notes.txt"
D:\argus2.o\start_argus_local.bat delete "$HOME\Downloads\old-file.txt"
```

### Call the local Argus MCP server directly

The terminal can also act as an MCP client. In PowerShell, use the `.ps1`
launcher for MCP commands (especially commands with JSON arguments). These
commands start the Argus desktop server locally over stdio, so they do not
need Copilot, Ollama, or internet:

```powershell
& D:\argus2.o\start_argus_local.ps1 mcp list --filter system
& D:\argus2.o\start_argus_local.ps1 mcp call get_system_info
& D:\argus2.o\start_argus_local.ps1 mcp call open_app_with_url --args '{"app_name":"notepad"}'
```

Bluetooth is available through two local MCP tools. First list the adapter
names and instance IDs:

```powershell
& D:\argus2.o\start_argus_local.ps1 mcp call bluetooth_status
```

Then copy the exact Bluetooth `InstanceId` from the result and request the
specific change (`false` turns it off; `true` turns it on). Use the PowerShell
launcher for JSON arguments so PowerShell passes the quotes and `&` characters
intact:

```powershell
& D:\argus2.o\start_argus_local.ps1 mcp call bluetooth_set_enabled --args '{"instance_id":"PASTE_EXACT_INSTANCE_ID","enabled":false}'
```

The terminal shows the target and arguments and requires typing `CALL`.
Disabling a Bluetooth adapter disconnects Bluetooth peripherals and may require
an elevated PowerShell process. The tool refuses IDs that do not identify a
Windows Bluetooth-class adapter (by its device name) and does not disable every
Bluetooth device at once.
The current Copilot CLI MCP allowlist does not include these Bluetooth tools;
use `start_argus_local.ps1 mcp ...` in PowerShell for this offline capability.

Tool calls show the exact tool name and JSON arguments and require typing
`CALL`. The MCP server may expose actions that change files or system settings;
review the tool and arguments before confirming. To discover a tool's exact
name and description, use `mcp list`; use `mcp list --filter TEXT` to narrow the
results. The CLI passes JSON arguments directly to the tool and does not infer
or translate natural language. The server runs locally, but some individual
tools (such as website or network speed checks) still need internet; choose a
local system/file/app tool for offline work.

For other offline tasks, use PowerShell commands directly in the terminal.
PowerShell can perform general local tasks, but commands must be explicit; a
terminal does not understand arbitrary natural-language instructions by itself.
The `argus-local` CLI also provides direct MCP access to the full Argus desktop
server. Tools that inherently contact websites, remote services, or online
databases still need a network connection.

The default `argus` server exposes the desktop, browser, and Windows automation
tools. It does not connect to MySQL. Database and ML tools are separate servers
and are not needed for normal PC automation.

The optional `argus-sqlserver` server connects to the local SQL Server Express
instance with Windows Integrated Authentication and uses `MyDatabase` by
default. It provides connection/database/table inspection and single-statement
guarded `SELECT` queries, limited to 200 returned rows. Write statements and
multiple SQL statements are rejected. The server uses the current Windows
account's SQL Server permissions; query validation is not a substitute for
least-privilege database permissions. The Python SQL Server ODBC driver must be
installed in the project virtual environment. It is also registered in
`.vscode/mcp.json`; reload VS Code after changing that file.

The local Copilot CLI MCP configuration can register it with:

```bat
copilot mcp add --env "ARGUS_SQLSERVER_INSTANCE=.\SQLEXPRESS" --env "ARGUS_SQLSERVER_DATABASE=MyDatabase" argus-sqlserver -- "D:\argus2.o\.venv\Scripts\python.exe" "D:\argus2.o\argus_sqlserver.py"
copilot mcp list
```

In Copilot, ask for database tasks such as “Use Argus SQL Server to list tables
in MyDatabase” or “Count the rows in dbo.customers.” Restart an existing Copilot
CLI session if the newly added tools do not appear.

## Online natural-language tasks with Copilot CLI

Use this launcher from a terminal for online Copilot tasks:

```powershell
D:\argus2.o\start_argus_copilot.bat
```

It starts Copilot with its online `auto` model and clears temporary offline
provider environment variables from that process. The CLI must be installed,
signed in, and connected to the internet. This launcher loads the project
instructions and uses the existing user-level Argus MCP registration. It does
not install Copilot or change your account configuration.

If Copilot CLI is not installed, install it while online and then open a new
terminal:

```powershell
winget install GitHub.Copilot
```

The Argus MCP server is registered in the current user's Copilot CLI
configuration. Check it with `copilot mcp list`; the current setup uses a
small tool allowlist to reduce model context. `argus-sqlserver` is optional and
is disabled unless explicitly enabled.

The `.vscode/mcp.json` file configures VS Code's MCP client. Copilot CLI does
not load that VS Code-specific file; it uses the user-level Copilot MCP
configuration. The CLI may display an informational migration notice about
`.vscode/mcp.json`; it does not mean the user-level Argus registration failed.
Do not register the same Argus server in multiple CLI configuration sources.

Examples of online natural-language tasks:

```text
Use Argus to open YouTube in Chrome.
Use Argus to open Notepad.
Use Argus to check my system health.
```

Copilot interprets the request using an online model and Argus performs the
supported operation. The current user-level Copilot registration exposes only
five Argus tools: system information, uptime, health, opening an app with an
optional URL, and running a command. Other Argus features are available through
the direct offline MCP client, but are not exposed to Copilot by that
allowlist. By contrast, the local terminal CLI and PowerShell do not need a
model but require explicit commands.

## Optional MySQL tools

The installed database engine on this computer is SQL Server Express, not
MySQL. Do not add the MySQL server for SQL Server databases. Its tools remain
separate and require an independently configured, reachable MySQL instance.

For Copilot CLI, edit `%USERPROFILE%\.copilot\mcp-config.json` and add a second
entry under `mcpServers`:

```json
"argus-mysql": {
  "type": "local",
  "command": "D:\\argus2.o\\.venv\\Scripts\\python.exe",
  "args": [
    "D:\\argus2.o\\argus_server.py",
    "--server",
    "mysql"
  ],
  "tools": ["*"],
  "env": {
    "MYSQL_HOST": "localhost",
    "MYSQL_PORT": "3306",
    "MYSQL_USER": "your-user",
    "MYSQL_PASSWORD": "your-password",
    "MYSQL_DATABASE": "your-database"
  }
}
```

Replace the example connection values with your database settings. This config
stores the password as local plaintext; protect the file and do not commit it.
The same command-line option can be used to configure another MCP client:
`argus_server.py --server mysql`.

The separate `ml` server can be run with `argus_server.py --server ml`.
