<p align="center">
  <img src="../assets/argus-hero.png" alt="Argus 2.0 visual identity" width="100%">
</p><h1 align="center">ARGUS · TERMINAL CONTROL CENTER</h1><p align="center">
  <strong>One project. Three ways to interact.</strong><br>
  Local commands · Direct MCP · Copilot CLI
</p><p align="center">
  <img src="https://img.shields.io/badge/Windows-Ready-76B900?style=for-the-badge&logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/MCP-stdio-76B900?style=for-the-badge" alt="MCP stdio">
  <img src="https://img.shields.io/badge/Copilot-Optional-76B900?style=for-the-badge&logo=githubcopilot&logoColor=white" alt="Copilot optional">
</p><p align="center">
  <a href="../README.md">← Main README</a> ·
  <a href="../README_CORE.md">Learned-agent lab</a> ·
  <a href="#pick-your-interface">Pick your interface</a>
</p>[!TIP]
Not sure which one to use? Start with the chooser below. Local CLI is for explicit commands, direct MCP is for calling a named tool yourself, and Copilot CLI adds the natural-language model interface.

Argus terminal & MCP guide
Pick the interface that matches your task. Argus has three different ways to work from the terminal. They are not interchangeable.

Pick your interface
<p align="center">
  <img src="../assets/argus-architecture.png" alt="Argus architecture" width="100%">
</p>Choose your path
<details open>
<summary><strong>🟢 I want to run a simple local command (no AI)</strong></summary>Use the Local CLI. It accepts explicit commands such as health, open notepad, and list downloads; it does not interpret arbitrary natural-language requests.
.\start_argus_local.bat --help
.\start_argus_local.bat info
.\start_argus_local.bat health
.\start_argus_local.bat open notepad
.\start_argus_local.bat open calculator
.\start_argus_local.bat open downloads
.\start_argus_local.bat list downloads
</details><details>
<summary><strong>🔧 I want to discover and call a specific MCP tool myself</strong></summary>Use the direct MCP client in the PowerShell launcher. It connects to the desktop server over stdio and does not need Copilot or an AI model.
& .\start_argus_local.ps1 mcp list
& .\start_argus_local.ps1 mcp list --filter system
& .\start_argus_local.ps1 mcp call get_system_info
& .\start_argus_local.ps1 mcp call open_app_with_url --args '{"app_name":"notepad"}'
Every call displays the tool name and JSON arguments and requires typing CALL. Inspect the operation before confirming. The desktop server includes tools that can change files or Windows settings.
</details><details>
<summary><strong>🤖 I want natural-language interaction through GitHub Copilot CLI</strong></summary>
Use Copilot CLI. It provides the model/chat interface and selects tools from the user's configured MCP allowlist.
.\start_argus_copilot.bat
Copilot CLI must already be installed, signed in, and online. The launcher does not install Copilot or create the user's MCP registration.
</details><p align="center">
  <img src="https://img.shields.io/badge/01-Local%20CLI-76B900?style=flat-square" alt="Local CLI">
  <img src="https://img.shields.io/badge/02-Direct%20MCP-76B900?style=flat-square" alt="Direct MCP">
  <img src="https://img.shields.io/badge/03-Copilot-76B900?style=flat-square" alt="Copilot">
</p>1. Local CLI — explicit commands, no AI model
The Windows launcher uses the checkout's .venv:
.\start_argus_local.bat --help
.\start_argus_local.bat info
.\start_argus_local.bat health
.\start_argus_local.bat open notepad
.\start_argus_local.bat open calculator
.\start_argus_local.bat open downloads
.\start_argus_local.bat list downloads
It supports system information and health, Notepad and Calculator, common folders or an exact path, directory listing, and deletion of one exact file. It refuses executable and script files through open. Deletion refuses directories and symbolic links and requires typing DELETE after displaying the resolved file path.
<details>
<summary><strong>Open a document or delete one exact file</strong></summary>.\start_argus_local.bat open "$HOME\Documents\notes.txt"
.\start_argus_local.bat delete "$HOME\Downloads\old-file.txt"
Deletion is permanent. Check the displayed resolved path carefully before typing DELETE.
</details>2. Direct MCP access from PowerShell
& .\start_argus_local.ps1 mcp list
& .\start_argus_local.ps1 mcp list --filter system
& .\start_argus_local.ps1 mcp call get_system_info
& .\start_argus_local.ps1 mcp call open_app_with_url --args '{"app_name":"notepad"}'
<details>
<summary><strong>Bluetooth tools — inspect before changing an adapter</strong></summary>The desktop MCP server includes bluetooth_status and bluetooth_set_enabled. First inspect the devices and copy the exact adapter InstanceId:
& .\start_argus_local.ps1 mcp call bluetooth_status
Then pass the exact ID and requested state:
& .\start_argus_local.ps1 mcp call bluetooth_set_enabled --args '{"instance_id":"PASTE_EXACT_INSTANCE_ID","enabled":false}'
false disables and true enables the selected Bluetooth adapter. Disabling an adapter can disconnect peripherals and may require elevated PowerShell. The tool checks that the selected device is identified as a Bluetooth-class adapter; do not use an ID copied from another device.

The current Copilot CLI allowlist documented by this repository does not include these Bluetooth tools. Use the direct local MCP client for them.
</details>3. MCP server modes
The main argus_server.py entry point supports three separate server modes:
<details open>
<summary><strong>Desktop MCP — default mode</strong></summary>python argus_server.py --server desktop
The default is desktop. It provides Windows desktop and automation tools.
</details><details>
<summary><strong>MySQL MCP — separate database server</strong></summary>
Set connection settings in the current PowerShell session:
$env:MYSQL_HOST = "localhost"
$env:MYSQL_PORT = "3306"
$env:MYSQL_USER = "your-user"
$env:MYSQL_PASSWORD = "your-password"
$env:MYSQL_DATABASE = "your-database"
python argus_server.py --server mysql
The MySQL server includes administrative and write-capable tools as well as inspection tools. Use a dedicated least-privilege account, review actions before invoking them, and never commit credentials.
</details><details>
<summary><strong>ML monitoring MCP</strong></summary>python argus_server.py --server ml
The ml mode exposes monitoring tools for the project's ML/data-collection environment. It is separate from the learned policy pipeline in the argus/ package.
</details>Dependency note: the main server module imports the MySQL connector before selecting a mode. Install the full requirements.txt before starting the main server, even if you select desktop.

4. SQL Server MCP server
SQL Server runs as a separate MCP process:
$env:ARGUS_SQLSERVER_INSTANCE = ".\SQLEXPRESS"
$env:ARGUS_SQLSERVER_DATABASE = "MyDatabase"
python argus_sqlserver.py
Defaults are instance .\SQLEXPRESS and database MyDatabase. The server uses Windows Integrated Authentication and requires Microsoft ODBC Driver 17 or 18 for SQL Server.

It exposes connection/database/table/schema inspection plus execute_read_query. The query tool accepts one statement beginning with SELECT, rejects semicolons, comments, and a list of write/administrative keywords, uses a query timeout, and returns at most 200 rows. These application-side checks are not a substitute for a least-privilege SQL Server login.
<details>
<summary><strong>SQL Server setup checklist</strong></summary>- [ ] SQL Server instance is running.
- [ ] Microsoft ODBC Driver 17 or 18 is installed.
- [ ] The Windows account has the required database permissions.
- [ ] ARGUS_SQLSERVER_INSTANCE points to the intended instance.
- [ ] ARGUS_SQLSERVER_DATABASE points to the intended database.
</details>5. GitHub Copilot CLI — online natural-language interaction
Launch Copilot from the repository root:
.\start_argus_copilot.bat
The launcher checks that copilot is on PATH, starts it with the auto model, clears temporary offline-provider environment variables for that process, and loads the repository's .github/instructions. Copilot CLI must already be installed, signed in, and online. The launcher does not install Copilot or create the user's MCP registration.
<details>
<summary><strong>Which Argus tools can Copilot currently call?</strong></summary>
The repository's current instructions document a user-level Copilot MCP allowlist of five Argus tools:
- get_system_info
- system_uptime
- system_health_check
- open_app_with_url
- run_command
Do not assume Copilot can call every tool registered by the desktop MCP server. Check the user-level registration with:
copilot mcp list
The Local CLI and PowerShell accept explicit commands; they do not translate natural language. Copilot CLI is the model-driven interpretation layer.
</details>6. VS Code MCP configuration is separate
.vscode/mcp.json is for the VS Code MCP client. It is not the source of the user's Copilot CLI MCP registration. The checked-in file contains checkout-specific executable and working-directory paths; update those values for your own machine rather than copying them as a portable template. Do not put secrets in the file or commit credentials.
7. Troubleshooting — start with the symptom
<details>
<summary><strong>Local launcher says the virtual environment is missing</strong></summary>Create the environment in the repository root and install the required profile:
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
</details><details>
<summary><strong>Main MCP server fails during startup</strong></summary>The main server imports the MySQL connector before it selects a mode. Install the full requirements, including mcp and mysql-connector-python, even when starting the Desktop mode:
python -m pip install -r requirements.txt
python argus_server.py --help
</details><details>
<summary><strong>SQL Server connection fails</strong></summary>Check that SQL Server is running, ODBC Driver 17 or 18 is installed, the Windows account has access, and the ARGUS_SQLSERVER_INSTANCE / ARGUS_SQLSERVER_DATABASE values are correct.
</details><details>
<summary><strong>Copilot does not show Argus tools</strong></summary>
Check that Copilot CLI is installed and signed in, then inspect the user-level MCP registration:
copilot mcp list
The VS Code .vscode/mcp.json file and Copilot CLI's user-level registration are separate configurations.
</details><details>
<summary><strong>A tool is not available in Copilot</strong></summary>
The Copilot user-level allowlist is smaller than the full desktop MCP tool set. Discover tools from the direct local MCP client:
& .\start_argus_local.ps1 mcp list
</details>Remember: MCP exposes tools to a client; the Local CLI is explicit-command driven; Copilot CLI provides the natural-language layer. Review all tool arguments before confirming system-changing operations.

<p align="center">
  <strong>ARGUS 2.0</strong><br>
  <sub>Local-first tools · Explicit permissions · Review before execution</sub>
</p>
