@echo off
setlocal
set "ARGUS_ROOT=%~dp0"
if not exist "%ARGUS_ROOT%.venv\Scripts\python.exe" (
    echo Argus virtual environment not found at "%ARGUS_ROOT%.venv".
    echo Create the project environment and install its requirements before using this launcher.
    exit /b 1
)
cd /d "%ARGUS_ROOT%"
if errorlevel 1 (
    echo Could not switch to the Argus project folder.
    exit /b 1
)
"%ARGUS_ROOT%.venv\Scripts\python.exe" -m argus.local_cli %*
exit /b %ERRORLEVEL%
