@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 (
    echo Could not switch to the Argus project folder.
    exit /b 1
)
where copilot >nul 2>nul
if errorlevel 1 (
    echo GitHub Copilot CLI was not found. Install or add "copilot" to PATH.
    exit /b 1
)
set "COPILOT_OFFLINE="
set "COPILOT_PROVIDER_BASE_URL="
set "COPILOT_PROVIDER_TYPE="
set "COPILOT_PROVIDER_WIRE_API="
set "COPILOT_PROVIDER_TRANSPORT="
set "COPILOT_PROVIDER_MODEL_ID="
set "COPILOT_PROVIDER_WIRE_MODEL="
set "COPILOT_PROVIDER_MAX_PROMPT_TOKENS="
set "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS="
set "COPILOT_MODEL="
if defined COPILOT_CUSTOM_INSTRUCTIONS_DIRS (
    set "COPILOT_CUSTOM_INSTRUCTIONS_DIRS=%~dp0.github\instructions,%COPILOT_CUSTOM_INSTRUCTIONS_DIRS%"
) else (
    set "COPILOT_CUSTOM_INSTRUCTIONS_DIRS=%~dp0.github\instructions"
)
copilot --model auto %*
exit /b %ERRORLEVEL%
