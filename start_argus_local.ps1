$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Argus virtual environment not found at '$python'."
    exit 1
}

Push-Location $projectRoot
try {
    & $python -m argus.local_cli @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
