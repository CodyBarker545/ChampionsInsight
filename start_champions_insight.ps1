param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$FrontendUrl = "https://localhost:5173"

function Assert-Exists {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Message
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw $Message
    }
}

Assert-Exists $BackendPython "Python virtual environment was not found. Run .\setup_test_machine.ps1 first."
Assert-Exists (Join-Path $FrontendDir "node_modules") "Frontend dependencies were not found. Run .\setup_test_machine.ps1 first."

$backendCommand = @"
Set-Location '$ProjectRoot'
& '$BackendPython' 'backend\app.py'
"@

$frontendCommand = @"
Set-Location '$FrontendDir'
npm run dev -- --host 0.0.0.0
"@

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $backendCommand
) -WindowStyle Normal

Start-Sleep -Seconds 4

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $frontendCommand
) -WindowStyle Normal

if (-not $NoBrowser) {
    Start-Sleep -Seconds 5
    Start-Process $FrontendUrl
}

Write-Host "Champions Insight is starting."
Write-Host "Frontend: $FrontendUrl"
Write-Host "Backend:  https://localhost:5000"
