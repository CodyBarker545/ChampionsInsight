param(
    [switch]$SkipFrontend,
    [switch]$SkipData,
    [switch]$WithDebugReport
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating Python virtual environment..."
    Invoke-Step "python" "-m" "venv" ".venv"
}

Write-Host "Upgrading pip..."
Invoke-Step $VenvPython "-m" "pip" "install" "--upgrade" "pip"

Write-Host "Installing backend requirements..."
Invoke-Step $VenvPip "install" "-r" "backend\requirements.txt"

if (-not $SkipFrontend) {
    Write-Host "Installing frontend requirements..."
    Push-Location frontend
    try {
        Invoke-Step "npm" "install"
    }
    finally {
        Pop-Location
    }
}

if (-not $SkipData) {
    Write-Host "Building derived data artifacts..."
    $dataArgs = @("backend\scripts\build_all_data.py")
    if ($WithDebugReport) {
        $dataArgs += "--with-debug-report"
    }
    Invoke-Step $VenvPython @dataArgs
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Backend:  .\.venv\Scripts\python.exe backend\app.py"
Write-Host "Frontend: cd frontend; npm run dev"
