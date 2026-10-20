param(
    [switch]$SkipFrontend,
    [switch]$SkipData,
    [switch]$WithDebugReport,
    [switch]$SkipShortcut,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
$LauncherPath = Join-Path $ProjectRoot "start_champions_insight.ps1"

function Assert-CommandAvailable {
    param(
        [Parameter(Mandatory = $true)][string]$CommandName,
        [Parameter(Mandatory = $true)][string]$InstallHint
    )

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "$CommandName was not found. $InstallHint"
    }
}

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

Assert-CommandAvailable "python" "Install Python 3.11+ from https://www.python.org/downloads/ and enable 'Add Python to PATH'."
if (-not $SkipFrontend) {
    Assert-CommandAvailable "npm" "Install Node.js LTS from https://nodejs.org/."
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating Python virtual environment..."
    Invoke-Step "python" "-m" "venv" ".venv"
}

Write-Host "Upgrading pip..."
Invoke-Step $VenvPython "-m" "pip" "install" "--upgrade" "pip"

Write-Host "Installing backend Python requirements..."
Invoke-Step $VenvPip "install" "-r" "requirements.txt"

if (-not $SkipFrontend) {
    Write-Host "Installing frontend requirements..."
    Push-Location (Join-Path $ProjectRoot "frontend")
    try {
        Invoke-Step "npm" "install"
    }
    finally {
        Pop-Location
    }

    Write-Host "Installing bundled damage calculator requirements..."
    Push-Location (Join-Path $ProjectRoot "backend\pokemon_damage_calculator\damage-calc-master")
    try {
        Invoke-Step "npm" "install"
    }
    finally {
        Pop-Location
    }
}

if (-not $SkipData) {
    Write-Host "Validating shipped AI artifacts..."
    Invoke-Step $VenvPython "backend\scripts\validate_release_artifacts.py"

    Write-Host "Building derived Pokedex and Battle data caches..."
    $dataArgs = @("backend\scripts\build_all_data.py", "--skip-ai-artifacts")
    if ($WithDebugReport) {
        $dataArgs += "--with-debug-report"
    }
    Invoke-Step $VenvPython @dataArgs
}

if (-not $SkipShortcut) {
    Write-Host "Creating desktop launcher shortcut..."
    if (-not (Test-Path -LiteralPath $LauncherPath)) {
        throw "Launcher script was not found: $LauncherPath"
    }

    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktopPath "Champions Insight.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "powershell.exe"
    $shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$LauncherPath`""
    $shortcut.WorkingDirectory = $ProjectRoot
    $shortcut.IconLocation = "powershell.exe,0"
    $shortcut.Description = "Start Champions Insight frontend and backend"
    $shortcut.Save()
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Start with: .\start_champions_insight.ps1"
Write-Host "Desktop shortcut: Champions Insight"

if ($Launch) {
    Write-Host "Launching Champions Insight..."
    & $LauncherPath
}
