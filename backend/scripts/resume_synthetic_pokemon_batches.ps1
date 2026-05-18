param(
    [int]$BatchSize = 6,
    [int]$TargetPerVariant = 1000,
    [int]$PauseSeconds = 5
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$generator = Join-Path $PSScriptRoot "generate_synthetic_pokemon_cards.py"
$inputDir = Join-Path $repoRoot "backend\data\pokemon\champions_sprites"
$outputDir = Join-Path $repoRoot "backend\data\training_dataset\slot_pokemon_synthetic"
$logDir = Join-Path $repoRoot "backend\test-output"
$logPath = Join-Path $logDir "synthetic_pokemon_batches.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# The generator filters out Mega forms before slicing, leaving 522 source sprites:
# 261 normal + 261 shiny.
$totalSprites = 522

for ($start = 0; $start -lt $totalSprites; $start += $BatchSize) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] Starting batch at sprite index $start"

    & $python $generator `
        --input $inputDir `
        --output $outputDir `
        --target-per-variant $TargetPerVariant `
        --seed 10 `
        --start-index $start `
        --max-sprites $BatchSize 2>&1 |
        Add-Content -Path $logPath

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] Finished batch at sprite index $start"

    Start-Sleep -Seconds $PauseSeconds
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] All batches complete."
